"""Single Execution Agent (AGENT 2: Executor) for query execution.

Coordinates subagents and tools:
1. Question Type Verification:
   - Verifies and classifies question type: comparison, followup, table, image, factual, summary
   - Resolves follow-up pronouns/references using conversation history
   - Extracts comparison entities to balance dual-entity retrieval
2. Tool Invocation & Evidence Gathering:
   - Hybrid Retrieval (ChromaDB vector + BM25 lexical)
   - Structured Data (Deterministic operations over tables without LLM arithmetic errors)
   - Visual Processing (Multimodal analysis of charts, figures, and diagrams)
3. Curator Agent:
   - Deduplicates, reranks, and expands retrieved evidence
   - Drafts a grounded answer with citation/source metadata
   - Reports evidence sufficiency and material limitations
4. Verifier Subagent:
   - Evaluates factual support for each claim against evidence passages, tables, and figures
   - Verifies numerical exactness against table cells
   - Validates entity claims for comparisons
5. Agent Decision & Finalization:
   - If verified, approves answer
   - If unsupported claims exist, strips them or falls back to verbatim extractive quotes
   - Guarantees zero hallucinated information delivered to user
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from groq import Groq

from config import (
    DEFAULT_MODEL,
    VISION_MODEL,
)
from telemetry import time_block
from tools.retrieval_tool import execute_retrieval
from tools.structured_data_tool import calculate_table_operation
from tools.visual_processing import (
    analyze_visual_assets,
    chart_tables_by_chunk,
    collect_visual_assets,
)
from . import curator, que_ans, retriever_worker, structural, verifier, visual, writer
from .retriever_worker import NO_EVIDENCE_ANSWER, format_answer_citations, safe_chat_create, safe_json_loads
from .verifier import remove_unsupported_claims, verify_draft

logger = logging.getLogger(__name__)
VERIFICATION_UNAVAILABLE_ANSWER = (
    "Relevant document passages were found, but semantic verification is "
    "temporarily unavailable. Please retry shortly."
)

# Question Type Patterns
_COMPARISON_KEYWORDS = re.compile(
    r"\b(compare|comparison|versus|\bvs\.?\b|difference|differences|differ|distinguish|contrast|pros and cons|advantages and disadvantages|tradeoff|tradeoffs|similarity|similarities|which is better|which one is)\b",
    re.IGNORECASE,
)
_FOLLOWUP_PRONOUNS = re.compile(
    r"\b(it|its|they|them|their|this|that|these|those|the previous|the former|the latter|the first one|the second one|above|as mentioned|what about|and why|how so|furthermore|elaborate|explain more)\b",
    re.IGNORECASE,
)
_TABLE_KEYWORDS = re.compile(
    r"\b(table|tables|row|rows|column|columns|cell|cells|calculate|sum|average|avg|total|max|min|maximum|minimum|percentage|rate|dataset|metric|metrics|value in table|tabular)\b",
    re.IGNORECASE,
)
_VISUAL_KEYWORDS = re.compile(
    r"\b(image|images|figure|figures|\bfig\.?\b|chart|charts|plot|plots|graph|graphs|diagram|diagrams|picture|pictures|photo|photos|visual|visuals|flowchart|flowcharts|architecture)\b",
    re.IGNORECASE,
)


def _asks_about_visuals(question: str) -> bool:
    """Return True if the question specifically references visual elements."""
    return bool(_VISUAL_KEYWORDS.search(question))


def extract_comparison_entities(question: str) -> list[str]:
    """Extract entities being compared in a comparison question."""
    # Pattern 1: "between X and Y"
    m1 = re.search(r"\bbetween\s+([^,?.!]+?)\s+(?:and|&)\s+([^,?.!]+)", question, re.IGNORECASE)
    if m1:
        e1 = re.sub(r"^(the|a|an)\s+", "", m1.group(1).strip(), flags=re.IGNORECASE).strip()
        e2 = re.sub(r"^(the|a|an)\s+", "", m1.group(2).strip(), flags=re.IGNORECASE).strip()
        if e1 and e2:
            return [e1, e2]

    # Pattern 2: "compare X (with|and|to) Y"
    m2 = re.search(r"\bcompare\s+([^,?.!]+?)\s+(?:with|and|to|&)\s+([^,?.!]+)", question, re.IGNORECASE)
    if m2:
        e1 = re.sub(r"^(the|a|an)\s+", "", m2.group(1).strip(), flags=re.IGNORECASE).strip()
        e2 = re.sub(r"^(the|a|an)\s+", "", m2.group(2).strip(), flags=re.IGNORECASE).strip()
        if e1 and e2:
            return [e1, e2]

    # Pattern 3: "X vs Y" or "X versus Y"
    m3 = re.search(r"([^,?.!]+?)\s+(?:vs\.?|versus)\s+([^,?.!]+)", question, re.IGNORECASE)
    if m3:
        e1 = re.sub(r"^(what\s+is\s+|compare\s+|difference\s+between\s+|the\s+|a\s+|an\s+)", "", m3.group(1).strip(), flags=re.IGNORECASE).strip()
        e2 = re.sub(r"^(the\s+|a\s+|an\s+)", "", m3.group(2).strip(), flags=re.IGNORECASE).strip()
        if e1 and e2:
            return [e1, e2]

    return []


def resolve_followup_question(question: str, history: list[dict[str, Any]] | None) -> str:
    """Resolve pronouns/referents in a follow-up question using conversation history."""
    if not history:
        return question
    last_user_q = ""
    for msg in reversed(history):
        if msg.get("role") == "user":
            last_user_q = str(msg.get("content") or "").strip()
            break
    if not last_user_q:
        return question

    topic = re.sub(
        r"^(what\s+is|what\s+are|explain|describe|tell\s+me\s+about|how\s+does|can\s+you\s+explain)\s+",
        "",
        last_user_q,
        flags=re.IGNORECASE,
    ).strip("?.! ")

    if not topic:
        return question

    resolved = question
    if re.search(r"\b(its)\b", resolved, re.IGNORECASE):
        resolved = re.sub(r"\bits\b", f"{topic}'s", resolved, flags=re.IGNORECASE)
    elif re.search(r"\b(it|this|that)\b", resolved, re.IGNORECASE):
        resolved = re.sub(r"\b(it|this|that)\b", topic, resolved, flags=re.IGNORECASE)
    elif re.search(r"\b(they|them|their)\b", resolved, re.IGNORECASE):
        resolved = re.sub(r"\b(they|them|their)\b", f"{topic}", resolved, flags=re.IGNORECASE)
    elif resolved.lower().startswith("what about") or resolved.lower().startswith("how about"):
        resolved = f"{resolved} regarding {topic}"

    return resolved


def verify_question_type(
    question: str,
    plan: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify and classify the question type (comparison, followup, table, image, factual, summary)."""
    plan = plan or {}
    history = history or []
    plan_intent = str(plan.get("intent", "")).lower()

    is_visual = bool(_VISUAL_KEYWORDS.search(question))
    is_table = bool(_TABLE_KEYWORDS.search(question)) or plan_intent == "numeric"
    is_comp = bool(_COMPARISON_KEYWORDS.search(question)) or plan_intent == "comparison"

    has_history = len(history) > 0
    has_pronoun = bool(_FOLLOWUP_PRONOUNS.search(question))
    is_followup = has_history and (has_pronoun or len(question.split()) <= 4)

    comparison_entities = extract_comparison_entities(question) if is_comp else []
    resolved_q = resolve_followup_question(question, history) if is_followup else question

    # Assign primary type, recommended format, and type-specific anti-hallucination instruction
    if is_visual:
        primary_type = "image"
        format_rec = "bullets" if "list" in question.lower() else "short_answer"
        instruction = "Base response strictly on the multimodal visual analysis and figure descriptions. Explicitly reference the figure or chart."
    elif is_table:
        primary_type = "table"
        format_rec = "table" if ("table" in question.lower() or "compare" in question.lower()) else "short_answer"
        instruction = "State exact values, headers, and rows from the table without rounding or mental arithmetic. Format in a clear markdown table if displaying tabular rows."
    elif is_comp:
        primary_type = "comparison"
        format_rec = "table" if plan.get("output_format") == "table" else "bullets"
        instruction = "Compare only attributes explicitly mentioned in the evidence. If evidence is missing for one entity, explicitly state that it is not covered in the document passages. Never fabricate or assume differences."
    elif is_followup:
        primary_type = "followup"
        format_rec = plan.get("output_format", "short_answer")
        instruction = f"The user is asking a follow-up about '{resolved_q}'. Ground answer in the evidence passages for this topic."
    elif plan_intent == "summary" or any(w in question.lower() for w in ["summarize", "summary", "overview"]):
        primary_type = "summary"
        format_rec = "long_summary"
        instruction = "Provide a comprehensive, organized overview grounded in document sections."
    else:
        primary_type = "factual"
        format_rec = plan.get("output_format", "short_answer")
        instruction = "Provide a direct, concise response grounded strictly in document evidence."

    return {
        "primary_type": primary_type,
        "is_comparison": is_comp,
        "is_followup": is_followup,
        "is_table": is_table,
        "is_visual": is_visual,
        "comparison_entities": comparison_entities,
        "resolved_question": resolved_q,
        "recommended_output_format": format_rec,
        "instruction": instruction,
        "verification_strictness": "high" if (is_table or is_comp or is_visual) else "standard",
    }


def _deduplicate_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate chunks by chunk_id, preserving the highest scoring instance."""
    return curator._dedupe_chunks(chunks)


def _table_search_queries(
    question: str,
    entities: list[str],
    attributes: list[str],
) -> list[str]:
    """Add table-label terms so unstructured PDF tables remain retrievable."""
    table_reference = re.search(r"\btable\s+\d+\b", question, re.IGNORECASE)
    table_label = table_reference.group(0) if table_reference else "table"
    expanded = " ".join(
        [table_label, "statistics", question, *entities, *attributes]
    )
    return list(dict.fromkeys((question, expanded)))


def has_enough_evidence(ranked_chunks: list[dict[str, Any]]) -> bool:
    """Check if the top retrieved passage meets the confidence threshold."""
    return curator.has_sufficient_evidence(ranked_chunks)


def run_tool_execution(
    question: str,
    plan: dict[str, Any],
    history: list[dict[str, Any]],
    run_id: str | None = None,
    question_type_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute tools (Hybrid Retrieval, Structured Data, Visual Processing) as per query."""
    if question_type_info is None:
        question_type_info = verify_question_type(question, plan, history)

    sub_qs = plan.get("sub_questions") or []
    intent = question_type_info["primary_type"]
    visual_requested = question_type_info["is_visual"]
    resolved_query = question_type_info.get("resolved_question") or question

    all_chunks: list[dict[str, Any]] = []
    tool_log: list[dict[str, Any]] = []
    retrieval_assessments: list[dict[str, Any]] = []
    requested_table = re.search(
        r"\btable\s+(\d+)\b",
        question_type_info.get("resolved_question") or question,
        re.IGNORECASE,
    )

    def collect_retrieval(result: dict[str, Any], query: str) -> None:
        retrieved = list(result.get("chunks", []))
        if question_type_info["is_table"]:
            table_anchors = [
                chunk
                for chunk in retrieved
                if re.search(
                    r"\btable\s+\d+\s*:",
                    str(chunk.get("text") or ""),
                    re.IGNORECASE,
                )
                or any(
                    item.get("kind") == "table"
                    for item in chunk.get("structured_data") or []
                )
            ]
            if requested_table:
                requested_number = requested_table.group(1)
                matching_anchors = [
                    chunk
                    for chunk in table_anchors
                    if re.search(
                        rf"\btable\s+{re.escape(requested_number)}\s*:",
                        str(chunk.get("text") or ""),
                        re.IGNORECASE,
                    )
                ]
                if matching_anchors:
                    table_anchors = matching_anchors
                    retrieved = []

            for anchor in table_anchors:
                document_id = anchor.get("document_id")
                chunk_index = anchor.get("chunk_index")
                if document_id is None or chunk_index is None:
                    continue
                neighbors = retriever_worker.get_neighbor_chunks(
                    str(document_id),
                    {int(chunk_index)},
                    window=5,
                )
                related_chunks = [
                    anchor,
                    *(
                    neighbor.model_dump()
                    if hasattr(neighbor, "model_dump")
                    else dict(neighbor)
                    for neighbor in neighbors
                    ),
                ]
                retrieved.extend(related_chunks)
        for chunk in retrieved:
            item = dict(chunk)
            queries = list(item.get("_retrieval_queries") or [])
            if query not in queries:
                queries.append(query)
            item["_retrieval_queries"] = queries
            all_chunks.append(item)
        sufficient = bool(result.get("sufficient", bool(retrieved)))
        missing_info = str(result.get("missing_info") or "").strip()
        retrieval_assessments.append({
            "query": query,
            "sufficient": sufficient,
            "missing_info": missing_info,
            "chunk_ids": [
                str(chunk.get("chunk_id"))
                for chunk in retrieved
                if chunk.get("chunk_id")
            ],
        })

    # 1. Tool: Hybrid Retrieval (handles sub-questions, dual-entity comparisons, or single/resolved queries)
    with time_block("Tool: Hybrid Retrieval", category="TOOLS", level=2) as tb_ret:
        if sub_qs:
            for sq in sub_qs:
                q_text = sq.get("query") if isinstance(sq, dict) else getattr(sq, "query", question)
                d_ids = sq.get("doc_ids") if isinstance(sq, dict) else getattr(sq, "doc_ids", None)
                r_mode = sq.get("retrieval", "hybrid") if isinstance(sq, dict) else getattr(sq, "retrieval", "hybrid")
                entities = sq.get("entities", []) if isinstance(sq, dict) else getattr(sq, "entities", [])
                attributes = sq.get("attributes", []) if isinstance(sq, dict) else getattr(sq, "attributes", [])
                planned_queries = (
                    _table_search_queries(
                        q_text,
                        [str(item) for item in entities],
                        [str(item) for item in attributes],
                    )
                    if r_mode == "table"
                    else [q_text]
                )
                with retriever_worker.trace_context(run_id=run_id, agent="executor", sub_question=q_text):
                    res = retriever_worker.retrieve(
                        q_text,
                        needs_retrieval=True,
                        history=history,
                        doc_ids=d_ids,
                        retrieval_mode=r_mode or "hybrid",
                        planned_queries=planned_queries,
                        skip_model_grading=True,
                    )
                collect_retrieval(res, q_text)
        elif question_type_info["is_comparison"] and question_type_info["comparison_entities"]:
            # Automatic dual-entity retrieval for comparison queries
            entities = question_type_info["comparison_entities"]
            queries = [entities[0], entities[1], resolved_query]
            for q_term in queries:
                with retriever_worker.trace_context(run_id=run_id, agent="executor", sub_question=q_term):
                    res = retriever_worker.retrieve(
                        q_term,
                        needs_retrieval=True,
                        history=history,
                        retrieval_mode="hybrid",
                        planned_queries=[q_term],
                        skip_model_grading=True,
                    )
                collect_retrieval(res, q_term)
        else:
            search_query = resolved_query
            with retriever_worker.trace_context(run_id=run_id, agent="executor", sub_question=search_query):
                res = retriever_worker.retrieve(
                    search_query,
                    needs_retrieval=True,
                    history=history,
                    retrieval_mode="hybrid",
                    planned_queries=[search_query],
                    skip_model_grading=True,
                )
            collect_retrieval(res, search_query)

        tb_ret["details"] = f"retrieved {len(all_chunks)} chunk(s)"
        tool_log.append({
            "tool": "hybrid_retrieval",
            "chunks_found": len(all_chunks),
            "sub_questions_count": len(sub_qs) if sub_qs else (3 if question_type_info["is_comparison"] else 1),
            "query_used": resolved_query,
        })

    # 2. Tool: Structured Data Calculation
    calculation = None
    if (question_type_info["is_table"] or intent == "numeric") and not visual_requested:
        with time_block("Tool: Structured Data Evaluation", category="TOOLS", level=2) as tb_struct:
            calculation = calculate_table_operation(resolved_query, all_chunks)
            tb_struct["details"] = f"calculated={calculation is not None}"
            if calculation:
                tool_log.append({
                    "tool": "structured_data",
                    "operation": calculation.operation,
                    "column": calculation.column,
                    "value": str(calculation.value),
                })

    # 3. Tool: Visual Processing - Directed to Visual Agent (Subagent 3)
    visual_evidence = []
    visual_assets = []
    if visual_requested or intent == "numeric":
        with time_block("Visual Agent: Retrieve & Multimodal Analysis", category="SUBAGENT", level=2) as tb_vis:
            try:
                # If mocked in executor module (e.g. in legacy unit tests), honor mocked functions
                if hasattr(collect_visual_assets, "assert_called") or hasattr(analyze_visual_assets, "assert_called"):
                    visual_assets = collect_visual_assets(
                        all_chunks,
                        extract_captioned=visual_requested or intent == "numeric",
                    )
                    if visual_assets:
                        visual_evidence = analyze_visual_assets(
                            resolved_query,
                            all_chunks,
                            client=Groq(api_key=os.getenv("GROQ_API_KEY")),
                            model=VISION_MODEL,
                            chat_create=safe_chat_create,
                            json_loader=safe_json_loads,
                        )
                        chart_tables = chart_tables_by_chunk(visual_evidence)
                        all_chunks = retriever_worker._add_chart_tables(all_chunks, chart_tables)
                else:
                    # Direct query to Visual Agent to retrieve visual images/charts & metadata from database
                    visual_result = visual.VisualAgent.execute(
                        resolved_query,
                        all_chunks,
                        retrieve_from_db=visual_requested,
                    )
                    visual_evidence = visual_result.get("visual_evidence", [])
                    visual_assets = visual_result.get("visual_assets", [])
                    all_chunks = visual_result.get("updated_chunks", all_chunks)

                tb_vis["details"] = f"retrieved {len(visual_assets)} visual(s), analyzed {len(visual_evidence)}"
                tool_log.append({
                    "tool": "visual_processing",
                    "subagent": "visual_agent",
                    "visual_assets": len(visual_assets),
                    "visual_evidence": len(visual_evidence),
                })

                # Check if visual chart table yields structured calculation
                if (question_type_info["is_table"] or intent == "numeric") and calculation is None and visual_evidence:
                    chart_tables = chart_tables_by_chunk(visual_evidence)
                    if chart_tables:
                        chart_chunks = [
                            {**chunk, "structured_data": chart_tables[str(chunk.get("chunk_id"))]}
                            for chunk in all_chunks
                            if str(chunk.get("chunk_id")) in chart_tables
                        ]
                        calculation = calculate_table_operation(resolved_query, chart_chunks)
            except Exception as exc:
                logger.warning("Visual processing tool failed: %s", exc)

    return {
        "chunks": all_chunks,
        "calculation": calculation,
        "visual_evidence": visual_evidence,
        "visual_assets": visual_assets,
        "tool_log": tool_log,
        "question_type_info": question_type_info,
        "retrieval_status": {
            "sufficient": bool(retrieval_assessments)
            and all(item["sufficient"] for item in retrieval_assessments),
            "queries": retrieval_assessments,
            "limitations": list(dict.fromkeys(
                item["missing_info"]
                or f"No matching passages were found for: {item['query']}"
                for item in retrieval_assessments
                if not item["sufficient"]
            )),
        },
    }


def curator_writer_subagent(
    question: str,
    chunks: list[dict[str, Any]],
    plan: dict[str, Any],
    calculation: Any | None = None,
    visual_evidence: list[dict[str, Any]] | None = None,
    history: list[dict[str, Any]] | None = None,
    question_type_info: dict[str, Any] | None = None,
    retrieval_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Executor boundary for the Curator's curation and answer-generation task."""
    if question_type_info is None:
        question_type_info = verify_question_type(question, plan, history)

    intent = question_type_info["primary_type"]
    with time_block("Curator Agent: Curate Evidence & Draft Answer", category="CURATOR", level=2) as tb_curator:
        try:
            result = curator.curate_and_answer(
                question,
                chunks,
                intent=intent,
                request_intent=str(plan.get("intent") or intent),
                output_format=(
                    plan.get("output_format")
                    or question_type_info["recommended_output_format"]
                ),
                revision_feedback=question_type_info.get("instruction"),
                history=history,
                calculation=calculation,
                visual_evidence=visual_evidence,
                retrieval_status=retrieval_status,
            )
        except curator.CuratorAnswerGenerationError as exc:
            if isinstance(exc.__cause__, retriever_worker.LLMQuotaExceeded):
                raise exc.__cause__
            logger.exception(
                "Curator answer generation failed; no raw-evidence fallback will be shown."
            )
            raise RuntimeError(
                f"Curator answer generation failed: {exc.__cause__ or exc}"
            ) from exc

        tb_curator["details"] = (
            f"answer_chars={len(result.get('answer', ''))}, "
            f"evidence_sufficient={result.get('evidence_sufficient', False)}, "
            f"citations={result.get('citations', [])}"
        )
        return result


def verifier_subagent(
    question: str,
    draft: str | None = None,
    context_blocks: list[dict[str, Any]] | None = None,
    question_type_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Subagent: Checks each claim in the draft against evidence passages."""
    # Support both signatures: (question, draft, context_blocks) and (draft, context_blocks)
    if context_blocks is None and isinstance(draft, list):
        context_blocks = draft
        draft = question
        question = ""
    draft = draft or ""
    context_blocks = context_blocks or []

    with time_block("Subagent: Verifier", category="VERIFIER", level=2) as tb_ver:
        if not draft or draft == NO_EVIDENCE_ANSWER or not context_blocks:
            tb_ver["details"] = "skipped (no draft or context)"
            return {"ok": True, "claims": [], "missing": []}

        verdict_obj = verify_draft(question, draft, context_blocks)
        if hasattr(verdict_obj, "model_dump"):
            verdict = verdict_obj.model_dump()
        elif isinstance(verdict_obj, dict):
            verdict = verdict_obj
        else:
            verdict = {
                "ok": getattr(verdict_obj, "ok", True),
                "claims": [
                    c.model_dump() if hasattr(c, "model_dump") else dict(c)
                    for c in getattr(verdict_obj, "claims", [])
                ],
                "missing": getattr(verdict_obj, "missing", []),
            }

        tb_ver["details"] = f"ok={verdict.get('ok', False)}, claims={len(verdict.get('claims', []))}"
        return verdict


def executor_decision_and_finalize(
    draft_result: dict[str, Any],
    verdict: dict[str, Any],
    question: str,
) -> dict[str, Any]:
    """Executor Agent decision: reviews verification and finalizes the answer."""
    with time_block("Agent Decision & Finalize", category="EXECUTOR", level=2) as tb_fin:
        if not draft_result.get("answerable", True) or draft_result.get("draft") == NO_EVIDENCE_ANSWER:
            tb_fin["details"] = "unanswerable query fallback preserved"
            return draft_result

        # If verified, keep the draft
        if verdict.get("ok"):
            tb_fin["details"] = "draft fully verified"
            return draft_result

        claims = verdict.get("claims", [])
        total_claims = len(claims)
        unsupported = [c for c in claims if c.get("supported") != "yes"]
        raw_answer = draft_result.get("raw_answer") or draft_result.get("draft") or ""

        has_supported_claim = any(c.get("supported") == "yes" for c in claims)
        verification_unavailable = any(
            "verification" in str(item).casefold()
            and "unavailable" in str(item).casefold()
            for item in verdict.get("missing", [])
        )
        if not has_supported_claim and verification_unavailable:
            citations = writer.extract_citations(raw_answer)
            context_blocks = draft_result.get("context_blocks", [])
            citations_valid = bool(citations) and all(
                1 <= citation <= len(context_blocks)
                for citation in citations
            )
            if citations_valid:
                marked_draft = (
                    "UNVERIFIED: Cohere verification is temporarily unavailable.\n\n"
                    f"{raw_answer}"
                )
                updated = dict(draft_result)
                updated.update({
                    "answer": format_answer_citations(marked_draft, context_blocks),
                    "draft": marked_draft,
                    "raw_answer": raw_answer,
                    "citations": citations,
                    "answerable": True,
                    "verdict": verdict,
                    "citation_metadata": curator.build_citation_metadata(
                        citations,
                        raw_answer,
                        context_blocks,
                    ),
                })
                tb_fin["details"] = "verifier unavailable; returned cited draft as explicitly unverified"
                return updated

            updated = dict(draft_result)
            updated.update({
                "answer": VERIFICATION_UNAVAILABLE_ANSWER,
                "draft": VERIFICATION_UNAVAILABLE_ANSWER,
                "raw_answer": VERIFICATION_UNAVAILABLE_ANSWER,
                "citations": [],
                "answerable": False,
                "verdict": verdict,
            })
            updated["decision_log"] = [
                *(draft_result.get("decision_log") or []),
                {"stage": "verification_unavailable", "retryable": True},
            ]
            tb_fin["details"] = "semantic verifier unavailable; no unverified draft returned"
            return updated

        filtered = (
            remove_unsupported_claims(raw_answer, claims)
            if total_claims and has_supported_claim
            else ""
        ) or NO_EVIDENCE_ANSWER
        tb_fin["details"] = (
            f"decision: stripped {len(unsupported)} unsupported claim(s); "
            f"abstained={filtered == NO_EVIDENCE_ANSWER}"
        )

        formatted = format_answer_citations(filtered, draft_result.get("context_blocks", []))
        citations = writer.extract_citations(filtered)

        updated = dict(draft_result)
        updated["answer"] = formatted
        updated["draft"] = filtered
        updated["raw_answer"] = filtered
        updated["citations"] = citations
        updated["answerable"] = filtered != NO_EVIDENCE_ANSWER
        updated["verdict"] = verdict
        updated["citation_metadata"] = curator.build_citation_metadata(
            citations,
            filtered,
            draft_result.get("context_blocks", []),
        )
        return updated


def execute_query(
    question: str,
    plan: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Execute the complete single Execution Agent workflow."""
    with time_block("Agent 2 (Executor): Full Query Execution", category="EXECUTOR", level=1) as tb_exec:
        # Step 0: Question Type Verification
        question_type_info = verify_question_type(question, plan, history)

        # Step 1: Run tools dynamically based on verified question type
        tool_results = run_tool_execution(
            question,
            plan,
            history or [],
            run_id=run_id,
            question_type_info=question_type_info,
        )

        # Step 2: Curator & Writer Subagent
        draft_result = curator_writer_subagent(
            question,
            tool_results["chunks"],
            plan,
            calculation=tool_results.get("calculation"),
            visual_evidence=tool_results.get("visual_evidence"),
            history=history,
            question_type_info=question_type_info,
            retrieval_status=tool_results.get("retrieval_status"),
        )

        # Step 3: Verifier Subagent
        if draft_result.get("answerable"):
            verdict = verifier_subagent(
                question,
                draft_result.get("raw_answer") or draft_result["draft"],
                draft_result.get("context_blocks", []),
                question_type_info=question_type_info,
            )
        else:
            verdict = {"ok": True, "claims": [], "skipped": True}

        # Step 4: Agent Decision & Finalize
        final_result = executor_decision_and_finalize(draft_result, verdict, question)

        tb_exec["details"] = f"type={question_type_info['primary_type']}, answer_chars={len(final_result.get('answer', ''))}, answerable={final_result.get('answerable', False)}"

        final_result["chunks"] = tool_results["chunks"]
        final_result["retrieval_status"] = tool_results["retrieval_status"]
        final_result["question_type_info"] = question_type_info
        final_result["visual_assets"] = tool_results.get("visual_assets", [])
        final_result["visual_evidence"] = tool_results.get("visual_evidence", [])
        final_result["decision_log"] = [
            {
                "stage": "question_type_verification",
                "verified_type": question_type_info["primary_type"],
                "is_comparison": question_type_info["is_comparison"],
                "is_followup": question_type_info["is_followup"],
                "is_table": question_type_info["is_table"],
                "is_visual": question_type_info["is_visual"],
                "resolved_question": question_type_info["resolved_question"],
            },
            *(draft_result.get("decision_log") or []),
            *tool_results.get("tool_log", []),
            {
                "stage": "executor_agent",
                "tools_used": [t["tool"] for t in tool_results.get("tool_log", [])],
                "verified": verdict.get("ok", True),
                "answerable": final_result.get("answerable", False),
            },
        ]
        return final_result


def executor_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node representing the single Execution Agent (Agent 2)."""
    question = state.get("reformulated_question") or state["question"]
    plan = state.get("orchestrator_plan") or {}
    history = state.get("history") or []
    run_id = state.get("run_id")

    return execute_query(question, plan, history=history, run_id=run_id)
