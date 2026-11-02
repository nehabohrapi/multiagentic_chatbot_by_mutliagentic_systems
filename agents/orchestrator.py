"""Single Orchestrator Agent for Nexus local-document research system.

Replaces the legacy separate router and decomposer with a unified supervisor
that plans routing, clarification, direct responses, query rewriting, and
sub-question retrieval in a single pass.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from groq import Groq

from config import ROUTER_MODEL
from tools.document_store import get_document_catalog
from .executor import _asks_about_visuals
from .retriever_worker import safe_chat_create, safe_json_loads
from .schemas import OrchestratorPlan, SubQuestion

logger = logging.getLogger(__name__)

ORCHESTRATOR_SYSTEM_PROMPT = """You are the Orchestrator of a document question-answering system. Every user
question reaches you first. You do NOT answer the question. You decide what the
system should search and output one JSON query-understanding plan. Your only job
is to classify, decompose, resolve references, and form standalone search
queries. Do not select downstream agents, answer the question, or verify claims.

INPUTS
- question: the user's latest message
- history: the last few turns (use only to resolve references like "it", "that
  table", "the second one")
- documents: list of {doc_id, title, pages, has_summary_cache}

STEP 1: CHOOSE AN ACTION
- respond_directly: greetings, thanks, questions about the system itself, or
  anything unrelated to the documents. Put the reply in "direct_reply".
- clarify: the question cannot be made answerable even after using history and
  the document list (e.g. "which one?" with several candidate documents and
  no clue). Ask ONE short question in "clarifying_question". Do not clarify if
  you can reasonably infer the meaning.
- retrieve: the answer must come from document chunks, tables, images, or charts.
  If the user demands an image, chart, figure, or visual data, choose "retrieve"
  so it redirects to the Executor Agent.
- summarize: the user wants an overview of a whole document or section.

STEP 2: REWRITE IF AMBIGUOUS (only for retrieve / summarize)
If the question is vague, uses pronouns, or depends on history:
- Produce a standalone "rewritten_query" that is fully self-contained, uses
  specific terms likely to appear in the document, and keeps the user's intent.
- Never add facts or assumptions that are not in the question or history.
- If the question is already clear, copy it unchanged and set
  ambiguity = "none".

STEP 3: DECOMPOSE ONLY IF NEEDED
- Single fact or single topic: exactly 1 sub-question.
- Comparison, numeric across items, or multi-part: split into the smallest
  set of independent sub-questions (max 4). Each must be answerable on its own.
- Do not split just because a sentence is long.

STEP 4: SET RETRIEVAL DETAILS
For each sub-question set:
- doc_ids: which documents to search (all if unknown)
- retrieval: "hybrid" (default) | "table" (numbers, rows, columns) |
  "summary_cache" (whole-document questions when has_summary_cache = true)
- question_type: the type of this sub-question
- direction: the evidence direction, such as "find_specific_fact",
  "identify_entities", "compare_common_attributes", "extract_values",
  "find_causes_and_effects", or "resolve_reference"

STEP 5: SET OUTPUT CONTROLS
- output_format: "short_answer" | "bullets" | "table" | "long_summary"
- needs_verification: true for numeric, legal, medical, financial or
  comparison answers; false for simple lookups and summaries.

RULES
- Minimize work: fewer sub-questions, no retrieval if not needed.
- Never invent document names, page numbers, or facts.
- If the user names a document, restrict doc_ids to it.
- Output valid JSON only, no prose, no markdown.

OUTPUT SCHEMA
{
  "action": "respond_directly | clarify | retrieve | summarize",
  "intent": "factual | comparison | numeric | multi_part | summary | other",
  "ambiguity": "none | resolved_by_rewrite | needs_clarification",
  "rewritten_query": "string",
  "sub_questions": [
    {"id": "q1", "query": "string", "doc_ids": ["string"],
     "retrieval": "hybrid | table | summary_cache",
     "question_type": "string", "direction": "string",
     "entities": ["string"], "attributes": ["string"]}
  ],
  "output_format": "short_answer | bullets | table | long_summary",
  "needs_verification": true,
  "direct_reply": "string or null",
  "clarifying_question": "string or null",
  "reason": "max 15 words"
}
"""


def plan_orchestration(
    question: str,
    history: list[dict[str, str]] | None = None,
    documents: list[dict[str, Any]] | None = None,
) -> OrchestratorPlan:
    """Generate an execution plan for the user question in a single LLM pass."""
    normalized = (question or "").strip()
    if not normalized:
        raise ValueError("Question cannot be empty.")

    if documents is None:
        documents = get_document_catalog()

    if _is_follow_up_question(normalized, history or []):
        recent_history = history or []
        resolved_question = _resolve_follow_up(normalized, recent_history) or normalized
        plan = _fallback_plan(resolved_question, intent_question=normalized)

        previous_workflow = next(
            (
                message.get("workflow")
                for message in reversed(recent_history)
                if message.get("role") == "assistant"
                and isinstance(message.get("workflow"), dict)
            ),
            {},
        )
        previous_plan = previous_workflow.get("orchestrator_plan", {})
        previous_sub_questions = (
            previous_plan.get("sub_questions", [])
            if isinstance(previous_plan, dict)
            else []
        )
        previous_doc_ids = list(dict.fromkeys(
            doc_id
            for sub_question in previous_sub_questions
            if isinstance(sub_question, dict)
            and isinstance(sub_question.get("doc_ids"), list)
            for doc_id in sub_question.get("doc_ids", [])
            if isinstance(doc_id, str) and doc_id
        ))
        if not previous_doc_ids:
            previous_doc_ids = list(dict.fromkeys(
                chunk.get("document_id")
                for chunk in previous_workflow.get("evidence", [])
                if isinstance(chunk, dict)
                and isinstance(chunk.get("document_id"), str)
                and chunk.get("document_id")
            ))
        if not previous_doc_ids:
            previous_user_question = next(
                (
                    message.get("content", "")
                    for message in reversed(recent_history)
                    if message.get("role") == "user"
                ),
                "",
            )
            previous_doc_ids = _match_documents_from_query(
                str(previous_user_question), documents
            )
        for sub_question in plan.sub_questions:
            sub_question.doc_ids = previous_doc_ids
        return plan

    deterministic = _deterministic_plan(normalized, history or [], documents)
    if deterministic is not None:
        return deterministic

    if not _requires_complex_planning(normalized):
        return _fallback_plan(normalized)

    user_payload = {
        "question": normalized,
        "history": (history or [])[-6:],
        "documents": documents,
    }

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = safe_chat_create(
            client,
            model=ROUTER_MODEL,
            messages=[
                {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            max_attempts=1,
            temperature=0.0,
            max_tokens=600,
        )
        raw_text = response.choices[0].message.content or ""
        data = safe_json_loads(raw_text)
        return OrchestratorPlan.model_validate(data)
    except Exception as exc:
        logger.warning("Orchestrator model call failed, using heuristic plan: %s", exc)
        return _fallback_plan(normalized)


def _is_clear_retrieval_question(question: str) -> bool:
    """Avoid planning when a question has an explicit, self-contained intent."""
    lowered = question.casefold()
    words = set(re.findall(r"[a-z0-9]+", lowered))
    ambiguous_terms = {"it", "this", "that", "these", "those", "they", "them"}
    explicit_intent = (
        {"compare", "comparison", "versus", "difference"} & words
        or "vs" in words
        or {"summarize", "summary", "overview", "briefing"} & words
        or {"average", "percentage", "range", "total", "sum", "maximum", "minimum"} & words
    )
    return (
        len(words) >= 4
        and not ambiguous_terms.intersection(words)
        and (
            bool(explicit_intent)
            or not {"what", "which", "how"} & words
            or len(words) >= 6
        )
    )


def _contains_any(text: str, values: tuple[str, ...]) -> bool:
    lowered = text.casefold()
    return any(value in lowered for value in values)


def _is_summary_request(question: str) -> bool:
    """Match requests to produce a summary, not questions that mention one."""
    lowered = question.casefold().strip()
    return bool(
        re.search(r"\bsummari[sz](?:e(?:s|d)?|ing)\b", lowered)
        or re.search(
            r"\b(?:give|get|make|write|create|prepare|generate|provide|show|tell)"
            r"\s+(?:me\s+)?(?:(?:a|an)\s+)?(?:(?:brief|short|quick|concise)\s+)?"
            r"(?:summary|overview|briefing)\b",
            lowered,
        )
        or re.search(r"\b(?:summary|overview|briefing)\s+(?:of|on|for)\b", lowered)
    )


def _is_follow_up_question(
    question: str,
    history: list[dict[str, str]],
) -> bool:
    words = set(re.findall(r"[a-z0-9]+", question.casefold()))
    references = {"it", "this", "that", "these", "those", "they", "them"}
    return bool(history) and (
        bool(references.intersection(words))
        or len(words) <= 4
        or question.casefold().startswith(("what about", "how about"))
    )


def _requires_complex_planning(question: str) -> bool:
    lowered = question.casefold()
    comparison_terms = (
        "compare",
        "comparison",
        "versus",
        "difference",
        "differences",
        "contrast",
        " vs ",
    )
    return _contains_any(lowered, comparison_terms) or len(
        _split_multi_part_question(question)
    ) > 1


def _deterministic_plan(
    question: str,
    history: list[dict[str, str]],
    documents: list[dict[str, Any]],
) -> OrchestratorPlan | None:
    """Build query-only plans for clear requests without spending an LLM call."""
    original_question = question
    lowered = question.casefold()
    words = set(re.findall(r"[a-z0-9]+", lowered))
    ambiguous_terms = {"it", "this", "that", "these", "those", "they", "them"}

    if not question:
        return None
    is_summary_request = _is_summary_request(original_question)
    if not is_summary_request and _has_unresolved_definite_reference(question, history):
        return OrchestratorPlan(
            action="clarify",
            intent="other",
            ambiguity="needs_clarification",
            rewritten_query="",
            clarifying_question=_clarification_for_reference(question),
            reason="Unresolved document reference",
        )
    if re.search(r"\b(?:hello|h+i+|hey|thanks|thank you)\b", lowered) and len(words) <= 4:
        return OrchestratorPlan(
            action="respond_directly",
            intent="other",
            ambiguity="none",
            rewritten_query=question,
            direct_reply="Hello! I am ready to help you research your documents.",
            reason="Clear conversational request",
        )
    if ambiguous_terms.intersection(words):
        resolved = _resolve_follow_up(question, history)
        if resolved is None and not is_summary_request:
            return OrchestratorPlan(
                action="clarify",
                intent="other",
                ambiguity="needs_clarification",
                rewritten_query="",
                clarifying_question=_clarification_for_reference(question),
                reason="Follow-up reference has no conversation context",
            )
        if resolved is not None:
            question = resolved
            lowered = question.casefold()

    if is_summary_request:
        return OrchestratorPlan(
            action="summarize",
            intent="summary",
            ambiguity="resolved_by_rewrite" if question != original_question else "none",
            rewritten_query=question,
            sub_questions=[
                SubQuestion(
                    id="q1", query=question, retrieval="summary_cache",
                    question_type="summary", direction="summarize_context",
                )
            ],
            output_format="long_summary",
            needs_verification=False,
            reason="Clear summary request",
        )

    if _asks_about_visuals(question):
        return _single_query_plan(
            question,
            intent="factual",
            retrieval="hybrid",
            question_type="visual",
            direction="find_visual_assets",
            reason="Visual or image request redirected to executor agent",
        )

    if _requires_complex_planning(question):
        return None

    definition_patterns = (
        r"\bwhat\s+is\s+(?:an?\s+)?(?P<term>[A-Za-z][A-Za-z0-9+#.-]{1,9})[?.!]*$",
        r"\bwhat\s+is\s+mean\s+by\s+(?P<term>.+?)[?.!]*$",
        r"\bwhat\s+do\s+you\s+mean\s+by\s+(?P<term>.+?)[?.!]*$",
        r"\bwhat\s+does\s+(?P<term>.+?)\s+mean[?.!]*$",
        r"\bwhat\s+is\s+the\s+meaning\s+of\s+(?P<term>.+?)[?.!]*$",
        r"\bdefinition\s+of\s+(?P<term>.+?)[?.!]*$",
        r"\bdefine\s+(?P<term>.+?)[?.!]*$",
    )
    definition_term = next(
        (
            match.group("term").strip()
            for pattern in definition_patterns
            if (match := re.search(pattern, question, flags=re.IGNORECASE))
        ),
        None,
    )
    if definition_term:
        return _single_query_plan(
            f"{definition_term} definition",
            question_type="definition",
            direction="find_definition",
            reason="Clear definition request",
        )

    if _contains_any(
        lowered,
        ("average", "percentage", "range", "total", "sum", "maximum",
         "minimum", "how many", "number of"),
    ):
        return _single_query_plan(
            question, intent="numeric", retrieval="table", output_format="table",
            needs_verification=True, question_type="numeric",
            direction="extract_values_and_calculate", reason="Clear numeric request",
        )
    if _contains_any(lowered, ("when ", "what year", "how long", "before ", "after ")):
        return _single_query_plan(
            question, question_type="temporal",
            direction="extract_dates_and_order_events", reason="Clear temporal request",
        )
    if _contains_any(
        lowered,
        ("why ", "cause", "caused", "because", "consequence", "impact", "effect"),
    ):
        return _single_query_plan(
            question, question_type="causal", direction="find_causes_and_effects",
            reason="Clear causal request",
        )
    if _contains_any(lowered, ("how did ", "how does ", "explain ", "describe ")):
        return _single_query_plan(
            question, question_type="explanatory",
            direction="find_explanation_and_evidence",
            reason="Clear explanatory request",
        )
    if _contains_any(lowered, ("define ", "definition of", "what does", "meaning of")):
        return _single_query_plan(
            question, question_type="definition", direction="find_definition",
            reason="Clear definition request",
        )
    if _contains_any(lowered, ("extract ", "give me the ", "provide the ", "fields ", "details of")):
        return _single_query_plan(
            question, question_type="extraction", direction="extract_requested_fields",
            output_format="bullets", reason="Clear extraction request",
        )
    if _contains_any(lowered, ("classify ", "category", "categorize", "type of")):
        return _single_query_plan(
            question, question_type="classification",
            direction="classify_entities_using_evidence",
            output_format="table", reason="Clear classification request",
        )
    if _contains_any(lowered, ("list ", "which ", "all ", "names of")):
        return _single_query_plan(
            question, question_type="list", direction="enumerate_entities",
            output_format="bullets", reason="Clear list request",
        )
    if _contains_any(lowered, ("how to ", "how do i ", "steps ", "process")):
        return _single_query_plan(
            question, question_type="procedural", direction="find_ordered_steps",
            output_format="bullets", reason="Clear procedural request",
        )
    if re.search(r"^\s*(?:is|are|did|does)\b", lowered) or "true or false" in lowered:
        return _single_query_plan(
            question, question_type="verification", direction="verify_claim",
            needs_verification=True, reason="Clear verification request",
        )

    matched_doc_ids = _match_documents_from_query(question, documents)
    target_docs = matched_doc_ids if len(matched_doc_ids) == 1 else None

    if len(words) >= 4:
        return _single_query_plan(
            question, question_type="factual", direction="find_specific_fact",
            reason="Clear factual request",
            doc_ids=target_docs,
        )
    return None


def _match_documents_from_query(query: str, documents: list[dict[str, Any]]) -> list[str]:
    """Find any document IDs explicitly named or referenced via title/alias in the query."""
    lowered = query.casefold()
    matched = []
    for doc in documents or []:
        did = doc.get("doc_id")
        if not did:
            continue
        title = str(doc.get("title") or "")
        aliases = [title] + list(doc.get("aliases") or [])
        for alias in aliases:
            if alias and len(alias) >= 3 and re.search(rf"\b{re.escape(alias.casefold())}\b", lowered):
                matched.append(did)
                break
    return matched


def _single_query_plan(
    question: str,
    *,
    intent: str = "factual",
    retrieval: str = "hybrid",
    output_format: str = "short_answer",
    needs_verification: bool = False,
    question_type: str = "factual",
    direction: str = "find_specific_fact",
    reason: str,
    doc_ids: list[str] | None = None,
) -> OrchestratorPlan:
    return OrchestratorPlan(
        action="retrieve", intent=intent, ambiguity="none",
        rewritten_query=question,
        sub_questions=[SubQuestion(
            id="q1", query=question, retrieval=retrieval,
            doc_ids=doc_ids or [],
            question_type=question_type, direction=direction,
        )],
        output_format=output_format, needs_verification=needs_verification,
        reason=reason,
    )


def _comparison_plan(question: str, documents: list[dict[str, Any]]) -> OrchestratorPlan:
    """Build comparison queries from the user's question without assuming a topic."""
    explicit_entities = re.findall(
        r"\b(?:world war|war|battle|conflict)\s*(?:i{1,3}|iv|v|[0-9]{1,3})\b",
        question.casefold(),
    )
    sub_questions = [
        SubQuestion(
            id="q1",
            query=f"{question} Identify the entities being compared.",
            question_type="entity_discovery", direction="identify_entities",
            entities=explicit_entities,
        ),
        SubQuestion(
            id="q2",
            query=f"{question} Find the comparison points and differences asked for.",
            question_type="comparison", direction="compare_common_attributes",
            entities=explicit_entities,
        ),
    ]
    return OrchestratorPlan(
        action="retrieve", intent="comparison", ambiguity="none",
        rewritten_query=question, sub_questions=sub_questions,
        output_format="table", needs_verification=True,
        reason="Comparison queries are grounded in the user's question",
    )


def _resolve_follow_up(question: str, history: list[dict[str, str]]) -> str | None:
    recent_user = next(
        (
            message.get("content", "").strip()
            for message in reversed(history)
            if message.get("role") == "user" and message.get("content", "").strip()
        ),
        "",
    )
    if not recent_user:
        return None
    return f"{question} Context from the previous question: {recent_user}"


def _has_unresolved_definite_reference(
    question: str,
    history: list[dict[str, str]],
) -> bool:
    """Detect generic references that cannot be grounded without context."""
    if history:
        return False
    # If user asks for visual/image/chart data, redirect to executor instead of clarifying
    if _asks_about_visuals(question):
        return False
    # Let explicit multi-part plans preserve all requested operations; the
    # individual sub-question can still be clarified by retrieval feedback.
    if re.search(r"\b(?:and|also)\b", question.casefold()):
        return False
    return bool(re.search(
        r"\bthe\s+(?:one|other|war|conflict|document|table|figure|chart|"
        r"image|model|crop|paper)\b",
        question.casefold(),
    ))


def _clarification_for_reference(question: str) -> str:
    lowered = question.casefold()
    if re.search(r"\bthe\s+(?:table|figure|chart|image)\b", lowered):
        return "Which table, figure, or image are you referring to?"
    if re.search(r"\bthe\s+(?:war|conflict)\b", lowered):
        return "Which war or conflict are you referring to?"
    if re.search(r"\bthe\s+(?:model|crop|paper)\b", lowered):
        return "Which model, crop, or document are you referring to?"
    return "What specific document or item are you referring to?"


def _split_multi_part_question(question: str) -> list[str]:
    parts = re.split(
        r"\s*(?:;|,\s*and\s+|\s+and\s+also\s+|\s+and\s+|\s+also\s+)\s*",
        question,
        flags=re.IGNORECASE,
    )
    return [part.strip() for part in parts if len(part.strip().split()) >= 3]


def _fallback_plan(
    question: str,
    *,
    intent_question: str | None = None,
) -> OrchestratorPlan:
    """Deterministic fallback if the orchestrator model call fails."""
    lowered = question.lower()
    intent_text = (intent_question or question).casefold()
    if _is_summary_request(intent_text):
        return OrchestratorPlan(
            action="summarize",
            intent="summary",
            ambiguity="none",
            rewritten_query=question,
            output_format="long_summary",
            needs_verification=False,
            reason="Heuristic fallback summary",
        )
    if any(token in lowered for token in ("compare", "versus", "vs", "difference")):
        return OrchestratorPlan(
            action="retrieve",
            intent="comparison",
            ambiguity="none",
            rewritten_query=question,
            sub_questions=[
                SubQuestion(id="q1", query=question, retrieval="hybrid"),
            ],
            output_format="table",
            needs_verification=True,
            reason="Heuristic fallback comparison",
        )
    if any(
        token in lowered
        for token in ("average", "percentage", "range", "total", "sum", "maximum", "minimum")
    ):
        return OrchestratorPlan(
            action="retrieve",
            intent="numeric",
            ambiguity="none",
            rewritten_query=question,
            sub_questions=[
                SubQuestion(id="q1", query=question, retrieval="table"),
            ],
            output_format="table",
            needs_verification=True,
            reason="Heuristic fallback numeric",
        )
    return OrchestratorPlan(
        action="retrieve",
        intent="factual",
        ambiguity="none",
        rewritten_query=question,
        sub_questions=[
            SubQuestion(id="q1", query=question, retrieval="hybrid"),
        ],
        output_format="short_answer",
        needs_verification=True,
        reason="Heuristic fallback factual",
    )


def orchestrator_node(state: dict[str, Any]) -> dict[str, Any]:
    """Single Orchestrator entry node in the LangGraph workflow."""
    question = state["question"]
    history = state.get("history", [])
    plan = plan_orchestration(question, history=history)

    # Sub-question strings for retriever workers
    sub_qs = [sq.query for sq in plan.sub_questions] if plan.sub_questions else [question]

    return {
        "route": plan.intent,
        "intent": plan.intent,
        "orchestrator_plan": plan.model_dump(),
        "action": plan.action,
        "sub_questions": sub_qs,
        "reformulated_question": plan.rewritten_query or question,
        "needs_verification": plan.needs_verification,
        "decision_log": [{
            "stage": "orchestrator",
            "action": plan.action,
            "intent": plan.intent,
            "sub_questions_count": len(sub_qs),
            "reason": plan.reason,
        }],
    }
