"""LangGraph orchestration for Nexus retrieval agents."""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from config import (
    MAX_TOTAL_LLM_CALLS,
    MIN_EVIDENCE_SCORE,
    SUMMARY_MAX_CONTEXT_CHARS,
)

from telemetry import aggregate_trace_usage, capture_timings
from . import curator, retriever_worker, writer
from .executor import _asks_about_visuals, executor_node
from .input_guardrails import blocked_notice_node, input_guardrail_node
from .orchestrator import orchestrator_node
from gurdrails.nemo_output_rails import nemo_output_rails_node
from .router import reformulate_query
from .schemas import Verdict
from .summarizer import summarize_document
from .verifier import remove_unsupported_claims, verifier_node
from memory.state_store import record_run

logger = logging.getLogger(__name__)


class MultiAgentState(TypedDict, total=False):
    """State shared by orchestrator, retrieval, and answer-generation nodes."""

    question: str
    reformulated_question: str
    history: list[dict[str, str]]
    route: str
    action: str
    is_safe: bool
    input_cleared: bool
    orchestrator_plan: dict[str, Any]
    sub_questions: list[str]
    worker_question: str
    worker_doc_ids: list[str] | None
    worker_retrieval_mode: str
    evidence: Annotated[list[dict[str, Any]], operator.add]
    curated: list[dict[str, Any]]
    context_blocks: list[dict[str, Any]]
    draft: str
    raw_answer: str
    answer: str
    verdict: dict[str, Any]
    verification_report: dict[str, Any]
    rounds: int
    llm_calls: int
    summary_model_calls: int
    decision_log: Annotated[list[dict[str, Any]], operator.add]
    answerable: bool
    citations: list[int]
    chunks: list[dict[str, Any]]
    ranked_chunks: list[dict[str, Any]]
    rerank_confidence: str
    sufficient: bool
    missing_info: str
    intent: str
    search_queries: list[str]
    attempts: int
    run_id: str
    clarification: str
    question_type_info: dict[str, Any]
    visual_assets: list[Any]
    visual_evidence: list[Any]
    timings: list[dict[str, Any]]
    token_usage: dict[str, Any]
    output_validation: dict[str, Any]
    summary_coverage: dict[str, Any] | None



def _worker_node(state: MultiAgentState) -> dict[str, Any]:
    """Run the unchanged retrieval subgraph for one question."""
    worker_question = (
        state.get("worker_question")
        or state.get("reformulated_question")
        or state["question"]
    )
    doc_ids = state.get("worker_doc_ids")
    retrieval_mode = state.get("worker_retrieval_mode")
    if doc_ids is None or retrieval_mode is None:
        plan = state.get("orchestrator_plan") or {}
        sub_qs = plan.get("sub_questions") or []
        if sub_qs:
            first_sq = sub_qs[0]
            if isinstance(first_sq, dict):
                doc_ids = doc_ids if doc_ids is not None else first_sq.get("doc_ids")
                retrieval_mode = retrieval_mode or first_sq.get("retrieval", "hybrid")
            else:
                doc_ids = doc_ids if doc_ids is not None else getattr(first_sq, "doc_ids", None)
                retrieval_mode = retrieval_mode or getattr(first_sq, "retrieval", "hybrid")

    with retriever_worker.trace_context(
        run_id=state.get("run_id"),
        agent="retriever_worker",
        sub_question=worker_question,
    ):
        result = retriever_worker.retrieve(
            worker_question,
            needs_retrieval=True,
            history=state.get("history", []),
            doc_ids=doc_ids,
            retrieval_mode=retrieval_mode or "hybrid",
            planned_queries=[worker_question],
            skip_model_grading=True,
        )
    evidence = list(result.get("chunks", []))
    return {
        "evidence": evidence,
        "decision_log": [{
            "stage": "retriever_worker",
            "question": worker_question,
            "chunks": len(evidence),
            "attempts": result.get("attempts", 0),
            "doc_ids": doc_ids,
            "retrieval_mode": retrieval_mode or "hybrid",
        }],
    }


def _fan_out(state: MultiAgentState) -> list[Send]:
    """Create one synchronous LangGraph worker invocation per sub-question."""
    plan = state.get("orchestrator_plan") or {}
    sub_qs = plan.get("sub_questions") or []
    if sub_qs:
        sends = []
        for sq in sub_qs:
            q_text = sq.get("query") if isinstance(sq, dict) else getattr(sq, "query", state["question"])
            d_ids = sq.get("doc_ids") if isinstance(sq, dict) else getattr(sq, "doc_ids", None)
            r_mode = sq.get("retrieval", "hybrid") if isinstance(sq, dict) else getattr(sq, "retrieval", "hybrid")
            sends.append(
                Send(
                    "retriever_worker",
                    {
                        "question": state.get("reformulated_question", state["question"]),
                        "history": state.get("history", []),
                        "worker_question": q_text,
                        "worker_doc_ids": d_ids,
                        "worker_retrieval_mode": r_mode,
                    },
                )
            )
        return sends

    questions = state.get("sub_questions") or [state["question"]]
    return [
        Send(
            "retriever_worker",
            {
                "question": state.get("reformulated_question", state["question"]),
                "history": state.get("history", []),
                "worker_question": question,
            },
        )
        for question in questions
    ]



def _direct_reply_node(state: MultiAgentState) -> dict[str, Any]:
    plan = state.get("orchestrator_plan") or {}
    reply = plan.get("direct_reply") or "Hello! I am ready to help you research and answer questions about your documents."
    return {
        "answer": reply,
        "draft": reply,
        "raw_answer": reply,
        "citations": [],
        "answerable": True,
        "decision_log": [{"stage": "direct_reply", "reply": reply}],
    }


def _clarify_node(state: MultiAgentState) -> dict[str, Any]:
    plan = state.get("orchestrator_plan") or {}
    question = plan.get("clarifying_question") or "Could you please specify which document or topic you mean?"
    return {
        "answer": question,
        "draft": question,
        "raw_answer": question,
        "citations": [],
        "answerable": False,
        "decision_log": [{"stage": "clarify", "question": question}],
    }


def _after_orchestrator(state: MultiAgentState) -> str | list[Send]:
    """Single Orchestrator routing decision."""
    plan = state.get("orchestrator_plan") or {}
    action = plan.get("action") or state.get("action")
    question = state.get("question", "")

    # Ensure all visual/image/chart queries redirect to executor agent
    if (
        _asks_about_visuals(question)
        or plan.get("intent") in {"visual", "image"}
        or any(
            (getattr(sq, "question_type", "") == "visual" if hasattr(sq, "question_type") else sq.get("question_type") == "visual")
            for sq in (plan.get("sub_questions") or [])
        )
    ):
        return "executor"

    if action == "respond_directly":
        return "direct_reply"
    if action == "clarify":
        return "clarify"
    if action == "summarize" or state.get("route") == "summary":
        return "summarizer"

    return "executor"


def _after_router(state: MultiAgentState) -> str | list[Send]:
    """Backward-compatible router helper for tests."""
    return _after_orchestrator(state)


def has_enough_evidence(ranked_chunks: list[dict[str, Any]]) -> bool:
    """Apply the initial hybrid-reranker floor before generating an answer."""
    if not ranked_chunks:
        return False
    best = ranked_chunks[0]
    score = float(
        best.get(
            "rerank_score",
            best.get("hybrid_score")
            if best.get("hybrid_score") is not None
            else 1.0 - float(best.get("distance", 1.0)),
        )
    )
    return score >= MIN_EVIDENCE_SCORE


def _writer_node(state: MultiAgentState) -> dict[str, Any]:
    """Curate retrieved evidence and generate a grounded answer in one step."""
    best_by_id: dict[str, dict[str, Any]] = {}
    for chunk in state.get("evidence", []):
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            continue
        score = float(
            chunk.get("hybrid_score")
            if chunk.get("hybrid_score") is not None
            else 1.0 - float(chunk.get("distance", 1.0))
        )
        previous = best_by_id.get(chunk_id)
        if previous is None or score > float(previous.get("_merge_score", -1.0)):
            item = dict(chunk)
            item["_merge_score"] = score
            best_by_id[chunk_id] = item

    evidence = [
        {key: value for key, value in chunk.items() if key != "_merge_score"}
        for chunk in best_by_id.values()
    ]

    intent = state.get("intent") or state.get("route", "")
    if intent in {"comparison", "multi_part", "numeric"}:
        dynamic_top_k = 8
    elif intent == "summary":
        dynamic_top_k = 15
    else:
        dynamic_top_k = 8

    curator_query = state.get("reformulated_question") or state["question"]
    ranked, confidence = curator.rerank_chunks(curator_query, evidence, top_k=dynamic_top_k)
    context = curator.expand_chunks(
        ranked,
        max_chars=(
            SUMMARY_MAX_CONTEXT_CHARS
            if state.get("route") == "summary"
            else 14000
        ),
    )
    if not context and state.get("context_blocks"):
        context = state["context_blocks"]
    top_score = (
        float(
            ranked[0].get(
                "rerank_score",
                ranked[0].get("hybrid_score")
                if ranked[0].get("hybrid_score") is not None
                else 1.0 - float(ranked[0].get("distance", 1.0)),
            )
        )
        if ranked
        else 0.0
    )
    if not context or (ranked and not has_enough_evidence(ranked)):
        fallback = retriever_worker.NO_EVIDENCE_ANSWER
        return {
            "answer": fallback,
            "draft": fallback,
            "raw_answer": fallback,
            "citations": [],
            "chunks": evidence,
            "curated": ranked,
            "ranked_chunks": ranked,
            "context_blocks": context,
            "rerank_confidence": confidence,
            "answerable": False,
            "llm_calls": _current_llm_calls(),
            "decision_log": [{
                "stage": "curator_writer",
                "input_chunks": len(state.get("evidence", [])),
                "deduped_chunks": len(evidence),
                "context_chunks": len(context),
                "top_score": top_score,
                "minimum_score": MIN_EVIDENCE_SCORE,
                "answerable": False,
            }],
        }
    retrieval = dict(state)
    if context:
        retrieval["chunks"] = evidence
        retrieval["curated"] = ranked
        retrieval["ranked_chunks"] = ranked
        retrieval["context_blocks"] = context
    else:
        context = retrieval.get("context_blocks", [])
        retrieval["context_blocks"] = context
        retrieval["ranked_chunks"] = state.get("ranked_chunks", [])
    retrieval["rerank_confidence"] = confidence
    verdict = state.get("verdict") or {}
    feedback = _revision_feedback(verdict)
    writer_question = state.get("reformulated_question") or state["question"]
    plan = state.get("orchestrator_plan") or {}
    output_format = plan.get("output_format", "short_answer")
    # The plan is authoritative at this boundary. Preserve the numeric intent
    # even when a nested retrieval worker returned a generic retrieval intent.
    retrieval["intent"] = plan.get("intent") or state.get("intent") or state.get("route")
    try:
        generated = writer.answer_from_retrieval(
            writer_question,
            retrieval,
            revision_feedback=feedback,
            output_format=output_format,
            return_details=True,
        )
    except retriever_worker.LLMBudgetExceeded:
        raw_answer = state.get("raw_answer") or state.get("answer") or (
            "I couldn't complete a grounded answer within the model-call budget."
        )
        generated = {
            "answer": writer.strip_source_references(raw_answer),
            "raw_answer": raw_answer,
            "citations": writer.extract_citations(raw_answer),
        }
    if isinstance(generated, dict):
        answer = str(generated.get("answer") or "")
        raw_answer = str(generated.get("raw_answer") or answer)
        citations = list(generated.get("citations") or [])
    else:
        raw_answer = str(generated)
        answer = writer.strip_source_references(raw_answer)
        citations = writer.extract_citations(raw_answer)
    context = retrieval.get("context_blocks", [])
    return {
        "answer": answer,
        "draft": answer,
        "raw_answer": raw_answer,
        "chunks": evidence,
        "curated": ranked,
        "ranked_chunks": ranked,
        "context_blocks": context or retrieval.get("context_blocks", []),
        "rerank_confidence": confidence,
        "answerable": bool(context or retrieval.get("context_blocks")),
        "llm_calls": _current_llm_calls(),
        "citations": citations,
        "decision_log": [{
            "stage": "curator_writer",
            "input_chunks": len(state.get("evidence", [])),
            "deduped_chunks": len(evidence),
            "context_chunks": len(context or retrieval.get("context_blocks", [])),
            "context_chunk_ids": [
                chunk.get("chunk_id")
                for chunk in (context or retrieval.get("context_blocks", []))
                if chunk.get("chunk_id")
            ],
            "top_k": dynamic_top_k,
            "rerank_confidence": confidence,
            "citations": citations,
            "output_format": output_format,
        }],
    }



def _current_llm_calls() -> int:
    budget = retriever_worker._llm_call_budget.get()
    return int(budget["calls"]) if budget is not None else 0


def _revision_feedback(verdict: dict[str, Any]) -> str | None:
    if not verdict:
        return None
    unsupported = [
        claim.get("claim", "")
        for claim in verdict.get("claims", [])
        if claim.get("supported") != "yes"
    ]
    missing = [item for item in verdict.get("missing", []) if item]
    feedback = []
    if unsupported:
        feedback.append(
            "Remove or qualify claims that are not fully supported: "
            + "; ".join(unsupported)
        )
    if missing:
        feedback.append("Evidence is missing for: " + "; ".join(missing))
    return " ".join(feedback) or None


def _after_input_guardrail(state: MultiAgentState) -> str:
    """Route based on input security boundary clearance."""
    if not state.get("input_cleared", True) or not state.get("is_safe", True):
        return "blocked_notice"
    return "orchestrator"


def _after_writer(state: MultiAgentState) -> str:
    if not state.get("answerable", True):
        return "nemo_output_rails"
    return "verifier"


def _after_verifier(state: MultiAgentState) -> str | list[Send]:
    verdict = state.get("verdict") or {}
    if verdict.get("ok"):
        return "nemo_output_rails"
    return "finalize"


def _finalize_node(state: MultiAgentState) -> dict[str, Any]:
    """Remove claims that remained unsupported at the verification limit."""
    verdict = state.get("verdict") or {}
    claims = verdict.get("claims", [])
    raw_answer = state.get("raw_answer") or state.get("draft") or state.get("answer", "")
    filtered = remove_unsupported_claims(raw_answer, claims)
    if not filtered:
        filtered = retriever_worker.NO_EVIDENCE_ANSWER
    return {
        "answer": retriever_worker.format_answer_citations(
            filtered,
            state.get("context_blocks", []),
        ),
        "draft": filtered,
        "raw_answer": filtered,
        "citations": writer.extract_citations(filtered),
        "answerable": filtered != retriever_worker.NO_EVIDENCE_ANSWER,
    }


def _summarizer_node(state: MultiAgentState) -> dict[str, Any]:
    """Run full-document Gemini map/reduce summarization."""
    plan = state.get("orchestrator_plan") or {}
    sub_questions = plan.get("sub_questions") or []
    first_sub_question = sub_questions[0] if sub_questions else {}
    doc_ids = (
        first_sub_question.get("doc_ids", [])
        if isinstance(first_sub_question, dict)
        else getattr(first_sub_question, "doc_ids", [])
    )
    result = summarize_document(
        state["question"],
        document_query=state.get("reformulated_question"),
        doc_ids=doc_ids,
        history=state.get("history", []),
        run_id=state.get("run_id"),
    )
    clarification = result.get("clarifying_question")
    return {
        "answer": clarification or result["answer"] or (
            "I couldn't find a matching local document to summarize."
        ),
        "draft": clarification or result["answer"],
        "raw_answer": clarification or result.get("raw_answer", result["answer"]),
        "citations": [] if clarification else result["citations"],
        "chunks": result["chunks"],
        "context_blocks": result["chunks"],
        "answerable": bool(result["answer"]) and not clarification,
        "clarification": clarification or "",
        "summary_model_calls": result.get("model_calls", 0),
        "decision_log": [{
            "stage": "summarizer",
            "cached": result["cached"],
            "chunks": len(result["chunks"]),
            "model_calls": result.get("model_calls", 0),
            "clarification": bool(clarification),
        }],
    }


def _after_summarizer(state: MultiAgentState) -> str:
    """Send summaries and clarifications directly to the output rails."""
    return END


_input_guardrail_node = input_guardrail_node
_blocked_notice_node = blocked_notice_node
_nemo_output_rails_node = nemo_output_rails_node


def _build_graph():
    graph = StateGraph(MultiAgentState)
    graph.add_node("input_guardrails", input_guardrail_node)
    graph.add_node("blocked_notice", blocked_notice_node)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("direct_reply", _direct_reply_node)
    graph.add_node("clarify", _clarify_node)
    graph.add_node("executor", executor_node)
    graph.add_node("retriever_worker", _worker_node)
    graph.add_node("writer", _writer_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("finalize", _finalize_node)
    graph.add_node("summarizer", _summarizer_node)
    graph.add_node("nemo_output_rails", nemo_output_rails_node)

    graph.add_edge(START, "input_guardrails")
    graph.add_conditional_edges(
        "input_guardrails",
        _after_input_guardrail,
        {
            "blocked_notice": "blocked_notice",
            "orchestrator": "orchestrator",
        },
    )
    graph.add_edge("blocked_notice", "nemo_output_rails")
    graph.add_conditional_edges(
        "orchestrator",
        _after_orchestrator,
        {
            "direct_reply": "direct_reply",
            "clarify": "clarify",
            "summarizer": "summarizer",
            "executor": "executor",
            "retriever_worker": "retriever_worker",
        },
    )
    graph.add_edge("direct_reply", "nemo_output_rails")
    graph.add_edge("clarify", "nemo_output_rails")
    graph.add_edge("executor", "nemo_output_rails")
    graph.add_conditional_edges(
        "summarizer",
        _after_summarizer,
        {END: "nemo_output_rails"},
    )
    graph.add_edge("retriever_worker", "writer")
    graph.add_conditional_edges(
        "writer",
        _after_writer,
        {"verifier": "verifier", "nemo_output_rails": "nemo_output_rails"},
    )
    graph.add_conditional_edges(
        "verifier",
        _after_verifier,
        {"writer": "writer", "finalize": "finalize", "nemo_output_rails": "nemo_output_rails"},
    )
    graph.add_edge("finalize", "nemo_output_rails")
    graph.add_edge("nemo_output_rails", END)
    return graph.compile()


multi_agent_graph = _build_graph()


def run_multi_agent(
    question: str,
    *,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Run the router, retrieval, combined curation/answer, and verification graph."""
    normalized = (question or "").strip()
    if not normalized:
        raise ValueError("Question cannot be empty.")
    run_id = uuid4().hex
    record_run(run_id, normalized, status="started")
    with (
        retriever_worker.llm_budget(MAX_TOTAL_LLM_CALLS) as budget,
        capture_timings() as timing_records,
        retriever_worker.trace_context(run_id=run_id),
    ):
        try:
            reformulated = reformulate_query(normalized, history)
            result = multi_agent_graph.invoke({
                "question": normalized,
                "reformulated_question": reformulated,
                "history": history or [],
                "run_id": run_id,
                "sub_questions": [],
                "evidence": [],
                "decision_log": [],
                "rounds": 0,
                "llm_calls": 0,
            })
        except retriever_worker.LLMBudgetExceeded:
            result = {
                "question": normalized,
                "history": history or [],
                "run_id": run_id,
                "answer": (
                    "I couldn't complete a grounded answer within the configured "
                    "model-call budget."
                ),
                "draft": "",
                "answerable": False,
                "citations": [],
                "evidence": [],
                "context_blocks": [],
                "llm_calls": budget["calls"],
                "decision_log": [{
                    "stage": "budget",
                    "limit": MAX_TOTAL_LLM_CALLS,
                    "calls": budget["calls"],
                }],
            }
            status = "budget_exceeded"
        except Exception as exc:
            logger.exception("Multi-agent run %s failed", run_id)
            try:
                record_run(
                    run_id,
                    normalized,
                    status="failed",
                    calls=budget["calls"],
                    result={"error": f"{type(exc).__name__}: {exc}"},
                )
            except Exception:
                logger.exception("Could not persist failure for run %s", run_id)
            raise
        else:
            status = "completed"
    token_usage = aggregate_trace_usage(timing_records)
    result["summary_model_calls"] = int(result.get("summary_model_calls", 0))
    result["llm_calls"] = token_usage["model_calls"]
    result["timings"] = timing_records
    result["token_usage"] = token_usage
    record_run(
        run_id,
        normalized,
        route=result.get("route"),
        status=status,
        calls=result["llm_calls"],
        result=result,
    )
    return dict(result)
