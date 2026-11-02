"""Backward-compatible exports for the original retrieval-agent API."""

from __future__ import annotations

from typing import Any

from config import USE_MULTI_AGENT_RAG

from . import curator, graph, retriever_worker, writer
from .verifier import remove_unsupported_claims, verify_draft

MAX_RETRIEVAL_ATTEMPTS = retriever_worker.MAX_RETRIEVAL_ATTEMPTS
RetrievalGrade = retriever_worker.RetrievalGrade
RetrievalPlan = retriever_worker.RetrievalPlan
RetrievalState = retriever_worker.RetrievalState
safe_json_loads = retriever_worker.safe_json_loads
wrap_data_for_prompt = retriever_worker.wrap_data_for_prompt

_plan_queries = retriever_worker._plan_queries
_grade_chunks = retriever_worker._grade_chunks
search_document_chunks = retriever_worker.search_document_chunks
get_neighbor_chunks = retriever_worker.get_neighbor_chunks
safe_chat_create = retriever_worker.safe_chat_create


def retrieve(
    question: str,
    *,
    needs_retrieval: bool = True,
    history: list[dict[str, str]] | None = None,
    planned_queries: list[str] | None = None,
    skip_model_grading: bool = False,
) -> RetrievalState:
    """Run retrieval while preserving the original test and import surface."""
    patched = {
        "_plan_queries": _plan_queries,
        "_grade_chunks": _grade_chunks,
        "search_document_chunks": search_document_chunks,
    }
    original = {
        name: getattr(retriever_worker, name)
        for name in patched
    }
    try:
        for name, value in patched.items():
            setattr(retriever_worker, name, value)
        return retriever_worker.retrieve(
            question,
            needs_retrieval=needs_retrieval,
            history=history,
            planned_queries=planned_queries,
            skip_model_grading=skip_model_grading,
        )
    finally:
        for name, value in original.items():
            setattr(retriever_worker, name, value)


def answer_from_retrieval(
    question: str,
    retrieval: RetrievalState,
    *,
    revision_feedback: str | None = None,
    output_format: str = "short_answer",
    return_details: bool = False,
) -> str | dict[str, Any]:
    """Generate an answer while preserving the original patch point."""
    original = retriever_worker.safe_chat_create
    retriever_worker.safe_chat_create = safe_chat_create
    try:
        return writer.answer_from_retrieval(
            question,
            retrieval,
            revision_feedback=revision_feedback,
            output_format=output_format,
            return_details=return_details,
        )
    finally:
        retriever_worker.safe_chat_create = original


def rerank_chunks(
    question: str,
    chunks: list[dict[str, Any]],
    *,
    top_k: int = 8,
) -> tuple[list[dict[str, Any]], str]:
    """Preserve the original helper while sharing semantic reranking."""
    return curator.rerank_chunks(question, chunks, top_k=top_k)


def expand_chunks(
    ranked_chunks: list[dict[str, Any]],
    *,
    window: int = 1,
    max_chars: int = 12000,
) -> list[dict[str, Any]]:
    """Preserve the original neighbor-expansion helper and patch point."""
    original = retriever_worker.get_neighbor_chunks
    retriever_worker.get_neighbor_chunks = get_neighbor_chunks
    try:
        return retriever_worker.expand_chunks(
            ranked_chunks,
            window=window,
            max_chars=max_chars,
        )
    finally:
        retriever_worker.get_neighbor_chunks = original


def run_rag(
    question: str,
    *,
    history: list[dict[str, str]] | None = None,
) -> RetrievalState:
    """Run the graph entry point, with an opt-in fallback to Phase 1 behavior."""
    if USE_MULTI_AGENT_RAG:
        return graph.run_multi_agent(question, history=history)
    state = retrieve(question, needs_retrieval=True, history=history)
    ranked, confidence = rerank_chunks(
        question,
        state.get("chunks", []),
        top_k=8,
    )
    context = expand_chunks(
        ranked,
        max_chars=(
            retriever_worker.SUMMARY_MAX_CONTEXT_CHARS
            if state.get("intent") == "summary"
            else 12000
        ),
    )
    state["ranked_chunks"] = ranked
    state["context_blocks"] = context
    state["rerank_confidence"] = confidence
    state["decision_log"] = list(state.get("decision_log", []))
    if not context or (ranked and not graph.has_enough_evidence(ranked)):
        raw_answer = retriever_worker.NO_EVIDENCE_ANSWER
        state.update({
            "answer": raw_answer,
            "raw_answer": raw_answer,
            "answerable": False,
            "citations": [],
        })
        return state

    retrieval_context = dict(state)
    details = answer_from_retrieval(
        question,
        retrieval_context,
        return_details=True,
    )
    context = retrieval_context.get("context_blocks") or context
    state["context_blocks"] = context
    if isinstance(details, dict):
        raw_answer = str(details.get("raw_answer") or details.get("answer") or "")
    else:
        raw_answer = str(details)
    verdict = verify_draft(question, raw_answer, context)
    round_number = 1
    if not verdict.ok:
        raw_answer = remove_unsupported_claims(
            raw_answer,
            [claim.model_dump() for claim in verdict.claims],
        ) or retriever_worker.NO_EVIDENCE_ANSWER
    answer = retriever_worker.format_answer_citations(raw_answer, context)
    state.update({
        "answer": answer,
        "raw_answer": raw_answer,
        "answerable": bool(context) and answer != retriever_worker.NO_EVIDENCE_ANSWER,
        "citations": writer.extract_citations(raw_answer),
        "verdict": verdict.model_dump(),
        "rounds": round_number,
        "decision_log": [
            *state["decision_log"],
            {
                "stage": "verify",
                "ok": verdict.ok,
                "rounds": round_number,
                "missing": verdict.missing,
            },
        ],
    })
    return state
