"""Answer generation stage for the local-document retrieval pipeline."""

from __future__ import annotations

from typing import Any

from . import retriever_worker


def answer_from_retrieval(
    question: str,
    retrieval: dict[str, Any],
    *,
    revision_feedback: str | None = None,
    output_format: str = "short_answer",
    return_details: bool = False,
) -> str | dict[str, Any]:
    """Generate a grounded answer from retrieved local-document evidence."""
    return retriever_worker.answer_from_retrieval(
        question,
        retrieval,
        revision_feedback=revision_feedback,
        output_format=output_format,
        return_details=return_details,
    )


extract_citations = retriever_worker.extract_citations
strip_source_references = retriever_worker.strip_source_references

safe_chat_create = retriever_worker.safe_chat_create
