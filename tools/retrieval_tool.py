"""High-level agentic retrieval tool for Nexus multi-agent document research.

Provides optimized, mode-specific evidence retrieval for the Orchestrator
and Retriever Worker agents:
- 'hybrid': Dense vector similarity + SQLite FTS5 lexical matching.
- 'table': Structured tabular retrieval prioritizing markdown tables & metadata.
- 'summary_cache': Target document-level summaries or full chunk sequences.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal
from pydantic import BaseModel, ConfigDict, Field

from tools.document_store import (
    DocumentChunkSearchResult,
    get_document_summary_chunks,
    get_neighbor_chunks,
    search_document_chunks,
)

logger = logging.getLogger(__name__)

RetrievalMode = Literal["hybrid", "table", "summary_cache"]


class RetrievalToolInput(BaseModel):
    """Input parameters for the retrieval tool."""

    model_config = ConfigDict(extra="ignore")

    query: str = Field(description="Search question or keyword phrase")
    mode: RetrievalMode = Field(
        default="hybrid",
        description="Retrieval strategy: 'hybrid' for facts, 'table' for numbers/metrics/comparisons, 'summary_cache' for document briefings",
    )
    doc_ids: list[str] = Field(
        default_factory=list,
        description="Optional list of document IDs to restrict search scope to",
    )
    file_type: str | None = Field(
        default=None,
        description="Optional file extension filter: pdf, docx, txt, md, json",
    )
    max_results: int = Field(default=8, ge=1, le=30)


class RetrievalToolOutput(BaseModel):
    """Structured response from the retrieval tool."""

    model_config = ConfigDict(extra="ignore")

    query: str
    mode: str
    doc_ids: list[str]
    total_found: int
    chunks: list[dict[str, Any]]


def execute_retrieval(
    query: str,
    *,
    mode: RetrievalMode = "hybrid",
    doc_ids: list[str] | None = None,
    file_type: str | None = None,
    max_results: int = 8,
    search_fn: Callable[..., list[DocumentChunkSearchResult]] | None = None,
    summary_fn: Callable[..., list[DocumentChunkSearchResult]] | None = None,
) -> list[DocumentChunkSearchResult]:
    """Execute high-level agentic retrieval with mode-specific optimization.

    Parameters:
        query: Search query or question.
        mode: Retrieval mode ('hybrid', 'table', or 'summary_cache').
        doc_ids: Optional document IDs to restrict search scope.
        file_type: Optional file type filter (pdf, docx, txt, md, json).
        max_results: Maximum number of chunks to return.
        search_fn: Override for search function (default: search_document_chunks).
        summary_fn: Override for summary function (default: get_document_summary_chunks).

    Returns:
        List of ranked DocumentChunkSearchResult objects.
    """
    normalized = (query or "").strip()
    if not normalized:
        return []

    search = search_fn or search_document_chunks
    summary = summary_fn or get_document_summary_chunks

    cleaned_doc_ids = [d.strip() for d in (doc_ids or []) if d and d.strip()] or None

    # Mode 1: Summary cache / full document retrieval
    if mode == "summary_cache":
        if cleaned_doc_ids:
            # First attempt targeted fetch within specified doc_ids
            summary_search_kwargs: dict[str, Any] = {
                "max_results": max_results * 2,
                "file_type": file_type,
                "doc_ids": cleaned_doc_ids,
            }
            chunks = search(normalized, **summary_search_kwargs)
            if chunks:
                return chunks[:max_results]
        summary_kwargs: dict[str, Any] = {"file_type": file_type}
        if cleaned_doc_ids:
            summary_kwargs["doc_ids"] = cleaned_doc_ids
        return summary(normalized, **summary_kwargs)[:max_results]

    # Mode 2: Structured table retrieval
    if mode == "table":
        table_kwargs: dict[str, Any] = {
            "max_results": max_results,
            "file_type": file_type,
            "table_only": True,
        }
        if cleaned_doc_ids:
            table_kwargs["doc_ids"] = cleaned_doc_ids

        table_results = search(normalized, **table_kwargs)
        if len(table_results) >= max_results:
            return table_results

        # Augment with regular hybrid results to ensure complete context
        hybrid_kwargs: dict[str, Any] = {
            "max_results": max_results,
            "file_type": file_type,
        }
        if cleaned_doc_ids:
            hybrid_kwargs["doc_ids"] = cleaned_doc_ids

        hybrid_results = search(normalized, **hybrid_kwargs)
        seen_ids = {c.chunk_id for c in table_results}
        combined = list(table_results)
        for chunk in hybrid_results:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                combined.append(chunk)
                if len(combined) >= max_results:
                    break
        return combined

    # Mode 3: Standard hybrid retrieval (vector + FTS5)
    default_kwargs: dict[str, Any] = {
        "max_results": max_results,
        "file_type": file_type,
    }
    if cleaned_doc_ids:
        default_kwargs["doc_ids"] = cleaned_doc_ids

    return search(normalized, **default_kwargs)



def retrieve_for_subquestion(
    sub_question: dict[str, Any] | Any,
    *,
    max_results: int = 8,
    file_type: str | None = None,
    search_fn: Callable[..., list[DocumentChunkSearchResult]] | None = None,
    summary_fn: Callable[..., list[DocumentChunkSearchResult]] | None = None,
) -> list[DocumentChunkSearchResult]:
    """Execute retrieval directly from an Orchestrator SubQuestion specification."""
    if isinstance(sub_question, dict):
        query = sub_question.get("query", "")
        mode = sub_question.get("retrieval", "hybrid")
        doc_ids = sub_question.get("doc_ids", [])
    else:
        query = getattr(sub_question, "query", "")
        mode = getattr(sub_question, "retrieval", "hybrid")
        doc_ids = getattr(sub_question, "doc_ids", [])

    return execute_retrieval(
        query=query,
        mode=mode,
        doc_ids=doc_ids or None,
        file_type=file_type,
        max_results=max_results,
        search_fn=search_fn,
        summary_fn=summary_fn,
    )


def retrieve_table_evidence(
    query: str,
    *,
    doc_ids: list[str] | None = None,
    file_type: str | None = None,
    max_results: int = 8,
) -> list[DocumentChunkSearchResult]:
    """Targeted retrieval for financial figures, comparison tables, and metrics."""
    return execute_retrieval(
        query=query,
        mode="table",
        doc_ids=doc_ids,
        file_type=file_type,
        max_results=max_results,
    )


def retrieve_document_summary(
    query: str,
    *,
    doc_ids: list[str] | None = None,
    file_type: str | None = None,
    max_results: int = 8,
) -> list[DocumentChunkSearchResult]:
    """Retrieve full document context or cached summary chunks."""
    return execute_retrieval(
        query=query,
        mode="summary_cache",
        doc_ids=doc_ids,
        file_type=file_type,
        max_results=max_results,
    )


def expand_evidence_context(
    chunks: list[DocumentChunkSearchResult],
    window: int = 1,
) -> list[DocumentChunkSearchResult]:
    """Expand retrieved chunks with adjacent context passages from the same document."""
    if not chunks or window <= 0:
        return chunks

    by_doc: dict[str, set[int]] = {}
    for c in chunks:
        if c.document_id:
            by_doc.setdefault(c.document_id, set()).add(c.chunk_index)

    expanded: list[DocumentChunkSearchResult] = []
    seen_ids: set[str] = set()
    for doc_id, indices in by_doc.items():
        neighbors = get_neighbor_chunks(doc_id, indices, window=window)
        for neighbor in neighbors:
            if neighbor.chunk_id not in seen_ids:
                seen_ids.add(neighbor.chunk_id)
                expanded.append(neighbor)

    return expanded or chunks
