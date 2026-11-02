"""Tools package for Nexus multi-agent document research system.

Exposes tools used by the Orchestrator, Retriever Workers, and Summarizer agents.
"""

from .document_store import (
    DocumentSearchResult,
    DocumentChunkSearchResult,
    get_document_catalog,
    get_document_chunks,
    get_document_summary_chunks,
    get_neighbor_chunks,
    search_document_chunks,
    search_documents,
    read_document,
    delete_document_chunks,
)
from .document_ingestion import (
    ingest_documents,
    save_uploaded_documents,
    get_vector_collection,
    get_vector_store_stats,
    ensure_documents_indexed,
)
from .retrieval_tool import (
    RetrievalMode,
    RetrievalToolInput,
    RetrievalToolOutput,
    execute_retrieval,
    retrieve_for_subquestion,
    retrieve_table_evidence,
    retrieve_document_summary,
    expand_evidence_context,
)

__all__ = [
    "DocumentSearchResult",
    "DocumentChunkSearchResult",
    "get_document_catalog",
    "get_document_chunks",
    "get_document_summary_chunks",
    "get_neighbor_chunks",
    "search_document_chunks",
    "search_documents",
    "read_document",
    "delete_document_chunks",
    "ingest_documents",
    "save_uploaded_documents",
    "get_vector_collection",
    "get_vector_store_stats",
    "ensure_documents_indexed",
    "RetrievalMode",
    "RetrievalToolInput",
    "RetrievalToolOutput",
    "execute_retrieval",
    "retrieve_for_subquestion",
    "retrieve_table_evidence",
    "retrieve_document_summary",
    "expand_evidence_context",
]
