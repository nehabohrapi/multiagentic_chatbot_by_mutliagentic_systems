import logging
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, List

from pydantic import BaseModel, Field

from config import (
    DOCUMENTS_DIR,
    MAX_DOCUMENT_BYTES,
    MAX_DOCUMENT_SEARCH_DISTANCE,
    MAX_EXTRACTED_CONTENT_LENGTH,
    HYBRID_LEXICAL_RRF_WEIGHT,
    HYBRID_RRF_K,
    HYBRID_VECTOR_RRF_WEIGHT,
    RETRIEVAL_CANDIDATE_MULTIPLIER,
    RETRIEVAL_MIN_CANDIDATES,
    SUMMARY_MAX_RELEVANT_CHUNKS,
    SUMMARY_RELEVANT_CHUNKS_PER_QUERY,
)
from document_processing.document_loader import extract_document_pages
from embeddings.local_model import embed_query

logger = logging.getLogger(__name__)
_document_catalog_cache: tuple[dict[str, Any], ...] | None = None
_document_catalog_lock = threading.Lock()

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md", ".json", ".csv",
    ".png", ".jpg", ".jpeg", ".webp", ".gif",
}
STOP_WORDS = {
    "about", "after", "again", "also", "and", "are", "been", "before",
    "being", "between", "but", "can", "could", "does", "from", "have",
    "into", "just", "more", "most", "other", "over", "should", "some",
    "such", "than", "that", "the", "their", "there", "these", "they",
    "this", "through", "under", "was", "were", "what", "when", "where",
    "which", "while", "with", "would",
}


class DocumentSearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class DocumentChunkSearchResult(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    source_path: str
    file_type: str
    chunk_index: int
    page_number: int | None = None
    text: str
    distance: float
    match_type: str = "vector"
    hybrid_score: float | None = None
    vector_score: float | None = None
    lexical_score: float | None = None
    vector_rank: int | None = None
    lexical_rank: int | None = None
    structured_data: list[dict] = Field(default_factory=list)


def _structured_data_from_metadata(metadata: dict[str, Any]) -> list[dict]:
    value = metadata.get("structured_data")
    if not value:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _is_table_search_candidate(
    text: str,
    structured_blocks: list[dict[str, Any]],
) -> bool:
    """Recognize explicit PDF table captions even when extraction is plain text."""
    return "|" in text or bool(
        re.search(r"\btable\s+\d+\s*:", text, re.IGNORECASE)
    ) or any(
        block.get("kind") in {"table", "figure"}
        for block in structured_blocks
    )


def _safe_document_path(
    path: str | Path,
    documents_root: str | Path | None = None,
) -> Path:
    root = Path(documents_root or DOCUMENTS_DIR).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("Document path must be inside the documents folder.")
    if resolved.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document format: {resolved.suffix or 'no extension'}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Document not found: {resolved.name}")
    if resolved.stat().st_size > MAX_DOCUMENT_BYTES:
        raise ValueError(f"Document exceeds the {MAX_DOCUMENT_BYTES // (1024 * 1024)} MB size limit.")
    return resolved


def read_document(
    path: str | Path,
    max_chars: int = MAX_EXTRACTED_CONTENT_LENGTH,
    documents_root: str | Path | None = None,
) -> dict:
    """Read one user-provided PDF, JSON, DOCX, TXT, or Markdown file."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive.")
    resolved = _safe_document_path(path, documents_root=documents_root)
    pages = []
    try:
        extracted_pages = extract_document_pages(resolved)
        remaining = max_chars
        for page in extracted_pages:
            if remaining <= 0:
                break
            page_text = page.text[:remaining]
            if page_text:
                page_record: dict[str, Any] = {
                    "page_number": page.page_number,
                    "text": page_text,
                }
                if page.structured_data:
                    page_record["structured_data"] = list(page.structured_data)
                pages.append(page_record)
                remaining -= len(page_text) + 2
        content = "\n\n".join(page["text"] for page in pages)
    except Exception as exc:
        return {
            "url": str(resolved),
            "title": resolved.stem,
            "publisher": "Local document",
            "source_kind": "local_document",
            "content_type": resolved.suffix.lstrip("."),
            "is_pdf": resolved.suffix.lower() == ".pdf",
            "status": "failed",
            "content": "",
            "pages": [],
            "content_length": 0,
            "error": f"Could not extract document text: {exc}",
        }

    if not content:
        return {
            "url": str(resolved),
            "title": resolved.stem,
            "publisher": "Local document",
            "source_kind": "local_document",
            "content_type": resolved.suffix.lstrip("."),
            "is_pdf": resolved.suffix.lower() == ".pdf",
            "status": "failed",
            "content": "",
            "pages": [],
            "content_length": 0,
            "error": "Document contains no extractable text.",
        }

    return {
        "url": str(resolved),
        "title": resolved.stem,
        "publisher": "Local document",
        "source_kind": "local_document",
        "content_type": resolved.suffix.lstrip("."),
        "is_pdf": resolved.suffix.lower() == ".pdf",
        "status": "success",
        "content": content,
        "pages": pages,
        "content_length": len(content),
        "error": None,
    }


def _keywords(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-z0-9][a-z0-9+#.-]{2,}", text.lower())
        if word not in STOP_WORDS
    }


def _best_snippet(text: str, query_terms: set[str], max_chars: int = 700) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    ranked = sorted(
        enumerate(sentences),
        key=lambda pair: (
            len(query_terms & _keywords(pair[1])),
            -pair[0],
        ),
        reverse=True,
    )
    selected = [sentence.strip() for _, sentence in ranked[:3] if sentence.strip()]
    return " ".join(selected)[:max_chars]


def search_documents(query: str, max_results: int = 5) -> List[DocumentSearchResult]:
    """Find relevant local documents by vector similarity over indexed chunks."""
    query = (query or "").strip()
    if not query:
        raise ValueError("Search query cannot be empty.")
    if not 1 <= max_results <= 10:
        raise ValueError("max_results must be between 1 and 10.")

    from tools.document_ingestion import ensure_documents_indexed, get_vector_collection

    report = ensure_documents_indexed()
    if report["errors"]:
        logger_errors = "; ".join(
            f"{error['path']}: {error['error']}" for error in report["errors"]
        )
        raise RuntimeError(f"Could not index one or more local documents: {logger_errors}")

    collection = get_vector_collection()
    chunk_count = collection.count()
    if not chunk_count:
        return []

    matches = collection.query(
        query_embeddings=[list(embed_query(query))],
        n_results=min(max_results * 3, chunk_count),
        include=["documents", "metadatas", "distances"],
    )
    query_terms = _keywords(query)
    results = []
    seen_paths = set()
    documents = matches["documents"][0] or []
    metadatas = matches["metadatas"][0] or []
    distances = matches["distances"][0] or []
    for chunk, metadata, distance in zip(documents, metadatas, distances):
        if distance > MAX_DOCUMENT_SEARCH_DISTANCE:
            continue
        source_path = metadata.get("source_path")
        if not source_path or source_path in seen_paths:
            continue
        seen_paths.add(source_path)
        results.append(
            DocumentSearchResult(
                title=metadata.get("title") or Path(source_path).stem,
                url=source_path,
                snippet=_best_snippet(chunk, query_terms) if query_terms else chunk[:700],
            )
        )
        if len(results) >= max_results:
            break
    return results


def search_document_chunks(
    query: str,
    max_results: int = 8,
    file_type: str | None = None,
    doc_ids: list[str] | None = None,
    table_only: bool = False,
) -> list[DocumentChunkSearchResult]:
    """Return ranked Chroma chunks with source metadata for agentic retrieval."""
    query = (query or "").strip()
    if not query:
        raise ValueError("Search query cannot be empty.")
    if not 1 <= max_results <= 30:
        raise ValueError("max_results must be between 1 and 30.")
    if file_type is not None and file_type not in {
        extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS
    }:
        raise ValueError(f"Unsupported document file type filter: {file_type}")

    from tools.document_ingestion import ensure_documents_indexed, get_vector_collection

    report = ensure_documents_indexed()
    if report["errors"]:
        errors = "; ".join(
            f"{error['path']}: {error['error']}" for error in report["errors"]
        )
        raise RuntimeError(f"Could not index one or more local documents: {errors}")

    collection = get_vector_collection()
    chunk_count = collection.count()
    if not chunk_count:
        return []

    cleaned_doc_ids = list(dict.fromkeys(
        document_id.strip()
        for document_id in (doc_ids or [])
        if document_id and document_id.strip()
    ))
    where_conditions: list[dict[str, Any]] = []
    if file_type:
        where_conditions.append({"file_type": file_type})
    if len(cleaned_doc_ids) == 1:
        where_conditions.append({"document_id": cleaned_doc_ids[0]})
    elif len(cleaned_doc_ids) > 1:
        where_conditions.append({"document_id": {"$in": cleaned_doc_ids}})

    if len(where_conditions) == 1:
        where = where_conditions[0]
    elif len(where_conditions) > 1:
        where = {"$and": where_conditions}
    else:
        where = None

    search_started_at = time.perf_counter()
    candidate_limit = min(
        max(max_results * RETRIEVAL_CANDIDATE_MULTIPLIER, RETRIEVAL_MIN_CANDIDATES),
        chunk_count,
    )
    vector_started_at = time.perf_counter()
    matches = collection.query(
        query_embeddings=[list(embed_query(query))],
        n_results=candidate_limit,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    vector_candidates: dict[str, dict[str, Any]] = {}
    for chunk_id, text, metadata, distance in zip(
        matches["ids"][0] or [],
        matches["documents"][0] or [],
        matches["metadatas"][0] or [],
        matches["distances"][0] or [],
    ):
        if not isinstance(metadata, dict) or not isinstance(text, str):
            continue
        source_path = metadata.get("source_path")
        if not source_path:
            continue
        vector_candidates[chunk_id] = {
            "chunk_id": chunk_id,
            "text": text,
            "metadata": metadata,
            "distance": float(distance),
            "vector_score": 1.0 - float(distance),
            "vector_rank": len(vector_candidates) + 1,
        }
    vector_elapsed_ms = (time.perf_counter() - vector_started_at) * 1000
    _log_retrieval_channel(
        "vector",
        query,
        {"doc_ids": cleaned_doc_ids, "file_type": file_type},
        vector_elapsed_ms,
        [
            {
                "chunk_id": candidate["chunk_id"],
                "score": candidate["vector_score"],
                "match_type": "vector",
            }
            for candidate in list(vector_candidates.values())[:10]
        ],
    )

    from memory.state_store import search_fts_chunks

    fts_started_at = time.perf_counter()
    fts_candidates = search_fts_chunks(
        query,
        max_results=candidate_limit,
        file_type=file_type,
        doc_ids=cleaned_doc_ids or None,
    )
    fts_elapsed_ms = (time.perf_counter() - fts_started_at) * 1000
    _log_retrieval_channel(
        "fts5",
        query,
        {"doc_ids": cleaned_doc_ids, "file_type": file_type},
        fts_elapsed_ms,
        [
            {
                "chunk_id": candidate["chunk_id"],
                "score": candidate.get("bm25_score"),
                "match_type": "fts5",
            }
            for candidate in fts_candidates[:10]
        ],
    )

    # Reciprocal Rank Fusion combines channel orderings without comparing
    # cosine distance to SQLite's query-dependent BM25 score scale.
    rrf_constant = HYBRID_RRF_K
    vector_weight = HYBRID_VECTOR_RRF_WEIGHT
    lexical_weight = HYBRID_LEXICAL_RRF_WEIGHT
    candidates_by_id: dict[str, dict[str, Any]] = {}
    for candidate in vector_candidates.values():
        candidates_by_id[candidate["chunk_id"]] = candidate
    for rank, fts_match in enumerate(fts_candidates, start=1):
        chunk_id = fts_match["chunk_id"]
        candidate = candidates_by_id.get(chunk_id)
        if candidate is None:
            metadata = {
                "document_id": fts_match.get("document_id", ""),
                "title": fts_match.get("title", ""),
                "source_path": fts_match.get("source_path", ""),
                "file_type": fts_match.get("file_type", ""),
                "chunk_index": fts_match.get("chunk_index", 0),
                "page_number": fts_match.get("page_number"),
                "structured_data": fts_match.get("structured_data", ""),
            }
            candidate = {
                "chunk_id": chunk_id,
                "text": fts_match["text"],
                "metadata": metadata,
                "distance": 1.0,
                "vector_score": None,
                "vector_rank": None,
            }
            candidates_by_id[chunk_id] = candidate
        candidate["lexical_score"] = float(fts_match.get("bm25_score", 0.0))
        candidate["lexical_rank"] = rank
        candidate["rrf_score"] = candidate.get("rrf_score", 0.0) + (
            lexical_weight / (rrf_constant + rank)
        )

    maximum_rrf_score = (vector_weight + lexical_weight) / (rrf_constant + 1)
    for candidate in vector_candidates.values():
        candidate["rrf_score"] = candidate.get("rrf_score", 0.0) + (
            vector_weight / (rrf_constant + candidate["vector_rank"])
        )

    ranked = []
    for candidate in candidates_by_id.values():
        metadata = candidate["metadata"]
        structured_blocks = _structured_data_from_metadata(metadata)
        if table_only and not _is_table_search_candidate(
            candidate["text"], structured_blocks
        ):
            continue
        if (
            candidate.get("vector_rank") is not None
            and candidate["distance"] > MAX_DOCUMENT_SEARCH_DISTANCE
            and candidate.get("lexical_rank") is None
        ):
            continue
        candidate["hybrid_score"] = (
            candidate.get("rrf_score", 0.0) / maximum_rrf_score
        )
        ranked.append(candidate)
    ranked.sort(key=lambda candidate: candidate["hybrid_score"], reverse=True)

    results: list[DocumentChunkSearchResult] = []
    seen_ids: set[str] = set()
    for candidate in ranked[:max_results]:
        chunk_id = candidate["chunk_id"]
        metadata = candidate["metadata"]
        source_path = metadata.get("source_path") or ""
        if not source_path:
            continue
        if candidate.get("vector_rank") and candidate.get("lexical_rank"):
            match_type = "hybrid"
        elif candidate.get("lexical_rank"):
            match_type = "lexical"
        else:
            match_type = "vector"
        result_data = {
            "chunk_id": chunk_id,
            "document_id": metadata.get("document_id") or metadata.get("doc_id", ""),
            "title": metadata.get("title") or Path(source_path).stem,
            "source_path": source_path,
            "file_type": metadata.get("file_type", ""),
            "chunk_index": metadata.get("chunk_index", 0),
            "text": candidate["text"],
            "distance": candidate["distance"],
            "match_type": match_type,
            "hybrid_score": round(candidate["hybrid_score"], 6),
            "vector_score": (
                round(candidate["vector_score"], 6)
                if candidate.get("vector_score") is not None
                else None
            ),
            "lexical_score": (
                round(candidate["lexical_score"], 6)
                if candidate.get("lexical_score") is not None
                else None
            ),
            "vector_rank": candidate.get("vector_rank"),
            "lexical_rank": candidate.get("lexical_rank"),
            "structured_data": _structured_data_from_metadata(metadata),
        }
        if metadata.get("page_number") is not None:
            result_data["page_number"] = metadata["page_number"]
        results.append(DocumentChunkSearchResult(**result_data))
        seen_ids.add(chunk_id)

    _log_retrieval_channel(
        "hybrid",
        query,
        {"doc_ids": cleaned_doc_ids, "file_type": file_type},
        (time.perf_counter() - search_started_at) * 1000,
        [
            {
                "chunk_id": result.chunk_id,
                "score": result.hybrid_score,
                "match_type": result.match_type,
            }
            for result in results
        ],
    )

    # Keep a corpus-scan fallback for unusual terms that SQLite tokenization
    # or Chroma's ANN candidate window cannot return.
    if not results:
        query_terms = _keywords(query)
        if query_terms:
            minimum_overlap = max(1, (len(query_terms) + 1) // 2)
            stored = collection.get(include=["documents", "metadatas"])
            lexical_matches = []
            for chunk_id, text, metadata in zip(
                stored["ids"] or [],
                stored["documents"] or [],
                stored["metadatas"] or [],
            ):
                if chunk_id in seen_ids or (
                    file_type and metadata.get("file_type") != file_type
                ):
                    continue
                overlap = query_terms & _keywords(text)
                if len(overlap) < minimum_overlap:
                    continue
                lexical_matches.append(
                    (
                        len(overlap),
                        chunk_id,
                        text,
                        metadata,
                    )
                )

            lexical_matches.sort(key=lambda item: item[0], reverse=True)
            for overlap, chunk_id, text, metadata in lexical_matches[:max_results]:
                source_path = metadata.get("source_path")
                if not source_path:
                    continue
                result_data = {
                    "chunk_id": chunk_id,
                    "document_id": metadata.get("document_id", ""),
                    "title": metadata.get("title") or Path(source_path).stem,
                    "source_path": source_path,
                    "file_type": metadata.get("file_type", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "text": text,
                    # This is a normalized lexical score, not a Chroma
                    # distance. It is only used for deterministic fallback
                    # ranking when vector retrieval found no result.
                    "distance": 1 - (overlap / len(query_terms)),
                    "match_type": "lexical",
                    "hybrid_score": overlap / len(query_terms),
                    "lexical_score": overlap / len(query_terms),
                    "structured_data": _structured_data_from_metadata(metadata),
                }
                if metadata.get("page_number") is not None:
                    result_data["page_number"] = metadata["page_number"]
                results.append(DocumentChunkSearchResult(**result_data))
    return results


def _log_retrieval_channel(
    channel: str,
    query: str,
    scope: dict[str, Any],
    elapsed_ms: float,
    hits: list[dict[str, Any]],
) -> None:
    logger.info(
        "retrieval_channel %s",
        json.dumps(
            {
                "channel": channel,
                "query": query,
                "scope": scope,
                "latency_ms": round(elapsed_ms, 3),
                "matches": hits,
            },
            ensure_ascii=False,
        ),
    )


def get_neighbor_chunks(
    document_id: str,
    chunk_indices: set[int],
    window: int = 1,
) -> list[DocumentChunkSearchResult]:
    """Load nearby chunks from one document for context expansion."""
    if not document_id or not chunk_indices or window < 0:
        return []

    from tools.document_ingestion import get_vector_collection

    collection = get_vector_collection()
    stored = collection.get(
        where={"document_id": document_id},
        include=["documents", "metadatas"],
    )
    wanted = {
        index
        for hit in chunk_indices
        for index in range(max(0, hit - window), hit + window + 1)
    }
    results = []
    for chunk_id, text, metadata in zip(
        stored["ids"] or [],
        stored["documents"] or [],
        stored["metadatas"] or [],
    ):
        index = metadata.get("chunk_index")
        if index not in wanted:
            continue
        source_path = metadata.get("source_path")
        if not source_path:
            continue
        result_data = {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "title": metadata.get("title") or Path(source_path).stem,
            "source_path": source_path,
            "file_type": metadata.get("file_type", ""),
            "chunk_index": index,
            "text": text,
            "distance": 0.0,
            "match_type": "neighbor",
            "structured_data": _structured_data_from_metadata(metadata),
        }
        if metadata.get("page_number") is not None:
            result_data["page_number"] = metadata["page_number"]
        results.append(DocumentChunkSearchResult(**result_data))
    return sorted(results, key=lambda item: item.chunk_index)


def get_document_summary_chunks(
    query: str,
    *,
    file_type: str | None = None,
    doc_ids: list[str] | None = None,
) -> list[DocumentChunkSearchResult]:
    """Select one document, then retrieve its relevant summary chunks in source order."""
    from tools.document_ingestion import ensure_documents_indexed

    report = ensure_documents_indexed()
    if report["errors"]:
        raise RuntimeError("; ".join(error["error"] for error in report["errors"]))

    requested_doc_ids = [doc_id for doc_id in (doc_ids or []) if doc_id]
    if len(requested_doc_ids) == 1:
        document_id = requested_doc_ids[0]
    else:
        matches = search_document_chunks(
            query,
            max_results=1,
            file_type=file_type,
            doc_ids=requested_doc_ids or None,
        )
        if not matches or not matches[0].document_id:
            return []
        document_id = matches[0].document_id

    search_queries = [
        query,
        f"{query} purpose objective scope overview",
        f"{query} main topics methodology contributions",
        f"{query} key findings results observations",
        f"{query} challenges limitations open research questions future work conclusion "
        "LLM architecture pre-training adaptation fine-tuning utilization evaluation",
    ]
    unique_queries = list(dict.fromkeys(item.strip() for item in search_queries if item.strip()))
    candidates: dict[str, dict[str, Any]] = {}
    for search_query in unique_queries:
        for result in search_document_chunks(
            search_query,
            max_results=SUMMARY_RELEVANT_CHUNKS_PER_QUERY,
            file_type=file_type,
            doc_ids=[document_id],
        ):
            if result.document_id != document_id:
                continue
            candidate = candidates.setdefault(
                result.chunk_id,
                {"chunk": result, "query_matches": 0, "score": 0.0},
            )
            candidate["query_matches"] += 1
            candidate["score"] += float(
                result.hybrid_score
                if result.hybrid_score is not None
                else max(0.0, 1.0 - result.distance)
            )

    ranked = sorted(
        candidates.values(),
        key=lambda item: (item["query_matches"], item["score"]),
        reverse=True,
    )[:SUMMARY_MAX_RELEVANT_CHUNKS]
    chunks = [item["chunk"] for item in ranked]
    return sorted(
        chunks,
        key=lambda chunk: (
            chunk.page_number is None,
            chunk.page_number if chunk.page_number is not None else chunk.chunk_index,
            chunk.chunk_index,
        ),
    )


def get_document_chunks(document_id: str) -> list[DocumentChunkSearchResult]:
    """Return every indexed chunk for one document in source order."""
    normalized_id = (document_id or "").strip()
    if not normalized_id:
        raise ValueError("A document ID is required to load all document chunks.")

    from tools.document_ingestion import ensure_documents_indexed, get_vector_collection

    report = ensure_documents_indexed()
    if report["errors"]:
        errors = "; ".join(
            f"{error['path']}: {error['error']}" for error in report["errors"]
        )
        raise RuntimeError(f"Could not index one or more local documents: {errors}")

    stored = get_vector_collection().get(
        where={"document_id": normalized_id},
        include=["documents", "metadatas"],
    )
    chunks: list[DocumentChunkSearchResult] = []
    for chunk_id, text, metadata in zip(
        stored.get("ids") or [],
        stored.get("documents") or [],
        stored.get("metadatas") or [],
    ):
        if not isinstance(metadata, dict) or not isinstance(text, str):
            continue
        source_path = str(metadata.get("source_path") or "")
        if not source_path:
            continue
        chunks.append(
            DocumentChunkSearchResult(
                chunk_id=str(chunk_id),
                document_id=normalized_id,
                title=str(metadata.get("title") or Path(source_path).stem),
                source_path=source_path,
                file_type=str(metadata.get("file_type") or Path(source_path).suffix.lstrip(".")),
                chunk_index=int(metadata.get("chunk_index") or 0),
                page_number=(
                    int(metadata["page_number"])
                    if metadata.get("page_number") is not None
                    else None
                ),
                text=text,
                distance=0.0,
                match_type="document",
                structured_data=_structured_data_from_metadata(metadata),
            )
        )

    return sorted(
        chunks,
        key=lambda chunk: (
            chunk.page_number is None,
            chunk.page_number if chunk.page_number is not None else chunk.chunk_index,
            chunk.chunk_index,
        ),
    )


def delete_document_chunks(source_path: str | Path) -> int:
    """Delete a local document and every corresponding Chroma chunk."""
    path = _safe_document_path(source_path)
    from tools.document_ingestion import get_vector_collection

    collection = get_vector_collection()
    stored = collection.get(
        where={"source_path": str(path)},
        include=["metadatas"],
    )
    ids = stored["ids"] or []
    if ids:
        collection.delete(ids=ids)
    path.unlink()
    invalidate_document_catalog_cache()
    return len(ids)


def _extract_document_aliases(title: str, first_chunk_text: str = "") -> list[str]:
    """Extract recognizable abbreviations, acronyms, and title tokens for catalog routing."""
    aliases: set[str] = set()
    if title:
        aliases.add(title.strip())
        parts = re.split(r"[_\s-]+", title)
        for part in parts:
            cleaned = re.sub(r"^[0-9.]+|[0-9.]+$", "", part).strip()
            if len(cleaned) >= 3 and cleaned.lower() not in {"pdf", "doc", "docx", "txt", "notes"}:
                aliases.add(cleaned)
    if first_chunk_text:
        colon_match = re.match(r"^\s*([A-Za-z0-9_-]{3,15})\s*:", first_chunk_text)
        if colon_match:
            aliases.add(colon_match.group(1))
    return sorted(aliases, key=len, reverse=True)


def invalidate_document_catalog_cache() -> None:
    """Invalidate the in-memory document catalog cache when documents change."""
    global _document_catalog_cache
    with _document_catalog_lock:
        _document_catalog_cache = None


def get_document_catalog() -> list[dict[str, Any]]:
    """Return a catalog of indexed documents for the Orchestrator with summary cache status."""
    global _document_catalog_cache
    from tools.document_ingestion import ensure_documents_indexed, get_vector_collection
    from memory.state_store import _connection

    with _document_catalog_lock:
        if _document_catalog_cache is not None:
            return [dict(doc) for doc in _document_catalog_cache]

        try:
            report = ensure_documents_indexed()
        except Exception:
            report = {"inventory_changed": True, "errors": ["Document sync failed"]}

        cache_valid = (
            _document_catalog_cache is not None
            and not report.get("inventory_changed", True)
            and not report.get("errors")
        )
        if cache_valid:
            docs = [dict(doc) for doc in _document_catalog_cache]
        else:

            _document_catalog_cache = None
            try:
                collection = get_vector_collection()
                stored = collection.get(include=["metadatas"])
            except Exception:
                return []

            metadatas = stored.get("metadatas") or []
            docs_by_id: dict[str, dict[str, Any]] = {}
            for meta in metadatas:
                doc_id = meta.get("document_id") or meta.get("doc_id")
                if not doc_id:
                    continue
                page = meta.get("page_number")
                if doc_id not in docs_by_id:
                    docs_by_id[doc_id] = {
                        "doc_id": doc_id,
                        "title": meta.get("title") or Path(meta.get("source_path", "")).stem or "document",
                        "pages": page if page is not None else 1,
                    }
                elif page is not None and page > docs_by_id[doc_id]["pages"]:
                    docs_by_id[doc_id]["pages"] = page
            docs = list(docs_by_id.values())

        cached_doc_ids: set[str] = set()
        first_chunk_texts: dict[str, str] = {}
        try:
            with _connection() as conn:
                rows = conn.execute("SELECT DISTINCT document_id FROM summary_cache").fetchall()
                cached_doc_ids = {row[0] for row in rows}
                fc_rows = conn.execute("SELECT document_id, text FROM chunks_fts WHERE chunk_index = 0").fetchall()
                first_chunk_texts = {row[0]: row[1] for row in fc_rows}
        except Exception:
            pass

        catalog = [
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "aliases": doc.get("aliases") or _extract_document_aliases(doc["title"], first_chunk_texts.get(doc["doc_id"], "")),
                "pages": doc["pages"],
                "has_summary_cache": doc["doc_id"] in cached_doc_ids,
            }
            for doc in docs
        ]
        if not report.get("errors"):
            _document_catalog_cache = tuple(dict(item) for item in catalog)
        return catalog
