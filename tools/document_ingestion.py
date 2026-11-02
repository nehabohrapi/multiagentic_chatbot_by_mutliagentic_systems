import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    CHROMA_UPSERT_BATCH_SIZE,
    DOCUMENT_EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    MAX_INDEXED_DOCUMENT_LENGTH,
    MAX_DOCUMENT_BYTES,
    DOCUMENTS_DIR,
)
from document_processing.chunker import chunk_document_pages
from embeddings.local_model import embed_documents, get_embedding_function
from tools.document_store import SUPPORTED_EXTENSIONS, read_document

_ingestion_lock = threading.Lock()
_last_inventory_signature: tuple[tuple[str, int, int], ...] | None = None
_INDEX_SCHEMA_VERSION = 2


def _print_timing(activity: str, started_at: float, **details: Any) -> None:
    elapsed = time.perf_counter() - started_at
    detail_text = " ".join(f"{key}={value}" for key, value in details.items())
    suffix = f" {detail_text}" if detail_text else ""
    print(f"[document-ingestion] {activity}: {elapsed:.3f}s{suffix}", flush=True)


def save_uploaded_documents(uploaded_files: list[Any]) -> dict[str, Any]:
    """Safely copy Streamlit-style uploaded files into the local document library."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"saved": [], "unchanged": [], "errors": []}

    for uploaded_file in uploaded_files:
        raw_name = str(getattr(uploaded_file, "name", "")).replace("\\", "/")
        original_name = Path(raw_name).name
        suffix = Path(original_name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            report["errors"].append({
                "name": original_name or "unnamed file",
                "error": (
                    "Unsupported format. Upload PDF, JSON, DOCX, TXT, Markdown, "
                    "CSV, PNG, JPG, WEBP, or GIF files."
                ),
            })
            continue

        stem = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(original_name).stem).strip(" ._")
        if not stem:
            stem = "document"
        safe_name = f"{stem}{suffix}"
        content = uploaded_file.getvalue()
        if not content:
            report["errors"].append({"name": original_name, "error": "The file is empty."})
            continue
        if len(content) > MAX_DOCUMENT_BYTES:
            report["errors"].append({
                "name": original_name,
                "error": f"File exceeds the {MAX_DOCUMENT_BYTES // (1024 * 1024)} MB size limit.",
            })
            continue

        target = DOCUMENTS_DIR / safe_name
        if target.exists():
            if target.read_bytes() == content:
                report["unchanged"].append(safe_name)
                continue
            content_hash = hashlib.sha256(content).hexdigest()[:10]
            target = DOCUMENTS_DIR / f"{stem}-{content_hash}{suffix}"
            if target.exists() and target.read_bytes() == content:
                report["unchanged"].append(target.name)
                continue

        temporary = target.with_name(f".{target.name}.upload")
        temporary.write_bytes(content)
        temporary.replace(target)
        report["saved"].append(target.name)

    return report


def _file_inventory(documents_dir: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in documents_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _inventory_signature(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (str(path), path.stat().st_size, path.stat().st_mtime_ns)
        for path in paths
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _open_vector_collection(chroma_dir: str, collection_name: str):
    """Open and retain a persistent Chroma collection for this process."""
    Path(chroma_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_dir)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=get_embedding_function(),
    )


def get_vector_collection():
    """Open the persistent Chroma collection used for local document chunks."""
    return _open_vector_collection(str(CHROMA_DIR.resolve()), CHROMA_COLLECTION_NAME)


def _embedding_cache_key(text: str) -> str:
    cache_material = f"{EMBEDDING_MODEL_NAME}\0{text}".encode("utf-8")
    return hashlib.sha256(cache_material).hexdigest()


def _prepare_embeddings(records: list[dict[str, Any]]) -> tuple[list[list[float]], int, int]:
    """Load cached vectors and embed only texts missing from the persistent cache."""
    from memory.state_store import (
        get_cached_chunk_embeddings,
        put_cached_chunk_embeddings,
    )

    started_at = time.perf_counter()
    cache_keys = [_embedding_cache_key(record["text"]) for record in records]
    cached_embeddings: dict[str, list[float]] = {}
    for offset in range(0, len(cache_keys), DOCUMENT_EMBEDDING_BATCH_SIZE):
        cached_embeddings.update(
            get_cached_chunk_embeddings(
                cache_keys[offset : offset + DOCUMENT_EMBEDDING_BATCH_SIZE]
            )
        )
    _print_timing(
        "Embedding cache lookup",
        started_at,
        chunks=len(records),
        hits=len(cached_embeddings),
        misses=len(records) - len(cached_embeddings),
    )

    embeddings_by_key = dict(cached_embeddings)
    missing_records = [
        (cache_key, record)
        for cache_key, record in zip(cache_keys, records)
        if cache_key not in cached_embeddings
    ]
    for offset in range(0, len(missing_records), DOCUMENT_EMBEDDING_BATCH_SIZE):
        batch = missing_records[offset : offset + DOCUMENT_EMBEDDING_BATCH_SIZE]
        batch_started_at = time.perf_counter()
        vectors = embed_documents([record["text"] for _, record in batch])
        _print_timing(
            "Document embedding",
            batch_started_at,
            batch=offset // DOCUMENT_EMBEDDING_BATCH_SIZE + 1,
            chunks=len(batch),
            model=EMBEDDING_MODEL_NAME,
        )
        cache_started_at = time.perf_counter()
        generated = {
            cache_key: vector
            for (cache_key, _), vector in zip(batch, vectors)
        }
        put_cached_chunk_embeddings(generated)
        embeddings_by_key.update(generated)
        _print_timing(
            "Embedding cache write",
            cache_started_at,
            chunks=len(generated),
        )

    return (
        [embeddings_by_key[cache_key] for cache_key in cache_keys],
        len(missing_records),
        len(records) - len(missing_records),
    )


def _upsert_prepared_document(
    collection: Any,
    records: list[dict[str, Any]],
    embeddings: list[list[float]],
    source_path: str,
) -> None:
    ids = [record["id"] for record in records]
    texts = [record["text"] for record in records]
    metadatas = [record["metadata"] for record in records]
    upsert_started_at = time.perf_counter()
    for offset in range(0, len(ids), CHROMA_UPSERT_BATCH_SIZE):
        batch_end = min(offset + CHROMA_UPSERT_BATCH_SIZE, len(ids))
        batch_started_at = time.perf_counter()
        try:
            collection.upsert(
                ids=ids[offset:batch_end],
                documents=texts[offset:batch_end],
                embeddings=embeddings[offset:batch_end],
                metadatas=metadatas[offset:batch_end],
            )
        finally:
            _print_timing(
                "Chroma upsert (precomputed vectors and vector write)",
                batch_started_at,
                file=Path(source_path).name,
                batch=offset // CHROMA_UPSERT_BATCH_SIZE + 1,
                chunks=batch_end - offset,
            )
    _print_timing(
        "Chroma upsert total",
        upsert_started_at,
        file=Path(source_path).name,
        chunks=len(ids),
    )


def _index_prepared_document_fts(prepared: dict[str, Any]) -> None:
    from memory.state_store import index_fts_chunks

    started_at = time.perf_counter()
    try:
        index_fts_chunks(prepared["fts_records"])
    finally:
        _print_timing(
            "SQLite FTS indexing",
            started_at,
            file=Path(prepared["path"]).name,
            chunks=len(prepared["fts_records"]),
        )


def _prepare_document_record(path: Path, root: Path) -> dict[str, Any]:
    """Read, chunk, and prepare a single document for vector indexing."""
    source_path = str(path)
    started_at = time.perf_counter()
    source_hash = _file_sha256(path)
    _print_timing("Preparation file hash", started_at, file=path.name)

    started_at = time.perf_counter()
    result = read_document(
        path,
        max_chars=MAX_INDEXED_DOCUMENT_LENGTH,
        documents_root=root,
    )
    _print_timing("Document extraction", started_at, file=path.name)
    if result["status"] != "success":
        return {"path": source_path, "error": result["error"]}

    content = result["content"]
    source_pages = result.get("pages") or [{"page_number": None, "text": content}]
    started_at = time.perf_counter()
    chunks = chunk_document_pages(source_pages)
    _print_timing("Document chunking", started_at, file=path.name, chunks=len(chunks))
    if not chunks:
        return {"path": source_path, "error": "Document contains no text to index."}

    started_at = time.perf_counter()
    document_id = hashlib.sha256(source_path.encode("utf-8")).hexdigest()
    records = []
    for chunk in chunks:
        chunk_id = hashlib.sha256(
            f"{source_path}\0{source_hash}\0{chunk.chunk_index}".encode("utf-8")
        ).hexdigest()
        metadata = {
            "document_id": document_id,
            "doc_id": document_id,
            "source_path": source_path,
            "source": source_path,
            "original_filename": path.name,
            "file_type": path.suffix.lower().lstrip("."),
            "title": path.stem,
            "chunk_index": chunk.chunk_index,
            "source_hash": source_hash,
            "index_schema_version": _INDEX_SCHEMA_VERSION,
            "page": chunk.page_number if chunk.page_number is not None else 0,
            "structured_data": json.dumps(
                list(chunk.structured_data), ensure_ascii=False
            ),
        }
        if chunk.page_number is not None:
            metadata["page_number"] = chunk.page_number
        records.append({
            "id": chunk_id,
            "text": chunk.text,
            "metadata": metadata,
        })

    fts_records = [
        {
            "chunk_id": record["id"],
            "document_id": document_id,
            "title": path.stem,
            "source_path": source_path,
            "file_type": path.suffix.lower().lstrip("."),
            "chunk_index": chunk.chunk_index,
            "page_number": chunk.page_number,
            "text": record["text"],
            "structured_data": json.dumps(list(chunk.structured_data), ensure_ascii=False),
        }
        for chunk, record in zip(chunks, records)
    ]
    _print_timing(
        "Chunk ID and metadata preparation",
        started_at,
        file=path.name,
        chunks=len(chunks),
    )

    return {
        "path": source_path,
        "document_id": document_id,
        "records": records,
        "fts_records": fts_records,
    }


def ingest_documents(
    documents_dir: Path | str | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Extract, chunk, embed, and persist all supported documents under a folder."""
    ingestion_started_at = time.perf_counter()
    global _last_inventory_signature
    root = Path(documents_dir or DOCUMENTS_DIR).resolve()
    started_at = time.perf_counter()
    root.mkdir(parents=True, exist_ok=True)
    _print_timing("Prepare documents directory", started_at)
    started_at = time.perf_counter()
    paths = _file_inventory(root)
    _print_timing("Scan document files", started_at, files=len(paths))
    started_at = time.perf_counter()
    signature = _inventory_signature(paths)
    _print_timing("Build file inventory signature", started_at)

    lock_started_at = time.perf_counter()
    with _ingestion_lock:
        _print_timing("Wait for ingestion lock", lock_started_at)
        inventory_changed = (
            root == DOCUMENTS_DIR.resolve()
            and signature != _last_inventory_signature
        )
        if not force and root == DOCUMENTS_DIR.resolve() and signature == _last_inventory_signature:
            _print_timing("Skip unchanged inventory", ingestion_started_at, files=len(paths))
            return {
                "documents_seen": len(paths),
                "documents_indexed": 0,
                "chunks_upserted": 0,
                "inventory_changed": False,
                "errors": [],
            }

        started_at = time.perf_counter()
        collection = get_vector_collection()
        existing = collection.get(include=["metadatas"])
        _print_timing("Open Chroma collection and load existing metadata", started_at)
        from memory.state_store import get_fts_index_records

        indexed_fts_ids = {
            record["chunk_id"] for record in get_fts_index_records()
        }
        started_at = time.perf_counter()
        existing_by_path: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for chunk_id, metadata in zip(existing["ids"], existing["metadatas"] or []):
            source_path = metadata.get("source_path")
            if source_path:
                existing_by_path.setdefault(source_path, []).append((chunk_id, metadata))
        _print_timing("Group indexed chunks by source file", started_at)

        live_paths = {str(path) for path in paths}
        report: dict[str, Any] = {
            "documents_seen": len(paths),
            "documents_indexed": 0,
            "chunks_upserted": 0,
            "inventory_changed": inventory_changed,
            "chunks_embedded": 0,
            "embedding_cache_hits": 0,
            "errors": [],
        }

        max_workers = min(4, max(1, len(paths)))
        prepared_records: dict[str, dict[str, Any]] = {}
        futures = []

        prepare_started_at = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for path in paths:
                source_path = str(path)
                old_chunks = existing_by_path.get(source_path, [])
                started_at = time.perf_counter()
                source_hash = _file_sha256(path)
                _print_timing("Check file content hash", started_at, file=path.name)

                if (
                    not force
                    and old_chunks
                    and all(chunk_id in indexed_fts_ids for chunk_id, _ in old_chunks)
                    and all(
                        metadata.get("index_schema_version") == _INDEX_SCHEMA_VERSION
                        for _, metadata in old_chunks
                    )
                    and all(metadata.get("source_hash") == source_hash for _, metadata in old_chunks)
                ):
                    continue

                futures.append((source_path, executor.submit(_prepare_document_record, path, root)))

            for source_path, future in futures:
                try:
                    prepared = future.result()
                except Exception as exc:  # pragma: no cover - defensive safety net
                    report["errors"].append({"path": source_path, "error": str(exc)})
                    continue

                if "error" in prepared:
                    report["errors"].append({"path": prepared["path"], "error": prepared["error"]})
                    continue

                prepared_records[source_path] = prepared

        _print_timing(
            "Prepare changed documents",
            prepare_started_at,
            files=len(prepared_records),
            errors=len(report["errors"]),
        )

        fts_changed = False
        for source_path, prepared in prepared_records.items():
            record_list = prepared["records"]
            ids = [record["id"] for record in record_list]
            old_chunks = existing_by_path.get(source_path, [])
            try:
                embeddings, embedded_count, cache_hits = _prepare_embeddings(record_list)
            except Exception as exc:
                report["errors"].append({"path": source_path, "error": f"Embedding failed: {exc}"})
                continue

            report["chunks_embedded"] += embedded_count
            report["embedding_cache_hits"] += cache_hits
            vector_future = None
            fts_future = None
            with ThreadPoolExecutor(max_workers=2) as writer_pool:
                vector_future = writer_pool.submit(
                    _upsert_prepared_document,
                    collection,
                    record_list,
                    embeddings,
                    source_path,
                )
                fts_future = writer_pool.submit(_index_prepared_document_fts, prepared)

                file_errors = []
                for step, future in (("Chroma upsert", vector_future), ("SQLite FTS indexing", fts_future)):
                    try:
                        future.result()
                    except Exception as exc:
                        file_errors.append(f"{step} failed: {exc}")

            if file_errors:
                report["errors"].append(
                    {"path": source_path, "error": "; ".join(file_errors)}
                )
                continue
            fts_changed = True

            obsolete_ids = [chunk_id for chunk_id, _ in old_chunks if chunk_id not in ids]
            if obsolete_ids:
                try:
                    delete_started_at = time.perf_counter()
                    collection.delete(ids=obsolete_ids)
                    _print_timing(
                        "Delete obsolete Chroma chunks",
                        delete_started_at,
                        file=Path(source_path).name,
                        chunks=len(obsolete_ids),
                    )
                except Exception as exc:
                    report["errors"].append(
                        {
                            "path": source_path,
                            "error": f"Could not delete obsolete Chroma chunks: {exc}",
                        }
                    )
                    continue

            report["documents_indexed"] += 1
            report["chunks_upserted"] += len(ids)

        stale_ids = [
            chunk_id
            for source_path, chunks in existing_by_path.items()
            if source_path not in live_paths
            for chunk_id, _ in chunks
        ]
        if stale_ids:
            try:
                delete_started_at = time.perf_counter()
                collection.delete(ids=stale_ids)
                _print_timing(
                    "Delete stale Chroma chunks",
                    delete_started_at,
                    chunks=len(stale_ids),
                )
                fts_delete_started_at = time.perf_counter()
                from memory.state_store import delete_fts_chunks_by_ids
                delete_fts_chunks_by_ids(stale_ids)
                _print_timing(
                    "Delete stale SQLite FTS chunks",
                    fts_delete_started_at,
                    chunks=len(stale_ids),
                )
                fts_changed = True
            except Exception as exc:
                report["errors"].append(
                    {"path": "<stale documents>", "error": f"Cleanup failed: {exc}"}
                )

        reconcile_started_at = time.perf_counter()
        try:
            vector_ids = set(collection.get(include=["metadatas"])["ids"])
            from memory.state_store import (
                delete_fts_chunks_by_ids,
                get_fts_index_records,
            )

            orphaned_fts_ids = [
                record["chunk_id"]
                for record in get_fts_index_records()
                if record["chunk_id"] not in vector_ids
            ]
            if orphaned_fts_ids:
                delete_fts_chunks_by_ids(orphaned_fts_ids)
                fts_changed = True
                _print_timing(
                    "Reconcile FTS-only chunks",
                    reconcile_started_at,
                    chunks=len(orphaned_fts_ids),
                )
        except Exception as exc:
            report["errors"].append({
                "path": "<index consistency>",
                "error": f"Could not reconcile ChromaDB and FTS5 chunk IDs: {exc}",
            })

        if fts_changed:
            try:
                from memory.state_store import optimize_fts_index

                optimize_started_at = time.perf_counter()
                optimize_fts_index()
                _print_timing("SQLite FTS optimize", optimize_started_at)
            except Exception as exc:
                report["errors"].append(
                    {"path": "<FTS index>", "error": f"FTS optimize failed: {exc}"}
                )

        if root == DOCUMENTS_DIR.resolve() and not report["errors"]:
            _last_inventory_signature = signature
        _print_timing(
            "Ingestion total",
            ingestion_started_at,
            files=len(paths),
            indexed=report["documents_indexed"],
            chunks=report["chunks_upserted"],
            errors=len(report["errors"]),
        )
        return report


def ensure_documents_indexed() -> dict[str, Any]:
    """Update the default corpus index only when the document inventory changes."""
    return ingest_documents()


def get_vector_store_stats() -> dict[str, Any]:
    """Return persisted chunk and embedding counts for verification."""
    collection = get_vector_collection()
    stored = collection.get(include=["embeddings"])
    embeddings = stored["embeddings"]
    embedding_count = 0 if embeddings is None else len(embeddings)
    dimensions = 0
    if embedding_count:
        dimensions = len(embeddings[0])
    return {
        "collection": CHROMA_COLLECTION_NAME,
        "persist_directory": str(CHROMA_DIR.resolve()),
        "chunks": len(stored["ids"]),
        "chunks_with_embeddings": embedding_count,
        "embedding_dimensions": dimensions,
    }


if __name__ == "__main__":
    ingestion_report = ingest_documents(force=True)
    stats = get_vector_store_stats()
    print(json.dumps({"ingestion": ingestion_report, "vector_store": stats}, indent=2))
    if ingestion_report["errors"]:
        raise SystemExit(1)
    if stats["chunks"] != stats["chunks_with_embeddings"]:
        raise SystemExit("ChromaDB verification failed: a stored chunk has no embedding.")
