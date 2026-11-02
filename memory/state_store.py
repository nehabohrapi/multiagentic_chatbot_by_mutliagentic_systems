"""Small SQLite store for request state and reusable document summaries."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from config import STATE_DB_FILE

logger = logging.getLogger(__name__)
_DEFAULT_STATE_DB_PATH = Path(STATE_DB_FILE).resolve()
_connection_local = threading.local()


def _figure_records(structured_data: Any) -> list[dict[str, Any]]:
    if isinstance(structured_data, str):
        try:
            structured_data = json.loads(structured_data)
        except json.JSONDecodeError:
            return []
    if not isinstance(structured_data, list):
        return []
    return [
        item
        for item in structured_data
        if isinstance(item, dict)
        and item.get("kind") == "figure"
        and str(item.get("image_path") or "").strip()
    ]


def _sync_chunk_image_assets(
    connection: sqlite3.Connection,
    chunk: dict[str, Any],
) -> None:
    chunk_id = str(chunk.get("chunk_id") or "")
    if not chunk_id:
        return
    connection.execute(
        "DELETE FROM image_assets WHERE chunk_id = ?",
        (chunk_id,),
    )
    for index, figure in enumerate(_figure_records(chunk.get("structured_data"))):
        image_path = str(figure["image_path"]).strip()
        asset_id = hashlib.sha256(
            f"{chunk_id}\0{image_path}\0{index}".encode("utf-8")
        ).hexdigest()
        connection.execute(
            """
            INSERT OR REPLACE INTO image_assets
                (asset_id, chunk_id, document_id, title, source_path, image_path,
                 file_name, page_number, description, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asset_id,
                chunk_id,
                str(chunk.get("document_id") or ""),
                str(chunk.get("title") or ""),
                str(chunk.get("source_path") or ""),
                image_path,
                str(figure.get("source") or Path(image_path).name),
                figure.get("page_number", chunk.get("page_number")),
                str(figure.get("description") or ""),
                json.dumps(figure, ensure_ascii=False),
            ),
        )


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    db_path = Path(STATE_DB_FILE).resolve()
    reuse_connection = db_path == _DEFAULT_STATE_DB_PATH
    connection = getattr(_connection_local, "connection", None) if reuse_connection else None
    initialize = connection is None or getattr(
        _connection_local, "db_path", None
    ) != db_path
    if initialize:
        if connection is not None:
            connection.close()
        connection = sqlite3.connect(db_path, timeout=30)
        if reuse_connection:
            _connection_local.connection = connection
            _connection_local.db_path = db_path

    try:
        if initialize:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS summary_cache (
                    cache_key TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    raw_answer TEXT NOT NULL DEFAULT '',
                    citations TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            summary_columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(summary_cache)")
            }
            if "raw_answer" not in summary_columns:
                connection.execute(
                    "ALTER TABLE summary_cache "
                    "ADD COLUMN raw_answer TEXT NOT NULL DEFAULT ''"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS summary_progress (
                    document_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    batch_index INTEGER NOT NULL,
                    summary TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (document_id, model, content_hash, batch_index)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS gemini_token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    tokens INTEGER NOT NULL CHECK (tokens >= 0)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_gemini_token_usage_created_at "
                "ON gemini_token_usage(created_at)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_state (
                    run_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    route TEXT,
                    status TEXT NOT NULL,
                    calls INTEGER NOT NULL DEFAULT 0,
                    result TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    document_id UNINDEXED,
                    title,
                    source_path UNINDEXED,
                    file_type UNINDEXED,
                    chunk_index UNINDEXED,
                    page_number UNINDEXED,
                    text,
                    structured_data UNINDEXED
                )
                """
            )
            image_assets_table_exists = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'image_assets'
                """
            ).fetchone() is not None
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS image_assets (
                    asset_id TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    page_number INTEGER,
                    description TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_image_assets_chunk "
                "ON image_assets(chunk_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_image_assets_document "
                "ON image_assets(document_id)"
            )
            if not image_assets_table_exists:
                for row in connection.execute(
                    """
                    SELECT chunk_id, document_id, title, source_path, page_number,
                           structured_data
                    FROM chunks_fts
                    """
                ):
                    _sync_chunk_image_assets(
                        connection,
                        {
                            "chunk_id": row[0],
                            "document_id": row[1],
                            "title": row[2],
                            "source_path": row[3],
                            "page_number": row[4],
                            "structured_data": row[5],
                        },
                    )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunk_embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    embedding_json TEXT NOT NULL
                )
                """
            )
            connection.commit()
        yield connection
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        if reuse_connection and initialize:
            connection.close()
            del _connection_local.connection
            del _connection_local.db_path
        raise
    else:
        if connection.in_transaction:
            connection.rollback()
    finally:
        if not reuse_connection:
            connection.close()


def content_hash(chunks: list[dict[str, Any]]) -> str:
    """Return a stable digest for the ordered document chunks."""
    payload = "\n".join(
        f"{chunk.get('chunk_id')}:{chunk.get('text', '')}" for chunk in chunks
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_summary(
    *,
    document_id: str,
    model: str,
    digest: str,
    ttl_seconds: int | None = None,
) -> dict[str, Any] | None:
    """Return a cached summary, optionally limiting how long it remains valid."""
    key = f"{document_id}:{model}:{digest}"
    with _connection() as connection:
        row = connection.execute(
            """
            SELECT answer, raw_answer, citations, created_at
            FROM summary_cache WHERE cache_key = ?
            """,
            (key,),
        ).fetchone()
    if row is None or (
        ttl_seconds is not None and time.time() - float(row[3]) > ttl_seconds
    ):
        return None
    return {
        "answer": row[0],
        "raw_answer": row[1] or row[0],
        "citations": json.loads(row[2]),
    }


def put_summary(
    *,
    document_id: str,
    model: str,
    digest: str,
    answer: str,
    raw_answer: str,
    citations: list[int],
) -> None:
    """Persist a summary for the current document content and model."""
    key = f"{document_id}:{model}:{digest}"
    with _connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO summary_cache
            (cache_key, document_id, model, content_hash, answer, raw_answer, citations, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                document_id,
                model,
                digest,
                answer,
                raw_answer,
                json.dumps(citations),
                time.time(),
            ),
        )
        connection.commit()


def get_summary_progress(
    *,
    document_id: str,
    model: str,
    digest: str,
    batch_index: int,
) -> str | None:
    """Return a saved map-stage result for one document batch."""
    with _connection() as connection:
        row = connection.execute(
            """
            SELECT summary FROM summary_progress
            WHERE document_id = ? AND model = ? AND content_hash = ?
                AND batch_index = ?
            """,
            (document_id, model, digest, batch_index),
        ).fetchone()
    return str(row[0]) if row else None


def put_summary_progress(
    *,
    document_id: str,
    model: str,
    digest: str,
    batch_index: int,
    summary: str,
) -> None:
    """Persist one successful map-stage result for crash-safe resumption."""
    with _connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO summary_progress
            (document_id, model, content_hash, batch_index, summary, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (document_id, model, digest, batch_index, summary, time.time()),
        )
        connection.commit()


def clear_summary_progress(
    *,
    document_id: str,
    model: str,
    digest: str,
) -> None:
    """Remove map-stage checkpoints after a complete summary is persisted."""
    with _connection() as connection:
        connection.execute(
            """
            DELETE FROM summary_progress
            WHERE document_id = ? AND model = ? AND content_hash = ?
            """,
            (document_id, model, digest),
        )
        connection.commit()


def reserve_gemini_tokens(
    tokens: int,
    *,
    limit: int,
    hard_limit: int | None = None,
    window_seconds: int = 60,
) -> float:
    """Reserve shared rolling-window Gemini input TPM across app processes."""
    request_limit = hard_limit if hard_limit is not None else limit
    if tokens < 0:
        raise ValueError("Gemini token reservation cannot be negative.")
    if tokens > request_limit:
        raise ValueError(
            f"A Gemini request uses {tokens:,} input tokens, exceeding the "
            f"configured {request_limit:,}-token single-request quota ceiling."
        )

    started = time.perf_counter()
    while True:
        wait_seconds = 0.0
        with _connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = time.time()
            cutoff = now - window_seconds
            connection.execute(
                "DELETE FROM gemini_token_usage WHERE created_at <= ?",
                (cutoff,),
            )
            used, oldest = connection.execute(
                "SELECT COALESCE(SUM(tokens), 0), MIN(created_at) "
                "FROM gemini_token_usage"
            ).fetchone()
            if (
                int(used) + tokens <= limit
                or (int(used) == 0 and tokens <= request_limit)
            ):
                connection.execute(
                    "INSERT INTO gemini_token_usage (created_at, tokens) VALUES (?, ?)",
                    (now, tokens),
                )
                connection.commit()
                return time.perf_counter() - started

            connection.rollback()
            wait_seconds = max(
                0.01,
                window_seconds - (now - float(oldest)),
            )
        time.sleep(wait_seconds)


def record_run(
    run_id: str,
    question: str,
    *,
    route: str | None = None,
    status: str = "started",
    calls: int = 0,
    result: dict[str, Any] | None = None,
) -> None:
    """Create or update the durable state record for one graph invocation."""
    with _connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO run_state
            (run_id, question, route, status, calls, result, created_at)
            VALUES (?, ?, ?, ?, ?, ?, COALESCE(
                (SELECT created_at FROM run_state WHERE run_id = ?), ?
            ))
            """,
            (
                run_id,
                question,
                route,
                status,
                calls,
                json.dumps(result, default=str) if result is not None else None,
                run_id,
                time.time(),
            ),
        )
        connection.commit()


_FTS_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could",
    "did", "do", "does", "doing", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "itself", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "would", "you", "your",
    "yours", "yourself", "yourselves", "tell", "give", "explain",
    "describe", "find", "show",
}


def _sanitize_fts_query(query: str) -> str:
    """Prepare a user query for safe SQLite FTS5 matching, filtering common stop words."""
    tokens = re.findall(r"[A-Za-z0-9_#$%.+-]+", query)
    if not tokens:
        return ""
    clean_tokens = [re.sub(r'["\']', "", t) for t in tokens if len(t) >= 2]
    if not clean_tokens:
        return ""
    content_tokens = [t for t in clean_tokens if t.lower() not in _FTS_STOP_WORDS]
    selected_tokens = content_tokens if content_tokens else clean_tokens
    return " OR ".join(f'"{t}"' for t in selected_tokens)


def index_fts_chunks(chunks: list[dict[str, Any]]) -> None:
    """Replace one file's chunks in one SQLite transaction."""
    if not chunks:
        return
    with _connection() as connection:
        source_path = chunks[0].get("source_path", "")
        connection.execute(
            "DELETE FROM chunks_fts WHERE source_path = ?",
            (source_path,),
        )
        connection.execute(
            "DELETE FROM image_assets WHERE source_path = ?",
            (source_path,),
        )
        connection.executemany(
            """
                INSERT INTO chunks_fts
                (chunk_id, document_id, title, source_path, file_type, chunk_index, page_number, text, structured_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
            [
                (
                    chunk["chunk_id"],
                    chunk.get("document_id", ""),
                    chunk.get("title", ""),
                    chunk.get("source_path", ""),
                    chunk.get("file_type", ""),
                    chunk.get("chunk_index", 0),
                    chunk.get("page_number"),
                    chunk.get("text", ""),
                    chunk.get("structured_data", ""),
                )
                for chunk in chunks
            ],
        )
        for chunk in chunks:
            _sync_chunk_image_assets(connection, chunk)
        connection.commit()


def update_fts_chunk_metadata(chunk_id: str, structured_data: str) -> None:
    """Update one indexed chunk's structured metadata without reindexing its file."""
    with _connection() as connection:
        connection.execute(
            "UPDATE chunks_fts SET structured_data = ? WHERE chunk_id = ?",
            (structured_data, chunk_id),
        )
        row = connection.execute(
            """
            SELECT chunk_id, document_id, title, source_path, page_number
            FROM chunks_fts
            WHERE chunk_id = ?
            """,
            (chunk_id,),
        ).fetchone()
        if row:
            _sync_chunk_image_assets(
                connection,
                {
                    "chunk_id": row[0],
                    "document_id": row[1],
                    "title": row[2],
                    "source_path": row[3],
                    "page_number": row[4],
                    "structured_data": structured_data,
                },
            )
        connection.commit()


def get_cached_chunk_embeddings(cache_keys: list[str]) -> dict[str, list[float]]:
    """Load embedding vectors for a bounded set of cache keys."""
    if not cache_keys:
        return {}
    with _connection() as connection:
        placeholders = ",".join("?" for _ in cache_keys)
        rows = connection.execute(
            f"""
            SELECT cache_key, embedding_json
            FROM chunk_embedding_cache
            WHERE cache_key IN ({placeholders})
            """,
            cache_keys,
        ).fetchall()
    return {
        cache_key: [float(value) for value in json.loads(embedding_json)]
        for cache_key, embedding_json in rows
    }


def put_cached_chunk_embeddings(embeddings: dict[str, list[float]]) -> None:
    """Persist generated embedding vectors in one SQLite transaction."""
    if not embeddings:
        return
    with _connection() as connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO chunk_embedding_cache (cache_key, embedding_json)
            VALUES (?, ?)
            """,
            [
                (cache_key, json.dumps(vector, separators=(",", ":")))
                for cache_key, vector in embeddings.items()
            ],
        )
        connection.commit()


def optimize_fts_index() -> None:
    """Merge FTS5 index segments after a complete ingestion run."""
    with _connection() as connection:
        connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")
        connection.commit()


def delete_fts_chunks_by_ids(chunk_ids: list[str]) -> None:
    """Delete chunks from the FTS5 index by their chunk IDs."""
    if not chunk_ids:
        return
    with _connection() as connection:
        connection.executemany(
            "DELETE FROM chunks_fts WHERE chunk_id = ?",
            [(chunk_id,) for chunk_id in chunk_ids],
        )
        connection.executemany(
            "DELETE FROM image_assets WHERE chunk_id = ?",
            [(chunk_id,) for chunk_id in chunk_ids],
        )
        connection.commit()


def delete_fts_chunks_by_paths(paths: list[str]) -> None:
    """Delete chunks from the FTS5 index for specified source file paths."""
    if not paths:
        return
    with _connection() as connection:
        connection.executemany(
            "DELETE FROM chunks_fts WHERE source_path = ?",
            [(path,) for path in paths],
        )
        connection.executemany(
            "DELETE FROM image_assets WHERE source_path = ?",
            [(path,) for path in paths],
        )
        connection.commit()

def search_fts_chunks(
    query: str,
    max_results: int = 10,
    file_type: str | None = None,
    doc_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Search chunks using SQLite FTS5 BM25 ranking with optional filters."""
    sanitized = _sanitize_fts_query(query)
    if not sanitized:
        return []

    conditions = ["chunks_fts MATCH ?"]
    params: list[Any] = [sanitized]

    if file_type:
        conditions.append("file_type = ?")
        params.append(file_type)

    if doc_ids:
        placeholders = ",".join("?" for _ in doc_ids)
        conditions.append(f"document_id IN ({placeholders})")
        params.extend(doc_ids)

    where_clause = " AND ".join(conditions)
    params.append(max_results)

    with _connection() as connection:
        try:
            cursor = connection.execute(
                f"""
                SELECT chunk_id, document_id, title, source_path, file_type,
                       chunk_index, page_number, text, structured_data, rank
                FROM chunks_fts
                WHERE {where_clause}
                ORDER BY rank
                LIMIT ?
                """,
                params,
            )
            rows = cursor.fetchall()
        except sqlite3.OperationalError as exc:
            logger.exception("FTS5 chunk search failed for query %r.", query)
            raise RuntimeError("FTS5 chunk search failed.") from exc

    results = []
    for row in rows:
        results.append({
            "chunk_id": row[0],
            "document_id": row[1],
            "title": row[2],
            "source_path": row[3],
            "file_type": row[4],
            "chunk_index": row[5],
            "page_number": row[6],
            "text": row[7],
            "structured_data": row[8],
            "bm25_score": max(0.01, -float(row[9])),
        })
    return results


def get_fts_index_records() -> list[dict[str, Any]]:
    """Return chunk identity metadata for cross-index consistency checks."""
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT chunk_id, document_id, source_path, chunk_index, page_number
            FROM chunks_fts
            """
        ).fetchall()
    return [
        {
            "chunk_id": row[0],
            "document_id": row[1],
            "source_path": row[2],
            "chunk_index": row[3],
            "page_number": row[4],
        }
        for row in rows
    ]


def search_visual_chunks(
    query: str = "",
    doc_ids: list[str] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Retrieve indexed image paths and source metadata with matching chunks."""
    with _connection() as connection:
        conditions = [
            "EXISTS (SELECT 1 FROM image_assets ia "
            "WHERE ia.chunk_id = chunks_fts.chunk_id)"
        ]
        params: list[Any] = []

        if doc_ids:
            placeholders = ",".join("?" for _ in doc_ids)
            conditions.append(f"document_id IN ({placeholders})")
            params.extend(doc_ids)

        tokens = re.findall(r"[A-Za-z0-9_#$%.+-]+", query)
        ignore_words = {
            "image", "images", "chart", "charts", "figure", "figures",
            "plot", "plots", "diagram", "diagrams", "photo", "photos",
            "picture", "pictures", "visual", "visuals", "show", "what",
            "find", "give", "me", "the", "in", "of", "and", "or", "for", "is", "are",
        }
        content_tokens = [
            re.sub(r'["\']', "", t)
            for t in tokens
            if len(t) >= 2 and t.lower() not in ignore_words
        ]

        where_sql = " AND ".join(conditions)
        rows: list[Any] = []

        if content_tokens:
            fts_query = " OR ".join(f'"{t}"' for t in content_tokens)
            try:
                cur = connection.execute(
                    f"""
                    SELECT chunk_id, document_id, title, source_path, file_type,
                           chunk_index, page_number, text, structured_data, rank
                    FROM chunks_fts
                    WHERE {where_sql} AND chunks_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    [*params, fts_query, limit],
                )
                rows = cur.fetchall()
            except sqlite3.OperationalError:
                rows = []

        if not rows:
            cur = connection.execute(
                f"""
                SELECT chunk_id, document_id, title, source_path, file_type,
                       chunk_index, page_number, text, structured_data, 0.0 as rank
                FROM chunks_fts
                WHERE {where_sql}
                ORDER BY chunk_index ASC
                LIMIT ?
                """,
                [*params, limit],
            )
            rows = cur.fetchall()

        asset_rows_by_chunk: dict[str, list[Any]] = {}
        if rows:
            placeholders = ",".join("?" for _ in rows)
            asset_rows = connection.execute(
                f"""
                SELECT asset_id, chunk_id, image_path, file_name, page_number,
                       description, metadata_json
                FROM image_assets
                WHERE chunk_id IN (
                    SELECT chunk_id FROM chunks_fts
                    WHERE chunk_id IN ({placeholders})
                )
                ORDER BY page_number, file_name
                """,
                [row[0] for row in rows],
            ).fetchall()
            for asset_row in asset_rows:
                asset_rows_by_chunk.setdefault(asset_row[1], []).append(asset_row)

    results = []
    for row in rows:
        image_assets = []
        for asset_row in asset_rows_by_chunk.get(row[0], []):
            try:
                metadata = json.loads(asset_row[6])
            except json.JSONDecodeError:
                metadata = {}
            metadata = {
                **metadata,
                "asset_id": asset_row[0],
                "image_path": asset_row[2],
                "source": asset_row[3],
                "page_number": asset_row[4],
                "description": asset_row[5],
            }
            image_assets.append(metadata)
        results.append({
            "chunk_id": row[0],
            "document_id": row[1],
            "title": row[2],
            "source_path": row[3],
            "file_type": row[4],
            "chunk_index": row[5],
            "page_number": row[6],
            "text": row[7],
            "structured_data": image_assets,
            "image_assets": image_assets,
        })
    return results
