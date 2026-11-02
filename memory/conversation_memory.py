"""SQLite-backed conversation memory store for Nexus with JSON fallback support."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List

from config import CONVERSATIONS_FILE, DATA_DIR

logger = logging.getLogger(__name__)

CONVERSATIONS_DB_FILE = getattr(
    __import__("config"), "CONVERSATIONS_DB_FILE", DATA_DIR / "conversations.sqlite3"
)
MEMORY_FILE: Path = CONVERSATIONS_FILE


def _is_json_mode() -> bool:
    """Detect if MEMORY_FILE has been patched to a JSON path for tests or legacy use."""
    return str(MEMORY_FILE).lower().endswith(".json")


@contextmanager
def _get_connection(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = db_path or MEMORY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=30.0)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                image_paths TEXT,
                workflow TEXT,
                created_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_session_id "
            "ON conversations(session_id, id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_created_at "
            "ON conversations(created_at)"
        )
        yield connection
    finally:
        connection.close()


def _sanitize_for_json(obj: Any) -> Any:
    """Ensure any nested object can be serialized cleanly to JSON."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, "model_dump"):
        return _sanitize_for_json(obj.model_dump())
    if hasattr(obj, "dict"):
        return _sanitize_for_json(obj.dict())
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    return str(obj)


def _sanitize_workflow_for_storage(workflow: dict[str, Any] | None) -> dict[str, Any] | None:
    """Strip recursive history and ensure clean serialization before persisting."""
    if not workflow or not isinstance(workflow, dict):
        return None
    cleaned = dict(workflow)
    # Crucial: Never persist history inside workflow to prevent 2^N exponential recursion
    cleaned.pop("history", None)
    return _sanitize_for_json(cleaned)


def load_history(session_id: str) -> list[dict[str, Any]]:
    """Load saved messages and optional image paths / workflow metadata for a session."""
    if _is_json_mode():
        return _load_history_json(session_id)

    if not MEMORY_FILE.exists():
        return []

    try:
        with _get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT role, content, image_paths, workflow, created_at
                FROM conversations
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            )
            rows = cursor.fetchall()

        messages: list[dict[str, Any]] = []
        for row in rows:
            msg: dict[str, Any] = {
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["created_at"],
            }
            if row["image_paths"]:
                try:
                    msg["image_paths"] = json.loads(row["image_paths"])
                except Exception:
                    msg["image_paths"] = []
            if row["workflow"]:
                try:
                    msg["workflow"] = json.loads(row["workflow"])
                except Exception:
                    pass
            messages.append(msg)
        return messages
    except Exception as exc:
        logger.warning("Failed to load history from SQLite for %s: %s", session_id, exc)
        return []


def save_message(
    session_id: str,
    role: str,
    content: str,
    *,
    image_paths: list[str] | None = None,
    workflow: dict | None = None,
) -> None:
    """Append a text message and optional image paths / workflow metadata to a conversation."""
    if _is_json_mode():
        _save_message_json(
            session_id,
            role,
            content,
            image_paths=image_paths,
            workflow=workflow,
        )
        return

    deduped_images = list(dict.fromkeys(image_paths)) if image_paths else None
    images_json = json.dumps(deduped_images, ensure_ascii=False) if deduped_images else None
    sanitized_wf = _sanitize_workflow_for_storage(workflow)
    workflow_json = json.dumps(sanitized_wf, ensure_ascii=False) if sanitized_wf else None
    now = time.time()

    with _get_connection() as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO conversations (session_id, role, content, image_paths, workflow, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, role, content, images_json, workflow_json, now),
            )


def list_conversations() -> List[str]:
    """Return all conversation/session IDs recorded in the database."""
    if _is_json_mode():
        return _list_conversations_json()

    if not MEMORY_FILE.exists():
        return []

    try:
        with _get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT session_id, MAX(created_at) as last_active
                FROM conversations
                GROUP BY session_id
                ORDER BY last_active DESC
                """
            )
            return [str(row["session_id"]) for row in cursor.fetchall()]
    except Exception as exc:
        logger.warning("Failed to list conversations from SQLite: %s", exc)
        return []


def create_new_conversation(prefix: str = "conv") -> str:
    """Generate a clean, unique conversation ID."""
    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:6]}"


def clear_conversation(session_id: str) -> bool:
    """Remove a conversation from the database."""
    if _is_json_mode():
        return _clear_conversation_json(session_id)

    if not MEMORY_FILE.exists():
        return False

    try:
        with _get_connection() as conn:
            with conn:
                cursor = conn.execute(
                    "DELETE FROM conversations WHERE session_id = ?",
                    (session_id,),
                )
                return cursor.rowcount > 0
    except Exception as exc:
        logger.warning("Failed to clear conversation %s from SQLite: %s", session_id, exc)
        return False


# ==============================================================================
# JSON Fallback for legacy mode or tests patching MEMORY_FILE with a .json path
# ==============================================================================

def _load_history_json(session_id: str) -> list[dict[str, Any]]:
    if not MEMORY_FILE.exists():
        return []
    try:
        with MEMORY_FILE.open("r", encoding="utf-8") as file:
            conversations = json.load(file)
        return conversations.get(session_id, [])
    except Exception:
        return []


def _save_message_json(
    session_id: str,
    role: str,
    content: str,
    *,
    image_paths: list[str] | None = None,
    workflow: dict | None = None,
) -> None:
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if MEMORY_FILE.exists():
        try:
            with MEMORY_FILE.open("r", encoding="utf-8") as file:
                conversations = json.load(file)
        except Exception:
            conversations = {}
    else:
        conversations = {}

    history = conversations.setdefault(session_id, [])
    message: dict[str, Any] = {
        "role": role,
        "content": content,
        "timestamp": time.time(),
    }
    if image_paths:
        message["image_paths"] = list(dict.fromkeys(image_paths))
    if workflow:
        sanitized_wf = _sanitize_workflow_for_storage(workflow)
        if sanitized_wf:
            message["workflow"] = sanitized_wf
    history.append(message)

    with MEMORY_FILE.open("w", encoding="utf-8") as file:
        json.dump(conversations, file, indent=2, ensure_ascii=False)


def _list_conversations_json() -> list[str]:
    if not MEMORY_FILE.exists():
        return []
    try:
        with MEMORY_FILE.open("r", encoding="utf-8") as file:
            conversations = json.load(file)
            return list(conversations.keys())
    except Exception:
        return []


def _clear_conversation_json(session_id: str) -> bool:
    if not MEMORY_FILE.exists():
        return False
    try:
        with MEMORY_FILE.open("r", encoding="utf-8") as file:
            conversations = json.load(file)
        if session_id in conversations:
            del conversations[session_id]
            with MEMORY_FILE.open("w", encoding="utf-8") as file:
                json.dump(conversations, file, indent=2, ensure_ascii=False)
            return True
    except Exception:
        pass
    return False
