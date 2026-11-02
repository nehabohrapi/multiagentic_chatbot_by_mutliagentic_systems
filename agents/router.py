"""Backward-compatible question routing shim that delegates to the Orchestrator."""

from __future__ import annotations

import re
from typing import Any

from .orchestrator import orchestrator_node, plan_orchestration
from .schemas import Route


def reformulate_query(
    question: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Make short follow-up questions searchable using recent user context."""
    normalized = (question or "").strip()
    if not normalized:
        raise ValueError("Question cannot be empty.")
    recent_user = next(
        (
            message.get("content", "").strip()
            for message in reversed(history or [])
            if message.get("role") == "user" and message.get("content", "").strip()
        ),
        "",
    )
    if recent_user and len(normalized.split()) <= 8:
        lowered = normalized.casefold()
        if any(
            token in re.findall(r"[a-z]+", lowered)
            for token in ("it", "this", "that", "they", "these")
        ):
            return f"{normalized} Context from the previous question: {recent_user}"
    return normalized


def route_question(question: str) -> Route:
    """Classify a question using the unified Orchestrator."""
    plan = plan_orchestration(question)
    route_name = plan.intent if plan.intent in ("factual", "multi_part", "comparison", "summary", "numeric") else (
        "summary" if plan.action == "summarize" else "factual"
    )
    return Route(route=route_name, reason=plan.reason or "Orchestrator plan")


router_node = orchestrator_node
