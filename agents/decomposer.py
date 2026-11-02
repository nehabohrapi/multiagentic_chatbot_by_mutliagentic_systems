"""Backward-compatible decomposer shim that delegates to the Orchestrator."""

from __future__ import annotations

from typing import Any

from .orchestrator import plan_orchestration
from .schemas import SubQuestions


def decompose_question(question: str, route: str = "multi_part") -> SubQuestions:
    """Extract independent sub-questions using the unified Orchestrator."""
    plan = plan_orchestration(question)
    queries = [sq.query for sq in plan.sub_questions] if plan.sub_questions else [question]
    return SubQuestions(questions=queries)


def decomposer_node(state: dict[str, Any]) -> dict[str, Any]:
    """Decomposition node compatible with legacy graphs."""
    plan = state.get("orchestrator_plan") or {}
    sub_qs = (
        state.get("sub_questions")
        or [sq.get("query") for sq in plan.get("sub_questions", []) if sq.get("query")]
        or [state["question"]]
    )
    return {"sub_questions": sub_qs}
