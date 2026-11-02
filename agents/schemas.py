"""Validated data contracts shared by the multi-agent graph."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Route(BaseModel):
    """Classification returned by the router agent."""

    model_config = ConfigDict(extra="forbid", strict=True)

    route: Literal["factual", "multi_part", "comparison", "summary", "numeric"]
    reason: str = Field(min_length=1)


class SubQuestions(BaseModel):
    """Independent retrieval questions produced by the decomposer."""

    model_config = ConfigDict(extra="forbid", strict=True)

    questions: list[str] = Field(min_length=1, max_length=4)

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, questions: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for question in questions:
            normalized = question.strip()
            if not normalized:
                raise ValueError("Sub-questions cannot be empty.")
            key = normalized.casefold()
            if key not in seen:
                seen.add(key)
                cleaned.append(normalized)
        if not cleaned:
            raise ValueError("At least one sub-question is required.")
        return cleaned


class ClaimCheck(BaseModel):
    """Verification result for one claim in a generated answer."""

    model_config = ConfigDict(extra="forbid", strict=True)

    claim: str = Field(min_length=1)
    passage_ids: list[int]
    supported: Literal["yes", "partial", "no"]

    @field_validator("passage_ids", mode="before")
    @classmethod
    def normalize_passage_ids(cls, passage_ids: Any) -> Any:
        """Accept JSON numeric strings while keeping strict integer validation."""
        if not isinstance(passage_ids, list):
            return passage_ids
        normalized: list[Any] = []
        for passage_id in passage_ids:
            if isinstance(passage_id, str) and passage_id.strip().isdigit():
                normalized.append(int(passage_id.strip()))
            else:
                normalized.append(passage_id)
        return normalized


class Verdict(BaseModel):
    """Structured verifier result."""

    model_config = ConfigDict(extra="forbid", strict=True)

    ok: bool
    claims: list[ClaimCheck]
    missing: list[str]


class SubQuestion(BaseModel):
    """Sub-question detail in an orchestrator plan."""

    model_config = ConfigDict(extra="ignore")

    id: str = "q1"
    query: str
    doc_ids: list[str] = Field(default_factory=list)
    retrieval: Literal["hybrid", "table", "summary_cache"] = "hybrid"
    question_type: str = "factual"
    direction: str = "find_specific_fact"
    entities: list[str] = Field(default_factory=list)
    attributes: list[str] = Field(default_factory=list)


class OrchestratorPlan(BaseModel):
    """Structured execution plan produced by the single Orchestrator."""

    model_config = ConfigDict(extra="ignore")

    action: Literal["respond_directly", "clarify", "retrieve", "summarize"]
    intent: Literal["factual", "comparison", "numeric", "multi_part", "summary", "other"]
    ambiguity: Literal["none", "resolved_by_rewrite", "needs_clarification"] = "none"
    rewritten_query: str = ""
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    output_format: Literal["short_answer", "bullets", "table", "long_summary"] = "short_answer"
    needs_verification: bool = False
    direct_reply: str | None = None
    clarifying_question: str | None = None
    reason: str = Field(default="")
