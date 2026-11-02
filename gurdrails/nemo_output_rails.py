"""Local NeMo-style output validation, provenance, coverage, and telemetry rails."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from telemetry import current_trace_usage, time_block

from agents import writer
from agents.retriever_worker import (
    NO_EVIDENCE_ANSWER,
    format_answer_citations,
)


def _load_output_rail_config() -> dict[str, Any]:
    config_path = Path(__file__).with_name("config.yml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_rails = (
        config.get("rails", {})
        .get("config", {})
        .get("output_rails")
    )
    if not isinstance(output_rails, dict):
        raise ValueError(f"Missing rails.config.output_rails in {config_path}")
    patterns = output_rails.get("prompt_leak_patterns")
    if not isinstance(patterns, list) or not all(isinstance(item, str) for item in patterns):
        raise ValueError(f"Invalid prompt_leak_patterns in {config_path}")
    citation_options = output_rails.get("citation_validation", {})
    summary_options = output_rails.get("summary_coverage", {})
    structure_options = output_rails.get("structure", {})
    required_fields = structure_options.get("required_fields", [])
    if not isinstance(required_fields, list) or not all(
        isinstance(item, str) for item in required_fields
    ):
        raise ValueError(f"Invalid output structure.required_fields in {config_path}")
    max_facts = int(summary_options.get("max_facts", 100))
    min_fact_overlap = float(summary_options.get("min_fact_overlap", 0.5))
    max_retries = int(output_rails.get("max_retries", 0))
    if max_facts < 0 or not 0 <= min_fact_overlap <= 1 or max_retries != 0:
        raise ValueError(f"Invalid output rail limits in {config_path}")
    return {
        "normalize_unicode": bool(output_rails.get("normalize_unicode", False)),
        "strip_control_characters": bool(
            output_rails.get("strip_control_characters", False)
        ),
        "prompt_leak_patterns": tuple(
            re.compile(pattern, re.IGNORECASE) for pattern in patterns
        ),
        "citation_validation": {
            "enabled": bool(citation_options.get("enabled", True)),
            "verify_document_id": bool(citation_options.get("verify_document_id", True)),
            "verify_chunk_id": bool(citation_options.get("verify_chunk_id", True)),
            "verify_page_number": bool(citation_options.get("verify_page_number", True)),
        },
        "summary_coverage": {
            "enabled": bool(summary_options.get("enabled", True)),
            "max_facts": max_facts,
            "min_fact_overlap": min_fact_overlap,
        },
        "required_fields": tuple(required_fields),
        "max_retries": max_retries,
    }


_OUTPUT_RAIL_CONFIG = _load_output_rail_config()


@dataclass(frozen=True)
class OutputRailsResult:
    """Sanitized answer and citation metadata returned by the local output rails."""

    answer: str
    citations: list[int]
    citation_metadata: list[dict[str, Any]]
    validation: dict[str, Any]
    summary_coverage: dict[str, Any] | None = None
    answerable: bool = True
    changed: bool = False
    reason: str | None = None


def sanitize_output_text(text: str) -> str:
    """Remove prompt artifacts and non-printing controls without an LLM call."""
    cleaned = text or ""
    if _OUTPUT_RAIL_CONFIG["normalize_unicode"]:
        cleaned = unicodedata.normalize("NFKC", cleaned)
    if _OUTPUT_RAIL_CONFIG["strip_control_characters"]:
        cleaned = "".join(
            char
            for char in cleaned
            if char in "\n\t" or not unicodedata.category(char).startswith("C")
        )
    for pattern in _OUTPUT_RAIL_CONFIG["prompt_leak_patterns"]:
        cleaned = pattern.sub("", cleaned)
    return cleaned.strip()


def _citation_source_metadata(
    citation_id: int,
    context_blocks: list[dict[str, Any]],
    provided_metadata: dict[int, dict[str, Any]],
    answer: str,
) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    if citation_id < 1 or citation_id > len(context_blocks):
        return None, [f"citation_{citation_id}_has_no_matching_context"], []

    chunk = context_blocks[citation_id - 1]
    prior = provided_metadata.get(citation_id, {})
    issues: list[str] = []
    warnings: list[str] = []
    source_path = str(
        chunk.get("source_path") or chunk.get("title") or chunk.get("filename") or ""
    )
    document = source_path.replace("\\", "/").rsplit("/", 1)[-1] or None
    document_id = chunk.get("document_id") or chunk.get("doc_id")
    chunk_id = chunk.get("chunk_id") or chunk.get("id")
    page_number = chunk.get("page_number")
    if page_number is None:
        page_number = chunk.get("page")

    checks = (
        ("verify_document_id", "document_id", document_id),
        ("verify_chunk_id", "chunk_id", chunk_id),
        ("verify_page_number", "page", page_number),
    )
    for config_key, field, expected in checks:
        if not _OUTPUT_RAIL_CONFIG["citation_validation"][config_key]:
            continue
        if expected is None or expected == "":
            if field == "page":
                warnings.append(f"citation_{citation_id}_page_unavailable")
            else:
                warnings.append(f"citation_{citation_id}_{field}_unavailable")
            continue
        supplied = prior.get(field)
        if supplied is not None and str(supplied) != str(expected):
            warnings.append(f"citation_{citation_id}_{field}_metadata_repaired")

    if page_number is not None:
        try:
            page_number = int(page_number)
        except (TypeError, ValueError):
            warnings.append(f"citation_{citation_id}_source_page_invalid")
            page_number = None
        if page_number is not None and page_number < 1:
            warnings.append(f"citation_{citation_id}_source_page_invalid")
            page_number = None

    canonical = {
        "citation": citation_id,
        "document": document or prior.get("document") or "Unknown document",
        "document_id": document_id,
        "chunk_id": chunk_id,
        "page": page_number,
        "supporting_claim": str(prior.get("supporting_claim") or ""),
    }
    if not canonical["supporting_claim"]:
        for sentence in re.split(r"(?<=[.!?])\s+", answer):
            if re.search(
                rf"[\[【](?:chunk\s*)?{citation_id}(?:\s*[,，]|\s*[\]】])",
                sentence,
                flags=re.IGNORECASE,
            ):
                canonical["supporting_claim"] = re.sub(
                    rf"[\[【](?:chunk\s*)?{citation_id}(?:\s*[,，][^\]】]*)?[\]】]",
                    "",
                    sentence,
                    flags=re.IGNORECASE,
                ).strip()
                canonical["supporting_claim"] = re.sub(
                    r"\s+([,.;:!?])",
                    r"\1",
                    canonical["supporting_claim"],
                )
                break
    return canonical, issues, warnings


def _remove_invalid_citations(answer: str, invalid_ids: set[int]) -> str:
    if not invalid_ids:
        return answer

    def replace(match: re.Match[str]) -> str:
        return "" if int(match.group(1)) in invalid_ids else match.group(0)

    cleaned = re.sub(
        r"[\[【](?:chunk\s*)?(\d+)(?:\s*[,，][^\]】]*)?[\]】]",
        replace,
        answer,
        flags=re.IGNORECASE,
    )
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def apply_output_rails(
    answer: str,
    context_blocks: list[dict[str, Any]] | None = None,
    citations: list[int] | None = None,
    route: str = "",
    citation_metadata: list[dict[str, Any]] | None = None,
    is_summary: bool = False,
) -> OutputRailsResult:
    """Run local structure, provenance, coverage, and text-cleanup rails."""
    invalid_context_type = context_blocks is not None and not isinstance(context_blocks, list)
    invalid_citation_type = citations is not None and not isinstance(citations, list)
    invalid_metadata_type = (
        citation_metadata is not None and not isinstance(citation_metadata, list)
    )
    if not isinstance(context_blocks, list):
        context_blocks = []
    if not isinstance(citation_metadata, list):
        citation_metadata = []
    if not isinstance(citations, list):
        citations = []
    provided_metadata = {
        int(item["citation"]): item
        for item in citation_metadata
        if isinstance(item, dict) and str(item.get("citation", "")).isdigit()
    }
    issues: list[str] = []
    warnings: list[str] = []
    if invalid_context_type:
        issues.append("invalid_context_blocks_type")
    if invalid_citation_type:
        issues.append("invalid_citations_type")
    if invalid_metadata_type:
        issues.append("invalid_citation_metadata_type")
    answerable = True
    if not isinstance(answer, str):
        issues.append("invalid_answer_type")
        cleaned = NO_EVIDENCE_ANSWER
        answerable = False
    elif not answer.strip():
        issues.append("empty_answer_replaced")
        cleaned = NO_EVIDENCE_ANSWER
        answerable = False
    else:
        cleaned = sanitize_output_text(answer)
        if not cleaned:
            issues.append("answer_empty_after_sanitization")
            cleaned = NO_EVIDENCE_ANSWER
            answerable = False

    if route in {"respond_directly", "clarify", "direct_reply"}:
        citation_ids: list[int] = []
        metadata: list[dict[str, Any]] = []
        formatted = cleaned
    else:
        answer_citations = writer.extract_citations(cleaned)
        requested_citations = (answer_citations or citations) if answerable else []
        citation_ids = sorted({int(item) for item in requested_citations if str(item).isdigit()})
        citation_count = len(citation_ids)
        metadata = []
        invalid_ids: set[int] = set()
        if _OUTPUT_RAIL_CONFIG["citation_validation"]["enabled"]:
            for citation_id in citation_ids:
                canonical, citation_issues, citation_warnings = _citation_source_metadata(
                    citation_id,
                    context_blocks,
                    provided_metadata,
                    cleaned,
                )
                issues.extend(citation_issues)
                warnings.extend(citation_warnings)
                if canonical is None:
                    invalid_ids.add(citation_id)
                else:
                    metadata.append(canonical)
            citation_ids = [item for item in citation_ids if item not in invalid_ids]
            cleaned = _remove_invalid_citations(cleaned, invalid_ids)
        else:
            metadata = [
                provided_metadata[item]
                for item in citation_ids
                if item in provided_metadata
            ]
        formatted = (
            format_answer_citations(cleaned, context_blocks)
            if context_blocks and cleaned != NO_EVIDENCE_ANSWER
            else cleaned
        )

    coverage = None
    if is_summary and _OUTPUT_RAIL_CONFIG["summary_coverage"]["enabled"]:
        from agents.summarizer import assess_summary_coverage

        coverage = assess_summary_coverage(
            cleaned,
            context_blocks,
            max_facts=_OUTPUT_RAIL_CONFIG["summary_coverage"]["max_facts"],
            min_fact_overlap=_OUTPUT_RAIL_CONFIG["summary_coverage"]["min_fact_overlap"],
        )
        if not coverage["complete"]:
            warnings.append("summary_coverage_incomplete")

    structure = {
        "answer": isinstance(formatted, str) and bool(formatted.strip()),
        "citations": isinstance(citation_ids, list)
        and all(isinstance(item, int) and item > 0 for item in citation_ids),
        "citation_metadata": isinstance(metadata, list)
        and all(isinstance(item, dict) for item in metadata),
        "validation": True,
    }
    missing_fields = [
        field for field in _OUTPUT_RAIL_CONFIG["required_fields"]
        if not structure.get(field, False)
    ]
    if missing_fields:
        issues.extend(f"required_output_field_invalid:{field}" for field in missing_fields)

    validation = {
        "valid": not issues and all(structure.values()),
        "structure": structure,
        "citation_validation": {
            "enabled": _OUTPUT_RAIL_CONFIG["citation_validation"]["enabled"],
            "citations_checked": citation_count if route not in {
                "respond_directly", "clarify", "direct_reply"
            } else 0,
            "complete": _OUTPUT_RAIL_CONFIG["citation_validation"]["enabled"] and not any(
                warning.endswith("_unavailable") or warning.endswith("_invalid")
                for warning in warnings
            ) and not any(
                issue.startswith("citation_") for issue in issues
            ),
        },
        "issues": list(dict.fromkeys(issues)),
        "warnings": list(dict.fromkeys(warnings)),
    }
    return OutputRailsResult(
        answer=formatted,
        citations=citation_ids,
        citation_metadata=metadata,
        validation=validation,
        summary_coverage=coverage,
        answerable=answerable and formatted != NO_EVIDENCE_ANSWER,
        changed=formatted != answer,
        reason="; ".join(validation["issues"]) or None,
    )


def nemo_output_rails_node(state: dict[str, Any]) -> dict[str, Any]:
    """Apply local output rails and attach validation and execution metrics."""
    started = perf_counter()
    plan = state.get("orchestrator_plan") or {}
    is_summary = bool(
        state.get("is_summary")
        or state.get("summary_model_calls")
        or state.get("route") == "summary"
        or (
            isinstance(plan, dict)
            and plan.get("action") == "summarize"
        )
    )
    with time_block(
        "NeMo Output Rails: Validate & Format",
        category="GUARDRAIL",
        level=1,
    ) as timing:
        raw_answer = state.get("raw_answer") or state.get("draft") or state.get("answer") or ""
        result = apply_output_rails(
            raw_answer,
            context_blocks=state.get("context_blocks") or [],
            citations=state.get("citations"),
            route=state.get("action") or state.get("route") or "",
            citation_metadata=state.get("citation_metadata") or [],
            is_summary=is_summary,
        )
        elapsed = round(perf_counter() - started, 6)
        token_usage = state.get("token_usage") or current_trace_usage()
        timing["details"] = (
            f"changed={result.changed}, chars={len(result.answer)}, "
            f"citations={len(result.citations)}, llm_calls=0, retries=0, "
            f"elapsed={elapsed:.6f}s"
        )

    decision = {
        "stage": "nemo_output_rails",
        "engine": "local_deterministic",
        "changed": result.changed,
        "validation": result.validation,
        "summary_coverage": result.summary_coverage,
        "elapsed_seconds": elapsed,
        "token_usage": {
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "total_tokens": token_usage.get("total_tokens", 0),
        },
        "query_llm_calls": (
            int(token_usage.get("calls", 0))
            + int(state.get("summary_model_calls", 0))
        ),
        "citations_count": len(result.citations),
        "llm_calls": 0,
        "retry_count": 0,
        "retry_limit": _OUTPUT_RAIL_CONFIG["max_retries"],
    }
    if result.reason:
        decision["reason"] = result.reason

    return {
        "answer": result.answer,
        "draft": result.answer,
        "citations": result.citations,
        "citation_metadata": result.citation_metadata,
        "evidence_sufficient": state.get("evidence_sufficient", False),
        "limitations": state.get("limitations", []),
        "answerable": bool(state.get("answerable", True)) and result.answerable,
        "output_validation": result.validation,
        "summary_coverage": result.summary_coverage,
        "decision_log": [decision],
    }
