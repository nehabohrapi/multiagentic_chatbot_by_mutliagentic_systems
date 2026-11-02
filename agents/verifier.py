"""Cohere-backed claim verification shared by the application pipelines."""

from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any

from config import (
    BASE_DIR,
    VERIFIER_BATCH_SIZE,
    VERIFIER_MAX_CONCURRENCY,
    VERIFIER_SUMMARY_BATCH_SIZE,
    VERIFIER_SUMMARY_LARGE_BATCH_SIZE,
)
from verifier_agent import CohereVerifier, verify_passages

from . import retriever_worker
from .schemas import ClaimCheck, Verdict

VERIFICATION_DB = BASE_DIR / "verifier.db"


def _as_verdict(report: dict[str, Any]) -> Verdict:
    claim_checks: list[ClaimCheck] = []
    missing: list[str] = []
    for result in report["claims"]:
        supported = (
            result.get("structured_verdict") == "supported"
            or result.get("status") in {"SUPPORTED", "SUPPORTED_WEB"}
        )
        sentence = str(result.get("sentence") or result["claim"])
        structured_status = str(result.get("structured_verdict") or result.get("status") or "unsupported")
        claim_checks.append(
            ClaimCheck(
                claim=sentence,
                passage_ids=list(result.get("verified_citation_ids") or []),
                supported="yes" if supported else "no",
            )
        )
        if not supported:
            last_attempt = (
                result.get("evidence_trail")[-1]
                if result.get("evidence_trail")
                else {}
            )
            reason = str(last_attempt.get("reason") or result["status"])
            missing.append(f"{sentence} ({result['status']}: {reason})")
    return Verdict(
        ok=(
            all(claim.supported == "yes" for claim in claim_checks)
            if claim_checks
            else report.get("run_id") is None
        ),
        claims=claim_checks,
        missing=missing,
    )


def verify_answer(
    draft: str,
    passages: list[dict[str, Any]],
    *,
    db_path: str | Path = VERIFICATION_DB,
    llm: CohereVerifier | None = None,
    allow_extractive: bool = True,
    batch_size: int | None = None,
    max_concurrency: int = VERIFIER_MAX_CONCURRENCY,
    is_summary: bool = False,
    include_saved_web: bool = True,
    revise: bool = False,
) -> dict[str, Any]:
    """Verify the full draft against retrieved passages and saved web sources."""
    if (draft or "").strip() == retriever_worker.NO_EVIDENCE_ANSWER:
        return {
            "run_id": None,
            "score": "0/0 claims verified",
            "verified_claims": 0,
            "total_claims": 0,
            "claims": [],
            "live_web_search": "disabled",
        }

    evidence_passages = []
    for citation_id, passage in enumerate(passages, 1):
        source_citation_id = passage.get("citation_id")
        text = str(passage.get("text") or "")
        structured_data = passage.get("structured_data")
        if structured_data:
            text += "\n\nStructured evidence:\n" + json.dumps(
                structured_data,
                ensure_ascii=False,
                default=str,
            )
        if text.strip():
            evidence_passages.append({
                "source_type": "document",
                "text": text,
                "citation_id": (
                    int(source_citation_id)
                    if source_citation_id is not None
                    else citation_id
                ),
                "citation_id_is_authoritative": bool(
                    passage.get("citation_id_is_authoritative")
                ),
                "chunk_id": passage.get("chunk_id"),
                "chunk_index": passage.get("chunk_index"),
                "document_id": passage.get("document_id"),
                "title": passage.get("title"),
                "source_path": passage.get("source_path"),
            })

    if batch_size is None:
        if is_summary:
            claims_est = len(re.split(r"(?<=[.!?])\s+", draft or ""))
            effective_batch_size = (
                VERIFIER_SUMMARY_LARGE_BATCH_SIZE
                if claims_est >= 25
                else VERIFIER_SUMMARY_BATCH_SIZE
            )
        else:
            effective_batch_size = VERIFIER_BATCH_SIZE
    else:
        effective_batch_size = batch_size

    return verify_passages(
        draft,
        evidence_passages,
        db_path,
        llm,
        allow_extractive=allow_extractive,
        batch_size=effective_batch_size,
        max_concurrency=max_concurrency,
        is_summary=is_summary,
        include_saved_web=include_saved_web,
        revise=revise,
    )


def verify_draft(
    question: str,
    draft: str,
    passages: list[dict[str, Any]],
    *,
    llm: CohereVerifier | None = None,
    db_path: str | Path = VERIFICATION_DB,
    batch_size: int = VERIFIER_BATCH_SIZE,
    max_concurrency: int = VERIFIER_MAX_CONCURRENCY,
) -> Verdict:
    """Preserve the graph's structured verdict contract with the new verifier."""
    is_comparison = bool(
        re.search(
            r"\b(compare|comparison|versus|vs\.?|difference|differences|contrast)\b",
            question,
            flags=re.IGNORECASE,
        )
    )
    report = verify_answer(
        draft,
        passages,
        db_path=db_path,
        llm=llm,
        allow_extractive=not is_comparison,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
    )
    return _as_verdict(report)


def remove_unsupported_claims(draft: str, claims: list[dict[str, Any]]) -> str:
    """Drop failed answer sentences while keeping supported prose and formatting."""
    unsupported = {
        _normalize_claim(str(claim.get("claim") or ""))
        for claim in claims
        if claim.get("supported") != "yes"
    }
    output_lines: list[str] = []
    kept_claim = False
    for line in (draft or "").splitlines():
        trimmed = line.strip()
        if not trimmed:
            output_lines.append("")
            continue
        if trimmed.startswith("#"):
            output_lines.append(line)
            continue
        if (
            (trimmed.startswith("**") and trimmed.endswith("**"))
            or (trimmed.startswith("__") and trimmed.endswith("__"))
        ) and not re.search(r"[\[【]\d+[\]】]", trimmed):
            output_lines.append(line)
            continue

        parts = re.split(r"(?<=[.!?।])\s+", line)
        kept_parts = []
        for part in parts:
            if not part.strip():
                continue
            if _normalize_claim(part) not in unsupported:
                kept_parts.append(part)
                kept_claim = True
        if kept_parts:
            output_lines.append(" ".join(kept_parts))
    return "\n".join(output_lines).strip() if kept_claim else ""


def _normalize_claim(claim: str) -> str:
    clean = re.sub(r"[\[【][^\]】]*[\]】]", "", claim)
    return re.sub(
        r"\s+",
        " ",
        clean.strip().lstrip("-*• ").casefold(),
    )


def verifier_node(state: dict[str, Any]) -> dict[str, Any]:
    """Verify a generated answer and retain the complete evidence trail."""
    node_start = perf_counter()
    raw_answer = state.get("raw_answer", state.get("draft", state.get("answer", "")))
    context_blocks = state.get("context_blocks", [])

    plan = state.get("orchestrator_plan") or {}
    plan_action = plan.get("action") if isinstance(plan, dict) else getattr(plan, "action", None)
    is_summary = bool(
        state.get("route") == "summary"
        or plan_action == "summarize"
        or state.get("summary_model_calls")
        or state.get("is_summary")
    )

    report = verify_answer(
        raw_answer,
        context_blocks,
        is_summary=is_summary,
    )
    node_elapsed = perf_counter() - node_start
    api_time = float(report.get("verification_metrics", {}).get("api_time_seconds", 0.0))
    usage = report.get("verification_metrics", {}).get("usage", {})
    in_tok = usage.get("input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)
    print(
        f"[timing] VERIFIER verifier_node completed: mode={'summary' if is_summary else 'normal'} | "
        f"total={node_elapsed:.3f}s | api_time={api_time:.3f}s | "
        f"claims={report.get('total_claims', 0)} | in_tok={in_tok} out_tok={out_tok} | "
        f"score={report.get('score')}",
        flush=True,
    )
    verdict = _as_verdict(report)
    retriever_worker._trace_event({
        "stage": "verify",
        "agent": "verifier",
        "ok": verdict.ok,
        "missing": verdict.missing,
        "run_id": report.get("run_id"),
        "score": report["score"],
        "total_time": round(node_elapsed, 4),
        "api_time": round(api_time, 4),
    })
    return {
        "verdict": verdict.model_dump(),
        "verification_report": report,
        "rounds": state.get("rounds", 0) + 1,
        "decision_log": [{
            "stage": "verifier",
            "ok": verdict.ok,
            "missing": verdict.missing,
            "score": report["score"],
            "verification_run_id": report.get("run_id"),
            "total_time_seconds": round(node_elapsed, 4),
            "api_time_seconds": round(api_time, 4),
            "unsupported": [
                claim.claim for claim in verdict.claims if claim.supported != "yes"
            ],
        }],
    }
