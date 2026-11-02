"""Input Guardrails Agent for Nexus Multi-Agent Research System.

Integrated with NVIDIA NeMo Guardrails to perform:
- Unicode NFKC normalization and zero-width/control character stripping
- High-confidence prompt injection and jailbreak pattern detection
- Presidio-powered PII detection and masking on input
- Per-user rate limiting and repeated attack tracking
- Zero-cost early threat gating before query execution to protect downstream agents.
"""

from __future__ import annotations

import logging
import threading
import unicodedata
from dataclasses import dataclass
from typing import Any

from telemetry import time_block
from gurdrails.gurdrailsmanager import GuardrailResult, GuardrailsManager

logger = logging.getLogger(__name__)

# Global singleton instance for NeMo Guardrails Manager
_GUARDRAILS_MANAGER: GuardrailsManager | None = None
_WARMUP_LOCK = threading.Lock()


def get_guardrails_manager() -> GuardrailsManager:
    """Retrieve or initialize the cached singleton GuardrailsManager instance."""
    global _GUARDRAILS_MANAGER
    if _GUARDRAILS_MANAGER is None:
        with _WARMUP_LOCK:
            if _GUARDRAILS_MANAGER is None:
                _GUARDRAILS_MANAGER = GuardrailsManager()
    return _GUARDRAILS_MANAGER


def _warmup_guardrails() -> None:
    """Run an initial check in the background to eliminate the 80s cold-start on the user's first query."""
    try:
        manager = get_guardrails_manager()
        # Warmup query MUST contain '@' or digits so that Smart Gating
        # (re.search(r"[@\d]", text)) does NOT skip the NeMo pipeline.
        # Without PII candidate tokens, Presidio's spaCy NLP models stay
        # unloaded and the first real PII query pays a ~30s cold-start.
        manager.check_input("warmup probe test@test.com 1234567890")
        # Clear the rate-limit / attack tracking entry created by the warmup probe.
        manager.reset_history("anonymous")
        logger.info("NeMo Guardrails background warmup completed successfully (Presidio models loaded).")
    except Exception as exc:
        logger.debug("NeMo Guardrails background warmup note: %s", exc)


_warmup_thread = threading.Thread(target=_warmup_guardrails, daemon=True, name="GuardrailsWarmup")
_warmup_thread.start()



@dataclass(frozen=True)
class InputGuardrailResult:
    """Detailed result of input validation and threat gating."""

    is_safe: bool
    normalized_question: str
    sanitized_question: str = ""
    threat_detected: bool = False
    threat_type: str | None = None
    reason: str | None = None


def normalize_input_text(text: str) -> str:
    """Apply Unicode NFKC normalization and clean excessive control characters."""
    if not text:
        return ""
    # Unicode NFKC normalization to standardize homoglyphs and compatibility chars
    normalized = unicodedata.normalize("NFKC", text)
    # Strip null bytes and non-printable control characters except standard whitespace
    cleaned = "".join(ch for ch in normalized if ch == "\n" or ch == "\t" or not unicodedata.category(ch).startswith("C"))
    return cleaned.strip()


def validate_user_input(question: str, user_id: str = "anonymous") -> InputGuardrailResult:
    """Validate user input using the NVIDIA NeMo Guardrails flow.

    Runs Unicode normalization, jailbreak & injection gating, PII masking,
    and rate limit checks before any downstream agent or LLM invocation.
    """
    manager = get_guardrails_manager()
    result: GuardrailResult = manager.check_input(question, user_id=user_id)

    if not result.allowed:
        norm_fallback = normalize_input_text(question)
        threat_type_str = result.reason.upper() if result.reason else "DISALLOWED_CONTENT"
        return InputGuardrailResult(
            is_safe=False,
            normalized_question=norm_fallback,
            sanitized_question="",
            threat_detected=True,
            threat_type=threat_type_str,
            reason=result.message or f"Triggered security check: {result.reason}",
        )

    sanitized = result.sanitized_input or normalize_input_text(question)
    return InputGuardrailResult(
        is_safe=True,
        normalized_question=sanitized,
        sanitized_question=sanitized,
        threat_detected=False,
        threat_type=None,
        reason=None,
    )


def input_guardrail_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node representing the NVIDIA NeMo Input Guardrail boundary."""
    with time_block("Input Guardrail: Validate & Gating", category="GUARDRAIL", level=1) as tb_guard:
        raw_question = state.get("question", "")
        user_id = state.get("user_id") or state.get("run_id") or "anonymous"
        validation = validate_user_input(raw_question, user_id=user_id)

        decision_entry = {
            "stage": "input_guardrails",
            "engine": "nemo_guardrails",
            "is_safe": validation.is_safe,
            "threat_detected": validation.threat_detected,
        }
        if validation.threat_type:
            decision_entry["threat_type"] = validation.threat_type
        if validation.reason:
            decision_entry["reason"] = validation.reason

        tb_guard["details"] = f"safe={validation.is_safe}, threat={validation.threat_type or 'None'}"

        if not validation.is_safe:
            rejection_message = (
                f"Blocked: Your request could not be processed because it triggered "
                f"security boundaries ({validation.reason or 'disallowed content'}). Zero LLM cost incurred."
            )
            return {
                "question": validation.normalized_question,
                "is_safe": False,
                "input_cleared": False,
                "answer": rejection_message,
                "draft": rejection_message,
                "raw_answer": rejection_message,
                "answerable": False,
                "citations": [],
                "decision_log": [decision_entry],
            }

        return {
            "question": validation.normalized_question,
            "is_safe": True,
            "input_cleared": True,
            "decision_log": [decision_entry],
        }


def blocked_notice_node(state: dict[str, Any]) -> dict[str, Any]:
    """Direct termination node for queries rejected at input guardrails."""
    raw_answer = state.get("answer") or "Request blocked by security guardrails."
    return {
        "answer": raw_answer,
        "draft": raw_answer,
        "raw_answer": raw_answer,
        "answerable": False,
        "citations": [],
        "decision_log": [{
            "stage": "blocked_notice",
            "status": "terminated_zero_cost",
        }],
    }
