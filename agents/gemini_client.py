"""Shared Gemini client and API-key resolution."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from config import (
    BASE_DIR,
    GEMINI_MAX_RETRIES,
    GEMINI_RETRY_BASE_SECONDS,
    GEMINI_TPM_HARD_LIMIT,
    GEMINI_TPM_LIMIT,
    GEMINI_TPM_WINDOW_SECONDS,
)
from memory.state_store import reserve_gemini_tokens
from telemetry import (
    model_request,
    record_model_retry,
    record_model_usage,
    time_block,
)


def gemini_api_key() -> str | None:
    """Prefer the project .env key over a stale inherited process value."""
    env_file = dotenv_values(Path(BASE_DIR) / ".env")
    file_key = env_file.get("GEMINI_API_KEY")
    if file_key and file_key.strip():
        return file_key.strip()
    process_key = os.getenv("GEMINI_API_KEY")
    return process_key.strip() if process_key and process_key.strip() else None


def new_gemini_client() -> genai.Client:
    api_key = gemini_api_key()
    if not api_key:
        raise RuntimeError("Gemini operations require GEMINI_API_KEY to be configured.")
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def _retry_delay_seconds(error: genai_errors.APIError, attempt: int) -> float:
    details = getattr(error, "details", None)
    if isinstance(details, dict):
        error_details = details.get("error", details)
        if isinstance(error_details, dict):
            for detail in error_details.get("details", []):
                if not isinstance(detail, dict):
                    continue
                if str(detail.get("@type", "")).endswith("RetryInfo"):
                    delay = str(detail.get("retryDelay", ""))
                    match = re.fullmatch(r"([\d.]+)s", delay)
                    if match:
                        return max(0.0, float(match.group(1)))

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {})
    retry_after = headers.get("Retry-After") if headers else None
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass

    return float(GEMINI_RETRY_BASE_SECONDS * (2 ** (attempt - 1)))


def _is_quota_exhausted(error: genai_errors.APIError) -> bool:
    details = getattr(error, "details", None)
    status = str(getattr(error, "status", "")).casefold()
    if status in {"resource_exhausted", "quota_exceeded"}:
        return True
    if isinstance(details, dict):
        error_details = details.get("error", details)
        if isinstance(error_details, dict) and (
            error_details.get("quotaFailure")
            or any("quota" in str(key).casefold() for key in error_details)
        ):
            return True
    message = " ".join((str(error), str(details or ""))).casefold()
    return any(
        marker in message
        for marker in ("quota exceeded", "quota_exceeded", "quota exhausted", "daily limit")
    )


def generate_gemini_content(
    client: genai.Client,
    *,
    model: str,
    contents: str,
    config: Any,
    stage: str,
    input_tokens: int | None = None,
    max_retries: int | None = None,
) -> Any:
    """Count, pace, and retry a Gemini generate-content request."""
    system_instruction = getattr(config, "system_instruction", None)
    if input_tokens is None:
        count_contents = (
            f"{system_instruction}\n\n{contents}"
            if system_instruction
            else contents
        )
        with time_block(
            f"Gemini token count ({stage})",
            category="GEMINI",
            level=2,
        ) as timing:
            with model_request(
                provider="google",
                model=model,
                operation="count_tokens",
                stage=stage,
                request_type="count_tokens",
            ) as span:
                token_response = client.models.count_tokens(
                    model=model,
                    contents=count_contents,
                )
            input_tokens = token_response.total_tokens
            if input_tokens is None:
                raise RuntimeError("Gemini did not return an input token count.")
            span.set_attribute("nexus.prompt_tokens.counted", int(input_tokens))
            timing["details"] = f"input_tokens={input_tokens:,}"

    retry_limit = GEMINI_MAX_RETRIES if max_retries is None else max(0, max_retries)
    for attempt in range(1, retry_limit + 2):
        with time_block(
            f"Gemini shared TPM pacing ({stage})",
            category="GEMINI",
            level=2,
        ) as timing:
            wait_seconds = reserve_gemini_tokens(
                int(input_tokens),
                limit=GEMINI_TPM_LIMIT,
                hard_limit=GEMINI_TPM_HARD_LIMIT,
                window_seconds=GEMINI_TPM_WINDOW_SECONDS,
            )
            timing["details"] = (
                f"input_tokens={input_tokens:,}; waited={wait_seconds:.3f}s"
            )

        try:
            with time_block(
                f"Gemini generate_content ({stage})",
                category="GEMINI",
                level=2,
            ):
                with model_request(
                    provider="google",
                    model=model,
                    operation="generate",
                    stage=stage,
                ) as span:
                    response = client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )
                    usage = getattr(response, "usage_metadata", None)
                    prompt_tokens = getattr(usage, "prompt_token_count", None)
                    output_tokens = getattr(usage, "candidates_token_count", None)
                    record_model_usage(
                        span,
                        input_tokens=(
                            int(prompt_tokens)
                            if prompt_tokens is not None
                            else int(input_tokens)
                        ),
                        output_tokens=(
                            int(output_tokens)
                            if output_tokens is not None
                            else None
                        ),
                        estimated_input=prompt_tokens is None,
                    )
                return response
        except genai_errors.APIError as exc:
            if (
                exc.code not in {429, 503}
                or attempt > retry_limit
                or _is_quota_exhausted(exc)
            ):
                raise
            record_model_retry("google", model, stage)
            delay = _retry_delay_seconds(exc, attempt)
            print(
                f"[timing] GEMINI retry {attempt}/{retry_limit} "
                f"for {stage} after HTTP {exc.code}; waiting {delay:.3f}s",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError("Gemini request retry loop exited unexpectedly.")
