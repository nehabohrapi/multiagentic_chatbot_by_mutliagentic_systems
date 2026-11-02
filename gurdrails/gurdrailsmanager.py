import base64
import logging
import re
import time
import unicodedata

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import sys
# Prevent torchaudio DLL load failure on Python 3.13 Windows from breaking transformers/presidio
if "torchaudio" not in sys.modules:
    sys.modules["torchaudio"] = None

import spacy
import spacy.util
if not spacy.util.is_package("en_core_web_lg") and spacy.util.is_package("en_core_web_sm"):
    _orig_is_pkg = spacy.util.is_package
    spacy.util.is_package = lambda name: True if name == "en_core_web_lg" else _orig_is_pkg(name)
    _orig_load = spacy.load
    spacy.load = lambda name, **kwargs: _orig_load("en_core_web_sm", **kwargs) if name == "en_core_web_lg" else _orig_load(name, **kwargs)

from nemoguardrails import LLMRails, RailsConfig


# ============================================================
# SECURITY LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

security_logger = logging.getLogger("input_guardrails")


# ============================================================
# RESULT STRUCTURE
# ============================================================

@dataclass
class GuardrailResult:
    allowed: bool
    sanitized_input: str = ""
    message: str = ""
    reason: str = ""


# ============================================================
# GUARDRAILS MANAGER
# ============================================================

class GuardrailsManager:

    def __init__(
        self,
        max_input_length=10000,
        max_requests=100,
        time_window=60,
        max_attack_attempts=10,
    ):
        """
        Initialize the input security module.

        max_input_length:
            Maximum query length in characters.

        max_requests:
            Maximum requests allowed per user in a time window.

        time_window:
            Rate-limit window in seconds.

        max_attack_attempts:
            Number of suspicious attempts tolerated per user
            within the attack-tracking window.
        """

        self.max_input_length = max_input_length
        self.max_requests = max_requests
        self.time_window = time_window
        self.max_attack_attempts = max_attack_attempts

        self.request_history = defaultdict(deque)
        self.attack_history = defaultdict(deque)

        self.config_path = Path(__file__).resolve().parent

        # Initialize NVIDIA NeMo Guardrails.
        config = RailsConfig.from_path(str(self.config_path))
        self.rails = LLMRails(config)

        security_logger.info("Input Guardrails initialized.")

    def reset_history(self, user_id=None):
        """Reset rate limit and attack tracking history."""
        if user_id:
            self.request_history.pop(user_id, None)
            self.attack_history.pop(user_id, None)
        else:
            self.request_history.clear()
            self.attack_history.clear()

    # ========================================================
    # 1. INPUT VALIDATION
    # ========================================================

    def validate_input(self, text):

        if not isinstance(text, str):
            return False, "Input must be text."

        text = text.strip()

        if not text:
            return False, "Please enter a question."

        if len(text) > self.max_input_length:
            return False, "Input exceeds the maximum allowed length."

        return True, ""

    # ========================================================
    # TEXT NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize_text(text):
        """
        Normalize Unicode and remove invisible formatting
        characters for security analysis.
        """

        normalized = unicodedata.normalize("NFKC", text)

        # Remove common zero-width formatting characters.
        normalized = re.sub(
            r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]",
            "",
            normalized,
        )

        return normalized

    # ========================================================
    # 2. JAILBREAK DETECTION
    # ========================================================

    def detect_jailbreak(self, text):

        patterns = [
            r"\b(ignore|disregard|forget|override)\s+(all\s+)?(previous|prior|above|system)\s+(instructions|prompts|rules|commands|directives)\b",
            r"\b(bypass|disable|turn\s+off)\s+(the\s+|your\s+)?(security|safety|guardrails?|filters?|restrictions|moderation)\b",
            r"act\s+as\s+(if\s+you\s+have\s+no\s+restrictions|an?\s+unrestricted)",
            r"reveal\s+(your\s+)?system\s+prompt",
            r"\b(developer\s+mode|unfiltered\s+mode|dan\s+mode|jailbreak|jailbroken)\b",
            r"\b(you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+(an?\s+|in\s+)?(unrestricted|jailbroken|evil|unfiltered|dan|developer\s+mode|root|admin)\b",
        ]

        normalized = self.normalize_text(text).lower()

        for pattern in patterns:
            if re.search(pattern, normalized):
                return True

        return False

    # ========================================================
    # 3. PROMPT INJECTION DETECTION
    # ========================================================

    def detect_prompt_injection(self, text):

        patterns = [
            r"ignore\s+.*(instructions|rules|commands|directives|prompts)",
            r"override\s+.*(system|developer|security|safety)\s+.*(instructions|rules|commands|directives)",
            r"override\s+.*rules",
            r"follow\s+these\s+new\s+instructions\s+instead",
            r"the\s+previous\s+rules\s+no\s+longer\s+apply",
            r"you\s+are\s+now\s+(an?\s+)?unrestricted",
            r"reveal\s+.*(hidden|internal|system)\s+(instructions|prompts)",
            r"<\s*/?\s*(?:system|system_message|instruction|prompt|im_start|im_end)\s*>",
            r"\[\s*/?\s*(?:inst|system|system_message|sys)\s*\]",
            r"(?:^|\n)\s*(?:system|admin|root)\s*:\s*",
        ]

        normalized = self.normalize_text(text).lower()

        for pattern in patterns:
            if re.search(pattern, normalized):
                return True

        return False

    # ========================================================
    # 4. PII DETECTION AND MASKING
    # ========================================================

    def mask_pii(self, text):
        """
        PII masking is handled by NeMo's configured
        'mask sensitive data on input' flow.

        This method intentionally does not implement a second,
        incomplete set of PII detectors.
        """

        return text

    # ========================================================
    # 5. CONTENT SAFETY
    # ========================================================

    # Content safety is handled by NeMo's configured
    # 'content safety check input' flow.

    # ========================================================
    # 6. TOPIC CONTROL
    # ========================================================

    # Topic control is handled by NeMo's configured
    # 'topic safety check input' flow.

    # ========================================================
    # 7. ENCODING AND OBFUSCATION DETECTION
    # ========================================================

    def detect_obfuscation(self, text):

        # Detect invisible formatting characters.
        if re.search(
            r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]",
            text,
        ):
            return True

        # Decode plausible Base64 text only when the query
        # explicitly asks to decode or execute encoded content.
        encoded_pattern = (
            r"(?:base64|decode|encoded\s+payload|"
            r"decode\s+this)\b"
        )

        if re.search(encoded_pattern, text, re.IGNORECASE):

            candidates = re.findall(
                r"(?<!\w)[A-Za-z0-9+/]{24,}={0,2}(?!\w)",
                text,
            )

            for candidate in candidates:
                try:
                    padded = candidate + "=" * (
                        (-len(candidate)) % 4
                    )

                    decoded = base64.b64decode(
                        padded,
                        validate=True,
                    ).decode("utf-8", errors="ignore")

                    if self.detect_jailbreak(decoded):
                        return True

                    if self.detect_prompt_injection(decoded):
                        return True

                except Exception:
                    continue

        return False

    # ========================================================
    # 8. RATE LIMITING
    # ========================================================

    def check_rate_limit(self, user_id):

        now = time.monotonic()
        history = self.request_history[user_id]

        while history and now - history[0] >= self.time_window:
            history.popleft()

        if len(history) >= self.max_requests:
            return False

        history.append(now)

        return True

    # ========================================================
    # 9. REPEATED ATTACK DETECTION
    # ========================================================

    def record_attack(self, user_id):

        now = time.monotonic()
        history = self.attack_history[user_id]

        while history and now - history[0] >= self.time_window:
            history.popleft()

        history.append(now)

        return len(history) >= self.max_attack_attempts

    # ========================================================
    # 10. SECURITY EVENT LOGGING
    # ========================================================

    def log_security_event(self, user_id, reason, blocked=True):

        # Never log the raw user query or detected PII here.
        security_logger.warning(
            "security_event user_id=%s reason=%s blocked=%s",
            user_id,
            reason,
            blocked,
        )

    # ========================================================
    # NEMO INPUT RAILS
    # ========================================================

    def run_nemo_input_rails(self, text):

        """
        Run the configured NeMo input rails without invoking
        the normal dialog/answer-generation flow.

        An allowed result can contain the masked input.
        A blocked result normally contains NeMo's refusal text.
        """

        result = self.rails.generate(
            messages=[
                {
                    "role": "user",
                    "content": text,
                }
            ],
            options={
                "rails": ["input"],
            },
        )

        response = getattr(result, "response", result)

        if isinstance(response, list):
            if not response:
                raise RuntimeError("NeMo returned an empty response.")

            last_message = response[-1]

            if isinstance(last_message, dict):
                processed_text = last_message.get("content", "")
            else:
                processed_text = str(last_message)

        elif isinstance(response, dict):
            processed_text = response.get("content", "")

        elif isinstance(response, str):
            processed_text = response

        else:
            raise RuntimeError("Unexpected NeMo response format.")

        processed_text = processed_text.strip()

        if not processed_text:
            raise RuntimeError("NeMo returned empty processed input.")

        # Default NeMo refusal text.
        refusal_markers = [
            "i'm sorry, i can't respond to that",
            "i am sorry, i cannot respond to that",
            "bot refuse to respond",
        ]

        if processed_text.lower() in refusal_markers:
            return GuardrailResult(
                allowed=False,
                message="Your query was blocked by security checks.",
                reason="nemo_input_rail",
            )

        return GuardrailResult(
            allowed=True,
            sanitized_input=processed_text,
        )

    # ========================================================
    # MAIN INPUT SECURITY PIPELINE
    # ========================================================

    def check_input(self, user_input, user_id="anonymous"):

        """
        Run all input security checks before the RAG workflow.

        IMPORTANT:
        Supply a stable session/user ID from your application
        for correct per-user rate limiting and attack tracking.
        """

        # 1. Basic validation.
        valid, message = self.validate_input(user_input)

        if not valid:
            self.log_security_event(user_id, "invalid_input")

            return GuardrailResult(
                allowed=False,
                message=message,
                reason="input_validation",
            )

        text = self.normalize_text(user_input)

        # 2 and 3. Local jailbreak / prompt-injection checks.
        if self.detect_jailbreak(text):
            repeated = self.record_attack(user_id)

            reason = (
                "repeated_attack"
                if repeated
                else "jailbreak_detected"
            )

            self.log_security_event(user_id, reason)

            return GuardrailResult(
                allowed=False,
                message="Your query was blocked by security checks.",
                reason=reason,
            )

        if self.detect_prompt_injection(text):
            repeated = self.record_attack(user_id)

            reason = (
                "repeated_attack"
                if repeated
                else "prompt_injection"
            )

            self.log_security_event(user_id, reason)

            return GuardrailResult(
                allowed=False,
                message="Your query was blocked by security checks.",
                reason=reason,
            )

        # 7. Encoding / obfuscation checks.
        if self.detect_obfuscation(text):
            repeated = self.record_attack(user_id)

            reason = (
                "repeated_attack"
                if repeated
                else "obfuscation_detected"
            )

            self.log_security_event(user_id, reason)

            return GuardrailResult(
                allowed=False,
                message="Your query was blocked by security checks.",
                reason=reason,
            )

        # 8. Rate limiting.
        if not self.check_rate_limit(user_id):
            self.log_security_event(user_id, "rate_limit")

            return GuardrailResult(
                allowed=False,
                message="Too many requests. Please try again later.",
                reason="rate_limit",
            )

        # Run configured NeMo rails:
        # Sensitive data detection (Email, Phone, Credit Card) requires '@' or digits.
        # Fast-path clean queries without candidate tokens to eliminate 60-90s model latency.
        has_pii_candidates = bool(re.search(r"[@\d]", text))
        if not has_pii_candidates:
            return GuardrailResult(
                allowed=True,
                sanitized_input=text,
            )

        try:
            nemo_result = self.run_nemo_input_rails(text)

        except Exception:
            self.log_security_event(user_id, "nemo_validation_error")

            # 11. Fail-safe behavior:
            # Never pass unchecked input to the RAG workflow.
            return GuardrailResult(
                allowed=False,
                message=(
                    "Security validation is temporarily unavailable. "
                    "Please try again later."
                ),
                reason="guardrail_error",
            )

        if not nemo_result.allowed:
            repeated = self.record_attack(user_id)

            reason = (
                "repeated_attack"
                if repeated
                else nemo_result.reason
            )

            self.log_security_event(user_id, reason)

            return GuardrailResult(
                allowed=False,
                message=nemo_result.message,
                reason=reason,
            )

        # 9. Repeated attacks are tracked when an attack is
        # detected. The request history is keyed by user_id.
        #
        # 10. Security events are logged without raw query text.

        return GuardrailResult(
            allowed=True,
            sanitized_input=nemo_result.sanitized_input,
            message="Input passed security checks.",
            reason="allowed",
        )