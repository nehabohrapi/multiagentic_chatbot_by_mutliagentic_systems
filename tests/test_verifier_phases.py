from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import verifier_agent
from agents import verifier
from verifier_agent import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SUMMARY_BATCH_SIZE,
    DEFAULT_SUMMARY_LARGE_BATCH_SIZE,
    CohereVerifier,
    _execute_batches,
    _judge_batch,
    _trim_evidence_text,
    verify_passages,
)


class MockRecordingLLM:
    """Mock LLM that records payload stats and responds with consistent mock verdicts."""

    def __init__(self, failure_on_first_call: bool = False, delay: float = 0.0):
        self.calls: list[dict] = []
        self.failure_on_first_call = failure_on_first_call
        self.delay = delay
        self.metrics = verifier_agent._empty_usage_metrics()

    def ask(self, system: str, prompt: str, max_tokens: int = 1200):
        if self.delay > 0:
            time.sleep(self.delay)

        self.metrics["requests"] += 1
        self.metrics["generation_requests"] += 1

        if self.failure_on_first_call and len(self.calls) == 0:
            self.calls.append({"failed": True})
            raise RuntimeError("Simulated transient network timeout")

        payload = json.loads(prompt)
        in_tok = len(prompt) // 4
        out_tok = 30 * len(payload.get("claims", []))
        self.metrics["input_tokens"] += in_tok
        self.metrics["output_tokens"] += out_tok

        self.calls.append({
            "system": system,
            "payload": payload,
            "max_tokens": max_tokens,
            "input_tokens": in_tok,
            "claims_count": len(payload.get("claims", [])),
        })

        results = []
        for claim in payload.get("claims", []):
            evidence = claim.get("evidence", [])
            evidence_id = evidence[0]["evidence_id"] if evidence else None
            quote = evidence[0]["text"][:25] if evidence else ""
            results.append({
                "claim_id": claim["claim_id"],
                "verdict": "SUPPORTED" if evidence else "NOT_FOUND",
                "evidence_id": evidence_id,
                "quote": quote,
                "reason": "Direct evidence supports claim.",
            })
        return {"claims": results}


class VerifierPhasesTests(unittest.TestCase):
    # Phase 2: Separate batch configurations
    def test_normal_query_uses_batch_size_ten(self):
        llm = MockRecordingLLM()
        with tempfile.TemporaryDirectory() as td:
            # 15 claims for a normal query -> 2 batches (10 + 5)
            answer = " ".join(f"Sentence {i} [{i}]." for i in range(1, 16))
            passages = [
                {"source_type": "document", "text": f"Fact statement {i}", "citation_id": i}
                for i in range(1, 16)
            ]
            report = verifier.verify_answer(
                answer,
                passages,
                db_path=Path(td) / "test.db",
                llm=llm,
                allow_extractive=False,
                is_summary=False,
            )

        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(llm.calls[0]["claims_count"], 10)
        self.assertEqual(llm.calls[1]["claims_count"], 5)
        self.assertEqual(report["verified_claims"], 15)

    def test_summary_query_uses_batch_size_twenty(self):
        llm = MockRecordingLLM()
        with tempfile.TemporaryDirectory() as td:
            # 22 claims for a summary -> batch size 20 -> 2 batches (20 + 2)
            answer = " ".join(f"Summary point {i} [chunk {i}]." for i in range(1, 23))
            passages = [
                {"source_type": "document", "text": f"Document text {i}", "citation_id": i, "chunk_index": i}
                for i in range(1, 23)
            ]
            report = verifier.verify_answer(
                answer,
                passages,
                db_path=Path(td) / "test.db",
                llm=llm,
                allow_extractive=False,
                is_summary=True,
                batch_size=20,
            )

        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(llm.calls[0]["claims_count"], 20)
        self.assertEqual(llm.calls[1]["claims_count"], 2)
        self.assertEqual(report["verified_claims"], 22)

    def test_large_summary_query_uses_batch_size_thirty(self):
        llm = MockRecordingLLM()
        with tempfile.TemporaryDirectory() as td:
            # 30 claims for a large summary -> 1 batch of 30!
            answer = " ".join(f"Deep summary point {i} [chunk {i}]." for i in range(1, 31))
            passages = [
                {"source_type": "document", "text": f"Document text {i}", "citation_id": i, "chunk_index": i}
                for i in range(1, 31)
            ]
            report = verifier.verify_answer(
                answer,
                passages,
                db_path=Path(td) / "test.db",
                llm=llm,
                allow_extractive=False,
                is_summary=True,
                batch_size=30,
            )

        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.calls[0]["claims_count"], 30)
        self.assertEqual(report["verified_claims"], 30)

    # Phase 3: Reduce input tokens
    def test_trim_evidence_text_limits_length(self):
        long_text = "Sentence one is here. " * 50  # ~1100 characters
        trimmed = _trim_evidence_text(long_text, max_chars=300)
        self.assertLessEqual(len(trimmed), 305)
        self.assertTrue(trimmed.endswith(".") or trimmed.endswith("..."))

    def test_cited_claims_do_not_send_unrelated_passages(self):
        llm = MockRecordingLLM()
        with tempfile.TemporaryDirectory() as td:
            answer = "Specific claim about transformers [1]."
            passages = [
                {"source_type": "document", "text": "Relevant transformers fact.", "citation_id": 1},
                {"source_type": "document", "text": "Unrelated passage 2.", "citation_id": 2},
                {"source_type": "document", "text": "Unrelated passage 3.", "citation_id": 3},
                {"source_type": "document", "text": "Unrelated passage 4.", "citation_id": 4},
            ]
            verifier.verify_answer(
                answer,
                passages,
                db_path=Path(td) / "test.db",
                llm=llm,
                allow_extractive=False,
            )

        claims_payload = llm.calls[0]["payload"]["claims"]
        evidence_list = claims_payload[0]["evidence"]
        # Only the cited passage is included, not 3-4 passages
        self.assertEqual(len(evidence_list), 1)
        self.assertEqual(evidence_list[0]["citation_id"], 1)

    # Phase 4: Verification reliability
    def test_missing_claim_in_model_response_is_safely_handled(self):
        class OmissionMockLLM:
            def __init__(self):
                self.metrics = verifier_agent._empty_usage_metrics()

            def ask(self, system, prompt, max_tokens=1200):
                payload = json.loads(prompt)
                # Intentionally omit the second claim
                first_claim = payload["claims"][0]
                return {
                    "claims": [
                        {
                            "claim_id": first_claim["claim_id"],
                            "verdict": "SUPPORTED",
                            "evidence_id": first_claim["evidence"][0]["evidence_id"],
                            "quote": first_claim["evidence"][0]["text"][:20],
                            "reason": "Direct support found.",
                        }
                    ]
                }

        llm = OmissionMockLLM()
        with tempfile.TemporaryDirectory() as td:
            answer = "Claim A [1]. Claim B [2]."
            passages = [
                {"source_type": "document", "text": "Claim A statement.", "citation_id": 1},
                {"source_type": "document", "text": "Claim B statement.", "citation_id": 2},
            ]
            report = verify_passages(
                answer,
                passages,
                Path(td) / "test.db",
                llm=llm,
                allow_extractive=False,
                batch_size=10,
            )

        claims = report["claims"]
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0]["status"], "SUPPORTED")
        # Omitted claim safely caught and marked UNVERIFIED with exact 1 result per claim
        self.assertEqual(claims[1]["status"], "UNVERIFIED")
        self.assertIn("omitted this claim", claims[1]["reason"])

    def test_failed_batch_retries_and_succeeds(self):
        llm = MockRecordingLLM(failure_on_first_call=True)
        with tempfile.TemporaryDirectory() as td:
            answer = "Reliable claim [1]."
            passages = [
                {"source_type": "document", "text": "Reliable text.", "citation_id": 1}
            ]
            report = verify_passages(
                answer,
                passages,
                Path(td) / "test.db",
                llm=llm,
                allow_extractive=False,
            )

        # First call failed, second call (retry) succeeded
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(report["verified_claims"], 1)

    # Phase 5: Benchmark and comparison across batch sizes 10, 20, 30
    def test_benchmark_batch_sizes_10_20_30(self):
        claims_count = 30
        answer = " ".join(f"Fact statement {i} [chunk {i}]." for i in range(1, claims_count + 1))
        passages = [
            {
                "source_type": "document",
                "text": f"Document chunk content confirming fact statement {i}.",
                "citation_id": i,
                "chunk_index": i,
            }
            for i in range(1, claims_count + 1)
        ]

        benchmarks = {}
        for bs in [10, 20, 30]:
            llm = MockRecordingLLM(delay=0.01)
            with tempfile.TemporaryDirectory() as td:
                t0 = time.perf_counter()
                report = verify_passages(
                    answer,
                    passages,
                    Path(td) / "test.db",
                    llm=llm,
                    allow_extractive=False,
                    batch_size=bs,
                    max_concurrency=1,
                    is_summary=True,
                )
                elapsed = time.perf_counter() - t0
                benchmarks[bs] = {
                    "api_calls": len(llm.calls),
                    "input_tokens": llm.metrics["input_tokens"],
                    "output_tokens": llm.metrics["output_tokens"],
                    "elapsed": elapsed,
                    "verified_claims": report["verified_claims"],
                }

        # Compare metrics
        # Batch size 10 takes 3 API calls
        self.assertEqual(benchmarks[10]["api_calls"], 3)
        # Batch size 20 takes 2 API calls
        self.assertEqual(benchmarks[20]["api_calls"], 2)
        # Batch size 30 takes only 1 API call
        self.assertEqual(benchmarks[30]["api_calls"], 1)

        # All configurations verify all 30 claims accurately
        self.assertEqual(benchmarks[10]["verified_claims"], 30)
        self.assertEqual(benchmarks[20]["verified_claims"], 30)
        self.assertEqual(benchmarks[30]["verified_claims"], 30)

        # Batch size 30 completes in fewer round trips and lower total latency
        self.assertLess(benchmarks[30]["api_calls"], benchmarks[10]["api_calls"])
        self.assertLess(benchmarks[30]["elapsed"], benchmarks[10]["elapsed"])


if __name__ == "__main__":
    unittest.main()
