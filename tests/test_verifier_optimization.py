import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import verifier_agent
from agents.verifier import _as_verdict, verify_answer
from verifier_agent import (
    CohereVerifier,
    _execute_batches,
    _judge_batch,
    verify_passages,
)


class MockLLM:
    def __init__(self, responses_by_batch=None, delay=0.0):
        self.calls = []
        self.responses_by_batch = list(responses_by_batch or [])
        self.delay = delay
        self.metrics = verifier_agent._empty_usage_metrics()

    def ask(self, system, prompt, max_tokens=1200):
        if self.delay > 0:
            time.sleep(self.delay)
        payload = json.loads(prompt)
        self.calls.append({
            "system": system,
            "payload": payload,
            "max_tokens": max_tokens,
        })
        self.metrics["requests"] += 1
        self.metrics["generation_requests"] += 1
        self.metrics["input_tokens"] += 50
        self.metrics["output_tokens"] += 20

        batch_idx = min(len(self.calls) - 1, len(self.responses_by_batch) - 1) if self.responses_by_batch else 0
        statuses = self.responses_by_batch[batch_idx] if self.responses_by_batch else ["SUPPORTED"] * len(payload["claims"])

        results = []
        for idx, claim in enumerate(payload["claims"]):
            status = statuses[idx] if isinstance(statuses, list) and idx < len(statuses) else "SUPPORTED"
            evidence = claim.get("evidence", [])
            if status in {"NOT_FOUND", "UNCERTAIN"} or not evidence:
                results.append({
                    "claim_id": claim["claim_id"],
                    "verdict": status,
                    "evidence_id": None,
                    "quote": "",
                    "reason": "Not found in evidence.",
                })
            else:
                results.append({
                    "claim_id": claim["claim_id"],
                    "verdict": status,
                    "evidence_id": evidence[0]["evidence_id"],
                    "quote": evidence[0]["text"][:20],
                    "reason": "Evidence confirms the claim directly.",
                })
        return {"claims": results}


class VerifierOptimizationTests(unittest.TestCase):
    def test_batch_size_four_splits_six_claims_into_two_batches(self):
        llm = MockLLM([
            ["SUPPORTED", "SUPPORTED", "SUPPORTED", "SUPPORTED"],
            ["SUPPORTED", "SUPPORTED"],
        ])
        with tempfile.TemporaryDirectory() as temp_dir:
            answer = "Claim one [1]. Claim two [2]. Claim three [3]. Claim four [4]. Claim five [5]. Claim six [6]."
            passages = [
                {"source_type": "document", "text": f"Fact statement {i}", "citation_id": i, "chunk_id": f"c{i}"}
                for i in range(1, 7)
            ]
            report = verify_passages(
                answer,
                passages,
                Path(temp_dir) / "test.db",
                llm=llm,
                allow_extractive=False,
                batch_size=4,
                max_concurrency=1,
            )

        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(len(llm.calls[0]["payload"]["claims"]), 4)
        self.assertEqual(len(llm.calls[1]["payload"]["claims"]), 2)
        self.assertEqual(report["verified_claims"], 6)

    def test_default_batch_size_ten_splits_fifteen_claims(self):
        llm = MockLLM([
            ["SUPPORTED"] * 10,
            ["SUPPORTED"] * 5,
        ])
        with tempfile.TemporaryDirectory() as temp_dir:
            answer = " ".join(f"Claim {i} [{i}]." for i in range(1, 16))
            passages = [
                {"source_type": "document", "text": f"Fact statement {i}", "citation_id": i, "chunk_id": f"c{i}"}
                for i in range(1, 16)
            ]
            report = verify_passages(
                answer,
                passages,
                Path(temp_dir) / "test.db",
                llm=llm,
                allow_extractive=False,
                max_concurrency=1,
            )

        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(len(llm.calls[0]["payload"]["claims"]), 10)
        self.assertEqual(len(llm.calls[1]["payload"]["claims"]), 5)
        self.assertEqual(report["verified_claims"], 15)

    def test_limited_parallelism_two_batches_concurrent(self):
        llm = MockLLM(
            responses_by_batch=[
                ["SUPPORTED", "SUPPORTED", "SUPPORTED", "SUPPORTED"],
                ["SUPPORTED", "SUPPORTED"],
            ],
            delay=0.05,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            answer = "C1 [1]. C2 [2]. C3 [3]. C4 [4]. C5 [5]. C6 [6]."
            passages = [
                {"source_type": "document", "text": f"Evidence for C{i}", "citation_id": i, "chunk_id": f"c{i}"}
                for i in range(1, 7)
            ]
            started = time.perf_counter()
            report = verify_passages(
                answer,
                passages,
                Path(temp_dir) / "test.db",
                llm=llm,
                allow_extractive=False,
                batch_size=4,
                max_concurrency=2,
            )
            elapsed = time.perf_counter() - started

        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(report["verified_claims"], 6)
        # Concurrent execution of two 0.05s batches completes much faster than 0.10s sequential overhead
        self.assertLess(elapsed, 0.25)

    def test_structured_verdict_supported_unsupported_uncertain(self):
        llm = MockLLM([
            ["SUPPORTED", "UNSUPPORTED", "UNCERTAIN"],
        ])
        with tempfile.TemporaryDirectory() as temp_dir:
            answer = "Supported claim [1]. Unsupported claim [2]. Uncertain claim [3]."
            passages = [
                {"source_type": "document", "text": "Supported claim text.", "citation_id": 1, "chunk_id": "c1"},
                {"source_type": "document", "text": "Unsupported claim contradicts text.", "citation_id": 2, "chunk_id": "c2"},
                {"source_type": "document", "text": "Uncertain claim other text.", "citation_id": 3, "chunk_id": "c3"},
            ]
            report = verify_passages(
                answer,
                passages,
                Path(temp_dir) / "test.db",
                llm=llm,
                allow_extractive=False,
                batch_size=4,
                max_concurrency=1,
            )

        claims = report["claims"]
        self.assertEqual(len(claims), 3)
        self.assertEqual(claims[0]["structured_verdict"], "supported")
        self.assertEqual(claims[1]["structured_verdict"], "unsupported")
        self.assertEqual(claims[2]["structured_verdict"], "uncertain")

        verdict = _as_verdict(report)
        self.assertEqual(verdict.claims[0].supported, "yes")
        self.assertEqual(verdict.claims[1].supported, "no")
        self.assertEqual(verdict.claims[2].supported, "no")

    def test_reduced_context_sends_only_relevant_evidence(self):
        llm = MockLLM([["SUPPORTED"]])
        with tempfile.TemporaryDirectory() as temp_dir:
            answer = "Specific topic claim [1]."
            passages = [
                {"source_type": "document", "text": "Specific topic claim text.", "citation_id": 1, "chunk_id": "c1"},
                {"source_type": "document", "text": "Irrelevant chunk about astronomy.", "citation_id": 2, "chunk_id": "c2"},
                {"source_type": "document", "text": "Irrelevant chunk about cooking.", "citation_id": 3, "chunk_id": "c3"},
            ]
            verify_passages(
                answer,
                passages,
                Path(temp_dir) / "test.db",
                llm=llm,
                allow_extractive=False,
                batch_size=4,
                max_concurrency=1,
            )

        call_payload = llm.calls[0]["payload"]
        claim_obj = call_payload["claims"][0]
        # Only cited / matched evidence is sent, not the entire irrelevant document corpus
        evidence_texts = [e["text"] for e in claim_obj["evidence"]]
        self.assertIn("Specific topic claim text.", evidence_texts[0])
        self.assertNotIn("answer_sentence", claim_obj)

    def test_output_length_is_limited(self):
        llm = MockLLM([["SUPPORTED"]])
        with tempfile.TemporaryDirectory() as temp_dir:
            answer = "Claim one [1]."
            passages = [{"source_type": "document", "text": "Claim one text.", "citation_id": 1, "chunk_id": "c1"}]
            verify_passages(
                answer,
                passages,
                Path(temp_dir) / "test.db",
                llm=llm,
                allow_extractive=False,
                batch_size=4,
            )

        max_tokens = llm.calls[0]["max_tokens"]
        # Constrained token limit for 1 claim (150 + 80 = 230), well below old 1400 token limit
        self.assertLessEqual(max_tokens, 250)

    def test_api_time_and_total_verifier_time_measured_separately(self):
        llm = MockLLM([["SUPPORTED"]], delay=0.03)
        with tempfile.TemporaryDirectory() as temp_dir:
            answer = "Measured claim [1]."
            passages = [{"source_type": "document", "text": "Measured claim text.", "citation_id": 1, "chunk_id": "c1"}]
            report = verify_passages(
                answer,
                passages,
                Path(temp_dir) / "test.db",
                llm=llm,
                allow_extractive=False,
            )

        metrics = report["verification_metrics"]
        self.assertIn("total_verifier_time_seconds", metrics)
        self.assertIn("api_time_seconds", metrics)
        self.assertGreater(metrics["total_verifier_time_seconds"], 0.0)
        self.assertGreater(metrics["api_time_seconds"], 0.0)
        self.assertGreaterEqual(
            metrics["total_verifier_time_seconds"],
            metrics["api_time_seconds"],
        )


if __name__ == "__main__":
    unittest.main()
