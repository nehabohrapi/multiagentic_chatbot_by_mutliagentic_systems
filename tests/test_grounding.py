import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents import curator, gemini_client, graph, retrieval_agent, verifier
from agents.retriever_worker import expand_chunks, extract_citations, rerank_chunks
from agents.schemas import ClaimCheck, Verdict
from agents.verifier import remove_unsupported_claims, verify_draft
import verifier_agent
from verifier_agent import DEFAULT_MODEL, CohereVerifier, verify_passages
from telemetry import recent_model_spans


class FakeCohereVerifier:
    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = []
        self.metrics = {
            "requests": 0,
            "count_tokens_requests": 0,
            "generation_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "retries": 0,
        }

    def ask(self, system, prompt, max_tokens=1200):
        payload = json.loads(prompt)
        self.calls.append(payload)
        batch_index = min(len(self.calls) - 1, len(self.batches) - 1)
        statuses = self.batches[batch_index]
        results = []
        for index, claim in enumerate(payload["claims"]):
            status = statuses[index] if isinstance(statuses, list) else statuses
            evidence = claim["evidence"]
            if status == "NOT_FOUND" or not evidence:
                results.append({
                    "claim_id": claim["claim_id"],
                    "verdict": "NOT_FOUND",
                    "evidence_id": None,
                    "quote": "",
                    "reason": "The supplied passages do not establish the claim.",
                })
                continue
            passage = next(
                (
                    item for item in evidence
                    if not claim["citation_ids"]
                    or item["citation_id"] in claim["citation_ids"]
                ),
                evidence[0],
            )
            results.append({
                "claim_id": claim["claim_id"],
                "verdict": status,
                "evidence_id": passage["evidence_id"],
                "quote": passage["text"][: min(40, len(passage["text"]))],
                "reason": "The quoted passage establishes this verdict.",
            })
        return {"claims": results}


class ClaimVerificationTests(unittest.TestCase):
    def setUp(self):
        self.passages = [
            {
                "chunk_index": 4,
                "text": "The trial reported a 20% reduction in risk.",
            },
            {
                "chunk_index": 5,
                "text": "The participants completed a follow-up survey.",
            },
        ]

    def test_verifier_uses_cohere_command_a_default(self):
        self.assertEqual(DEFAULT_MODEL, "command-a-03-2025")

    def test_project_env_cohere_key_takes_precedence_over_process_key(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("COHERE_API_KEY=project-test-key\n", encoding="utf-8")
            with (
                patch.object(verifier_agent, "PROJECT_ENV_FILE", env_file),
                patch.dict("os.environ", {"COHERE_API_KEY": "stale-process-key"}),
            ):
                self.assertEqual(
                    verifier_agent.cohere_api_key(),
                    "project-test-key",
                )

    def test_gemini_generation_retries_after_provider_retry_delay(self):
        from google.genai import errors

        retry_error = errors.APIError(
            429,
            {
                "error": {
                    "details": [{
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "0.25s",
                    }],
                },
            },
        )
        client = SimpleNamespace(
            models=SimpleNamespace(
                count_tokens=unittest.mock.Mock(
                    return_value=SimpleNamespace(total_tokens=42)
                ),
                generate_content=unittest.mock.Mock(side_effect=[
                    retry_error,
                    SimpleNamespace(text="ok"),
                ]),
            )
        )
        with (
            patch.object(
                gemini_client,
                "reserve_gemini_tokens",
                return_value=0.0,
            ) as reserve,
            patch.object(gemini_client.time, "sleep") as sleep,
        ):
            response = gemini_client.generate_gemini_content(
                client,
                model="gemini-test",
                contents="prompt",
                config=SimpleNamespace(system_instruction="system"),
                stage="test",
            )

        self.assertEqual(response.text, "ok")
        self.assertEqual(client.models.generate_content.call_count, 2)
        self.assertEqual(reserve.call_count, 2)
        self.assertEqual(reserve.call_args.kwargs["limit"], gemini_client.GEMINI_TPM_LIMIT)
        sleep.assert_called_once_with(0.25)

    def test_cohere_verifier_uses_command_a_json_chat(self):
        verifier = CohereVerifier.__new__(CohereVerifier)
        verifier.client = SimpleNamespace(chat=unittest.mock.Mock(return_value=SimpleNamespace(
            message=SimpleNamespace(content=[SimpleNamespace(text='{"claims":[]}')]),
            usage=SimpleNamespace(tokens=SimpleNamespace(input_tokens=80, output_tokens=12)),
        )))
        verifier.model = "command-a-03-2025"
        verifier.metrics = verifier_agent._empty_usage_metrics()

        result = verifier.ask("system prompt", "claim batch", max_tokens=321)

        self.assertEqual(result, {"claims": []})
        verifier.client.chat.assert_called_once_with(
            model="command-a-03-2025",
            messages=[
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "claim batch"},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=321,
        )
        self.assertEqual(verifier.metrics["requests"], 1)
        self.assertEqual(verifier.metrics["input_tokens"], 80)
        self.assertEqual(verifier.metrics["output_tokens"], 12)

    def test_gemini_quota_exhaustion_is_not_retried_and_is_counted(self):
        from google.genai import errors

        quota_error = errors.APIError(
            429,
            {"error": {"message": "Quota exceeded", "quotaFailure": [{"quotaId": "requests"}]}},
        )
        client = SimpleNamespace(
            models=SimpleNamespace(
                count_tokens=unittest.mock.Mock(
                    return_value=SimpleNamespace(total_tokens=42)
                ),
                generate_content=unittest.mock.Mock(side_effect=quota_error),
            )
        )
        before = len(recent_model_spans(provider="google", model="gemini-test"))
        with patch.object(gemini_client, "reserve_gemini_tokens", return_value=0.0):
            with self.assertRaises(errors.APIError):
                gemini_client.generate_gemini_content(
                    client,
                    model="gemini-test",
                    contents="prompt",
                    config=SimpleNamespace(system_instruction="system"),
                    stage="test",
                    max_retries=1,
                )

        client.models.generate_content.assert_called_once()
        spans = recent_model_spans(provider="google", model="gemini-test")[before:]
        self.assertEqual(
            [span["operation"] for span in spans],
            ["count_tokens", "generate"],
        )
        self.assertEqual(spans[-1]["status"], "ERROR")

    def test_exact_citation_backed_answer_skips_cohere(self):
        llm = FakeCohereVerifier([])
        with tempfile.TemporaryDirectory() as directory:
            report = verify_passages(
                "The trial reported a 20% reduction in risk. [1]",
                [{"source_type": "document", "text": "The trial reported a 20% reduction in risk.", "document_id": "doc-1"}],
                Path(directory) / "verifier.sqlite",
                llm,
            )
        self.assertEqual(llm.calls, [])
        self.assertEqual(report["score"], "1/1 claims verified")
        result = report["claims"][0]
        self.assertEqual(result["status"], "SUPPORTED")
        self.assertIn("20% reduction", result["quote"])
        self.assertEqual(report["verification_metrics"]["usage"]["generation_requests"], 0)

    def test_summary_verification_preserves_source_chunk_citation_ids(self):
        llm = FakeCohereVerifier([])
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                verifier_agent,
                "_saved_web_passages",
                side_effect=AssertionError("MAP verification must use source chunks only"),
            ):
                report = verifier.verify_answer(
                    "The trial reported a 20% reduction in risk. [chunk 201]",
                    [{
                        "source_type": "document",
                        "text": "The trial reported a 20% reduction in risk.",
                        "document_id": "doc-1",
                        "chunk_index": 200,
                        "citation_id": 201,
                        "citation_id_is_authoritative": True,
                    }],
                    db_path=Path(directory) / "verifier.sqlite",
                    llm=llm,
                    is_summary=True,
                    include_saved_web=False,
                    revise=True,
                )

        self.assertEqual(report["score"], "1/1 claims verified")
        self.assertEqual(report["claims"][0]["citation_ids"], [201])
        self.assertEqual(
            report["revised_answer"],
            "The trial reported a 20% reduction in risk. [chunk 201]",
        )
        self.assertEqual(llm.calls, [])

    def test_filename_page_citation_maps_to_the_matching_chunk(self):
        llm = FakeCohereVerifier([])
        with tempfile.TemporaryDirectory() as directory:
            report = verify_passages(
                "Harvest index is near 0.50 under favorable conditions. "
                "[agri_crop.pdf, p. 132]",
                [{
                    "source_type": "document",
                    "document_id": "crop-doc",
                    "title": "agri_crop.pdf",
                    "source_path": "D:/documents/agri_crop.pdf",
                    "page_number": 132,
                    "chunk_index": 31,
                    "citation_id": 1,
                    "citation_id_is_authoritative": True,
                    "text": "Harvest index is near 0.50 under favorable conditions.",
                }],
                Path(directory) / "verifier.sqlite",
                llm,
                is_summary=True,
            )

        self.assertEqual(llm.calls, [])
        self.assertEqual(report["score"], "1/1 claims verified")
        self.assertEqual(report["claims"][0]["status"], "SUPPORTED")
        self.assertEqual(report["claims"][0]["citation_ids"], [1])

    def test_unmapped_filename_page_citation_is_explicitly_unverified(self):
        llm = FakeCohereVerifier([])
        with tempfile.TemporaryDirectory() as directory:
            report = verify_passages(
                "Harvest index is near 0.50 under favorable conditions. "
                "[missing.pdf, p. 132]",
                [{
                    "source_type": "document",
                    "document_id": "crop-doc",
                    "title": "agri_crop.pdf",
                    "source_path": "D:/documents/agri_crop.pdf",
                    "page_number": 132,
                    "chunk_index": 31,
                    "citation_id": 1,
                    "citation_id_is_authoritative": True,
                    "text": "Harvest index is near 0.50 under favorable conditions.",
                }],
                Path(directory) / "verifier.sqlite",
                llm,
                is_summary=True,
            )

        self.assertEqual(llm.calls, [])
        self.assertEqual(report["claims"][0]["status"], "UNVERIFIED")
        self.assertIn("did not map", report["claims"][0]["reason"])

    def test_multiple_claims_are_verified_in_one_batched_request(self):
        llm = FakeCohereVerifier([["SUPPORTED", "CONTRADICTED"]])
        with tempfile.TemporaryDirectory() as directory:
            report = verify_passages(
                "Treatment reduced risk by 20% [1]. Participants finished the survey [2].",
                [
                    {"source_type": "document", "text": "The trial reported a 20% reduction in risk.", "document_id": "doc-1", "chunk_id": "c1"},
                    {"source_type": "document", "text": "Participants did not complete the follow-up survey.", "document_id": "doc-1", "chunk_id": "c2"},
                ],
                Path(directory) / "verifier.sqlite",
                llm,
            )
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(len(llm.calls[0]["claims"]), 2)
        self.assertEqual([item["status"] for item in report["claims"]], ["SUPPORTED", "CONTRADICTED"])

    def test_missing_evidence_fails_closed_without_cohere_call(self):
        llm = FakeCohereVerifier([])
        with tempfile.TemporaryDirectory() as directory:
            report = verify_passages(
                "The launch occurred in 2020.",
                [],
                Path(directory) / "verifier.sqlite",
                llm,
            )
        self.assertEqual(llm.calls, [])
        self.assertEqual(report["claims"][0]["status"], "UNVERIFIED")

    def test_invalid_citation_is_not_approved(self):
        llm = FakeCohereVerifier([["SUPPORTED"]])
        with tempfile.TemporaryDirectory() as directory:
            report = verify_passages(
                "The launch occurred in 2020 [8].",
                [{"source_type": "document", "text": "The launch occurred in 2020."}],
                Path(directory) / "verifier.sqlite",
                llm,
            )
        self.assertEqual(report["claims"][0]["status"], "UNVERIFIED")
        self.assertEqual(llm.calls, [])

    def test_comparison_is_semantically_verified_even_for_exact_text(self):
        llm = FakeCohereVerifier([["SUPPORTED"]])
        with tempfile.TemporaryDirectory() as directory:
            verdict = verify_draft(
                "Compare the two documents.",
                "The crop yield was 20 tonnes. [1]",
                [
                    {"text": "The crop yield was 20 tonnes.", "document_id": "doc-1"},
                    {"text": "The crop yield was 18 tonnes.", "document_id": "doc-2"},
                ],
                llm=llm,
                db_path=Path(directory) / "verifier.sqlite",
            )
        self.assertEqual(len(llm.calls), 1)
        self.assertTrue(verdict.ok)

    def test_quota_failure_does_not_trigger_recovery_batch(self):
        from google.genai import errors

        class QuotaVerifier(FakeCohereVerifier):
            def ask(self, system, prompt, max_tokens=1200):
                self.calls.append(json.loads(prompt))
                raise errors.APIError(429, {"error": {"message": "Quota exceeded"}})

        llm = QuotaVerifier([[]])
        with tempfile.TemporaryDirectory() as directory:
            report = verify_passages(
                "The launch took place in 2020.",
                [
                    {"text": f"The launch took place in 2020; supporting note {index}.", "document_id": "doc-1"}
                    for index in range(5)
                ],
                Path(directory) / "verifier.sqlite",
                llm,
            )
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(report["claims"][0]["status"], "UNVERIFIED")

    def test_malformed_batch_response_fails_closed(self):
        class MalformedVerifier(FakeCohereVerifier):
            def ask(self, system, prompt, max_tokens=1200):
                self.calls.append(json.loads(prompt))
                return {"claims": []}

        llm = MalformedVerifier([[]])
        with tempfile.TemporaryDirectory() as directory:
            report = verify_passages(
                "The launch occurred in 2020.",
                [
                    {"source_type": "document", "text": f"The product launch happened in the year 2020, note {index}."}
                    for index in range(5)
                ],
                Path(directory) / "verifier.sqlite",
                llm,
            )
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(report["claims"][0]["status"], "UNVERIFIED")

    def test_only_one_targeted_recovery_batch_is_allowed(self):
        llm = FakeCohereVerifier([["NOT_FOUND"], ["SUPPORTED"], ["CONTRADICTED"]])
        passages = [
            {"source_type": "document", "text": f"The product launch date is 2020; note {index}.", "document_id": "doc-1"}
            for index in range(6)
        ]
        with tempfile.TemporaryDirectory() as directory:
            report = verify_passages(
                "The product launched in 2020.",
                passages,
                Path(directory) / "verifier.sqlite",
                llm,
            )
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(report["claims"][0]["status"], "SUPPORTED")

    def test_recovery_is_not_attempted_after_three_supported_claims(self):
        llm = FakeCohereVerifier([["SUPPORTED", "SUPPORTED", "SUPPORTED", "NOT_FOUND"]])
        text = "Supported item one. Supported item two. Supported item three."
        passages = [
            {"source_type": "document", "text": f"{text} Additional evidence for the final claim.", "document_id": "doc-1"}
        ]
        with tempfile.TemporaryDirectory() as directory:
            report = verify_passages(
                "Claim one is supported [1]. Claim two is supported [1]. Claim three is supported [1]. Final claim unresolved [1].",
                passages,
                Path(directory) / "verifier.sqlite",
                llm,
            )
        self.assertEqual(len(llm.calls), 1)

    def test_application_adapter_uses_new_verdict_format(self):
        llm = FakeCohereVerifier([["CONTRADICTED"]])
        with tempfile.TemporaryDirectory() as directory:
            verdict = verify_draft(
                "Was the survey completed?",
                "The survey was completed.",
                [{"text": "Participants did not complete the survey."}],
                llm=llm,
                db_path=Path(directory) / "verifier.sqlite",
            )
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.claims[0].supported, "no")
        self.assertIn("CONTRADICTED", verdict.missing[0])

    def test_exact_no_evidence_response_is_a_valid_abstention(self):
        verdict = verify_draft(
            "What happened?",
            "The requested information isn't stated in the available document passages.",
            [],
        )
        self.assertTrue(verdict.ok)
        self.assertEqual(verdict.claims, [])

    def test_non_abstention_answer_with_no_claims_fails_closed(self):
        verdict = verify_draft(
            "Summarize the result.",
            "## Result",
            [{"text": "The study reports a positive result."}],
        )

        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.claims, [])

    def test_removes_unsupported_sentences_and_keeps_supported_ones(self):
        result = remove_unsupported_claims(
            "Supported fact [1]. Unsupported fact [2].",
            [
                {"claim": "Supported fact [1].", "supported": "yes"},
                {"claim": "Unsupported fact [2].", "supported": "no"},
            ],
        )

        self.assertEqual(result, "Supported fact [1].")


class CitationAndEvidenceGateTests(unittest.TestCase):
    def test_curator_preserves_definition_passage_added_by_writer(self):
        ranked = {
            "chunk_id": "ranked-context",
            "document_id": "doc-1",
            "chunk_index": 0,
            "text": "The term LLM appears in this section.",
            "hybrid_score": 0.8,
            "rerank_score": 0.8,
        }
        definition = {
            "chunk_id": "definition-context",
            "document_id": "doc-1",
            "chunk_index": 1,
            "text": "Large language models (LLM) are described in the survey.",
            "hybrid_score": 0.7,
            "rerank_score": 0.7,
        }

        def append_definition(question, retrieval, **kwargs):
            retrieval["context_blocks"].append(definition)
            return {
                "answer": "Large language models (LLM) [2]",
                "raw_answer": "Large language models (LLM) [2]",
                "citations": [2],
            }

        with (
            patch.object(curator, "rerank_chunks", return_value=([ranked], "high")),
            patch.object(curator, "expand_chunks", return_value=[ranked]),
            patch.object(curator.writer, "answer_from_retrieval", side_effect=append_definition),
        ):
            result = curator.curate_and_answer(
                "what is llm?",
                [ranked],
                intent="factual",
                request_intent="factual",
                retrieval_status={"sufficient": True},
            )

        self.assertEqual(
            [chunk["chunk_id"] for chunk in result["context_blocks"]],
            ["ranked-context", "definition-context"],
        )
        self.assertEqual(result["citation_metadata"][0]["chunk_id"], "definition-context")

    def test_extracts_citations_before_source_references_are_stripped(self):
        self.assertEqual(
            extract_citations("Fact [1]. Another [chunk 3, page 8]."),
            [1, 3],
        )

    def test_evidence_floor_uses_the_highest_reranker_score(self):
        self.assertTrue(graph.has_enough_evidence([{"rerank_score": 0.31}]))
        self.assertFalse(graph.has_enough_evidence([{"rerank_score": 0.29}]))
        self.assertFalse(graph.has_enough_evidence([]))

    def test_semantic_reranker_selects_best_passage_for_any_question(self):
        class FakeCrossEncoder:
            def predict(self, pairs):
                self.pairs = pairs
                return [-0.3, 1.2]

        encoder = FakeCrossEncoder()
        question = "What is the maximum fine for a first-class magistrate?"
        irrelevant = {
            "chunk_id": "unrelated",
            "text": "The magistrate supervises staff and assigns work.",
            "hybrid_score": 0.9,
        }
        relevant = {
            "chunk_id": "relevant",
            "text": "A first-class magistrate may impose a fine up to 50,000 rupees.",
            "hybrid_score": 0.5,
        }

        with patch.object(curator, "_get_cross_encoder", return_value=encoder):
            ranked, _ = curator.rerank_chunks(
                question,
                [irrelevant, relevant],
                top_k=1,
            )

        self.assertEqual(ranked[0]["chunk_id"], "relevant")
        self.assertGreater(ranked[0]["rerank_score"], 0.5)
        self.assertEqual(ranked[0]["cross_encoder_score"], 1.2)
        self.assertEqual(len(encoder.pairs), 2)

    def test_context_keeps_ranked_chunks_before_adding_neighbors(self):
        ranked = [
            {
                "chunk_id": f"ranked-{index}",
                "document_id": "doc",
                "chunk_index": index,
                "text": f"Relevant candidate {index}.",
                "rerank_score": 1.0 - index / 10,
            }
            for index in range(1, 7)
        ]
        neighbor = SimpleNamespace(
            chunk_id="neighbor-0",
            chunk_index=0,
            model_dump=lambda: {
                "chunk_id": "neighbor-0",
                "document_id": "doc",
                "chunk_index": 0,
                "text": "Adjacent context.",
            },
        )

        with patch(
            "agents.retriever_worker.get_neighbor_chunks",
            return_value=[neighbor],
        ):
            context = expand_chunks(ranked, max_chunks=8)

        self.assertEqual(
            [chunk["chunk_id"] for chunk in context[:6]],
            [chunk["chunk_id"] for chunk in ranked],
        )
        self.assertIn("neighbor-0", [chunk["chunk_id"] for chunk in context])

    def test_all_retrieved_answer_routes_run_through_verification(self):
        self.assertEqual(graph._after_writer({"route": "factual"}), "verifier")
        self.assertEqual(
            graph._after_writer({
                "route": "summary",
                "orchestrator_plan": {"needs_verification": False},
            }),
            "verifier",
        )

    def test_low_evidence_returns_abstention_without_calling_writer(self):
        chunk = {
            "chunk_id": "weak",
            "document_id": "doc",
            "chunk_index": 0,
            "text": "Unrelated text.",
            "distance": 0.9,
        }
        with (
            patch.object(graph.curator, "rerank_chunks", return_value=([{
                **chunk,
                "rerank_score": 0.1,
            }], "low")),
            patch.object(graph.curator, "expand_chunks", return_value=[chunk]),
            patch.object(graph.writer, "answer_from_retrieval") as writer,
        ):
            result = graph._writer_node({
                "question": "What is the answer?",
                "evidence": [chunk],
                "route": "factual",
            })

        writer.assert_not_called()
        self.assertEqual(result["answer"], graph.retriever_worker.NO_EVIDENCE_ANSWER)
        self.assertEqual(result["citations"], [])


class LegacyPipelineTests(unittest.TestCase):
    def test_legacy_entrypoint_keeps_raw_citations_and_verifies_them(self):
        chunk = {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "chunk_index": 0,
            "rerank_score": 0.8,
            "text": "A supported fact.",
        }
        retrieval = {
            "question": "What is the fact?",
            "chunks": [chunk],
            "decision_log": [],
        }
        verdict = Verdict(
            ok=True,
            claims=[
                ClaimCheck(
                    claim="A supported fact [1].",
                    passage_ids=[1],
                    supported="yes",
                )
            ],
            missing=[],
        )
        with (
            patch.object(retrieval_agent, "USE_MULTI_AGENT_RAG", False),
            patch.object(retrieval_agent, "retrieve", return_value=retrieval),
            patch.object(
                retrieval_agent,
                "rerank_chunks",
                return_value=([chunk], "high"),
            ),
            patch.object(retrieval_agent, "expand_chunks", return_value=[chunk]),
            patch.object(
                retrieval_agent,
                "answer_from_retrieval",
                return_value={
                    "answer": "A supported fact.",
                    "raw_answer": "A supported fact [1].",
                    "citations": [1],
                },
            ),
            patch.object(retrieval_agent, "verify_draft", return_value=verdict) as verify,
        ):
            result = retrieval_agent.run_rag("What is the fact?")

        verify.assert_called_once_with(
            "What is the fact?",
            "A supported fact [1].",
            [chunk],
        )
        self.assertEqual(result["answer"], "A supported fact.")
        self.assertEqual(result["raw_answer"], "A supported fact [1].")
        self.assertEqual(result["citations"], [1])


if __name__ == "__main__":
    unittest.main()
