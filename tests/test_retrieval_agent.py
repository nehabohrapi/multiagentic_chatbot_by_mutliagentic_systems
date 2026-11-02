import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.retrieval_agent import (
    MAX_RETRIEVAL_ATTEMPTS,
    RetrievalGrade,
    RetrievalPlan,
    answer_from_retrieval,
    expand_chunks,
    rerank_chunks,
    retrieve,
)
from agents.retriever_worker import (
    LLMQuotaExceeded,
    _grade_evidence_coverage,
    safe_chat_create,
)
from agents.verifier import verify_draft
from tools.document_store import DocumentChunkSearchResult


class RetrievalAgentTests(unittest.TestCase):
    @staticmethod
    def _chunk(chunk_id: str) -> DocumentChunkSearchResult:
        return DocumentChunkSearchResult(
            chunk_id=chunk_id,
            document_id="document-1",
            title="Research handbook",
            source_path="C:/documents/research-handbook.pdf",
            file_type="pdf",
            chunk_index=int(chunk_id[-1]),
            page_number=2,
            text=f"Evidence passage {chunk_id}",
            distance=0.2,
        )

    def test_safe_chat_error_reports_provider_reason_without_key(self):
        client = type("Client", (), {})()
        client.api_key = "test-secret"
        client.chat = type("Chat", (), {})()
        client.chat.completions = type("Completions", (), {})()
        client.chat.completions.create = unittest.mock.Mock(
            side_effect=RuntimeError("context window exceeded test-secret")
        )

        with (
            patch("agents.retriever_worker.FALLBACK_MODELS", []),
            patch("agents.retriever_worker.LLM_MAX_RETRIES", 3),
            patch("agents.retriever_worker.time.sleep") as sleep,
            self.assertRaisesRegex(RuntimeError, "context window exceeded \\[redacted\\]"),
        ):
            safe_chat_create(client, model="test-model", max_attempts=1)

        sleep.assert_not_called()

    def test_daily_quota_error_skips_same_model_retries_and_uses_fallback(self):
        class QuotaError(Exception):
            status_code = 429

        client = type("Client", (), {})()
        client.api_key = "test-key"
        client.chat = type("Chat", (), {})()
        client.chat.completions = type("Completions", (), {})()
        client.chat.completions.create = unittest.mock.Mock(
            side_effect=[QuotaError("tokens per day (TPD) rate_limit_exceeded"), "fallback-result"]
        )

        with (
            patch("agents.retriever_worker.FALLBACK_MODELS", ["fallback-model"]),
            patch("agents.retriever_worker.time.sleep") as sleep,
        ):
            result = safe_chat_create(
                client,
                model="primary-model",
                max_attempts=3,
            )

        self.assertEqual(result, "fallback-result")
        self.assertEqual(
            [call.kwargs["model"] for call in client.chat.completions.create.call_args_list],
            ["primary-model", "fallback-model"],
        )
        sleep.assert_not_called()

    def test_all_daily_quotas_fail_immediately_with_typed_error(self):
        class QuotaError(Exception):
            status_code = 429

        client = type("Client", (), {})()
        client.api_key = "test-key"
        client.chat = type("Chat", (), {})()
        client.chat.completions = type("Completions", (), {})()
        client.chat.completions.create = unittest.mock.Mock(
            side_effect=QuotaError(
                "tokens per day (TPD) limit reached; try again in 12m9s"
            )
        )

        with (
            patch("agents.retriever_worker.FALLBACK_MODELS", ["fallback-1", "fallback-2"]),
            patch("agents.retriever_worker.time.sleep") as sleep,
            self.assertRaisesRegex(LLMQuotaExceeded, "try again in 12m9s"),
        ):
            safe_chat_create(client, model="primary-model", max_attempts=3)

        self.assertEqual(client.chat.completions.create.call_count, 3)
        sleep.assert_not_called()

    def test_skips_planning_and_search_when_retrieval_is_not_needed(self):
        with (
            patch("agents.retrieval_agent._plan_queries") as planner,
            patch("agents.retrieval_agent.search_document_chunks") as search,
        ):
            result = retrieve("Hi", needs_retrieval=False)

        planner.assert_not_called()
        search.assert_not_called()
        self.assertEqual(result["attempts"], 0)
        self.assertEqual(result["chunks"], [])

    def test_planned_queries_skip_model_planning_and_grading(self):
        with (
            patch("agents.retrieval_agent._plan_queries") as planner,
            patch("agents.retrieval_agent._grade_chunks") as grader,
            patch(
                "agents.retrieval_agent.search_document_chunks",
                return_value=[
                    self._chunk("chunk-1").model_copy(
                        update={"text": "The refund window is thirty days."}
                    )
                ],
            ),
        ):
            result = retrieve(
                "What is the refund window?",
                planned_queries=["refund window"],
                skip_model_grading=True,
            )

        planner.assert_not_called()
        grader.assert_not_called()
        self.assertTrue(result["sufficient"])
        self.assertEqual(result["search_queries"], ["refund window"])

    def test_fallback_sufficiency_grade_checks_evidence_terms_not_chunk_count(self):
        sufficient = _grade_evidence_coverage(
            "What is the refund window?",
            [{"text": "The refund window is thirty days after purchase."}],
        )
        insufficient = _grade_evidence_coverage(
            "What is the refund window?",
            [{"text": "Garden tools and soil preparation."}],
        )

        self.assertTrue(sufficient.sufficient)
        self.assertFalse(insufficient.sufficient)
        self.assertIn("refund", insufficient.missing_info)
        self.assertIn("window", insufficient.missing_info)

    def test_stops_after_sufficient_retrieval_and_returns_chunk_metadata(self):
        plan = RetrievalPlan(queries=["refund window"], file_type="pdf")
        with (
            patch("agents.retrieval_agent._plan_queries", return_value=plan),
            patch(
                "agents.retrieval_agent.search_document_chunks",
                return_value=[self._chunk("chunk-1")],
            ) as search,
            patch(
                "agents.retrieval_agent._grade_chunks",
                return_value=RetrievalGrade(sufficient=True),
            ),
        ):
            result = retrieve("What is the refund window?")

        self.assertEqual(result["attempts"], 1)
        self.assertTrue(result["sufficient"])
        self.assertEqual(result["search_queries"], ["refund window"])
        self.assertEqual(result["chunks"][0]["page_number"], 2)
        search.assert_called_once_with(
            "refund window",
            max_results=20,
            file_type="pdf",
        )

    def test_replans_with_missing_information_and_stops_at_attempt_limit(self):
        def make_plan(state):
            attempt = state.get("attempts", 0)
            return RetrievalPlan(queries=[f"query attempt {attempt + 1}"])

        def make_grade(state):
            return RetrievalGrade(sufficient=False, missing_info="Need the effective date.")

        with (
            patch("agents.retrieval_agent._plan_queries", side_effect=make_plan) as planner,
            patch(
                "agents.retrieval_agent.search_document_chunks",
                side_effect=[
                    [self._chunk("chunk-1")],
                    [self._chunk("chunk-1"), self._chunk("chunk-2")],
                    [self._chunk("chunk-2"), self._chunk("chunk-3")],
                ],
            ),
            patch("agents.retrieval_agent._grade_chunks", side_effect=make_grade) as grader,
        ):
            result = retrieve("When does the policy take effect?")

        self.assertEqual(MAX_RETRIEVAL_ATTEMPTS, 2)
        self.assertEqual(result["attempts"], MAX_RETRIEVAL_ATTEMPTS)
        self.assertEqual(planner.call_count, MAX_RETRIEVAL_ATTEMPTS)
        self.assertEqual(grader.call_count, MAX_RETRIEVAL_ATTEMPTS)
        self.assertEqual(
            result["search_queries"],
            ["query attempt 1", "query attempt 2"],
        )
        self.assertEqual(
            {chunk["chunk_id"] for chunk in result["chunks"]},
            {"chunk-1", "chunk-2"},
        )
        self.assertFalse(result["sufficient"])
        self.assertEqual(result["missing_info"], "Need the effective date.")

    def test_answer_uses_evidence_without_exposing_source_references(self):
        retrieval = {
            "chunks": [self._chunk("chunk-1").model_dump()],
            "sufficient": True,
            "missing_info": "",
        }
        response = type(
            "Response",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type(
                                "Message",
                                (),
                                {"content": "The refund window is 30 days [1, Research handbook, p. 2]."},
                            )(),
                        },
                    )(),
                ],
            },
        )()
        with patch("agents.retrieval_agent.safe_chat_create", return_value=response) as chat:
            answer = answer_from_retrieval("What is the refund window?", retrieval)

        self.assertEqual(
            answer,
            "The refund window is 30 days [research-handbook.pdf, p. 2].",
        )
        prompt = chat.call_args.kwargs["messages"][1]["content"]
        self.assertIn("Evidence passage chunk-1", prompt)
        self.assertIn('"original": "What is the refund window?"', prompt)
        system_prompt = chat.call_args.kwargs["messages"][0]["content"]
        self.assertIn("Do not reveal chunk IDs", system_prompt)
        self.assertIn("do not answer from prior knowledge", system_prompt)
        self.assertIn("If the supplied passages do not state the requested information", system_prompt)
        self.assertIn("Do not add generic disclaimers", system_prompt)
        self.assertIn("explicitly addresses every one", system_prompt)
        self.assertIn("distinguish stated motivation from preparation or background", system_prompt)
        self.assertIn("implied motive as explicitly stated", system_prompt)
        self.assertIn("citations will be displayed with document names and page numbers", system_prompt)

    def test_high_confidence_single_passage_extracts_with_page_citation_without_llm(self):
        chunk = {
            **self._chunk("chunk-1").model_dump(),
            "text": "The annual plan refund window is thirty days after purchase.",
        }
        retrieval = {
            "chunks": [chunk],
            "rerank_confidence": "high",
            "intent": "factual",
        }

        with patch("agents.retrieval_agent.safe_chat_create") as chat:
            answer = answer_from_retrieval(
                "What is the annual plan refund window?",
                retrieval,
            )

        chat.assert_not_called()
        self.assertEqual(
            answer,
            "The annual plan refund window is thirty days after purchase. "
            "[research-handbook.pdf, p. 2]",
        )

    def test_acronym_colon_definition_is_extracted_without_llm(self):
        chunk = {
            **self._chunk("chunk-1").model_dump(),
            "text": (
                "DBMS: software that lets users define, create, query, update "
                "and administer databases (e.g. MySQL, PostgreSQL)."
            ),
        }

        with patch("agents.retrieval_agent.safe_chat_create") as chat:
            answer = answer_from_retrieval(
                "what is mean by DBMS?",
                {"context_blocks": [chunk], "intent": "factual"},
            )

        chat.assert_not_called()
        self.assertEqual(
            answer,
            "DBMS: software that lets users define, create, query, update "
            "and administer databases [research-handbook.pdf, p. 2]",
        )

    def test_acronym_parenthetical_expansion_is_extracted_without_llm(self):
        chunk = {
            **self._chunk("chunk-1").model_dump(),
            "text": "The term large language models (LLM) is used in this survey.",
        }

        with patch("agents.retrieval_agent.safe_chat_create") as chat:
            answer = answer_from_retrieval(
                "what is mean by llm",
                {"context_blocks": [chunk], "intent": "factual"},
            )

        chat.assert_not_called()
        self.assertEqual(
            answer,
            "large language models (LLM) [research-handbook.pdf, p. 2]",
        )

    def test_acronym_definition_uses_local_fts_when_top_context_omits_expansion(self):
        context = {
            **self._chunk("chunk-1").model_dump(),
            "text": "The LLM is discussed throughout this section.",
        }
        definition = {
            "chunk_id": "definition-chunk",
            "document_id": "document-1",
            "title": "Research handbook",
            "source_path": "C:/documents/research-handbook.pdf",
            "file_type": "pdf",
            "chunk_index": 3,
            "page_number": 4,
            "text": "The term large language models (LLM) is used in this survey.",
        }
        wrapped_product_name = {
            **definition,
            "chunk_id": "wrapped-product-name",
            "text": "Federatedscope-\nllm: A comprehensive package for fine-tuning large language models.",
        }

        with (
            patch("agents.retrieval_agent.safe_chat_create") as chat,
            patch(
                "memory.state_store.search_fts_chunks",
                return_value=[wrapped_product_name, definition],
            ) as search,
        ):
            answer = answer_from_retrieval(
                "what is llm?",
                {"context_blocks": [context], "intent": "factual"},
            )

        chat.assert_not_called()
        search.assert_called_once_with("llm", max_results=250, doc_ids=["document-1"])
        self.assertEqual(
            answer,
            "large language models (LLM) [research-handbook.pdf, p. 4]",
        )

    def test_definition_answer_is_cited_and_deterministically_verified(self):
        context = {
            **self._chunk("chunk-1").model_dump(),
            "text": "The LLM is discussed throughout this section.",
        }
        definition = {
            "chunk_id": "definition-chunk",
            "document_id": "document-1",
            "title": "Research handbook",
            "source_path": "C:/documents/research-handbook.pdf",
            "file_type": "pdf",
            "chunk_index": 3,
            "page_number": 4,
            "text": "The term large language models (LLM) is used in this survey.",
        }

        with (
            patch("agents.retriever_worker.safe_chat_create") as chat,
            patch("memory.state_store.search_fts_chunks", return_value=[definition]),
            tempfile.TemporaryDirectory() as directory,
        ):
            generated = answer_from_retrieval(
                "what is llm?",
                {"context_blocks": [context], "intent": "factual"},
                return_details=True,
            )
            verdict = verify_draft(
                "what is mean by llm",
                generated["raw_answer"],
                [context, definition],
                db_path=Path(directory) / "verifier.sqlite",
            )

        chat.assert_not_called()
        self.assertEqual(generated["citations"], [2])
        self.assertTrue(verdict.ok)
        self.assertEqual(verdict.claims[0].passage_ids, [2])

    def test_incomplete_single_passage_does_not_bypass_writer(self):
        chunk = {
            **self._chunk("chunk-1").model_dump(),
            "text": "The annual plan refund window is thirty days after purchase.",
        }
        retrieval = {
            "chunks": [chunk],
            "rerank_confidence": "high",
            "intent": "factual",
        }
        response = type(
            "Response",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type(
                                "Message",
                                (),
                                {"content": "The refund window is thirty days [1]."},
                            )(),
                        },
                    )(),
                ],
            },
        )()

        with patch("agents.retrieval_agent.safe_chat_create", return_value=response) as chat:
            answer = answer_from_retrieval(
                "What is the refund window and how many days can I file a claim for?",
                retrieval,
            )

        chat.assert_called_once()
        self.assertEqual(
            answer,
            "The refund window is thirty days [research-handbook.pdf, p. 2].",
        )
        self.assertIn(
            "explicitly addresses every one",
            chat.call_args.kwargs["messages"][0]["content"],
        )

    def test_answer_receives_structured_scoped_context_and_table_evidence(self):
        chunk = self._chunk("chunk-1").model_dump()
        chunk["structured_data"] = [{
            "kind": "table",
            "headers": ["Task", "Baseline Performance"],
            "rows": [["WikiFact", "23.61"]],
        }]
        retrieval = {
            "question": "What is the total baseline performance?",
            "orchestrator_plan": {
                "sub_questions": [{
                    "doc_ids": ["document-1"],
                }],
            },
            "context_blocks": [chunk],
            "sufficient": False,
            "missing_info": "No aggregate baseline is stated.",
        }
        response = type(
            "Response",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {"message": type("Message", (), {"content": "One value is 23.61 [1]."})()},
                    )(),
                ],
            },
        )()
        with patch("agents.retrieval_agent.safe_chat_create", return_value=response) as chat:
            answer_from_retrieval(
                "What is the total baseline performance?",
                retrieval,
            )

        prompt = chat.call_args.kwargs["messages"][1]["content"]
        serialized_context = prompt.split(
            "<DOCUMENT_CONTEXT>\n", 1
        )[1].split("\n</DOCUMENT_CONTEXT>", 1)[0]
        context = json.loads(serialized_context)
        self.assertEqual(context["question"]["original"], "What is the total baseline performance?")
        self.assertEqual(context["document_scope"]["scope_type"], "requested_documents")
        self.assertEqual(context["document_scope"]["requested_document_ids"], ["document-1"])
        self.assertEqual(context["retrieval_status"]["coverage"], "retrieved_passages_only")
        self.assertIn("No aggregate baseline is stated.", context["retrieval_status"]["limitations"])
        self.assertEqual(context["evidence"][0]["source_id"], 1)
        self.assertEqual(context["evidence"][0]["filename"], "research-handbook.pdf")
        self.assertEqual(context["evidence"][0]["page_number"], 2)
        self.assertEqual(context["evidence"][0]["tables"][0]["headers"], ["Task", "Baseline Performance"])
        self.assertEqual(context["evidence"][0]["tables"][0]["rows"], [["WikiFact", "23.61"]])

    def test_answer_refuses_to_invent_when_no_chunks_are_retrieved(self):
        with patch("agents.retrieval_agent.safe_chat_create") as chat:
            answer = answer_from_retrieval("What is the refund policy?", {"chunks": []})

        chat.assert_not_called()
        self.assertEqual(
            answer,
            "The requested information isn't stated in the available document passages.",
        )

    def test_low_confidence_evidence_returns_not_found_without_llm(self):
        retrieval = {
            "chunks": [self._chunk("chunk-1").model_dump()],
            "rerank_confidence": "low",
        }

        with patch("agents.retrieval_agent.safe_chat_create") as chat:
            answer = answer_from_retrieval("What caused the policy change?", retrieval)

        chat.assert_not_called()
        self.assertEqual(
            answer,
            "The requested information isn't stated in the available document passages.",
        )

    def test_rerank_prefers_vector_and_question_term_matches(self):
        chunks = [
            {**self._chunk("chunk-1").model_dump(), "distance": 0.2,
             "text": "The annual plan refund window is thirty days."},
            {**self._chunk("chunk-2").model_dump(), "distance": 0.5,
             "text": "This passage discusses unrelated gardening."},
        ]

        ranked, confidence = rerank_chunks("annual plan refund", chunks)

        self.assertEqual(ranked[0]["chunk_id"], "chunk-1")
        self.assertIn(confidence, {"high", "low"})
        self.assertIn("rerank_score", ranked[0])

    def test_expand_chunks_adds_adjacent_context(self):
        hit = {**self._chunk("chunk-1").model_dump(), "document_id": "document-1",
               "chunk_index": 1, "rerank_score": 0.8}
        neighbor = {**self._chunk("chunk-0").model_dump(), "document_id": "document-1",
                    "chunk_index": 0, "match_type": "neighbor"}
        with patch(
            "agents.retrieval_agent.get_neighbor_chunks",
            return_value=[
                DocumentChunkSearchResult(**neighbor),
                DocumentChunkSearchResult(**hit),
            ],
        ):
            blocks = expand_chunks([hit], window=1)

        self.assertEqual([block["chunk_index"] for block in blocks], [1, 0])

    def test_retrieval_plan_resilience_to_arbitrary_intents_and_nulls(self):
        # 1. Null intent from LLM should gracefully default to "question", not throw literal_error
        plan_null = RetrievalPlan.model_validate(
            {"queries": ["in which year war happened"], "file_type": None, "intent": None}
        )
        self.assertEqual(plan_null.intent, "question")
        self.assertEqual(plan_null.queries, ["in which year war happened"])

        # 2. Arbitrary intent labels (e.g. factual, historical) should not throw validation error
        plan_arbitrary = RetrievalPlan.model_validate(
            {"queries": ["in which year war happened"], "intent": "historical_factual"}
        )
        self.assertEqual(plan_arbitrary.intent, "question")

        # 3. Summary intent is correctly preserved
        plan_summary = RetrievalPlan.model_validate(
            {"queries": ["brief overview"], "intent": "summary"}
        )
        self.assertEqual(plan_summary.intent, "summary")

        # 4. Invalid file_type should gracefully default to None
        plan_bad_type = RetrievalPlan.model_validate(
            {"queries": ["war"], "file_type": "unsupported_xyz"}
        )
        self.assertIsNone(plan_bad_type.file_type)


if __name__ == "__main__":
    unittest.main()
