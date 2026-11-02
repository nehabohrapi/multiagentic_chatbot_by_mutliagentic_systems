import tempfile
import unittest
import json
import re
import time
from pathlib import Path
from unittest.mock import patch

from agents import gemini_client, graph, orchestrator, summarizer
from agents.router import reformulate_query
from memory import state_store
from tools.document_store import DocumentChunkSearchResult
from agents.verifier import verify_draft


class ArchitectureUpgradeTests(unittest.TestCase):
    def setUp(self):
        verifier = patch(
            "agents.summarizer.verify_answer",
            side_effect=lambda draft, *_args, **_kwargs: {
                "verified_claims": 0,
                "total_claims": 0,
                "revised_answer": draft,
            },
        )
        self.addCleanup(verifier.stop)
        verifier.start()

    def test_reduce_citations_map_to_their_map_batch(self):
        map_summaries = [
            {
                "summary": "First batch evidence [chunk 0].",
                "chunks": [{"chunk_index": 0}],
            },
            {
                "summary": "Second batch evidence [chunk 1].",
                "chunks": [{"chunk_index": 1}],
            },
        ]

        evidence, chunk_citations = summarizer._map_summary_evidence(map_summaries)
        reduce_draft = summarizer._remap_summary_citations(
            "Claim from first [chunk 0]. Claim from second [chunk 1].",
            chunk_citations,
        )

        self.assertEqual(len(evidence), 2)
        self.assertEqual(
            [passage["citation_id"] for passage in evidence],
            [1, 2],
        )
        self.assertEqual(
            reduce_draft,
            "Claim from first [1]. Claim from second [2].",
        )

    def _chunk(self, index: int) -> DocumentChunkSearchResult:
        return DocumentChunkSearchResult(
            chunk_id=f"chunk-{index}",
            document_id="document-1",
            title="Research paper",
            source_path="C:/documents/research-paper.pdf",
            file_type="pdf",
            chunk_index=index,
            page_number=index + 1,
            text=f"Research methods and findings for section {index}.",
            distance=0.0,
            match_type="summary",
        )

    def test_summary_route_bypasses_verification(self):
        self.assertEqual(graph._after_router({"route": "summary"}), "summarizer")
        self.assertEqual(
            graph._after_summarizer({"clarification": "", "answerable": True}),
            graph.END,
        )
        self.assertEqual(
            graph._after_summarizer({"clarification": "Choose a document"}),
            graph.END,
        )

    def test_unverified_non_summary_answer_finalizes_without_reverification(self):
        state = {
            "route": "factual",
            "rounds": 1,
            "verdict": {"ok": False},
        }

        self.assertEqual(graph._after_verifier(state), "finalize")

    def test_summarizer_receives_orchestrator_document_scope(self):
        summary_result = {
            "answer": "Document summary [chunk 0]",
            "citations": [0],
            "chunks": [],
            "cached": False,
        }
        state = {
            "question": "Summarize the study",
            "reformulated_question": "Summarize the study on wheat yield",
            "run_id": "run-1",
            "orchestrator_plan": {
                "sub_questions": [{"doc_ids": ["doc-1"]}],
            },
        }

        with patch.object(graph, "summarize_document", return_value=summary_result) as summarize:
            result = graph._summarizer_node(state)

        summarize.assert_called_once_with(
            "Summarize the study",
            document_query="Summarize the study on wheat yield",
            doc_ids=["doc-1"],
            history=[],
            run_id="run-1",
        )
        self.assertEqual(result["answer"], "Document summary [chunk 0]")

    def test_summarizer_receives_conversation_history(self):
        history = [
            {"role": "user", "content": "Tell me about wheat yield."},
            {"role": "assistant", "content": "The study reports wheat yield."},
        ]
        state = {
            "question": "Summarize that document",
            "history": history,
            "orchestrator_plan": {"sub_questions": []},
        }
        with patch.object(
            graph,
            "summarize_document",
            return_value={
                "answer": "Summary",
                "citations": [],
                "chunks": [],
                "cached": False,
            },
        ) as summarize:
            graph._summarizer_node(state)

        self.assertEqual(summarize.call_args.kwargs["history"], history)

    def test_summary_request_without_document_context_asks_clarification(self):
        result = {
            "answer": "Which document would you like summarized?",
            "clarifying_question": "Which document would you like summarized?",
            "citations": [],
            "chunks": [],
            "cached": False,
        }
        state = {
            "question": "Summarize the document",
            "orchestrator_plan": {"sub_questions": []},
        }
        with patch.object(graph, "summarize_document", return_value=result):
            summary_state = graph._summarizer_node(state)

        self.assertEqual(summary_state["clarification"], result["answer"])
        self.assertEqual(graph._after_summarizer(summary_state), graph.END)

    def test_reformulate_query_uses_recent_follow_up_context(self):
        result = reformulate_query(
            "What about this?",
            [{"role": "user", "content": "The research methodology"}],
        )
        self.assertIn("The research methodology", result)

    def test_summary_map_reduce_cache_and_gemini_prompts(self):
        chunks = [self._chunk(0), self._chunk(1)]
        from types import SimpleNamespace

        client = SimpleNamespace(
            models=SimpleNamespace(
                count_tokens=lambda **_: SimpleNamespace(total_tokens=100),
                generate_content=unittest.mock.Mock(side_effect=[
                    SimpleNamespace(text="Batch findings [chunk 1]"),
                    SimpleNamespace(text="Summary [chunk 1]"),
                    SimpleNamespace(text="Brief batch findings [chunk 1]"),
                    SimpleNamespace(text="Brief summary [chunk 1]"),
                ]),
            )
        )
        history = [{
            "role": "user",
            "content": (
                "Make the summary comprehensive, include examples, and provide "
                "source page references."
            ),
        }]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(state_store, "STATE_DB_FILE", Path(directory) / "state.db"):
                with (
                    patch.object(
                        summarizer,
                        "get_document_catalog",
                        return_value=[{"doc_id": "document-1", "title": "Research paper"}],
                    ),
                    patch.object(
                        summarizer,
                        "get_document_chunks",
                        return_value=chunks,
                    ),
                    patch.object(
                        summarizer.genai,
                        "Client",
                        return_value=client,
                    ) as create_client,
                    patch("agents.verifier.verify_answer") as verify_summary,
                ):
                    first = summarizer.summarize_document(
                        "Summarize the paper",
                        history=history,
                    )
                    second = summarizer.summarize_document(
                        "Summarize the paper",
                        history=history,
                    )
                    different_style = summarizer.summarize_document(
                        "Summarize the paper",
                        history=[{
                            "role": "user",
                            "content": "Please make this concise.",
                        }],
                    )

        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertFalse(different_style["cached"])
        self.assertEqual(create_client.call_count, 2)
        self.assertEqual(client.models.generate_content.call_count, 4)
        self.assertEqual(first["answer"], "Summary [research-paper.pdf, PDF p. 1]")
        self.assertEqual(first["citations"], [1])
        self.assertEqual(first["raw_answer"], "Summary [chunk 1]")
        self.assertEqual(second["raw_answer"], "Summary [chunk 1]")
        verify_summary.assert_not_called()
        self.assertNotIn("chunk", first["answer"].lower())
        self.assertIn("[research-paper.pdf, PDF p. 1]", first["answer"])
        prompts = [
            call.kwargs["contents"]
            for call in client.models.generate_content.call_args_list
        ]
        prompt = prompts[0]
        self.assertIn("Research methods and findings for section 0.", prompt)
        self.assertIn("Research methods and findings for section 1.", prompt)
        self.assertLess(
            prompt.index("Research methods and findings for section 0."),
            prompt.index("Research methods and findings for section 1."),
        )
        self.assertIn("Batch findings [chunk 1]", prompts[1])
        self.assertIn("[chunk 1]\nResearch methods and findings for section 0.", prompts[0])
        self.assertIn("[/chunk 1]", prompts[0])
        self.assertNotIn("Research paper", prompts[0])
        self.assertIn("complete coverage of major topics", summarizer._SUMMARY_SYSTEM_PROMPT)
        self.assertIn("do not turn a user's", summarizer._SUMMARY_SYSTEM_PROMPT)
        self.assertIn("source filename and page references", summarizer._SUMMARY_SYSTEM_PROMPT)
        self.assertIn("important examples or case studies", prompts[0])
        self.assertIn("introductory material overshadow later parts", prompts[0])
        self.assertIn("Coverage and accuracy matter more", prompts[1])
        self.assertIn("Do not use the user's summary criteria as headings", prompts[1])
        self.assertIn("comprehensive, include examples", prompts[1])
        self.assertIn("source page references", prompts[1])
        self.assertIn("Please make this concise", prompts[3])
        self.assertEqual(
            client.models.generate_content.call_args.kwargs["model"],
            summarizer.SUMMARY_MODEL,
        )
        self.assertNotIn("1. Overview", prompts[1])
        self.assertNotIn("Main Topics: cover", prompts[1])

    def test_summary_repairs_missing_source_sections_before_caching(self):
        from types import SimpleNamespace

        chunks = [
            DocumentChunkSearchResult(
                chunk_id=f"section-{index}",
                document_id="document-1",
                title="Research paper",
                source_path="C:/documents/research-paper.pdf",
                file_type="pdf",
                chunk_index=index - 1,
                page_number=index,
                text=f"{index}. {title}\nThe source explains {title.lower()}.",
                distance=0.0,
                match_type="summary",
            )
            for index, title in enumerate(
                ["Introduction", "Data Model", "Transactions"],
                start=1,
            )
        ]
        client = SimpleNamespace(
            models=SimpleNamespace(
                count_tokens=lambda **_: SimpleNamespace(total_tokens=100),
                generate_content=unittest.mock.Mock(side_effect=[
                    SimpleNamespace(text="1. Introduction\nOverview [chunk 1]"),
                    SimpleNamespace(text=(
                        "2. Data Model\nEntities describe data [chunk 2].\n"
                        "3. Transactions\nTransactions group database operations [chunk 3]."
                    )),
                    SimpleNamespace(text="1. Introduction\nOverview [chunk 1]"),
                    SimpleNamespace(text=(
                        "1. Introduction\nOverview [chunk 1]\n\n"
                        "2. Data Model\nEntities describe data [chunk 2].\n\n"
                        "3. Transactions\nTransactions group database operations [chunk 3]."
                    )),
                ]),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(state_store, "STATE_DB_FILE", Path(directory) / "state.db"):
                with (
                    patch.object(
                        summarizer,
                        "get_document_catalog",
                        return_value=[{"doc_id": "document-1", "title": "Research paper"}],
                    ),
                    patch.object(summarizer, "get_document_chunks", return_value=chunks),
                    patch.object(summarizer, "_new_client", return_value=client),
                ):
                    result = summarizer.summarize_document("Summarize the paper")

        self.assertIn("Introduction", result["answer"])
        self.assertIn("Data Model", result["answer"])
        self.assertIn("Transactions", result["answer"])
        self.assertEqual(client.models.generate_content.call_count, 4)
        map_prompt = client.models.generate_content.call_args_list[0].kwargs["contents"]
        reduce_prompt = client.models.generate_content.call_args_list[2].kwargs["contents"]
        self.assertIn("2. Data Model", map_prompt)
        self.assertIn("3. Transactions", map_prompt)
        self.assertIn("Source section headings that must each receive substantive coverage", reduce_prompt)

    def test_oversized_map_batch_is_split_and_summary_completes(self):
        from types import SimpleNamespace

        chunks = [self._chunk(index) for index in range(4)]
        client = SimpleNamespace(
            models=SimpleNamespace(
                count_tokens=lambda **kwargs: SimpleNamespace(
                    total_tokens=(
                        275_227
                        if kwargs["contents"].count("[/chunk ") > 2
                        else 100
                    )
                ),
            )
        )

        def generate_summary(_client, **kwargs):
            if kwargs["input_tokens"] > summarizer.GEMINI_TPM_HARD_LIMIT:
                raise ValueError(
                    "A Gemini request uses 275,227 input tokens, exceeding the "
                    "configured 250,000-token single-request quota ceiling."
                )
            match = re.search(r"\[chunk\s+(\d+)\]", kwargs["contents"])
            citation = match.group(1) if match else "1"
            return SimpleNamespace(text=f"Summary [chunk {citation}]")

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(state_store, "STATE_DB_FILE", Path(directory) / "state.db"):
                with (
                    patch.object(summarizer, "SUMMARY_BATCH_SIZE", 4),
                    patch.object(
                        summarizer,
                        "GEMINI_TPM_HARD_LIMIT",
                        250_000,
                    ),
                    patch.object(
                        summarizer,
                        "get_document_catalog",
                        return_value=[{"doc_id": "document-1", "title": "Research paper"}],
                    ),
                    patch.object(summarizer, "get_document_chunks", return_value=chunks),
                    patch.object(summarizer, "_new_client", return_value=client),
                    patch.object(
                        summarizer,
                        "generate_gemini_content",
                        side_effect=generate_summary,
                    ),
                ):
                    result = summarizer.summarize_document("Summarize the paper")

        self.assertIn("Summary", result["answer"])
        self.assertEqual(result["model_calls"], 3)

    def test_oversized_reduce_is_reduced_hierarchically(self):
        from types import SimpleNamespace

        chunks = [self._chunk(index) for index in range(4)]
        client = SimpleNamespace(
            models=SimpleNamespace(
                count_tokens=lambda **kwargs: SimpleNamespace(
                    total_tokens=(
                        275_227
                        if kwargs["contents"].count("Batch summary ") > 2
                        else 100
                    )
                ),
            )
        )

        def generate_summary(_client, **kwargs):
            if kwargs["input_tokens"] > summarizer.GEMINI_TPM_HARD_LIMIT:
                raise ValueError(
                    "A Gemini request uses 275,227 input tokens, exceeding the "
                    "configured 250,000-token single-request quota ceiling."
                )
            return SimpleNamespace(text="Summary [chunk 1]")

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(state_store, "STATE_DB_FILE", Path(directory) / "state.db"):
                with (
                    patch.object(summarizer, "SUMMARY_BATCH_SIZE", 1),
                    patch.object(
                        summarizer,
                        "get_document_catalog",
                        return_value=[{"doc_id": "document-1", "title": "Research paper"}],
                    ),
                    patch.object(summarizer, "get_document_chunks", return_value=chunks),
                    patch.object(summarizer, "_new_client", return_value=client),
                    patch.object(
                        summarizer,
                        "generate_gemini_content",
                        side_effect=generate_summary,
                    ),
                ):
                    result = summarizer.summarize_document("Summarize the paper")

        self.assertIn("Summary", result["answer"])
        self.assertEqual(result["model_calls"], 7)

    def test_summary_section_coverage_ignores_faq_numbered_questions(self):
        sections = summarizer._extract_summary_sections([
            {
                "citation_id": 1,
                "chunk_index": 0,
                "text": (
                    "L. canadensis (Wliiteaves), du Sénonien du Canada, par\n"
                    "C. Campillo\n"
                    "V. Results\n"
                    "IV. Important Results\n"
                    "13. Advanced Topics\n\n"
                    "14. Quick Revision: Frequently Asked Questions\n\n"
                    "5. Difference between clustered and non-clustered index?"
                ),
            },
        ])

        self.assertEqual(
            [section["title"] for section in sections],
            [
                "Results",
                "Important Results",
                "Advanced Topics",
                "Quick Revision: Frequently Asked Questions",
            ],
        )

    def test_summary_section_coverage_ignores_numbered_report_findings(self):
        findings = [
            "1. High confidence of decrease in the south-west of the state of Western Australia",
            "3. High confidence of increase in the south-west of the state of Western Australia",
            "4. For the Alps, conditions conducive to landslides are expected to increase",
            "7. Medium confidence of decrease in frequency and increase in intensities",
            "6. Except for Northern Baltic Sea coasts where relative sea levels fall",
        ]
        sections = summarizer._extract_summary_sections([
            {
                "citation_id": 1,
                "chunk_index": 0,
                "text": "\n".join(findings),
            },
        ])

        self.assertEqual(sections, [])

    def test_summary_section_coverage_ignores_numbered_legal_clauses(self):
        sections = summarizer._extract_summary_sections([
            {
                "citation_id": 1,
                "chunk_index": 0,
                "text": (
                    "15. The State Government may appoint an Executive\n"
                    "21. Subject to the other provisions of this Sanhita,—\n"
                    "36. Every police officer while making an arrest shall—\n"
                    "444. Save as otherwise expressly provided, no party has any right to\n"
                    "282. When in the course of the trial of a summons-case relating "
                    "to an offence punishable,\n"
                    "381. Any Court dealing with an application made to it for filing "
                    "a complaint under\n"
                    "513. For the purposes of this Chapter, unless the context "
                    "otherwise requires, \"period\n"
                    "6. Besides the High Courts and the Courts constituted under any "
                    "law, other than this\n"
                    "321. Upon receipt of the commission, the Chief Judicial "
                    "Magistrate or such Magistrate\n"
                    "334. In any inquiry, trial or other proceeding under this "
                    "Sanhita, a previous conviction\n"
                    "410. In every case so submitted, the confirmation of the "
                    "sentence, or any new sentence\n"
                    "427. After perusing such record and hearing the appellant or "
                    "his advocate, if he\n"
                    "527. A public servant having any duty to perform in connection "
                    "with the sale of any\n"
                    "1. Any Questions and Answers\n"
                    "## 2. Important Concepts"
                ),
            },
        ])

        self.assertEqual(
            [section["title"] for section in sections],
            ["Any Questions and Answers", "Important Concepts"],
        )

    def test_map_coverage_repair_does_not_abort_on_unmatched_heading(self):
        from types import SimpleNamespace

        chunk = DocumentChunkSearchResult(
            chunk_id="section-0",
            document_id="document-1",
            title="Research paper",
            source_path="C:/documents/research-paper.pdf",
            file_type="pdf",
            chunk_index=0,
            page_number=1,
            text="1. Important Topic\nSource content for an important topic.",
            distance=0.0,
            match_type="summary",
        )
        client = SimpleNamespace(
            models=SimpleNamespace(
                count_tokens=lambda **_: SimpleNamespace(total_tokens=100),
                generate_content=unittest.mock.Mock(side_effect=[
                    SimpleNamespace(text="Brief source summary [chunk 1]"),
                    SimpleNamespace(text="Brief source summary [chunk 1]"),
                    SimpleNamespace(text="Important Topic [chunk 1]"),
                ]),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(state_store, "STATE_DB_FILE", Path(directory) / "state.db"):
                with (
                    patch.object(
                        summarizer,
                        "get_document_catalog",
                        return_value=[{"doc_id": "document-1", "title": "Research paper"}],
                    ),
                    patch.object(summarizer, "get_document_chunks", return_value=[chunk]),
                    patch.object(summarizer, "_new_client", return_value=client),
                ):
                    result = summarizer.summarize_document("Summarize the paper")

        self.assertIn("Important Topic", result["answer"])
        self.assertEqual(client.models.generate_content.call_count, 3)

    def test_summary_section_coverage_ignores_table_of_contents_entries(self):
        sections = summarizer._extract_summary_sections([
            {
                "citation_id": 1,
                "chunk_index": 0,
                "text": (
                    "2. Yield Response to Water: the Original FAO Water "
                    "Production Function............................... 6\n"
                    "3. Yield Response to Water of Herbaceous Crops: the "
                    "AquaCrop simulation model........... 16\n"
                    "4. Actual Section\n"
                ),
            },
        ])

        self.assertEqual(
            [section["title"] for section in sections],
            ["Actual Section"],
        )

    def test_summary_source_excludes_table_of_contents_entries(self):
        formatted_chunk = summarizer._format_chunk({
            "citation_id": 1,
            "chunk_index": 0,
            "text": (
                "2. Yield Response to Water: the Original FAO Water "
                "Production Function............................... 6\n"
                "The section explains crop yield response to water stress."
            ),
        })

        self.assertNotIn("Yield Response to Water", formatted_chunk)
        self.assertNotIn("....... 6", formatted_chunk)
        self.assertIn("The section explains crop yield response", formatted_chunk)

    def test_summary_section_coverage_reads_markdown_headings_and_skips_references(self):
        sections = summarizer._extract_summary_sections([
            {
                "citation_id": 1,
                "chunk_index": 0,
                "text": (
                    "# Book title\n"
                    "## 1. Executive Summary\n"
                    "## 2. Important Concepts\n"
                    "## 3. References\n"
                ),
            },
        ])

        self.assertEqual(
            [section["title"] for section in sections],
            ["Executive Summary", "Important Concepts"],
        )

    def test_interrupted_summary_resumes_from_saved_map_batch(self):
        from types import SimpleNamespace

        chunks = [self._chunk(0), self._chunk(1)]
        first_client = SimpleNamespace(
            models=SimpleNamespace(
                count_tokens=lambda **_: SimpleNamespace(total_tokens=100),
                generate_content=unittest.mock.Mock(side_effect=[
                    SimpleNamespace(text="Saved first batch [chunk 1]"),
                    RuntimeError("temporary Gemini outage"),
                ]),
            )
        )
        resumed_client = SimpleNamespace(
            models=SimpleNamespace(
                count_tokens=lambda **_: SimpleNamespace(total_tokens=100),
                generate_content=unittest.mock.Mock(side_effect=[
                    SimpleNamespace(text="Resumed second batch [chunk 2]"),
                    SimpleNamespace(text="Complete summary [chunk 1] [chunk 2]"),
                ]),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(state_store, "STATE_DB_FILE", Path(directory) / "state.db"):
                with (
                    patch.object(summarizer, "SUMMARY_BATCH_SIZE", 1),
                    patch.object(
                        summarizer,
                        "get_document_catalog",
                        return_value=[{"doc_id": "document-1", "title": "Research paper"}],
                    ),
                    patch.object(summarizer, "get_document_chunks", return_value=chunks),
                    patch.object(
                        summarizer,
                        "_new_client",
                        side_effect=[first_client, resumed_client],
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, "temporary Gemini outage"):
                        summarizer.summarize_document("Summarize the paper")

                    result = summarizer.summarize_document("Summarize the paper")

        self.assertEqual(first_client.models.generate_content.call_count, 2)
        self.assertEqual(resumed_client.models.generate_content.call_count, 2)
        resumed_prompts = [
            call.kwargs["contents"]
            for call in resumed_client.models.generate_content.call_args_list
        ]
        self.assertIn("Source chunk batch 2", resumed_prompts[0])
        self.assertNotIn("Research methods and findings for section 0.", resumed_prompts[0])
        self.assertIn("Saved first batch [chunk 1]", resumed_prompts[1])
        self.assertTrue(result["answer"].startswith("Complete summary"))

    def test_completed_summary_cache_does_not_expire_by_default(self):
        now = time.time()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(state_store, "STATE_DB_FILE", Path(directory) / "state.db"):
                state_store.put_summary(
                    document_id="document-1",
                    model="gemini-test",
                    digest="content-digest",
                    answer="Permanent summary",
                    raw_answer="Permanent summary [chunk 0]",
                    citations=[0],
                )
                with patch.object(state_store.time, "time", return_value=now + 10 * 86400):
                    cached = state_store.get_summary(
                        document_id="document-1",
                        model="gemini-test",
                        digest="content-digest",
                    )

        self.assertIsNotNone(cached)
        self.assertEqual(cached["answer"], "Permanent summary")

    def test_gemini_token_reservations_share_persistent_rolling_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(state_store, "STATE_DB_FILE", Path(directory) / "state.db"):
                first_wait = state_store.reserve_gemini_tokens(70, limit=100)
                second_wait = state_store.reserve_gemini_tokens(20, limit=100)
                with state_store._connection() as connection:
                    reserved = connection.execute(
                        "SELECT SUM(tokens) FROM gemini_token_usage"
                    ).fetchone()[0]

        self.assertGreaterEqual(first_wait, 0)
        self.assertGreaterEqual(second_wait, 0)
        self.assertEqual(reserved, 90)

    def test_gemini_token_reservation_rejects_single_request_over_budget(self):
        with self.assertRaisesRegex(ValueError, "exceeding"):
            state_store.reserve_gemini_tokens(151, limit=100, hard_limit=150)

    def test_large_gemini_request_can_reserve_alone_under_provider_ceiling(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(state_store, "STATE_DB_FILE", Path(directory) / "state.db"):
                waited = state_store.reserve_gemini_tokens(
                    120,
                    limit=100,
                    hard_limit=150,
                )
                with state_store._connection() as connection:
                    reserved = connection.execute(
                        "SELECT SUM(tokens) FROM gemini_token_usage"
                    ).fetchone()[0]

        self.assertGreaterEqual(waited, 0)
        self.assertEqual(reserved, 120)

    def test_final_reduce_over_context_limit_asks_before_more_calls(self):
        from types import SimpleNamespace

        chunk = self._chunk(0)
        client = SimpleNamespace(
            models=SimpleNamespace(
                count_tokens=unittest.mock.Mock(side_effect=[
                    SimpleNamespace(total_tokens=50),
                    SimpleNamespace(total_tokens=101),
                ]),
                generate_content=unittest.mock.Mock(
                    return_value=SimpleNamespace(text="Batch summary [chunk 0]")
                ),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(state_store, "STATE_DB_FILE", Path(directory) / "state.db"):
                with (
                    patch.object(summarizer, "SUMMARY_MODEL_INPUT_TOKEN_LIMIT", 100),
                    patch.object(
                        summarizer,
                        "get_document_catalog",
                        return_value=[{"doc_id": "document-1", "title": "Research paper"}],
                    ),
                    patch.object(summarizer, "get_document_chunks", return_value=[chunk]),
                    patch.object(summarizer.genai, "Client", return_value=client),
                ):
                    result = summarizer.summarize_document("Summarize the paper")

        self.assertIn("May I continue", result["clarifying_question"])
        client.models.generate_content.assert_called_once()

    def test_summary_uses_all_chunks_in_configured_batches(self):
        from types import SimpleNamespace

        batch_size = summarizer.SUMMARY_BATCH_SIZE
        chunks = [self._chunk(index) for index in range(batch_size + 1)]
        client = SimpleNamespace(
            models=SimpleNamespace(
                count_tokens=lambda **_: SimpleNamespace(total_tokens=100),
                generate_content=unittest.mock.Mock(side_effect=[
                    SimpleNamespace(text="First batch [chunk 1]"),
                    SimpleNamespace(text=f"Second batch [chunk {batch_size + 1}]"),
                    SimpleNamespace(
                        text=f"Complete summary [chunk 1] [chunk {batch_size + 1}]"
                    ),
                ]),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(state_store, "STATE_DB_FILE", Path(directory) / "state.db"):
                with (
                    patch.object(
                        summarizer,
                        "get_document_catalog",
                        return_value=[{"doc_id": "document-1", "title": "Research paper"}],
                    ),
                    patch.object(summarizer, "get_document_chunks", return_value=chunks) as load,
                    patch.object(summarizer.genai, "Client", return_value=client),
                    patch(
                        "agents.summarizer.verify_answer",
                        side_effect=[
                            {
                                "verified_claims": 1,
                                "total_claims": 1,
                                "revised_answer": "Verified first batch [chunk 1]",
                            },
                            {
                                "verified_claims": 1,
                                "total_claims": 1,
                                "revised_answer": (
                                    f"Verified second batch [chunk {batch_size + 1}]"
                                ),
                            },
                        ],
                    ) as verify,
                ):
                    result = summarizer.summarize_document("Summarize the paper")

        load.assert_called_once_with("document-1")
        self.assertEqual(client.models.generate_content.call_count, 3)
        prompts = [
            call.kwargs["contents"]
            for call in client.models.generate_content.call_args_list
        ]
        self.assertIn(f"[chunk {batch_size}]", prompts[0])
        self.assertNotIn(f"[chunk {batch_size + 1}]", prompts[0])
        self.assertIn(f"[chunk {batch_size + 1}]", prompts[1])
        self.assertIn(f"Verified first batch [chunk 1]", prompts[2])
        self.assertIn(f"Verified second batch [chunk {batch_size + 1}]", prompts[2])
        self.assertEqual(len(result["chunks"]), batch_size + 1)
        self.assertEqual(verify.call_count, 2)
        verification_calls = verify.call_args_list
        self.assertEqual(len(verification_calls[0].args[1]), batch_size)
        self.assertEqual(len(verification_calls[1].args[1]), 1)
        self.assertEqual(verification_calls[0].args[1][0]["citation_id"], 1)
        self.assertEqual(
            verification_calls[0].args[1][-1]["citation_id"],
            batch_size,
        )
        self.assertEqual(
            verification_calls[1].args[1][0]["citation_id"],
            batch_size + 1,
        )
        for call in verification_calls:
            self.assertTrue(call.kwargs["revise"])
            self.assertTrue(call.kwargs["is_summary"])
            self.assertFalse(call.kwargs["include_saved_web"])
        self.assertIn(f"[chunk {batch_size + 1}]", result["raw_answer"])

    def test_summary_document_selection_uses_explicit_title(self):
        documents = [
            {"doc_id": "doc-1", "title": "wheat_study"},
            {"doc_id": "doc-2", "title": "barley_study"},
        ]

        selected = summarizer._select_document_ids(
            question="Summarize wheat_study",
            history=[],
            documents=documents,
            requested_doc_ids=None,
        )

        self.assertEqual(selected, ["doc-1"])

    def test_gemini_key_prefers_project_env_file_to_stale_process_env(self):
        with (
            patch.object(gemini_client, "BASE_DIR", Path("C:/project")),
            patch.object(
                gemini_client,
                "dotenv_values",
                return_value={"GEMINI_API_KEY": "file-key"},
            ),
            patch.dict("os.environ", {"GEMINI_API_KEY": "process-key"}),
        ):
            self.assertEqual(summarizer._gemini_api_key(), "file-key")

    def test_gemini_key_falls_back_to_process_env_without_env_file_key(self):
        with (
            patch.object(gemini_client, "BASE_DIR", Path("C:/project")),
            patch.object(gemini_client, "dotenv_values", return_value={}),
            patch.dict("os.environ", {"GEMINI_API_KEY": "process-key"}),
        ):
            self.assertEqual(summarizer._gemini_api_key(), "process-key")

    def test_new_gemini_client_disables_unmetered_sdk_retries(self):
        with (
            patch.object(gemini_client, "gemini_api_key", return_value="test-key"),
            patch.object(gemini_client.genai, "Client") as create_client,
        ):
            gemini_client.new_gemini_client()

        options = create_client.call_args.kwargs["http_options"]
        self.assertEqual(options.retry_options.attempts, 1)

    def test_summary_document_selection_uses_recent_conversation(self):
        documents = [
            {"doc_id": "doc-1", "title": "wheat_study"},
            {"doc_id": "doc-2", "title": "barley_study"},
        ]

        selected = summarizer._select_document_ids(
            question="Summarize that document",
            history=[{"role": "user", "content": "Tell me about wheat_study"}],
            documents=documents,
            requested_doc_ids=None,
        )

        self.assertEqual(selected, ["doc-1"])

    def test_ambiguous_summary_document_selection_asks_user(self):
        documents = [
            {"doc_id": "doc-1", "title": "wheat_study"},
            {"doc_id": "doc-2", "title": "barley_study"},
        ]

        clarification = summarizer._select_document_ids(
            question="Summarize the document",
            history=[],
            documents=documents,
            requested_doc_ids=None,
        )

        self.assertIsInstance(clarification, str)
        self.assertIn("wheat_study", clarification)
        self.assertIn("barley_study", clarification)

    def test_orchestrator_routes_direct_reply_and_clarify(self):
        direct_state = {
            "orchestrator_plan": {
                "action": "respond_directly",
                "direct_reply": "Hello!",
            }
        }
        self.assertEqual(graph._after_orchestrator(direct_state), "direct_reply")

        clarify_state = {
            "orchestrator_plan": {
                "action": "clarify",
                "clarifying_question": "Which document?",
            }
        }
        self.assertEqual(graph._after_orchestrator(clarify_state), "clarify")

    def test_orchestrator_routes_single_and_multipart_retrieval(self):
        single_state = {
            "question": "What is the policy?",
            "sub_questions": ["What is the policy?"],
            "orchestrator_plan": {"action": "retrieve"},
        }
        self.assertEqual(graph._after_orchestrator(single_state), "executor")

        multi_state = {
            "question": "Compare A and B",
            "sub_questions": ["What is A?", "What is B?"],
            "orchestrator_plan": {"action": "retrieve", "intent": "comparison"},
        }
        self.assertEqual(graph._after_orchestrator(multi_state), "executor")

    def test_graph_executes_non_summary_queries_through_executor_node(self):
        with (
            patch.object(
                graph,
                "orchestrator_node",
                return_value={
                    "orchestrator_plan": {
                        "action": "retrieve",
                        "intent": "factual",
                    },
                    "action": "retrieve",
                    "route": "factual",
                    "sub_questions": ["What is the policy?"],
                    "reformulated_question": "What is the policy?",
                },
            ),
            patch.object(
                graph,
                "executor_node",
                return_value={"answer": "The policy answer."},
            ) as run_executor,
        ):
            result = graph._build_graph().invoke({
                "question": "What is the policy?",
                "history": [],
                "sub_questions": [],
            })

        run_executor.assert_called_once()
        self.assertEqual(result["answer"], "The policy answer.")

    def test_orchestrator_does_not_decompose_numeric_queries(self):
        numeric_state = {
            "question": "Calculate the average for these values",
            "sub_questions": ["Extract the values", "Calculate the average"],
            "orchestrator_plan": {
                "action": "retrieve",
                "intent": "numeric",
            },
        }

        self.assertEqual(graph._after_orchestrator(numeric_state), "executor")

    def test_orchestrator_does_not_decompose_factual_queries(self):
        factual_state = {
            "question": "What is the policy?",
            "sub_questions": ["What is the policy?", "Find the policy details"],
            "orchestrator_plan": {
                "action": "retrieve",
                "intent": "factual",
            },
        }

        self.assertEqual(graph._after_orchestrator(factual_state), "executor")

    def test_numeric_writer_uses_orchestrator_intent_for_direct_calculation(self):
        state = {
            "question": "What is total baseline performance?",
            "reformulated_question": "What is total baseline performance?",
            "orchestrator_plan": {
                "intent": "numeric",
                "output_format": "table",
            },
            "route": "numeric",
            "context_blocks": [{
                "title": "llm_survey",
                "chunk_index": 1,
                "text": (
                    "| Task | Baseline Performance | Designed Prompt |\n"
                    "| --- | ---: | ---: |\n"
                    "| WikiFact | 23.61 | 28.47 |\n"
                    "| Colored Objects | 53.20 | 66.75 |\n"
                    "| WikiFact (General Description) | 29.25 | 31.21 |"
                ),
            }],
            "evidence": [],
            "verdict": {},
        }

        result = graph._writer_node(state)

        self.assertIn("106.06", result["answer"])

    def test_writer_node_combines_curation_and_answer_generation(self):
        evidence = self._chunk(0).model_dump()
        state = {
            "question": "What did the study find?",
            "evidence": [evidence],
            "orchestrator_plan": {"intent": "factual"},
            "route": "factual",
        }
        with (
            patch.object(
                graph.curator,
                "rerank_chunks",
                return_value=([evidence], "high"),
            ) as rerank,
            patch.object(
                graph.curator,
                "expand_chunks",
                return_value=[evidence],
            ) as expand,
            patch.object(
                graph.writer,
                "answer_from_retrieval",
                return_value="The study reports findings [1].",
            ) as answer,
        ):
            result = graph._writer_node(state)

        rerank.assert_called_once()
        expand.assert_called_once()
        answer.assert_called_once()
        self.assertEqual(result["decision_log"][0]["stage"], "curator_writer")
        self.assertEqual(result["context_blocks"], [evidence])
        self.assertNotIn("curator", graph.multi_agent_graph.nodes)

    def test_clear_question_uses_heuristic_plan_without_model_call(self):
        with patch.object(orchestrator, "safe_chat_create") as model:
            plan = orchestrator.plan_orchestration(
                "What is the wheat grain yield in the uploaded study?"
            )

        model.assert_not_called()
        self.assertEqual(plan.action, "retrieve")
        self.assertEqual(plan.sub_questions[0].query, "What is the wheat grain yield in the uploaded study?")

    def test_only_complex_queries_use_model_planning(self):
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
                                {
                                    "content": (
                                        '{"action":"retrieve","intent":"comparison",'
                                        '"rewritten_query":"Compare wheat and barley yield.",'
                                        '"sub_questions":[{"id":"q1","query":"Compare wheat and barley yield."}],'
                                        '"output_format":"table","needs_verification":true,'
                                        '"reason":"Comparison requires joint planning"}'
                                    )
                                },
                            )()
                        },
                    )()
                ]
            },
        )()
        with patch.object(orchestrator, "safe_chat_create") as model:
            model.return_value = response
            comparison = orchestrator.plan_orchestration(
                "Compare wheat and barley yield."
            )
            numeric = orchestrator.plan_orchestration("What is the average yield?")
            summary = orchestrator.plan_orchestration("Give an overview of the study.")

        model.assert_called_once()
        self.assertEqual(comparison.intent, "comparison")
        self.assertEqual(numeric.intent, "numeric")
        self.assertEqual(summary.action, "summarize")

    def test_anthropic_verifier_adapter_preserves_graph_verdict_shape(self):
        passages = [{"title": "Study", "chunk_index": 1, "text": "Evidence"}]

        with patch(
            "agents.verifier.verify_answer",
            return_value={
                "claims": [{
                    "sentence": "The result is positive.",
                    "claim": "The result is positive.",
                    "status": "SUPPORTED",
                    "evidence_trail": [],
                }],
            },
        ):
            verdict = verify_draft(
                "What is the result?",
                "The result is positive.",
                passages,
            )

        self.assertTrue(verdict.ok)
        self.assertEqual(verdict.claims[0].passage_ids, [])


if __name__ == "__main__":
    unittest.main()
