"""Unit tests for the single Execution Agent (Agent 2) and its subagents."""

import unittest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from agents.executor import (
    VERIFICATION_UNAVAILABLE_ANSWER,
    _asks_about_visuals,
    _deduplicate_chunks,
    curator_writer_subagent,
    execute_query,
    executor_decision_and_finalize,
    extract_comparison_entities,
    has_enough_evidence,
    resolve_followup_question,
    run_tool_execution,
    verifier_subagent,
    verify_question_type,
)
from gurdrails.nemo_output_rails import (
    apply_output_rails,
    nemo_output_rails_node,
    sanitize_output_text,
)
from agents.retriever_worker import NO_EVIDENCE_ANSWER
from tools.document_store import DocumentChunkSearchResult
from tools.structured_data_tool import CalculationResult


class ExecutorAgentTests(unittest.TestCase):
    def setUp(self):
        self.chunk1 = {
            "chunk_id": "c1",
            "document_id": "doc1",
            "chunk_index": 0,
            "title": "Doc 1",
            "text": "The project uses Python 3.11. Key metrics show 95% accuracy.",
            "distance": 0.1,
            "hybrid_score": 0.9,
            "rerank_score": 0.85,
        }
        self.chunk2 = {
            "chunk_id": "c2",
            "document_id": "doc1",
            "chunk_index": 1,
            "title": "Doc 1",
            "text": "System latency is 45 milliseconds under load.",
            "distance": 0.2,
            "hybrid_score": 0.8,
            "rerank_score": 0.75,
        }

    def test_deduplicate_chunks_preserves_highest_score(self):
        duplicate_low = dict(self.chunk1, hybrid_score=0.4)
        duplicate_high = dict(self.chunk1, hybrid_score=0.95)
        chunks = [duplicate_low, self.chunk2, duplicate_high]

        deduped = _deduplicate_chunks(chunks)
        self.assertEqual(len(deduped), 2)
        c1_item = next(c for c in deduped if c["chunk_id"] == "c1")
        self.assertEqual(c1_item["hybrid_score"], 0.95)

    def test_has_enough_evidence(self):
        self.assertTrue(has_enough_evidence([{"rerank_score": 0.35}]))
        self.assertFalse(has_enough_evidence([{"rerank_score": 0.10}]))
        self.assertFalse(has_enough_evidence([]))

    def test_asks_about_visuals_detection(self):
        self.assertTrue(_asks_about_visuals("Show me the chart on page 3"))
        self.assertTrue(_asks_about_visuals("What does figure 2 depict?"))
        self.assertTrue(_asks_about_visuals("Describe the diagram"))
        self.assertFalse(_asks_about_visuals("What is the main policy definition?"))

    @patch("agents.executor.retriever_worker.retrieve")
    def test_run_tool_execution_hybrid_retrieval(self, mock_retrieve):
        mock_retrieve.return_value = {"chunks": [self.chunk1]}
        plan = {
            "intent": "factual",
            "sub_questions": [{"query": "What is Python version?"}],
        }

        result = run_tool_execution("What is Python version?", plan, [])
        self.assertEqual(len(result["chunks"]), 1)
        self.assertEqual(result["chunks"][0]["chunk_id"], "c1")
        self.assertEqual(
            result["chunks"][0]["_retrieval_queries"],
            ["What is Python version?"],
        )
        self.assertEqual(
            result["retrieval_status"]["queries"][0]["chunk_ids"],
            ["c1"],
        )
        self.assertIn("hybrid_retrieval", [t["tool"] for t in result["tool_log"]])

    @patch("agents.executor.retriever_worker.retrieve")
    def test_table_retrieval_expands_query_with_table_and_entity_terms(
        self, mock_retrieve
    ):
        mock_retrieve.return_value = {"chunks": [self.chunk1]}
        question = "What parameter sizes are listed in Table 1?"
        plan = {
            "intent": "numeric",
            "sub_questions": [{
                "query": question,
                "retrieval": "table",
                "entities": ["GPT-3", "LLaMA"],
                "attributes": ["parameter sizes"],
            }],
        }

        run_tool_execution(question, plan, [])

        self.assertEqual(
            mock_retrieve.call_args.kwargs["planned_queries"],
            [
                question,
                "Table 1 statistics What parameter sizes are listed in Table 1? "
                "GPT-3 LLaMA parameter sizes",
            ],
        )

    @patch("agents.executor.retriever_worker.get_neighbor_chunks")
    @patch("agents.executor.retriever_worker.retrieve")
    def test_table_retrieval_adds_adjacent_chunks_for_caption(
        self, mock_retrieve, mock_neighbors
    ):
        caption = {
            **self.chunk1,
            "chunk_id": "table-caption",
            "chunk_index": 54,
            "text": "TABLE 1: Statistics of large language models",
        }
        row = {
            **self.chunk1,
            "chunk_id": "table-row",
            "chunk_index": 56,
            "text": "LLaMA [57] Feb-2023 65 ... GPT-3 [55] May-2020 175",
        }
        mock_retrieve.side_effect = [
            {"chunks": [caption, self.chunk2]},
            {"chunks": []},
        ]
        mock_neighbors.return_value = [row]
        question = "What parameter sizes are listed in Table 1?"
        plan = {
            "intent": "numeric",
            "sub_questions": [{
                "query": question,
                "retrieval": "table",
                "entities": ["GPT-3", "LLaMA"],
                "attributes": ["parameter sizes"],
            }],
        }

        result = run_tool_execution(question, plan, [])

        mock_neighbors.assert_called_once_with("doc1", {54}, window=5)
        self.assertEqual(
            {chunk["chunk_id"] for chunk in result["chunks"]},
            {"table-caption", "table-row"},
        )

    @patch("agents.executor.retriever_worker.retrieve")
    def test_run_tool_execution_reports_missing_subquestion_evidence(self, mock_retrieve):
        mock_retrieve.side_effect = [
            {"chunks": [self.chunk1], "sufficient": True},
            {"chunks": [], "sufficient": False, "missing_info": "No regional data found."},
        ]
        plan = {
            "intent": "multi_part",
            "sub_questions": [
                {"query": "What version?"},
                {"query": "Which region?"},
            ],
        }

        result = run_tool_execution("What version and region?", plan, [])

        self.assertFalse(result["retrieval_status"]["sufficient"])
        self.assertEqual(
            result["retrieval_status"]["limitations"],
            ["No regional data found."],
        )

    @patch("agents.executor.retriever_worker.retrieve")
    @patch("agents.executor.calculate_table_operation")
    def test_run_tool_execution_structured_data(self, mock_calc, mock_retrieve):
        mock_retrieve.return_value = {"chunks": [self.chunk1]}
        mock_calc.return_value = CalculationResult(
            operation="sum",
            column="total",
            values=(Decimal("10"), Decimal("20")),
            value=Decimal("30"),
            source_count=1,
        )
        plan = {"intent": "numeric", "sub_questions": []}

        result = run_tool_execution("What is the total revenue?", plan, [])
        self.assertIsNotNone(result["calculation"])
        self.assertEqual(result["calculation"].value, Decimal("30"))
        self.assertIn("structured_data", [t["tool"] for t in result["tool_log"]])

    @patch("agents.executor.retriever_worker.retrieve")
    @patch("agents.executor.collect_visual_assets")
    @patch("agents.executor.analyze_visual_assets")
    @patch("agents.executor.chart_tables_by_chunk")
    def test_run_tool_execution_visual_processing(
        self, mock_chart_tables, mock_analyze, mock_collect, mock_retrieve
    ):
        mock_retrieve.return_value = {"chunks": [self.chunk1]}
        mock_collect.return_value = [{"visual_id": "fig1"}]
        mock_analyze.return_value = [{"chart_data": {"x": 1}}]
        mock_chart_tables.return_value = {}

        plan = {"intent": "factual", "sub_questions": []}
        result = run_tool_execution("Explain the chart in section 2", plan, [])
        self.assertEqual(len(result["visual_evidence"]), 1)
        self.assertIn("visual_processing", [t["tool"] for t in result["tool_log"]])

    @patch("agents.executor.visual.VisualAgent.retrieve_visual_chunks_from_db")
    @patch("agents.executor.retriever_worker.retrieve")
    def test_image_demand_redirects_to_executor_and_visual_agent_retrieves_from_db(
        self, mock_retrieve, mock_retrieve_visual_db
    ):
        mock_retrieve.return_value = {"chunks": [self.chunk1]}
        visual_chunk = {
            "chunk_id": "c_vis",
            "document_id": "doc_crop",
            "page_number": 5,
            "title": "agri_crop",
            "text": "Figure 3 shows seasonal crop yields.",
            "structured_data": [{
                "kind": "figure",
                "source": "agri_crop.pdf",
                "page_number": 5,
                "image_path": "agri_crop_p5.png",
                "description": "Seasonal crop yield chart",
            }],
        }
        mock_retrieve_visual_db.return_value = [visual_chunk]

        question = "Show me the chart of crop yields and its image metadata"
        plan = {
            "intent": "factual",
            "sub_questions": [{"query": question, "question_type": "visual"}],
        }

        with patch("agents.executor.visual.VisualAgent.collect_assets") as mock_collect:
            mock_collect.return_value = [{
                "visual_id": "c_vis-figure-1",
                "chunk_id": "c_vis",
                "document_id": "doc_crop",
                "filename": "agri_crop.pdf",
                "page_number": 5,
                "caption": "Seasonal crop yield chart",
                "path": "agri_crop_p5.png",
            }]
            result = run_tool_execution(question, plan, [])

        mock_retrieve_visual_db.assert_called_once()
        self.assertTrue(result["question_type_info"]["is_visual"])
        self.assertEqual(result["question_type_info"]["primary_type"], "image")
        self.assertEqual(len(result["visual_assets"]), 1)
        self.assertEqual(result["visual_assets"][0]["filename"], "agri_crop.pdf")
        self.assertEqual(result["visual_assets"][0]["page_number"], 5)
        self.assertIn("visual_processing", [t["tool"] for t in result["tool_log"]])

    @patch("agents.executor.curator.rerank_chunks")
    @patch("agents.executor.curator.expand_chunks")
    @patch("agents.executor.writer.answer_from_retrieval")
    def test_curator_writer_subagent_drafts_answer(self, mock_writer, mock_expand, mock_rerank):
        mock_rerank.return_value = ([self.chunk1], "high")
        mock_expand.return_value = [self.chunk1]
        mock_writer.return_value = {
            "answer": "Python 3.11 is used [1].",
            "raw_answer": "Python 3.11 is used [1].",
            "citations": [1],
        }

        plan = {"intent": "factual", "output_format": "short_answer"}
        result = curator_writer_subagent("What is the Python version?", [self.chunk1], plan)

        self.assertTrue(result["answerable"])
        self.assertEqual(result["citations"], [1])
        self.assertIn("Python 3.11", result["draft"])
        self.assertTrue(result["evidence_sufficient"])

    @patch("agents.executor.curator.rerank_chunks")
    @patch("agents.executor.curator.expand_chunks")
    @patch("agents.executor.writer.answer_from_retrieval")
    def test_numeric_curator_keeps_more_ranked_chunks_for_table_rows(
        self, mock_writer, mock_expand, mock_rerank
    ):
        chunks = [dict(self.chunk1, chunk_id=f"c{index}") for index in range(12)]
        mock_rerank.return_value = (chunks, "high")
        mock_expand.return_value = chunks
        mock_writer.return_value = {
            "answer": "GPT-3: 175B; LLaMA: 65B [1].",
            "raw_answer": "GPT-3: 175B; LLaMA: 65B [1].",
            "citations": [1],
        }

        result = curator_writer_subagent(
            "What parameter sizes are listed in Table 1?",
            chunks,
            {"intent": "numeric", "output_format": "table"},
            question_type_info={"primary_type": "table"},
        )

        self.assertTrue(result["answerable"])
        self.assertEqual(mock_rerank.call_args.kwargs["top_k"], 12)
        self.assertEqual(mock_expand.call_args.kwargs["max_chunks"], 12)

    def test_curator_writer_subagent_low_evidence_fallback(self):
        low_chunk = dict(self.chunk1, rerank_score=0.1)
        with (
            patch("agents.executor.curator.rerank_chunks", return_value=([low_chunk], "low")),
            patch("agents.executor.curator.expand_chunks", return_value=[low_chunk]),
        ):
            plan = {"intent": "factual"}
            result = curator_writer_subagent("Random question", [low_chunk], plan)
            self.assertFalse(result["answerable"])
            self.assertEqual(result["draft"], NO_EVIDENCE_ANSWER)
            self.assertFalse(result["evidence_sufficient"])
            self.assertTrue(result["limitations"])

    @patch("agents.executor.curator.rerank_chunks")
    @patch("agents.executor.curator.expand_chunks")
    @patch("agents.executor.writer.answer_from_retrieval")
    def test_curator_returns_partial_answer_and_source_metadata(
        self, mock_writer, mock_expand, mock_rerank
    ):
        chunk = {
            **self.chunk1,
            "source_path": r"documents\research.pdf",
            "page_number": 4,
        }
        mock_rerank.return_value = ([chunk], "high")
        mock_expand.return_value = [chunk]
        mock_writer.return_value = {
            "answer": "Python 3.11 is used [research.pdf, p. 4].",
            "raw_answer": "Python 3.11 is used [1].",
            "citations": [1],
        }

        result = curator_writer_subagent(
            "What is the Python version and deployment region?",
            [chunk],
            {"intent": "multi_part"},
            retrieval_status={
                "sufficient": False,
                "limitations": ["No matching passages were found for: deployment region"],
            },
        )

        self.assertTrue(result["answerable"])
        self.assertFalse(result["evidence_sufficient"])
        self.assertIn("No matching passages", result["limitations"][0])
        self.assertEqual(
            result["citation_metadata"],
            [{
                "citation": 1,
                "document": "research.pdf",
                "page": 4,
                "chunk_id": "c1",
                "supporting_claim": "Python 3.11 is used",
            }],
        )
        mock_writer.assert_called_once()

    @patch("agents.executor.curator.rerank_chunks")
    @patch("agents.executor.curator.expand_chunks")
    @patch("agents.executor.writer.answer_from_retrieval")
    def test_compound_question_keeps_calculation_with_other_answer_parts(
        self, mock_writer, mock_expand, mock_rerank
    ):
        mock_rerank.return_value = ([self.chunk1], "high")
        mock_expand.return_value = [self.chunk1]
        mock_writer.return_value = {
            "answer": "Python 3.11 is used [1].",
            "raw_answer": "Python 3.11 is used [1].",
            "citations": [1],
        }
        calculation = CalculationResult(
            operation="average",
            column="latency",
            values=(Decimal("40"), Decimal("50")),
            value=Decimal("45"),
            source_count=2,
        )

        result = curator_writer_subagent(
            "What is the average latency and which Python version is used?",
            [self.chunk1],
            {"intent": "multi_part"},
            calculation=calculation,
        )

        self.assertTrue(result["answerable"])
        mock_writer.assert_called_once()
        retrieval = mock_writer.call_args.args[1]
        self.assertEqual(
            retrieval["calculated_evidence"],
            {
                "operation": "average",
                "column": "latency",
                "input_values": ["40", "50"],
                "result": "45",
                "source_count": 2,
            },
        )

    @patch("agents.executor.curator.rerank_chunks")
    @patch("agents.executor.curator.expand_chunks")
    @patch("agents.executor.writer.answer_from_retrieval", side_effect=RuntimeError("provider unavailable"))
    def test_curator_generation_failure_surfaces_instead_of_returning_raw_chunks(
        self, _mock_writer, mock_expand, mock_rerank
    ):
        mock_rerank.return_value = ([self.chunk1], "high")
        mock_expand.return_value = [self.chunk1]

        with self.assertLogs("agents.executor", level="ERROR"):
            with self.assertRaisesRegex(
                RuntimeError,
                "Curator answer generation failed: provider unavailable",
            ):
                curator_writer_subagent(
                    "What runtime does the project use?",
                    [self.chunk1],
                    {"intent": "factual"},
                )

    @patch("agents.executor.verify_draft")
    def test_verifier_subagent_checks_claims(self, mock_verify):
        mock_verify.return_value = {
            "ok": True,
            "claims": [{"claim": "Python 3.11 is used", "supported": "yes"}],
        }

        verdict = verifier_subagent("Python 3.11 is used [1].", [self.chunk1])
        self.assertTrue(verdict["ok"])
        self.assertEqual(len(verdict["claims"]), 1)

    @patch("agents.executor.verifier_subagent")
    @patch("agents.executor.curator_writer_subagent")
    @patch("agents.executor.run_tool_execution")
    @patch("agents.executor.verify_question_type")
    def test_executor_verifies_raw_citations_not_display_formatted_answer(
        self,
        mock_question_type,
        mock_tools,
        mock_curator,
        mock_verifier,
    ):
        mock_question_type.return_value = {
            "primary_type": "factual",
            "is_visual": False,
            "is_comparison": False,
            "is_followup": False,
            "is_table": False,
            "comparison_entities": [],
            "resolved_question": "Who is Percy?",
            "recommended_output_format": "bullets",
        }
        mock_tools.return_value = {
            "chunks": [self.chunk1],
            "calculation": None,
            "visual_evidence": [],
            "retrieval_status": {"sufficient": True},
        }
        mock_curator.return_value = {
            "draft": "Percy is a student [story.pdf, p. 6].",
            "raw_answer": "Percy is a student【1】.",
            "answer": "Percy is a student [story.pdf, p. 6].",
            "answerable": True,
            "context_blocks": [self.chunk1],
        }
        mock_verifier.return_value = {"ok": True, "claims": []}

        result = execute_query("Who is Percy?", {"needs_verification": True}, [])

        self.assertEqual(
            mock_verifier.call_args.args[1],
            "Percy is a student【1】.",
        )
        self.assertEqual(result["chunks"], [self.chunk1])
        self.assertEqual(result["retrieval_status"], {"sufficient": True})

    @patch("agents.executor.verifier_subagent")
    @patch("agents.executor.curator_writer_subagent")
    @patch("agents.executor.run_tool_execution")
    @patch("agents.executor.verify_question_type")
    def test_executor_still_verifies_when_plan_marks_question_simple(
        self,
        mock_question_type,
        mock_tools,
        mock_curator,
        mock_verifier,
    ):
        mock_question_type.return_value = {
            "primary_type": "factual",
            "is_visual": False,
            "is_comparison": False,
            "is_followup": False,
            "is_table": False,
            "comparison_entities": [],
            "resolved_question": "What do you mean by LLM?",
            "recommended_output_format": "short_answer",
        }
        mock_tools.return_value = {
            "chunks": [self.chunk1],
            "calculation": None,
            "visual_evidence": [],
            "retrieval_status": {"sufficient": True},
        }
        mock_curator.return_value = {
            "draft": "LLM means large language model.",
            "raw_answer": "LLM means large language model [1].",
            "answer": "LLM means large language model [1].",
            "answerable": True,
            "context_blocks": [self.chunk1],
        }

        result = execute_query(
            "What do you mean by LLM?",
            {"needs_verification": False},
            [],
        )

        mock_verifier.assert_called_once()
        self.assertIn("large language model", result["answer"])
        self.assertTrue(result["answerable"])

    def test_executor_decision_verified_passes_through(self):
        draft_result = {
            "draft": "Python 3.11 is used [1].",
            "raw_answer": "Python 3.11 is used [1].",
            "answer": "Python 3.11 is used [1].",
            "answerable": True,
            "citations": [1],
            "context_blocks": [self.chunk1],
        }
        verdict = {"ok": True, "claims": [{"claim": "Python 3.11 is used", "supported": "yes"}]}

        final = executor_decision_and_finalize(draft_result, verdict, "What is Python version?")
        self.assertEqual(final["draft"], "Python 3.11 is used [1].")

    def test_executor_decision_unverified_strips_claims(self):
        draft_result = {
            "draft": "Python 3.11 is used. Java is also used.",
            "raw_answer": "Python 3.11 is used. Java is also used.",
            "answer": "Python 3.11 is used. Java is also used.",
            "answerable": True,
            "citations": [],
            "context_blocks": [self.chunk1],
        }
        verdict = {
            "ok": False,
            "claims": [
                {"claim": "Python 3.11 is used.", "supported": "yes"},
                {"claim": "Java is also used.", "supported": "no"},
            ],
        }

        final = executor_decision_and_finalize(draft_result, verdict, "What languages are used?")
        # Java claim should be stripped, Python claim preserved
        self.assertNotIn("Java is also used", final["draft"])
        self.assertIn("Python 3.11 is used", final["draft"])

    def test_executor_decision_severe_failure_abstains_instead_of_dumping_chunks(self):
        draft_result = {
            "draft": "Completely hallucinated text that is wrong.",
            "raw_answer": "Completely hallucinated text that is wrong.",
            "answer": "Completely hallucinated text that is wrong.",
            "answerable": True,
            "citations": [],
            "context_blocks": [self.chunk1],
        }
        verdict = {
            "ok": False,
            "claims": [
                {"claim": "Completely hallucinated claim 1", "supported": "no"},
                {"claim": "Completely hallucinated claim 2", "supported": "no"},
            ],
        }

        final = executor_decision_and_finalize(draft_result, verdict, "What are metrics?")
        self.assertEqual(final["draft"], NO_EVIDENCE_ANSWER)
        self.assertNotIn("Python 3.11", final["answer"])
        self.assertFalse(final["answerable"])

    def test_verifier_unavailable_keeps_cited_draft_explicitly_unverified(self):
        draft_result = {
            "draft": "Python 3.11 is used [1].",
            "raw_answer": "Python 3.11 is used [1].",
            "answer": "Python 3.11 is used [1].",
            "answerable": True,
            "citations": [1],
            "context_blocks": [self.chunk1],
        }
        verdict = {
            "ok": False,
            "claims": [{"claim": "Python 3.11 is used.", "supported": "no"}],
            "missing": [
                "Python 3.11 is used. (UNVERIFIED: Verification response was unavailable or malformed; claim was not approved.)"
            ],
        }

        final = executor_decision_and_finalize(
            draft_result,
            verdict,
            "What is the Python version?",
        )

        self.assertTrue(final["answerable"])
        self.assertTrue(final["answer"].startswith("UNVERIFIED: Cohere verification is temporarily unavailable."))
        self.assertIn("Python 3.11 is used", final["answer"])
        self.assertEqual(final["citations"], [1])

    def test_verifier_unavailable_still_rejects_invalid_citations(self):
        draft_result = {
            "draft": "Python 3.11 is used [9].",
            "raw_answer": "Python 3.11 is used [9].",
            "answer": "Python 3.11 is used [9].",
            "answerable": True,
            "citations": [9],
            "context_blocks": [self.chunk1],
        }
        verdict = {
            "ok": False,
            "claims": [{"claim": "Python 3.11 is used.", "supported": "no"}],
            "missing": [
                "Python 3.11 is used. (UNVERIFIED: Verification response was unavailable or malformed; claim was not approved.)"
            ],
        }

        final = executor_decision_and_finalize(
            draft_result,
            verdict,
            "What is the Python version?",
        )

        self.assertFalse(final["answerable"])
        self.assertEqual(final["answer"], VERIFICATION_UNAVAILABLE_ANSWER)

    def test_verify_question_type_comparison(self):
        q = "Compare DBMS and File System"
        res = verify_question_type(q, {}, [])
        self.assertEqual(res["primary_type"], "comparison")
        self.assertTrue(res["is_comparison"])
        self.assertIn("DBMS", res["comparison_entities"])
        self.assertIn("File System", res["comparison_entities"])

    def test_verify_question_type_followup(self):
        history = [
            {"role": "user", "content": "What is database management system DBMS?"},
            {"role": "assistant", "content": "DBMS is a system software for databases."},
        ]
        q = "What are its main functions?"
        res = verify_question_type(q, {}, history)
        self.assertEqual(res["primary_type"], "followup")
        self.assertTrue(res["is_followup"])
        self.assertIn("database management system DBMS's", res["resolved_question"])

    def test_verify_question_type_table(self):
        q = "Calculate the average latency from the table"
        res = verify_question_type(q, {}, [])
        self.assertEqual(res["primary_type"], "table")
        self.assertTrue(res["is_table"])

    def test_verify_question_type_image(self):
        q = "Describe the system architecture diagram in Figure 2"
        res = verify_question_type(q, {}, [])
        self.assertEqual(res["primary_type"], "image")
        self.assertTrue(res["is_visual"])

    @patch("agents.executor.retriever_worker.retrieve")
    def test_run_tool_execution_comparison_retrieval(self, mock_retrieve):
        mock_retrieve.return_value = {"chunks": [self.chunk1]}
        plan = {}
        res = run_tool_execution("Compare DBMS and File System", plan, [])
        self.assertEqual(res["question_type_info"]["primary_type"], "comparison")
        # Should have retrieved for both entities and comparison query
        self.assertGreaterEqual(mock_retrieve.call_count, 2)

    @patch("agents.executor.curator.rerank_chunks")
    def test_zero_hallucination_floor_insufficient_evidence(self, mock_rerank):
        # When evidence confidence is below threshold, curator_writer returns NO_EVIDENCE_ANSWER
        low_chunk = dict(self.chunk1, hybrid_score=0.1, rerank_score=0.1)
        mock_rerank.return_value = ([low_chunk], "low")
        res = curator_writer_subagent("What is quantum computing?", [low_chunk], {})
        self.assertEqual(res["answer"], NO_EVIDENCE_ANSWER)
        self.assertFalse(res["answerable"])


class OutputGuardrailTests(unittest.TestCase):
    def test_sanitize_output_text_removes_prompt_delimiters(self):
        leaked = "<document_passage>Source text</document_passage>\nAssistant answer here."
        sanitized = sanitize_output_text(leaked)
        self.assertNotIn("<document_passage>", sanitized)
        self.assertNotIn("</document_passage>", sanitized)
        self.assertIn("Assistant answer here", sanitized)

    def test_output_rails_empty_answer_uses_fallback(self):
        result = apply_output_rails("")
        self.assertEqual(result.answer, NO_EVIDENCE_ANSWER)
        self.assertTrue(result.changed)
        self.assertFalse(result.answerable)
        self.assertIn("empty_answer_replaced", result.validation["issues"])

    def test_output_rails_reports_invalid_answer_type(self):
        result = apply_output_rails({"unexpected": "object"})

        self.assertEqual(result.answer, NO_EVIDENCE_ANSWER)
        self.assertFalse(result.answerable)
        self.assertIn("invalid_answer_type", result.validation["issues"])

    def test_nemo_output_rails_node_execution(self):
        state = {
            "answer": "Grounded answer [1].",
            "context_blocks": [{"title": "doc.pdf", "chunk_index": 0}],
            "citations": [1],
            "citation_metadata": [{"citation": 1, "document": "doc.pdf"}],
            "evidence_sufficient": True,
            "limitations": [],
            "decision_log": [],
            "token_usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
            },
        }
        output = nemo_output_rails_node(state)
        self.assertIn("Grounded answer", output["answer"])
        self.assertEqual(output["citations"], [1])
        self.assertEqual(output["citation_metadata"][0]["document"], "doc.pdf")
        self.assertEqual(output["citation_metadata"][0]["citation"], 1)
        self.assertTrue(output["evidence_sufficient"])
        self.assertEqual(output["decision_log"][-1]["stage"], "nemo_output_rails")
        self.assertEqual(output["decision_log"][-1]["llm_calls"], 0)
        self.assertEqual(output["decision_log"][-1]["retry_count"], 0)
        self.assertIn("elapsed_seconds", output["decision_log"][-1])
        self.assertEqual(
            output["decision_log"][-1]["token_usage"]["total_tokens"],
            20,
        )

    def test_citation_metadata_is_rebuilt_from_source_and_ids_are_checked(self):
        result = apply_output_rails(
            "Revenue increased by 10% [1].",
            context_blocks=[{
                "document_id": "doc-1",
                "chunk_id": "chunk-7",
                "page_number": 4,
                "source_path": "reports/revenue.pdf",
            }],
            citation_metadata=[{
                "citation": 1,
                "document_id": "wrong-doc",
                "chunk_id": "wrong-chunk",
                "page": 99,
            }],
        )

        citation = result.citation_metadata[0]
        self.assertEqual(citation["document_id"], "doc-1")
        self.assertEqual(citation["chunk_id"], "chunk-7")
        self.assertEqual(citation["page"], 4)
        self.assertEqual(citation["supporting_claim"], "Revenue increased by 10%.")
        self.assertTrue(result.validation["citation_validation"]["complete"])
        self.assertIn("citation_1_chunk_id_metadata_repaired", result.validation["warnings"])

    def test_unresolvable_citations_are_removed_and_reported(self):
        result = apply_output_rails(
            "Unsupported source [2].",
            context_blocks=[{"document_id": "doc-1", "chunk_id": "chunk-1"}],
        )

        self.assertEqual(result.citations, [])
        self.assertNotIn("[2]", result.answer)
        self.assertFalse(result.validation["valid"])
        self.assertIn("citation_2_has_no_matching_context", result.validation["issues"])

    def test_summary_coverage_checks_sections_and_important_facts(self):
        chunks = [{
            "document_id": "doc-1",
            "chunk_id": "chunk-1",
            "text": (
                "1. Findings\n"
                "The project reduced operating costs by 25% in 2024."
            ),
        }]
        result = apply_output_rails(
            "1. Findings: The project reduced operating costs by 25% in 2024.",
            context_blocks=chunks,
            citations=[],
            is_summary=True,
        )

        self.assertTrue(result.summary_coverage["complete"])
        self.assertEqual(result.summary_coverage["sections_checked"], 1)
        self.assertEqual(result.summary_coverage["facts_checked"], 1)

    def test_summary_coverage_reports_omitted_source_material(self):
        chunks = [{
            "document_id": "doc-1",
            "chunk_id": "chunk-1",
            "text": (
                "1. Findings\n"
                "The project reduced operating costs by 25% in 2024."
            ),
        }]
        result = apply_output_rails(
            "The report discusses the project.",
            context_blocks=chunks,
            citations=[],
            is_summary=True,
        )

        self.assertFalse(result.summary_coverage["complete"])
        self.assertEqual(result.summary_coverage["missing_sections"], ["1. Findings"])
        self.assertEqual(result.summary_coverage["missing_fact_count"], 1)


if __name__ == "__main__":
    unittest.main()
