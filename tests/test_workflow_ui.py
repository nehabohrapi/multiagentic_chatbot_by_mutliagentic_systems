"""Tests for workflow_ui component rendering and edge case handling."""

import unittest
from unittest.mock import MagicMock, patch

from workflow_ui import _format_seconds, _format_tokens, render_workflow_inspector


class WorkflowUITest(unittest.TestCase):
    def test_format_helpers(self):
        self.assertEqual(_format_seconds(None), "0.000s")
        self.assertEqual(_format_seconds(0.35), "350ms")
        self.assertEqual(_format_seconds(2.3456), "2.35s")
        self.assertEqual(_format_tokens(None), "0")
        self.assertEqual(_format_tokens(1500), "1,500")

    @patch("streamlit.info")
    @patch("streamlit.caption")
    @patch("streamlit.markdown")
    def test_render_empty_workflow(self, mock_markdown, mock_caption, mock_info):
        render_workflow_inspector(None, None)
        mock_info.assert_called_once()

    @patch("streamlit.expander")
    @patch("streamlit.selectbox")
    @patch("streamlit.columns")
    @patch("streamlit.metric")
    @patch("streamlit.caption")
    @patch("streamlit.markdown")
    def test_render_full_workflow(
        self,
        mock_markdown,
        mock_caption,
        mock_metric,
        mock_columns,
        mock_selectbox,
        mock_expander,
    ):
        mock_columns.side_effect = lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
        mock_exp = MagicMock()
        mock_expander.return_value.__enter__.return_value = mock_exp


        workflow = {
            "question": "What is SARTH?",
            "orchestrator_plan": {
                "action": "retrieve",
                "intent": "factual",
                "ambiguity": "none",
                "rewritten_query": "SARTH architecture and definition",
                "sub_questions": [
                    {
                        "id": "q1",
                        "query": "SARTH definition",
                        "retrieval": "hybrid",
                        "question_type": "definition",
                    }
                ],
                "reason": "Clear definition request",
                "output_format": "short_answer",
            },
            "question_type_info": {
                "primary_type": "factual",
                "is_comparison": False,
                "is_followup": False,
            },
            "chunks": [
                {
                    "chunk_id": "c1",
                    "title": "SARTH_Paper.pdf",
                    "page_number": 1,
                    "match_type": "hybrid",
                    "hybrid_score": 0.95,
                    "text": "SARTH is a multi-agent framework...",
                }
            ],
            "context_blocks": [
                {
                    "chunk_id": "c1",
                    "title": "SARTH_Paper.pdf",
                    "page_number": 1,
                    "text": "SARTH is a multi-agent framework...",
                }
            ],
            "raw_answer": "SARTH is a framework [1].",
            "answer": "SARTH is a framework [1].",
            "citations": [1],
            "verdict": {
                "ok": True,
                "claims": [
                    {"claim": "SARTH is a framework", "supported": "yes", "passage_ids": [1]}
                ],
                "missing": [],
            },
            "token_usage": {
                "prompt_tokens": 120,
                "completion_tokens": 40,
                "total_tokens": 160,
                "call_details": [
                    {"model": "llama-3.3-70b-versatile", "agent": "executor", "total_tokens": 160}
                ],
            },
            "timings": [
                {"label": "Query total", "category": "QUERY", "level": 0, "elapsed_seconds": 2.0},
                {"label": "Tool: Hybrid Retrieval", "category": "TOOLS", "level": 2, "elapsed_seconds": 0.45},
                {"label": "Verifier Subagent", "category": "VERIFIER", "level": 2, "elapsed_seconds": 1.8},
            ],
            "llm_calls": 2,
            "output_validation": {
                "valid": True,
                "citation_validation": {"complete": True, "citations_checked": 1},
                "issues": [],
                "warnings": [],
            },
            "summary_coverage": {
                "sections_checked": 2,
                "sections_covered": 2,
                "facts_checked": 4,
                "facts_covered": 3,
                "missing_sections": [],
                "missing_facts": [{"fact": "An omitted source fact."}],
            },
        }

        history = [
            {"role": "user", "content": "What is SARTH?"},
            {"role": "assistant", "content": "SARTH is a framework [1].", "workflow": workflow},
        ]

        workflow.pop("chunks")
        render_workflow_inspector(workflow, conversation_history=history)
        self.assertTrue(mock_markdown.called)
        self.assertTrue(mock_metric.called)
        self.assertIn(
            unittest.mock.call("Total Latency", "2.00s"),
            mock_metric.call_args_list,
        )
        self.assertTrue(
            any(
                "Retrieval Data (1 chunks)" in str(call.args[0])
                for call in mock_expander.call_args_list
            )
        )
        self.assertTrue(
            any("Output Rails" in str(call.args[0]) for call in mock_expander.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
