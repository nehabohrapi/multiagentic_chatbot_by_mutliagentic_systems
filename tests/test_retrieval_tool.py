"""Unit tests for the dedicated tools/retrieval_tool.py module."""

import unittest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from agents.schemas import SubQuestion
from tools.document_store import DocumentChunkSearchResult
from tools.retrieval_tool import (
    RetrievalToolInput,
    RetrievalToolOutput,
    execute_retrieval,
    expand_evidence_context,
    retrieve_document_summary,
    retrieve_for_subquestion,
    retrieve_table_evidence,
)
from tools.structured_data_tool import (
    calculate_table_operation,
    format_calculation_answer,
)


class RetrievalToolTests(unittest.TestCase):
    def _chunk(
        self,
        chunk_id: str,
        text: str = "text",
        doc_id: str = "doc_1",
        index: int = 0,
        structured_data: list[dict] | None = None,
    ) -> DocumentChunkSearchResult:
        return DocumentChunkSearchResult(
            chunk_id=chunk_id,
            document_id=doc_id,
            title="Sample Doc",
            source_path="sample.pdf",
            file_type="pdf",
            chunk_index=index,
            page_number=1,
            text=text,
            distance=0.1,
            match_type="hybrid",
            structured_data=structured_data or [],
        )


    def test_empty_query_returns_empty_list(self):
        self.assertEqual(execute_retrieval(""), [])
        self.assertEqual(execute_retrieval("   "), [])

    def test_execute_retrieval_hybrid_mode(self):
        mock_search = MagicMock(return_value=[self._chunk("c1")])
        results = execute_retrieval(
            "net profit",
            mode="hybrid",
            doc_ids=["doc_1"],
            file_type="pdf",
            max_results=5,
            search_fn=mock_search,
        )
        self.assertEqual(len(results), 1)
        mock_search.assert_called_once_with(
            "net profit",
            max_results=5,
            file_type="pdf",
            doc_ids=["doc_1"],
        )


    def test_execute_retrieval_table_mode_sufficient(self):
        mock_search = MagicMock(return_value=[
            self._chunk("t1", text="| col1 | col2 |"),
            self._chunk("t2", text="| colA | colB |"),
        ])
        results = execute_retrieval(
            "revenue numbers",
            mode="table",
            doc_ids=["doc_financials"],
            max_results=2,
            search_fn=mock_search,
        )
        self.assertEqual(len(results), 2)
        mock_search.assert_called_once_with(
            "revenue numbers",
            max_results=2,
            file_type=None,
            doc_ids=["doc_financials"],
            table_only=True,
        )

    def test_execute_retrieval_table_mode_fallback_combination(self):
        # table_only=True returns 1 chunk, fallback table_only=False returns additional chunk
        def side_effect(query, max_results=8, file_type=None, doc_ids=None, table_only=False):
            if table_only:
                return [self._chunk("t1", text="| Revenue | 100M |")]
            return [
                self._chunk("t1", text="| Revenue | 100M |"),
                self._chunk("h1", text="Company revenue explanation"),
            ]

        mock_search = MagicMock(side_effect=side_effect)
        results = execute_retrieval(
            "revenue details",
            mode="table",
            max_results=2,
            search_fn=mock_search,
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].chunk_id, "t1")
        self.assertEqual(results[1].chunk_id, "h1")
        self.assertEqual(mock_search.call_count, 2)

    def test_execute_retrieval_summary_cache_mode(self):
        mock_summary = MagicMock(return_value=[self._chunk("s1"), self._chunk("s2")])
        results = execute_retrieval(
            "summarize doc",
            mode="summary_cache",
            file_type="pdf",
            max_results=5,
            summary_fn=mock_summary,
        )
        self.assertEqual(len(results), 2)
        mock_summary.assert_called_once_with("summarize doc", file_type="pdf")

    def test_retrieve_for_subquestion_dict_and_schema(self):
        mock_search = MagicMock(return_value=[self._chunk("sq1")])

        # Test with SubQuestion model
        sq = SubQuestion(id="q1", query="Q3 expenses", doc_ids=["doc_q3"], retrieval="table")
        res1 = retrieve_for_subquestion(sq, max_results=4, search_fn=mock_search)
        self.assertEqual(len(res1), 1)

        # Test with dict
        res2 = retrieve_for_subquestion(
            {"query": "Q4 expenses", "doc_ids": ["doc_q4"], "retrieval": "hybrid"},
            max_results=4,
            search_fn=mock_search,
        )
        self.assertEqual(len(res2), 1)

    def test_convenience_functions(self):
        mock_search = MagicMock(return_value=[self._chunk("c1")])
        mock_summary = MagicMock(return_value=[self._chunk("s1")])

        with patch("tools.retrieval_tool.search_document_chunks", mock_search):
            table_res = retrieve_table_evidence("balance sheet", doc_ids=["d1"])
            self.assertEqual(len(table_res), 1)

        with patch("tools.retrieval_tool.get_document_summary_chunks", mock_summary):
            summary_res = retrieve_document_summary("executive briefing")
            self.assertEqual(len(summary_res), 1)

    def test_expand_evidence_context(self):
        mock_neighbors = [
            self._chunk("c0", doc_id="doc_1", index=0),
            self._chunk("c1", doc_id="doc_1", index=1),
            self._chunk("c2", doc_id="doc_1", index=2),
        ]
        with patch("tools.retrieval_tool.get_neighbor_chunks", return_value=mock_neighbors) as get_n:
            original = [self._chunk("c1", doc_id="doc_1", index=1)]
            expanded = expand_evidence_context(original, window=1)
            self.assertEqual(len(expanded), 3)
            get_n.assert_called_once_with("doc_1", {1}, window=1)

    def test_pydantic_tool_schemas(self):
        inp = RetrievalToolInput(
            query="test query",
            mode="table",
            doc_ids=["d1", "d2"],
            file_type="pdf",
            max_results=5,
        )
        self.assertEqual(inp.mode, "table")
        self.assertEqual(len(inp.doc_ids), 2)

        out = RetrievalToolOutput(
            query="test query",
            mode="table",
            doc_ids=["d1"],
            total_found=1,
            chunks=[{"chunk_id": "c1"}],
        )
        self.assertEqual(out.total_found, 1)

    def test_calculates_total_for_named_table_column(self):
        chunks = [{
            "structured_data": [{
                "kind": "table",
                "headers": ["Task", "Baseline Performance", "Designed Prompt"],
                "rows": [
                    ["WikiFact", "23.61", "28.47"],
                    ["Colored Objects", "53.20", "66.75"],
                    ["WikiFact (General Description)", "29.25", "31.21"],
                ],
            }],
        }]

        result = calculate_table_operation(
            "What is total baseline performance?",
            chunks,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.value, Decimal("106.06"))
        self.assertIn("106.06", format_calculation_answer(result))

    def test_calculates_total_from_markdown_table_in_retrieved_text(self):
        chunks = [{
            "text": (
                "| Task | Baseline Performance | Designed Prompt |\n"
                "| --- | ---: | ---: |\n"
                "| WikiFact | 23.61 | 28.47 |\n"
                "| Colored Objects | 53.20 | 66.75 |\n"
                "| WikiFact (General Description) | 29.25 | 31.21 |"
            ),
        }]

        result = calculate_table_operation(
            "What is total baseline performance in LLM models?",
            chunks,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.value, Decimal("106.06"))

    def test_calculates_first_numeric_column_without_domain_header(self):
        chunks = [{
            "structured_data": [{
                "kind": "table",
                "headers": ["Category", "Quarter A", "Quarter B"],
                "rows": [
                    ["North", "10", "12"],
                    ["South", "20", "18"],
                ],
            }],
        }]

        result = calculate_table_operation(
            "What is the total of the first numeric column?",
            chunks,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.value, Decimal("30"))


if __name__ == "__main__":
    unittest.main()
