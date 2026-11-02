from __future__ import annotations

import unittest
from unittest.mock import patch

from agents import summarizer
from tools.document_store import DocumentChunkSearchResult


class TestDocumentSelectionPlan(unittest.TestCase):
    def setUp(self):
        self.catalog = [
            {
                "doc_id": "doc-wheat",
                "title": "wheat_study.pdf",
                "aliases": ["wheat study", "wheat_study"],
                "source_path": "docs/wheat_study.pdf",
            },
            {
                "doc_id": "doc-barley",
                "title": "barley_study.pdf",
                "aliases": ["barley study", "barley_study"],
                "source_path": "docs/barley_study.pdf",
            },
            {
                "doc_id": "doc-corn",
                "title": "corn_genetics.pdf",
                "aliases": ["corn genetics", "corn_genetics"],
                "source_path": "docs/corn_genetics.pdf",
            },
        ]

    # 1. Validate doc_ids
    def test_validate_doc_ids_uses_valid_ids(self):
        res = summarizer._select_document_ids(
            question="Summarize the crop study",
            history=[],
            documents=self.catalog,
            requested_doc_ids=["doc-wheat"],
        )
        self.assertEqual(res, ["doc-wheat"])

    def test_validate_doc_ids_rejects_missing_ids(self):
        res = summarizer._select_document_ids(
            question="Summarize the crop study",
            history=[],
            documents=self.catalog,
            requested_doc_ids=["doc-unknown"],
        )
        self.assertIsInstance(res, str)
        self.assertIn("couldn't match that document", res)

    def test_validate_doc_ids_falls_through_if_conflicting_with_explicit_query(self):
        res = summarizer._select_document_ids(
            question="Summarize barley_study",
            history=[],
            documents=self.catalog,
            requested_doc_ids=["doc-wheat"],
        )
        # Should pick barley_study as explicitly requested in query
        self.assertEqual(res, ["doc-barley"])

    # 2. Single document
    def test_single_document_in_catalog(self):
        single_catalog = [self.catalog[0]]
        res = summarizer._select_document_ids(
            question="Summarize this",
            history=[],
            documents=single_catalog,
            requested_doc_ids=None,
        )
        self.assertEqual(res, ["doc-wheat"])

    # 3. Resolve references
    def test_resolve_reference_it_and_this_file(self):
        history = [
            {"role": "user", "content": "Can you check wheat_study.pdf?"},
            {"role": "assistant", "content": "I found wheat_study.pdf."},
        ]
        res1 = summarizer._select_document_ids(
            question="Summarize it",
            history=history,
            documents=self.catalog,
            requested_doc_ids=None,
        )
        self.assertEqual(res1, ["doc-wheat"])

        res2 = summarizer._select_document_ids(
            question="Summarize this file",
            history=history,
            documents=self.catalog,
            requested_doc_ids=None,
        )
        self.assertEqual(res2, ["doc-wheat"])

    def test_resolve_reference_previous_one(self):
        history = [
            {"role": "user", "content": "Tell me about barley_study"},
            {"role": "assistant", "content": "Barley study discusses yields."},
        ]
        res = summarizer._select_document_ids(
            question="Give me an overview of the previous one",
            history=history,
            documents=self.catalog,
            requested_doc_ids=None,
        )
        self.assertEqual(res, ["doc-barley"])

    def test_resolve_reference_ordinal_second_one(self):
        history = [
            {
                "role": "assistant",
                "content": "Which document would you like me to summarize? wheat_study.pdf, barley_study.pdf, corn_genetics.pdf.",
            }
        ]
        res = summarizer._select_document_ids(
            question="Summarize the second one",
            history=history,
            documents=self.catalog,
            requested_doc_ids=None,
        )
        self.assertEqual(res, ["doc-barley"])

    def test_no_mistaken_history_selection_for_unrelated_query(self):
        # When query has a specific title or topic, do NOT pick previous document from history
        history = [
            {"role": "user", "content": "Tell me about wheat_study"},
            {"role": "assistant", "content": "Wheat study details..."},
        ]
        res = summarizer._select_document_ids(
            question="Summarize corn_genetics",
            history=history,
            documents=self.catalog,
            requested_doc_ids=None,
        )
        self.assertEqual(res, ["doc-corn"])

    # 4. Retrieve and fuse
    def test_retrieve_and_fuse_skips_chunks_when_no_real_topic(self):
        with patch.object(summarizer, "search_document_chunks") as mock_search:
            summarizer._score_document_content("summarize the document please", self.catalog)
            mock_search.assert_not_called()

    def test_retrieve_and_fuse_calls_chunks_when_real_topic_present(self):
        with patch.object(summarizer, "search_document_chunks", return_value=[]) as mock_search:
            summarizer._score_document_content("summarize nitrogen fixation genetics", self.catalog)
            mock_search.assert_called_once()

    def test_retrieve_and_fuse_content_chunks_identifies_document(self):
        chunk_res = [
            DocumentChunkSearchResult(
                chunk_id="c1",
                document_id="doc-corn",
                title="corn_genetics.pdf",
                source_path="docs/corn_genetics.pdf",
                file_type="pdf",
                chunk_index=0,
                text="CRISPR mutations in maize and corn genetics.",
                distance=0.1,
            )
        ]
        with patch.object(summarizer, "search_document_chunks", return_value=chunk_res):
            res = summarizer._select_document_ids(
                question="Summarize CRISPR mutations in maize",
                history=[],
                documents=self.catalog,
                requested_doc_ids=None,
            )
            self.assertEqual(res, ["doc-corn"])

    # 5. Decide
    def test_decide_multi_document_intent(self):
        res = summarizer._select_document_ids(
            question="Compare wheat_study and barley_study",
            history=[],
            documents=self.catalog,
            requested_doc_ids=None,
        )
        self.assertIsInstance(res, list)
        self.assertIn("doc-wheat", res)
        self.assertIn("doc-barley", res)

    def test_decide_close_scores_asks_user_when_not_multi_doc(self):
        # Query matches "study" which appears equally in wheat_study and barley_study
        res = summarizer._select_document_ids(
            question="Summarize the crop study",
            history=[],
            documents=self.catalog,
            requested_doc_ids=None,
        )
        self.assertIsInstance(res, str)
        self.assertIn("Which document would you like me to summarize?", res)
        self.assertIn("wheat_study", res)
        self.assertIn("barley_study", res)

    # 6. Fallback
    def test_fallback_generic_query_asks_user_with_catalog_list(self):
        res = summarizer._select_document_ids(
            question="Summarize the document",
            history=[],
            documents=self.catalog,
            requested_doc_ids=None,
        )
        self.assertIsInstance(res, str)
        self.assertIn("Which document would you like me to summarize?", res)
        self.assertIn("wheat_study", res)
        self.assertIn("barley_study", res)
        self.assertIn("corn_genetics", res)


if __name__ == "__main__":
    unittest.main()
