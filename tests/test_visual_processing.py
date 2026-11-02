import json
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from agents import retriever_worker
from document_processing.chunker import chunk_document_pages
from document_processing.document_loader import extract_document_pages
from memory import conversation_memory, state_store
from tools import document_ingestion, document_store
from tools import visual_processing


class VisualProcessingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image_path = self.root / "chart.png"
        Image.new("RGB", (320, 180), color="white").save(self.image_path)
        self.addCleanup(self.temp_dir.cleanup)

    def test_standalone_image_extraction_preserves_searchable_figure_metadata(self):
        with (
            patch("tools.visual_processing.DOCUMENTS_DIR", self.root),
            patch("tools.visual_processing.EXTRACTED_IMAGES_DIR", self.root / "extracted"),
        ):
            pages = extract_document_pages(self.image_path)

        self.assertIn("Image asset", pages[0].text)
        self.assertEqual(pages[0].structured_data[0]["kind"], "figure")
        self.assertEqual(pages[0].structured_data[0]["image_path"], str(self.image_path.resolve()))
        chunks = chunk_document_pages(pages)
        self.assertEqual(chunks[0].structured_data[0]["kind"], "figure")

    def test_read_document_keeps_figure_metadata_for_indexing(self):
        with patch.object(document_store, "DOCUMENTS_DIR", self.root):
            result = document_store.read_document(self.image_path)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["pages"][0]["structured_data"][0]["kind"], "figure")

    def test_image_path_is_persisted_in_vector_and_fts_chunk_metadata(self):
        figure = {
            "kind": "figure",
            "source": self.image_path.name,
            "page_number": 1,
            "image_path": str(self.image_path.resolve()),
            "description": "Chart image.",
        }
        extracted = {
            "status": "success",
            "content": "Image asset chart.",
            "pages": [{
                "page_number": 1,
                "text": "Image asset chart.",
                "structured_data": [figure],
            }],
        }
        prepared = None
        with patch.object(document_ingestion, "read_document", return_value=extracted):
            prepared = document_ingestion._prepare_document_record(self.image_path, self.root)

        vector_metadata = json.loads(prepared["records"][0]["metadata"]["structured_data"])
        self.assertEqual(vector_metadata[0]["image_path"], str(self.image_path.resolve()))

        database_path = self.root / "image-index.sqlite3"
        with patch.object(state_store, "STATE_DB_FILE", database_path):
            document_ingestion._index_prepared_document_fts(prepared)
            indexed = state_store.search_fts_chunks("chart", max_results=1)

        self.assertEqual(len(indexed), 1)
        fts_metadata = json.loads(indexed[0]["structured_data"])
        self.assertEqual(fts_metadata[0]["image_path"], str(self.image_path.resolve()))

    def test_visual_asset_index_returns_paths_and_metadata_without_copying_image(self):
        image_size = self.image_path.stat().st_size
        prepared = {
            "chunk_id": "image-chunk",
            "document_id": "image-document",
            "title": "Chart",
            "source_path": str(self.image_path.resolve()),
            "file_type": "png",
            "chunk_index": 0,
            "page_number": 1,
            "text": "Chart with annual sales",
            "structured_data": json.dumps([{
                "kind": "figure",
                "source": self.image_path.name,
                "page_number": 1,
                "image_path": str(self.image_path.resolve()),
                "description": "Annual sales chart",
            }]),
        }
        database_path = self.root / "visual-assets.sqlite3"
        with (
            patch.object(state_store, "STATE_DB_FILE", database_path),
            patch.object(visual_processing, "DOCUMENTS_DIR", self.root),
        ):
            state_store.index_fts_chunks([prepared])
            results = state_store.search_visual_chunks("annual sales")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["image_assets"][0]["image_path"], str(self.image_path.resolve()))
            self.assertEqual(results[0]["image_assets"][0]["description"], "Annual sales chart")
            self.assertEqual(self.image_path.stat().st_size, image_size)
            self.assertEqual(
                visual_processing.collect_visual_assets(results)[0]["path"],
                self.image_path.resolve(),
            )

    def test_existing_fts_figure_metadata_is_backfilled_to_asset_index(self):
        database_path = self.root / "existing-index.sqlite3"
        figure = {
            "kind": "figure",
            "source": self.image_path.name,
            "page_number": 2,
            "image_path": str(self.image_path.resolve()),
            "description": "Existing indexed image",
        }
        connection = sqlite3.connect(database_path)
        try:
            connection.execute(
                """
                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                    chunk_id UNINDEXED, document_id UNINDEXED, title,
                    source_path UNINDEXED, file_type UNINDEXED,
                    chunk_index UNINDEXED, page_number UNINDEXED, text,
                    structured_data UNINDEXED
                )
                """
            )
            connection.execute(
                """
                INSERT INTO chunks_fts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "existing-image-chunk",
                    "existing-image-doc",
                    "Existing image",
                    str(self.root / "source.pdf"),
                    "pdf",
                    0,
                    2,
                    "Existing indexed image caption",
                    json.dumps([figure]),
                ),
            )
            connection.commit()
        finally:
            connection.close()

        with patch.object(state_store, "STATE_DB_FILE", database_path):
            results = state_store.search_visual_chunks("existing indexed image")
            self.assertEqual(len(results), 1)
            self.assertEqual(
                results[0]["image_assets"][0]["image_path"],
                str(self.image_path.resolve()),
            )


    def test_conversation_message_persists_image_paths(self):
        memory_path = self.root / "conversations.json"
        with patch.object(conversation_memory, "MEMORY_FILE", memory_path):
            conversation_memory.save_message(
                "session-1",
                "assistant",
                "Here is the chart.",
                image_paths=[str(self.image_path), str(self.image_path)],
            )
            history = conversation_memory.load_history("session-1")

        self.assertEqual(history[0]["image_paths"], [str(self.image_path)])

    def test_persisted_image_paths_are_restricted_to_local_image_roots(self):
        extracted_root = self.root / "extracted"
        extracted_root.mkdir()
        extracted_image = extracted_root / "embedded.png"
        Image.new("RGB", (16, 16), color="white").save(extracted_image)
        with tempfile.TemporaryDirectory() as outside_dir:
            outside_image = Path(outside_dir) / "outside.png"
            Image.new("RGB", (16, 16), color="white").save(outside_image)
            with (
                patch.object(visual_processing, "DOCUMENTS_DIR", self.root),
                patch.object(visual_processing, "EXTRACTED_IMAGES_DIR", extracted_root),
            ):
                safe = visual_processing.safe_image_paths([
                    str(self.image_path),
                    str(extracted_image),
                    str(outside_image),
                    str(self.root / "missing.png"),
                ])

        self.assertEqual(safe, [self.image_path.resolve(), extracted_image.resolve()])

    def test_captioned_pdf_image_is_extracted_and_path_persisted_on_demand(self):
        import fitz

        pdf_path = self.root / "captioned.pdf"
        document = fitz.open()
        page = document.new_page(width=500, height=500)
        page.insert_text((40, 30), "Photo 1 Root system of a self-rooted vine")
        page.insert_image(fitz.Rect(40, 60, 360, 240), filename=str(self.image_path))
        document.save(pdf_path)
        document.close()

        class FakeCollection:
            def __init__(self):
                self.updated_metadata = None

            def get(self, **_kwargs):
                return {"metadatas": [{"structured_data": "[]", "source_path": str(pdf_path)}]}

            def update(self, *, ids, metadatas):
                self.updated_metadata = (ids, metadatas)

        collection = FakeCollection()
        chunk = {
            "chunk_id": "captioned-chunk",
            "document_id": "doc-1",
            "title": "captioned",
            "source_path": str(pdf_path),
            "file_type": "pdf",
            "chunk_index": 0,
            "page_number": 1,
            "text": "Photo 1 Root system of a self-rooted vine",
            "structured_data": [],
        }
        fts_update = unittest.mock.Mock()
        extracted_root = self.root / "extracted"
        with (
            patch.object(visual_processing, "DOCUMENTS_DIR", self.root),
            patch.object(visual_processing, "EXTRACTED_IMAGES_DIR", extracted_root),
            patch("tools.document_ingestion.get_vector_collection", return_value=collection),
            patch("memory.state_store.update_fts_chunk_metadata", fts_update),
        ):
            assets = visual_processing.collect_visual_assets(
                [chunk],
                extract_captioned=True,
            )

        self.assertEqual(len(assets), 1)
        self.assertTrue(assets[0]["path"].is_file())
        self.assertIn(str(extracted_root.resolve()), str(assets[0]["path"]))
        self.assertEqual(len(chunk["structured_data"]), 1)
        self.assertEqual(
            chunk["structured_data"][0]["image_path"],
            str(assets[0]["path"]),
        )
        self.assertEqual(collection.updated_metadata[0], ["captioned-chunk"])
        stored_metadata = json.loads(collection.updated_metadata[1][0]["structured_data"])
        self.assertEqual(stored_metadata[0]["image_path"], str(assets[0]["path"]))
        fts_update.assert_called_once_with(
            "captioned-chunk",
            json.dumps(chunk["structured_data"], ensure_ascii=False),
        )

    def test_vision_request_includes_only_retrieved_local_image_and_returns_chart_data(self):
        chunks = [{
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "title": "Chart report",
            "source_path": str(self.root / "report.pdf"),
            "page_number": 2,
            "structured_data": [{
                "kind": "figure",
                "source": "report.pdf",
                "page_number": 2,
                "image_path": str(self.image_path),
                "description": "Bar chart.",
            }],
        }]
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
                                    "content": json.dumps({
                                        "visuals": [{
                                            "visual_id": "chunk-1-figure-1",
                                            "description": "A bar chart comparing two groups.",
                                            "chart_data": {
                                                "headers": ["Group", "Baseline"],
                                                "rows": [["A", "12"], ["B", "18"]],
                                            },
                                            "confidence": "high",
                                            "uncertainties": [],
                                        }],
                                    }),
                                },
                            )(),
                        },
                    )(),
                ],
            },
        )()
        request = {}

        def fake_chat_create(*_args, **kwargs):
            request.update(kwargs)
            return response

        with (
            patch.object(visual_processing, "DOCUMENTS_DIR", self.root),
            patch.object(visual_processing, "EXTRACTED_IMAGES_DIR", self.root / "extracted"),
        ):
            analyze = visual_processing.analyze_visual_assets(
                "Describe this chart.",
                chunks,
                client=object(),
                model="vision-model",
                chat_create=fake_chat_create,
                json_loader=json.loads,
            )

        self.assertEqual(len(analyze), 1)
        self.assertEqual(analyze[0]["source_id"], 1)
        self.assertEqual(analyze[0]["filename"], "report.pdf")
        self.assertEqual(analyze[0]["chart_data"]["rows"], [["A", "12"], ["B", "18"]])
        self.assertEqual(request["max_tokens"], 900)

    def test_answer_generation_reuses_executor_visual_analysis(self):
        visual_evidence = [{
            "visual_id": "chunk-1-figure-1",
            "chunk_id": "chunk-1",
            "filename": "report.pdf",
            "page_number": 2,
            "description": "A bar chart comparing two groups.",
            "chart_data": None,
            "confidence": "medium",
            "uncertainties": [],
        }]
        chunk = {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "title": "Chart report",
            "source_path": str(self.root / "report.pdf"),
            "page_number": 2,
            "text": "A bar chart compares two groups.",
        }
        retrieval = {
            "chunks": [chunk],
            "context_blocks": [chunk],
            "ranked_chunks": [chunk],
            "intent": "image",
            "rerank_confidence": "high",
            "visual_evidence": visual_evidence,
        }
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="The chart compares two groups [1].")
                )
            ]
        )

        with (
            patch.object(retriever_worker, "collect_visual_assets", return_value=[{}]),
            patch.object(retriever_worker, "_try_extractive_answer", return_value=None),
            patch.object(retriever_worker, "analyze_visual_assets") as analyze,
            patch.object(retriever_worker, "safe_chat_create", return_value=response),
        ):
            result = retriever_worker.answer_from_retrieval(
                "What does the chart show?",
                retrieval,
                return_details=True,
            )

        analyze.assert_not_called()
        self.assertEqual(retrieval["visual_evidence"], visual_evidence)
        self.assertIn("compares two groups", result["answer"])

    def test_chart_calculation_uses_existing_deterministic_calculator(self):
        chunks = [{
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "title": "Chart report",
            "source_path": str(self.root / "report.pdf"),
            "page_number": 2,
            "text": "[Figure on Page 2: bar chart]",
            "structured_data": [{
                "kind": "figure",
                "source": "report.pdf",
                "page_number": 2,
                "image_path": str(self.image_path),
                "description": "Bar chart.",
            }],
        }]
        visual_evidence = [{
            "visual_id": "chunk-1-figure-1",
            "source_id": 1,
            "chunk_id": "chunk-1",
            "filename": "report.pdf",
            "page_number": 2,
            "description": "A bar chart.",
            "chart_data": {
                "headers": ["Group", "Baseline Performance"],
                "rows": [["A", "12"], ["B", "18"]],
            },
            "confidence": "high",
            "uncertainties": [],
        }]
        retrieval = {
            "intent": "numeric",
            "context_blocks": chunks,
            "sufficient": True,
        }
        with (
            patch.object(retriever_worker, "collect_visual_assets", return_value=[{}]),
            patch.object(
                retriever_worker,
                "analyze_visual_assets",
                return_value=visual_evidence,
            ),
            patch.object(retriever_worker, "safe_chat_create") as writer_call,
        ):
            answer = retriever_worker.answer_from_retrieval(
                "What is the total baseline performance shown in the chart?",
                retrieval,
                return_details=True,
            )

        writer_call.assert_not_called()
        self.assertIn("30", answer["answer"])
        self.assertIn("[1, report.pdf, p. 2]", answer["raw_answer"])
        self.assertIn("[report.pdf, p. 2]", answer["answer"])
        self.assertEqual(answer["citations"], [1])
        self.assertIn(
            "table",
            [
                item.get("kind")
                for item in retrieval["context_blocks"][0]["structured_data"]
            ],
        )

    def test_uncertain_chart_transcription_is_not_sent_to_calculator(self):
        tables = visual_processing.chart_tables_by_chunk([{
            "chunk_id": "chunk-1",
            "confidence": "medium",
            "chart_data": {
                "headers": ["Group", "Value"],
                "rows": [["A", "12"]],
            },
        }])

        self.assertEqual(tables, {})


if __name__ == "__main__":
    unittest.main()
