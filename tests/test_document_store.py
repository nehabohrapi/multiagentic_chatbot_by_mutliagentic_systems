import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chromadb
from docx import Document

from document_processing.chunker import chunk_document_pages
from document_processing.document_loader import ExtractedPage
import memory.state_store as state_store
import tools.document_store as document_store
import tools.document_ingestion as document_ingestion


class DocumentStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.documents_dir = Path(self.temp_dir.name)
        self.chroma_dir = self.documents_dir / "chroma"
        self.documents_patch = patch.object(document_store, "DOCUMENTS_DIR", self.documents_dir)
        self.documents_patch.start()
        self.addCleanup(self.documents_patch.stop)
        self.ingestion_documents_patch = patch.object(
            document_ingestion, "DOCUMENTS_DIR", self.documents_dir
        )
        self.ingestion_documents_patch.start()
        self.addCleanup(self.ingestion_documents_patch.stop)
        self.chroma_patch = patch.object(document_ingestion, "CHROMA_DIR", self.chroma_dir)
        self.chroma_patch.start()
        self.addCleanup(self.chroma_patch.stop)
        self.state_db_patch = patch.object(
            state_store, "STATE_DB_FILE", self.documents_dir / "test_state.sqlite3"
        )
        self.state_db_patch.start()
        self.addCleanup(self.state_db_patch.stop)
        self.signature_patch = patch.object(document_ingestion, "_last_inventory_signature", None)
        self.signature_patch.start()
        self.addCleanup(self.signature_patch.stop)
        self.catalog_cache_patch = patch.object(document_store, "_document_catalog_cache", None)
        self.catalog_cache_patch.start()
        self.addCleanup(self.catalog_cache_patch.stop)
        self.upload_documents_patch = patch.object(
            document_ingestion, "DOCUMENTS_DIR", self.documents_dir
        )
        self.upload_documents_patch.start()
        self.addCleanup(self.upload_documents_patch.stop)
        self.addCleanup(self.temp_dir.cleanup)
        self.addCleanup(chromadb.api.client.SharedSystemClient.clear_system_cache)

    def test_table_search_recognizes_plain_text_pdf_table_captions(self):
        self.assertTrue(
            document_store._is_table_search_candidate(
                "8 TABLE 1: Statistics of large language models",
                [],
            )
        )
        self.assertFalse(
            document_store._is_table_search_candidate(
                "The report refers to Table 1 in its introduction.",
                [],
            )
        )

    def test_uploaded_documents_are_safely_saved_and_deduplicated(self):
        class UploadedFile:
            def __init__(self, name, content):
                self.name = name
                self._content = content

            def getvalue(self):
                return self._content

        uploaded = UploadedFile(r"..\research notes.txt", b"Local document content.")
        report = document_ingestion.save_uploaded_documents([uploaded])
        duplicate_report = document_ingestion.save_uploaded_documents([uploaded])

        self.assertEqual(report["saved"], ["research notes.txt"])
        self.assertEqual(report["errors"], [])
        self.assertEqual(duplicate_report["unchanged"], ["research notes.txt"])
        self.assertEqual(
            (self.documents_dir / "research notes.txt").read_bytes(),
            b"Local document content.",
        )

    def test_upload_accepts_files_over_previous_ten_megabyte_limit(self):
        class UploadedFile:
            name = "large-handbook.txt"

            def getvalue(self):
                return b"x" * (10 * 1024 * 1024 + 1)

        report = document_ingestion.save_uploaded_documents([UploadedFile()])

        self.assertEqual(report["saved"], ["large-handbook.txt"])
        self.assertEqual(report["errors"], [])
        self.assertEqual(
            (self.documents_dir / "large-handbook.txt").stat().st_size,
            10 * 1024 * 1024 + 1,
        )

    def test_search_and_read_local_markdown_documents(self):
        (self.documents_dir / "research-methods.md").write_text(
            "# Research Methods\nQualitative interviews are useful for exploratory research.",
            encoding="utf-8",
        )
        (self.documents_dir / "unrelated.txt").write_text(
            "A recipe for vegetable soup with carrots and potatoes.",
            encoding="utf-8",
        )

        results = document_store.search_documents("qualitative exploratory interviews")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "research-methods")
        self.assertIn("interviews", results[0].snippet.lower())

        source = document_store.read_document(results[0].url)
        self.assertEqual(source["status"], "success")
        self.assertEqual(source["source_kind"], "local_document")
        self.assertIn("Qualitative interviews", source["content"])

    def test_document_catalog_reuses_metadata_until_inventory_changes(self):
        collection = type("Collection", (), {})()
        collection.get = unittest.mock.Mock(
            return_value={
                "metadatas": [
                    {
                        "document_id": "doc-1",
                        "title": "Research Notes",
                        "page_number": 1,
                    },
                    {
                        "document_id": "doc-1",
                        "title": "Research Notes",
                        "page_number": 4,
                    },
                ]
            }
        )
        sync_reports = [
            {"inventory_changed": True, "errors": []},
            {"inventory_changed": False, "errors": []},
            {"inventory_changed": True, "errors": []},
        ]

        with (
            patch.object(
                document_ingestion,
                "ensure_documents_indexed",
                side_effect=sync_reports,
            ),
            patch.object(document_ingestion, "get_vector_collection", return_value=collection),
        ):
            first_catalog = document_store.get_document_catalog()
            second_catalog = document_store.get_document_catalog()

            self.assertEqual(first_catalog, second_catalog)
            self.assertEqual(first_catalog[0]["pages"], 4)
            collection.get.assert_called_once_with(include=["metadatas"])

            third_catalog = document_store.get_document_catalog()

        self.assertEqual(third_catalog, first_catalog)
        self.assertEqual(collection.get.call_count, 2)

    def test_summary_loader_selects_relevant_chunks_and_orders_them(self):
        chunks = [
            document_store.DocumentChunkSearchResult(
                chunk_id=chunk_id,
                document_id=document_id,
                title="Research Notes" if document_id == "doc-1" else "Other",
                source_path=(
                    "C:/documents/research-notes.pdf"
                    if document_id == "doc-1"
                    else "C:/documents/other.pdf"
                ),
                file_type="pdf",
                chunk_index=chunk_index,
                page_number=page_number,
                text=text,
                distance=0.1,
                hybrid_score=score,
            )
            for chunk_id, document_id, chunk_index, page_number, text, score in [
                ("chunk-2", "doc-1", 2, 2, "Page two.", 0.8),
                ("chunk-1b", "doc-1", 1, 1, "Page one second.", 0.9),
                ("chunk-1a", "doc-1", 0, 1, "Page one first.", 0.9),
                ("other-doc", "doc-2", 0, 1, "Other document.", 1.0),
            ]
        ]

        with (
            patch.object(
                document_ingestion,
                "ensure_documents_indexed",
                return_value={"errors": []},
            ),
            patch.object(document_store, "search_document_chunks", return_value=chunks) as search,
        ):
            selected = document_store.get_document_summary_chunks(
                "summarize research notes",
                doc_ids=["doc-1"],
            )

        self.assertEqual(search.call_count, 5)
        self.assertTrue(all(call.kwargs["doc_ids"] == ["doc-1"] for call in search.call_args_list))
        self.assertEqual([chunk.chunk_index for chunk in selected], [0, 1, 2])
        self.assertEqual(
            [chunk.text for chunk in selected],
            ["Page one first.", "Page one second.", "Page two."],
        )
        self.assertTrue(all(chunk.document_id == "doc-1" for chunk in selected))

    def test_full_document_loader_returns_all_chunks_in_source_order(self):
        collection = type("Collection", (), {})()
        collection.get = unittest.mock.Mock(return_value={
            "ids": ["chunk-2", "chunk-1", "chunk-0"],
            "documents": ["Third page", "Second page", "First page"],
            "metadatas": [
                {
                    "document_id": "doc-1",
                    "title": "Research Notes",
                    "source_path": str(self.documents_dir / "research.pdf"),
                    "file_type": "pdf",
                    "chunk_index": 2,
                    "page_number": 3,
                    "structured_data": "[]",
                },
                {
                    "document_id": "doc-1",
                    "title": "Research Notes",
                    "source_path": str(self.documents_dir / "research.pdf"),
                    "file_type": "pdf",
                    "chunk_index": 1,
                    "page_number": 2,
                    "structured_data": "[]",
                },
                {
                    "document_id": "doc-1",
                    "title": "Research Notes",
                    "source_path": str(self.documents_dir / "research.pdf"),
                    "file_type": "pdf",
                    "chunk_index": 0,
                    "page_number": 1,
                    "structured_data": "[]",
                },
            ],
        })

        with (
            patch.object(
                document_ingestion,
                "ensure_documents_indexed",
                return_value={"errors": []},
            ),
            patch.object(document_ingestion, "get_vector_collection", return_value=collection),
        ):
            chunks = document_store.get_document_chunks("doc-1")

        collection.get.assert_called_once_with(
            where={"document_id": "doc-1"},
            include=["documents", "metadatas"],
        )
        self.assertEqual([chunk.chunk_index for chunk in chunks], [0, 1, 2])
        self.assertEqual([chunk.text for chunk in chunks], [
            "First page",
            "Second page",
            "Third page",
        ])

    def test_full_document_loader_reads_every_chunk_from_chroma(self):
        source = self.documents_dir / "complete-document.txt"
        source.write_text(
            "The complete document contains ordered indexed evidence. " * 100,
            encoding="utf-8",
        )
        document_ingestion.ingest_documents(force=True)
        catalog = document_store.get_document_catalog()

        chunks = document_store.get_document_chunks(catalog[0]["doc_id"])

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.title == "complete-document" for chunk in chunks))
        self.assertEqual(
            [chunk.chunk_index for chunk in chunks],
            sorted(chunk.chunk_index for chunk in chunks),
        )
        self.assertIn("ordered indexed evidence", " ".join(chunk.text for chunk in chunks))

    def test_chunk_search_returns_ranked_chunk_metadata_and_applies_file_filter(self):
        source = self.documents_dir / "policy-evidence.txt"
        source.write_text(
            "The annual plan refund window is thirty days after purchase.",
            encoding="utf-8",
        )
        (self.documents_dir / "unrelated.json").write_text(
            '{"text": "The document describes unrelated gardening advice."}',
            encoding="utf-8",
        )
        document_ingestion.ingest_documents(force=True)

        results = document_store.search_document_chunks(
            "annual plan refund window",
            max_results=8,
            file_type="txt",
        )

        self.assertTrue(results)
        self.assertTrue(all(result.file_type == "txt" for result in results))
        self.assertEqual(results[0].source_path, str(source.resolve()))
        self.assertIn("refund window", results[0].text)
        self.assertIsInstance(results[0].distance, float)
        self.assertTrue(results[0].document_id)

    def test_chunk_search_skips_vector_results_with_missing_metadata(self):
        collection = type("Collection", (), {})()
        collection.count = lambda: 2
        collection.query = lambda **kwargs: {
            "ids": [["missing-metadata", "valid-chunk"]],
            "documents": [["unusable passage", "The refund window is thirty days."]],
            "metadatas": [[
                None,
                {
                    "document_id": "doc-1",
                    "title": "Policy",
                    "source_path": str(self.documents_dir / "policy.txt"),
                    "file_type": "txt",
                    "chunk_index": 0,
                },
            ]],
            "distances": [[0.1, 0.2]],
        }

        with (
            patch.object(
                document_ingestion,
                "ensure_documents_indexed",
                return_value={"errors": []},
            ),
            patch.object(document_ingestion, "get_vector_collection", return_value=collection),
            patch.object(document_store, "embed_query", return_value=(0.1, 0.2)),
        ):
            results = document_store.search_document_chunks("refund window", max_results=2)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "valid-chunk")

    def test_hybrid_search_fuses_and_deduplicates_vector_and_fts_candidates(self):
        source = str(self.documents_dir / "policy.txt")
        collection = type("Collection", (), {})()
        collection.count = lambda: 3
        collection.query = lambda **kwargs: {
            "ids": [["both", "vector-only"]],
            "documents": [["shared evidence", "vector evidence"]],
            "metadatas": [[
                {
                    "document_id": "doc-1",
                    "title": "Policy",
                    "source_path": source,
                    "file_type": "txt",
                    "chunk_index": 0,
                    "page_number": 2,
                },
                {
                    "document_id": "doc-1",
                    "title": "Policy",
                    "source_path": source,
                    "file_type": "txt",
                    "chunk_index": 1,
                    "page_number": 2,
                },
            ]],
            "distances": [[0.1, 0.2]],
        }
        fts_candidates = [
            {
                "chunk_id": "fts-only",
                "document_id": "doc-2",
                "title": "Other Policy",
                "source_path": str(self.documents_dir / "other.txt"),
                "file_type": "txt",
                "chunk_index": 0,
                "page_number": 4,
                "text": "lexical evidence",
                "bm25_score": 0.8,
            },
            {
                "chunk_id": "both",
                "document_id": "doc-1",
                "title": "Policy",
                "source_path": source,
                "file_type": "txt",
                "chunk_index": 0,
                "page_number": 2,
                "text": "shared evidence",
                "bm25_score": 0.5,
            },
        ]

        with (
            patch.object(
                document_ingestion,
                "ensure_documents_indexed",
                return_value={"errors": []},
            ),
            patch.object(document_ingestion, "get_vector_collection", return_value=collection),
            patch.object(document_store, "embed_query", return_value=(0.1, 0.2)),
            patch.object(state_store, "search_fts_chunks", return_value=fts_candidates),
        ):
            results = document_store.search_document_chunks(
                "shared evidence",
                max_results=3,
            )

        self.assertEqual(
            [result.chunk_id for result in results],
            ["both", "fts-only", "vector-only"],
        )
        self.assertEqual(len({result.chunk_id for result in results}), 3)
        self.assertEqual(results[0].match_type, "hybrid")
        self.assertEqual(results[0].vector_rank, 1)
        self.assertEqual(results[0].lexical_rank, 2)
        self.assertEqual(results[1].match_type, "lexical")
        self.assertEqual(results[1].document_id, "doc-2")

    def test_fts_search_surfaces_operational_errors_instead_of_returning_no_hits(self):
        class BrokenConnection:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def execute(self, *_args):
                raise sqlite3.OperationalError("FTS index unavailable")

        with (
            patch.object(state_store, "_connection", return_value=BrokenConnection()),
            self.assertRaisesRegex(RuntimeError, "FTS5 chunk search failed"),
        ):
            state_store.search_fts_chunks("refund window")

    def test_default_state_database_connection_is_reused_per_thread(self):
        state_db_path = self.documents_dir / "reusable_state.sqlite3"
        with (
            patch.object(state_store, "_DEFAULT_STATE_DB_PATH", state_db_path.resolve()),
            patch.object(state_store, "STATE_DB_FILE", state_db_path),
        ):
            try:
                with state_store._connection() as first:
                    first.execute("SELECT 1")
                with state_store._connection() as second:
                    second.execute("SELECT 1")

                self.assertIs(first, second)
            finally:
                cached_connection = getattr(
                    state_store._connection_local,
                    "connection",
                    None,
                )
                if cached_connection is not None:
                    cached_connection.close()
                    del state_store._connection_local.connection
                    del state_store._connection_local.db_path

    def test_chunk_search_uses_exact_term_fallback_when_vector_threshold_filters_all(self):
        source = self.documents_dir / "history.txt"
        source.write_text(
            "World War II began in 1939 and ended in 1945.",
            encoding="utf-8",
        )
        document_ingestion.ingest_documents(force=True)

        collection = document_ingestion.get_vector_collection()
        with (
            patch("tools.document_store.embed_query", return_value=(0.1, 0.2)),
            patch.object(
                collection,
                "query",
                return_value={
                    "ids": [["chunk-id"]],
                    "documents": [["unrelated passage"]],
                    "metadatas": [[{"source_path": str(source.resolve())}]],
                    "distances": [[0.99]],
                },
            ),
            patch.object(document_ingestion, "get_vector_collection", return_value=collection),
        ):
            results = document_store.search_document_chunks(
                "World War II dates",
                max_results=8,
            )

        self.assertTrue(results)
        self.assertEqual(results[0].match_type, "lexical")
        self.assertIn("1939", results[0].text)

    def test_docx_text_is_searchable(self):
        path = self.documents_dir / "project-handbook.docx"
        document = Document()
        document.add_paragraph("The project handbook defines the review process.")
        document.save(path)

        results = document_store.search_documents("handbook review process")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, str(path.resolve()))

    def test_json_document_extraction_preserves_page_numbers(self):
        path = self.documents_dir / "research-pages.json"
        path.write_text(
            '{"pages": ['
            '{"page_number": 3, "text": "First evidence passage.\\u0000"},'
            '{"page_number": 7, "text": "Second evidence passage."}'
            ']}',
            encoding="utf-8",
        )

        result = document_store.read_document(path)

        self.assertEqual(result["status"], "success")
        self.assertIn("First evidence passage.", result["content"])
        self.assertEqual(
            result["pages"],
            [
                {"page_number": 3, "text": "First evidence passage."},
                {"page_number": 7, "text": "Second evidence passage."},
            ],
        )

    def test_chunking_uses_configured_size_and_overlap(self):
        chunks = chunk_document_pages([ExtractedPage(page_number=4, text="x" * 5000)])

        self.assertEqual([chunk.chunk_index for chunk in chunks], [0, 1, 2, 3, 4, 5])
        self.assertEqual([len(chunk.text) for chunk in chunks], [1000, 1000, 1000, 1000, 1000, 750])
        self.assertEqual(chunks[0].text[-150:], chunks[1].text[:150])
        self.assertEqual(chunks[1].text[-150:], chunks[2].text[:150])
        self.assertTrue(all(chunk.page_number == 4 for chunk in chunks))

    def test_chunking_preserves_table_structure_and_newlines(self):
        table_text = (
            "| Quarter | Revenue | Profit |\n"
            "| --- | --- | --- |\n"
            "| Q1 | $10M | $2M |\n"
            "| Q2 | $12M | $3M |"
        )
        chunks = chunk_document_pages([ExtractedPage(page_number=1, text=table_text)])
        self.assertEqual(len(chunks), 1)
        # Newlines and pipes must be intact, not flattened into spaces
        self.assertIn("\n| Q1 | $10M | $2M |", chunks[0].text)
        self.assertIn("| Quarter | Revenue | Profit |", chunks[0].text)

    def test_chunking_preserves_structured_data(self):
        table = {
            "kind": "table",
            "headers": ["Name", "Score"],
            "rows": [["Ada", "10"]],
        }
        chunks = chunk_document_pages(
            [ExtractedPage(page_number=2, text="Ada scored 10.", structured_data=(table,))]
        )
        self.assertEqual(chunks[0].structured_data, (table,))

    def test_json_document_rejects_malformed_json(self):
        path = self.documents_dir / "broken.json"
        path.write_text('{"text": ', encoding="utf-8")

        result = document_store.read_document(path)

        self.assertEqual(result["status"], "failed")
        self.assertIn("Could not read valid JSON", result["error"])

    def test_read_document_caps_text_and_page_records_consistently(self):
        path = self.documents_dir / "long-notes.txt"
        path.write_text("A" * 50, encoding="utf-8")

        result = document_store.read_document(path, max_chars=12)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["content"], "A" * 12)
        self.assertEqual(result["pages"], [{"page_number": None, "text": "A" * 12}])
        self.assertEqual(result["content_length"], 12)

    def test_ingestion_persists_document_and_page_metadata(self):
        source = self.documents_dir / "research-pages.json"
        source.write_text(
            '{"pages": ['
            '{"page_number": 3, "text": "First page evidence. "},'
            '{"page_number": 7, "text": "Second page evidence."}'
            ']}',
            encoding="utf-8",
        )

        report = document_ingestion.ingest_documents(force=True)
        persisted = document_ingestion.get_vector_collection().get(
            include=["documents", "embeddings", "metadatas"]
        )

        self.assertEqual(report["documents_indexed"], 1)
        self.assertEqual(len(persisted["documents"]), len(persisted["embeddings"]))
        self.assertEqual(len(persisted["documents"]), 2)
        metadata_by_page = {metadata["page_number"]: metadata for metadata in persisted["metadatas"]}
        self.assertEqual(set(metadata_by_page), {3, 7})
        for metadata in metadata_by_page.values():
            self.assertTrue(metadata["document_id"])
            self.assertEqual(metadata["original_filename"], source.name)
            self.assertEqual(metadata["file_type"], "json")
            self.assertIn("chunk_index", metadata)

    def test_ingestion_persists_chunks_and_embeddings_in_chromadb(self):
        source = self.documents_dir / "vector-index.txt"
        source.write_text(
            " ".join(
                f"Section {index} describes vector databases and semantic document retrieval."
                for index in range(300)
            ),
            encoding="utf-8",
        )

        report = document_ingestion.ingest_documents(force=True)
        stats = document_ingestion.get_vector_store_stats()
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        collection = chromadb.PersistentClient(path=str(self.chroma_dir)).get_collection(
            document_ingestion.CHROMA_COLLECTION_NAME
        )
        persisted = collection.get(include=["documents", "embeddings", "metadatas"])

        self.assertEqual(report["documents_indexed"], 1)
        self.assertGreaterEqual(stats["chunks"], 2)
        self.assertEqual(stats["chunks"], stats["chunks_with_embeddings"])
        self.assertGreater(stats["embedding_dimensions"], 0)
        self.assertEqual(len(persisted["documents"]), stats["chunks"])
        self.assertEqual(len(persisted["embeddings"]), stats["chunks"])
        self.assertTrue(all(item["source_path"] == str(source.resolve()) for item in persisted["metadatas"]))
        self.assertIn("Section 299", " ".join(persisted["documents"]))

        second_run = document_ingestion.ingest_documents()
        self.assertEqual(second_run["chunks_upserted"], 0)

    def test_forced_ingestion_reuses_embedding_cache_and_replaces_fts_rows(self):
        source = self.documents_dir / "cache-test.txt"
        source.write_text(
            "A distinctive phrase validates persistent embedding cache behavior.",
            encoding="utf-8",
        )

        with patch.object(
            document_ingestion,
            "embed_documents",
            wraps=document_ingestion.embed_documents,
        ) as embed_documents:
            first_run = document_ingestion.ingest_documents(force=True)
            first_call_count = embed_documents.call_count
            second_run = document_ingestion.ingest_documents(force=True)

        self.assertEqual(first_run["chunks_embedded"], 1)
        self.assertEqual(first_run["embedding_cache_hits"], 0)
        self.assertEqual(second_run["chunks_embedded"], 0)
        self.assertEqual(second_run["embedding_cache_hits"], 1)
        self.assertEqual(embed_documents.call_count, first_call_count)
        fts_results = state_store.search_fts_chunks("distinctive phrase")
        self.assertEqual(len(fts_results), 1)
        self.assertEqual(fts_results[0]["source_path"], str(source.resolve()))
        vector_records = document_ingestion.get_vector_collection().get(
            include=["metadatas"]
        )
        fts_records = state_store.get_fts_index_records()
        self.assertEqual(set(vector_records["ids"]), {record["chunk_id"] for record in fts_records})
        vector_metadata = dict(zip(vector_records["ids"], vector_records["metadatas"]))
        for record in fts_records:
            metadata = vector_metadata[record["chunk_id"]]
            for field in ("document_id", "source_path", "chunk_index", "page_number"):
                self.assertEqual(str(record[field]), str(metadata.get(field)))

    def test_ingestion_removes_fts_chunks_with_no_vector_counterpart(self):
        state_store.index_fts_chunks([{
            "chunk_id": "stale-fts-chunk",
            "document_id": "removed-document",
            "title": "Removed document",
            "source_path": str(self.documents_dir / "removed.txt"),
            "file_type": "txt",
            "chunk_index": 0,
            "page_number": 1,
            "text": "Stale indexed text.",
            "structured_data": "",
        }])

        report = document_ingestion.ingest_documents(force=True)

        self.assertEqual(report["errors"], [])
        self.assertEqual(state_store.get_fts_index_records(), [])

    def test_read_document_rejects_paths_outside_documents_folder(self):
        outside_path = Path(self.temp_dir.name).parent / "outside.txt"
        with self.assertRaisesRegex(ValueError, "inside the documents folder"):
            document_store.read_document(outside_path)
