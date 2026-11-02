"""Visual Agent (Subagent 3) for multimodal vision and chart data extraction.

Coordinates vision model analysis (e.g. Groq Llama-3.2 Vision) over charts, figures,
diagrams, and extracts structured table data from visual assets.
"""

from __future__ import annotations

import os
from typing import Any

from groq import Groq

from config import VISION_MODEL
from telemetry import time_block
from tools.visual_processing import (
    analyze_visual_assets,
    chart_tables_by_chunk,
    collect_visual_assets,
)
from . import retriever_worker


class VisualAgent:
    """Subagent specialized in multimodal visual understanding, database retrieval of visual assets, and chart extraction."""

    @staticmethod
    def retrieve_visual_chunks_from_db(
        query: str,
        doc_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Retrieve chunks containing visual images or charts and their metadata directly from the database."""
        from memory.state_store import search_visual_chunks

        return search_visual_chunks(query=query, doc_ids=doc_ids, limit=limit)

    @staticmethod
    def collect_assets(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Collect all embedded visual assets (images, figures, chart paths) from chunks."""
        return collect_visual_assets(chunks, extract_captioned=True)

    @classmethod
    def retrieve_visual_assets(
        cls,
        query: str,
        chunks: list[dict[str, Any]] | None = None,
        doc_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Retrieve visual assets and their metadata from database and provided chunks."""
        all_chunks = list(chunks or [])
        db_chunks = cls.retrieve_visual_chunks_from_db(query, doc_ids=doc_ids, limit=limit)
        existing_ids = {str(c.get("chunk_id")) for c in all_chunks if c.get("chunk_id")}
        for chunk in db_chunks:
            cid = str(chunk.get("chunk_id"))
            if cid and cid not in existing_ids:
                existing_ids.add(cid)
                all_chunks.append(chunk)
        return cls.collect_assets(all_chunks)

    @classmethod
    def execute(
        cls,
        question: str,
        chunks: list[dict[str, Any]],
        *,
        retrieve_from_db: bool = False,
        doc_ids: list[str] | None = None,
        model: str = VISION_MODEL,
        client: Any = None,
    ) -> dict[str, Any]:
        """Analyze visual assets attached to retrieved chunks and extract chart tables.
        
        If retrieve_from_db is True, queries the database for visual assets/charts and their metadata.
        """
        all_chunks = list(chunks)
        db_visual_chunks = []
        if retrieve_from_db:
            db_visual_chunks = cls.retrieve_visual_chunks_from_db(question, doc_ids=doc_ids, limit=8)
            existing_ids = {str(c.get("chunk_id")) for c in all_chunks if c.get("chunk_id")}
            for chunk in db_visual_chunks:
                cid = str(chunk.get("chunk_id"))
                if cid and cid not in existing_ids:
                    existing_ids.add(cid)
                    all_chunks.append(chunk)

        visual_assets = cls.collect_assets(all_chunks)
        if not visual_assets:
            return {
                "has_visuals": False,
                "visual_assets": [],
                "visual_evidence": [],
                "updated_chunks": all_chunks,
                "log_entry": {
                    "subagent": "visual_agent",
                    "visuals_found": 0,
                    "db_chunks_retrieved": len(db_visual_chunks),
                },
            }

        with time_block("Visual Agent: Multimodal Analysis", category="SUBAGENT", level=2) as tb:
            groq_client = client or Groq(api_key=os.getenv("GROQ_API_KEY"))
            try:
                visual_evidence = analyze_visual_assets(
                    question,
                    all_chunks,
                    client=groq_client,
                    model=model,
                    chat_create=retriever_worker.safe_chat_create,
                    json_loader=retriever_worker.safe_json_loads,
                )
                chart_tables = chart_tables_by_chunk(visual_evidence)
                augmented_chunks = retriever_worker._add_chart_tables(all_chunks, chart_tables)
                tb["details"] = f"analyzed {len(visual_evidence)} image(s)/chart(s)"

                return {
                    "has_visuals": bool(visual_evidence or visual_assets),
                    "visual_assets": visual_assets,
                    "visual_evidence": visual_evidence,
                    "updated_chunks": augmented_chunks,
                    "log_entry": {
                        "subagent": "visual_agent",
                        "visuals_found": len(visual_assets),
                        "visual_evidence_count": len(visual_evidence),
                        "db_chunks_retrieved": len(db_visual_chunks),
                    },
                }
            except Exception as exc:
                tb["details"] = f"visual analysis error: {type(exc).__name__}"
                return {
                    "has_visuals": bool(visual_assets),
                    "visual_assets": visual_assets,
                    "visual_evidence": [],
                    "updated_chunks": all_chunks,
                    "log_entry": {
                        "subagent": "visual_agent",
                        "visuals_found": len(visual_assets),
                        "db_chunks_retrieved": len(db_visual_chunks),
                        "error": str(exc),
                    },
                }


run_visual_agent = VisualAgent.execute
