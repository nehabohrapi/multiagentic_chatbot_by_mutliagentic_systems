"""Que Ans Agent (Subagent 2) for hybrid document question-answering retrieval.

Performs Hybrid Search (ChromaDB dense vectors + BM25 lexical with Reciprocal Rank Fusion)
and Semantic Reranking across document chunks.
"""

from __future__ import annotations

from typing import Any

from telemetry import time_block
from . import curator, retriever_worker


class QueAnsAgent:
    """Subagent specialized in hybrid document retrieval and evidence collection."""

    @staticmethod
    def search_documents(
        query: str,
        *,
        history: list[dict[str, Any]] | None = None,
        doc_ids: list[str] | None = None,
        retrieval_mode: str = "hybrid",
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Perform hybrid search (dense ChromaDB vector + BM25 lexical) for a query."""
        with retriever_worker.trace_context(
            run_id=run_id,
            agent="que_ans_agent",
            sub_question=query,
        ):
            return retriever_worker.retrieve(
                query,
                needs_retrieval=True,
                history=history or [],
                doc_ids=doc_ids,
                retrieval_mode=retrieval_mode or "hybrid",
                planned_queries=[query],
                skip_model_grading=True,
            )

    @staticmethod
    def rerank_evidence(
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int = 8,
    ) -> tuple[list[dict[str, Any]], str]:
        """Apply cross-encoder / semantic reranking to retrieved passages."""
        return curator.rerank_chunks(query, chunks, top_k=top_k)

    @classmethod
    def execute(
        cls,
        question: str,
        *,
        plan: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        run_id: str | None = None,
        is_comparison: bool = False,
        comparison_entities: list[str] | None = None,
        resolved_question: str | None = None,
    ) -> dict[str, Any]:
        """Execute hybrid retrieval across sub-questions, comparison entities, or main query."""
        plan = plan or {}
        history = history or []
        sub_qs = plan.get("sub_questions") or []
        resolved_query = resolved_question or question
        comparison_entities = comparison_entities or []

        all_chunks: list[dict[str, Any]] = []
        retrieval_assessments: list[dict[str, Any]] = []

        def collect(res: dict[str, Any], q: str) -> None:
            retrieved = list(res.get("chunks", []))
            for chunk in retrieved:
                item = dict(chunk)
                queries = list(item.get("_retrieval_queries") or [])
                if q not in queries:
                    queries.append(q)
                item["_retrieval_queries"] = queries
                all_chunks.append(item)
            sufficient = bool(res.get("sufficient", bool(retrieved)))
            missing_info = str(res.get("missing_info") or "").strip()
            retrieval_assessments.append({
                "query": q,
                "sufficient": sufficient,
                "missing_info": missing_info,
                "chunk_ids": [str(c.get("chunk_id")) for c in retrieved if c.get("chunk_id")],
            })

        with time_block("Que Ans Agent: Hybrid Search", category="SUBAGENT", level=2) as tb:
            if sub_qs:
                for sq in sub_qs:
                    q_text = sq.get("query") if isinstance(sq, dict) else getattr(sq, "query", question)
                    d_ids = sq.get("doc_ids") if isinstance(sq, dict) else getattr(sq, "doc_ids", None)
                    r_mode = sq.get("retrieval", "hybrid") if isinstance(sq, dict) else getattr(sq, "retrieval", "hybrid")
                    res = cls.search_documents(
                        q_text,
                        history=history,
                        doc_ids=d_ids,
                        retrieval_mode=r_mode,
                        run_id=run_id,
                    )
                    collect(res, q_text)
            elif is_comparison and len(comparison_entities) >= 2:
                for q_term in [comparison_entities[0], comparison_entities[1], resolved_query]:
                    res = cls.search_documents(
                        q_term,
                        history=history,
                        retrieval_mode="hybrid",
                        run_id=run_id,
                    )
                    collect(res, q_term)
            else:
                res = cls.search_documents(
                    resolved_query,
                    history=history,
                    retrieval_mode="hybrid",
                    run_id=run_id,
                )
                collect(res, resolved_query)

            tb["details"] = f"retrieved {len(all_chunks)} chunk(s)"

        return {
            "chunks": all_chunks,
            "assessments": retrieval_assessments,
            "log_entry": {
                "subagent": "que_ans_agent",
                "chunks_found": len(all_chunks),
                "query_used": resolved_query,
            },
        }


run_que_ans_agent = QueAnsAgent.execute
