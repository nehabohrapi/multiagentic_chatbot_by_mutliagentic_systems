"""Measure vector, FTS5, and hybrid retrieval against labeled local questions."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import BASE_DIR, HYBRID_RRF_K, MAX_DOCUMENT_SEARCH_DISTANCE
from agents.curator import rerank_chunks
from embeddings.local_model import embed_query
from memory.state_store import get_fts_index_records, search_fts_chunks
from tools.document_ingestion import ensure_documents_indexed, get_vector_collection
from tools.document_store import search_document_chunks

logger = logging.getLogger(__name__)
DEFAULT_QUESTIONS = BASE_DIR / "data" / "retrieval_eval_questions.json"
DEFAULT_OUTPUT = BASE_DIR / "data" / "retrieval_eval_latest.json"


def _vector_search(query: str, k: int) -> list[dict[str, Any]]:
    collection = get_vector_collection()
    count = collection.count()
    if not count:
        return []
    response = collection.query(
        query_embeddings=[list(embed_query(query))],
        n_results=min(k, count),
        include=["documents", "metadatas", "distances"],
    )
    ids = response["ids"][0] or []
    documents = response["documents"][0] or []
    metadatas = response["metadatas"][0] or []
    distances = response["distances"][0] or []
    return [
        {
            "chunk_id": chunk_id,
            "document": Path(metadata.get("source_path") or "").name,
            "page": metadata.get("page_number"),
            "score": round(1.0 - float(distance), 6),
            "distance": float(distance),
            "match_type": "vector",
        }
        for chunk_id, metadata, distance in zip(ids, metadatas, distances)
    ]


def _fts_search(query: str, k: int) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": result["chunk_id"],
            "document": Path(result.get("source_path") or "").name,
            "page": result.get("page_number"),
            "score": float(result.get("bm25_score", 0.0)),
            "match_type": "fts5",
        }
        for result in search_fts_chunks(query, max_results=k)
    ]


def _hybrid_search(query: str, k: int):
    return search_document_chunks(query, max_results=k)


def _hybrid_hit(result: Any) -> dict[str, Any]:
    return {
        "chunk_id": result.chunk_id,
        "document": Path(result.source_path).name,
        "page": result.page_number,
        "score": result.hybrid_score,
        "match_type": result.match_type,
    }


def _reranked_hit(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": result["chunk_id"],
        "document": Path(result.get("source_path") or "").name,
        "page": result.get("page_number"),
        "score": result.get("rerank_score"),
        "match_type": result.get("match_type", "reranked"),
    }


def _timed_search(search, query: str, k: int) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    results = search(query, k)
    return results, (time.perf_counter() - started) * 1000


def _is_relevant(hit: dict[str, Any], question: dict[str, Any]) -> bool:
    return (
        hit["document"].casefold() == question["source"].casefold()
        and hit.get("page") in question["pages"]
    )


def _rrf_weighted_hits(
    vector_candidates: list[dict[str, Any]],
    fts_candidates: list[dict[str, Any]],
    lexical_weight: float,
    k: int,
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for rank, hit in enumerate(vector_candidates, start=1):
        candidates[hit["chunk_id"]] = {
            **hit,
            "vector_rank": rank,
            "rrf_score": 1.0 / (HYBRID_RRF_K + rank),
        }
    for rank, hit in enumerate(fts_candidates, start=1):
        candidate = candidates.get(hit["chunk_id"])
        if candidate is None:
            candidate = {**hit, "vector_rank": None, "rrf_score": 0.0}
            candidates[hit["chunk_id"]] = candidate
        candidate["lexical_rank"] = rank
        candidate["rrf_score"] += lexical_weight / (HYBRID_RRF_K + rank)

    max_score = (1.0 + lexical_weight) / (HYBRID_RRF_K + 1)
    ranked = []
    for candidate in candidates.values():
        if (
            candidate.get("vector_rank") is not None
            and candidate["distance"] > MAX_DOCUMENT_SEARCH_DISTANCE
            and candidate.get("lexical_rank") is None
        ):
            continue
        candidate["match_type"] = (
            "hybrid"
            if candidate.get("vector_rank") is not None
            and candidate.get("lexical_rank") is not None
            else "lexical"
            if candidate.get("lexical_rank") is not None
            else "vector"
        )
        candidate["score"] = round(candidate["rrf_score"] / max_score, 6)
        ranked.append(candidate)
    ranked.sort(key=lambda hit: hit["score"], reverse=True)
    return ranked[:k]


def _summarize(channel_results: list[dict[str, Any]], k: int) -> dict[str, Any]:
    latencies = [result["latency_ms"] for result in channel_results]
    return {
        "queries": len(channel_results),
        "hit_at_k": round(
            sum(result["evidence_found"] for result in channel_results)
            / max(len(channel_results), 1),
            4,
        ),
        "precision_at_k": round(
            sum(result["relevant_hits"] for result in channel_results)
            / max(len(channel_results) * k, 1),
            4,
        ),
        "mean_latency_ms": round(statistics.mean(latencies), 3) if latencies else 0,
        "median_latency_ms": round(statistics.median(latencies), 3) if latencies else 0,
        "p95_latency_ms": round(
            sorted(latencies)[min(len(latencies) - 1, int(len(latencies) * 0.95))]
            if latencies
            else 0,
            3,
        ),
    }


def _index_consistency_report() -> dict[str, Any]:
    """Compare chunk IDs and source metadata shared by ChromaDB and FTS5."""
    collection_records = get_vector_collection().get(include=["metadatas"])
    vector_by_id = {
        chunk_id: metadata
        for chunk_id, metadata in zip(
            collection_records["ids"] or [],
            collection_records["metadatas"] or [],
        )
    }
    fts_records = get_fts_index_records()
    fts_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for record in fts_records:
        chunk_id = str(record["chunk_id"])
        if chunk_id in fts_by_id:
            duplicate_ids.add(chunk_id)
        fts_by_id[chunk_id] = record

    vector_ids = set(vector_by_id)
    fts_ids = set(fts_by_id)
    metadata_fields = ("document_id", "source_path", "chunk_index", "page_number")
    mismatches = []
    for chunk_id in sorted(vector_ids & fts_ids):
        vector_metadata = vector_by_id[chunk_id] or {}
        fts_metadata = fts_by_id[chunk_id]
        differing_fields = [
            field
            for field in metadata_fields
            if (
                None if vector_metadata.get(field) is None else str(vector_metadata[field])
            ) != (
                None if fts_metadata.get(field) is None else str(fts_metadata[field])
            )
        ]
        if differing_fields:
            mismatches.append({
                "chunk_id": chunk_id,
                "fields": differing_fields,
            })

    missing_from_fts5 = sorted(vector_ids - fts_ids)
    missing_from_vector = sorted(fts_ids - vector_ids)
    return {
        "vector_chunks": len(vector_ids),
        "fts5_chunks": len(fts_records),
        "missing_from_fts5_count": len(missing_from_fts5),
        "missing_from_fts5_sample": missing_from_fts5[:10],
        "missing_from_vector_count": len(missing_from_vector),
        "missing_from_vector_sample": [
            fts_by_id[chunk_id] for chunk_id in missing_from_vector[:10]
        ],
        "metadata_mismatch_count": len(mismatches),
        "metadata_mismatch_sample": mismatches[:10],
        "duplicate_fts5_ids": sorted(duplicate_ids),
        "consistent": (
            vector_ids == fts_ids
            and not duplicate_ids
            and not mismatches
        ),
    }


def run_benchmark(
    questions_path: Path = DEFAULT_QUESTIONS,
    output_path: Path | None = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    payload = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = payload.get("questions")
    k = payload.get("k", 5)
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"No benchmark questions found in {questions_path}.")
    if not isinstance(k, int) or k < 1:
        raise ValueError("Benchmark k must be a positive integer.")
    if k > 30:
        raise ValueError("Benchmark k cannot exceed the retrieval limit of 30.")

    sync_report = ensure_documents_indexed()
    if sync_report["errors"]:
        raise RuntimeError(f"Document indexing failed before benchmark: {sync_report['errors']}")

    per_channel: dict[str, list[dict[str, Any]]] = {
        "vector": [],
        "fts5": [],
        "hybrid": [],
        "reranked": [],
    }
    rrf_sweep_records: dict[float, list[dict[str, Any]]] = {
        weight: [] for weight in (1.0, 2.0, 4.0, 8.0, 12.0, 16.0)
    }
    query_records = []
    for question in questions:
        query = question["query"]
        scope = {"source": question["source"], "pages": question["pages"]}
        channel_hits: dict[str, list[dict[str, Any]]] = {}
        channel_times: dict[str, float] = {}
        hybrid_candidates = []
        channel_candidates: dict[str, list[dict[str, Any]]] = {}
        for name, search in (
            ("vector", _vector_search),
            ("fts5", _fts_search),
            ("hybrid", _hybrid_search),
        ):
            candidate_count = max(k * 6, 30) if name == "hybrid" else max(k * 48, 240)
            results, latency_ms = _timed_search(search, query, candidate_count)
            if name in {"vector", "fts5"}:
                channel_candidates[name] = results
            if name == "hybrid":
                hybrid_candidates = results
                hits = [_hybrid_hit(result) for result in results[:k]]
            else:
                hits = results[:k]
            relevant_hits = sum(_is_relevant(hit, question) for hit in hits)
            evidence_found = any(_is_relevant(hit, question) for hit in hits)
            record = {
                "id": question["id"],
                "query": query,
                "scope": scope,
                "latency_ms": round(latency_ms, 3),
                "candidate_count": len(results),
                "chunk_ids": [hit["chunk_id"] for hit in hits],
                "scores": [hit["score"] for hit in hits],
                "match_types": [hit["match_type"] for hit in hits],
                "hits": hits,
                "relevant_hits": relevant_hits,
                "evidence_found": evidence_found,
            }
            per_channel[name].append(record)
            channel_hits[name] = hits
            channel_times[name] = record["latency_ms"]
            logger.info(
                "retrieval_benchmark query=%r scope=%s channel=%s latency_ms=%.3f "
                "chunk_ids=%s scores=%s match_types=%s evidence_found=%s",
                query,
                json.dumps(scope, sort_keys=True),
                name,
                latency_ms,
                record["chunk_ids"],
                record["scores"],
                record["match_types"],
                evidence_found,
            )
        for lexical_weight, sweep_records in rrf_sweep_records.items():
            sweep_hits = _rrf_weighted_hits(
                channel_candidates["vector"],
                channel_candidates["fts5"],
                lexical_weight,
                k,
            )
            sweep_records.append({
                "id": question["id"],
                "relevant_hits": sum(
                    _is_relevant(hit, question) for hit in sweep_hits
                ),
                "evidence_found": any(
                    _is_relevant(hit, question) for hit in sweep_hits
                ),
            })
        rerank_started = time.perf_counter()
        reranked, _ = rerank_chunks(
            query,
            [result.model_dump() for result in hybrid_candidates],
            top_k=k,
        )
        rerank_latency_ms = (time.perf_counter() - rerank_started) * 1000
        reranked_hits = [_reranked_hit(result) for result in reranked]
        reranked_record = {
            "id": question["id"],
            "query": query,
            "scope": scope,
            "latency_ms": round(channel_times["hybrid"] + rerank_latency_ms, 3),
            "rerank_latency_ms": round(rerank_latency_ms, 3),
            "chunk_ids": [hit["chunk_id"] for hit in reranked_hits],
            "scores": [hit["score"] for hit in reranked_hits],
            "match_types": [hit["match_type"] for hit in reranked_hits],
            "hits": reranked_hits,
            "relevant_hits": sum(_is_relevant(hit, question) for hit in reranked_hits),
            "evidence_found": any(_is_relevant(hit, question) for hit in reranked_hits),
        }
        per_channel["reranked"].append(reranked_record)
        channel_hits["reranked"] = reranked_hits
        channel_times["reranked"] = reranked_record["latency_ms"]
        logger.info(
            "retrieval_benchmark query=%r scope=%s channel=reranked "
            "latency_ms=%.3f rerank_latency_ms=%.3f chunk_ids=%s scores=%s "
            "match_types=%s evidence_found=%s",
            query,
            json.dumps(scope, sort_keys=True),
            reranked_record["latency_ms"],
            rerank_latency_ms,
            reranked_record["chunk_ids"],
            reranked_record["scores"],
            reranked_record["match_types"],
            reranked_record["evidence_found"],
        )
        query_records.append({
            "id": question["id"],
            "query": query,
            "scope": scope,
            "rerank_latency_ms": reranked_record["rerank_latency_ms"],
            "channels": {
                name: {
                    "latency_ms": channel_times[name],
                    "chunk_ids": [hit["chunk_id"] for hit in channel_hits[name]],
                    "scores": [hit["score"] for hit in channel_hits[name]],
                    "match_types": [hit["match_type"] for hit in channel_hits[name]],
                    "hits": channel_hits[name],
                    "relevant_hits": sum(
                        _is_relevant(hit, question) for hit in channel_hits[name]
                    ),
                    "evidence_found": any(
                        _is_relevant(hit, question) for hit in channel_hits[name]
                    ),
                }
                for name in channel_hits
            },
        })

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "questions_file": questions_path.name,
        "k": k,
        "scope": "all indexed local documents; expected source/page recorded per question",
        "index_consistency": _index_consistency_report(),
        "summary": {
            name: _summarize(records, k)
            for name, records in per_channel.items()
        },
        "rrf_weight_sweep": {
            str(weight): {
                "queries": len(records),
                "hit_at_k": round(
                    sum(record["evidence_found"] for record in records)
                    / max(len(records), 1),
                    4,
                ),
                "precision_at_k": round(
                    sum(record["relevant_hits"] for record in records)
                    / max(len(records) * k, 1),
                    4,
                ),
            }
            for weight, records in rrf_sweep_records.items()
        },
        "queries": query_records,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_benchmark(args.questions, args.output)
    print(json.dumps(result["summary"], indent=2))
    print(f"Detailed results: {args.output}")


if __name__ == "__main__":
    main()
