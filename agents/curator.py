"""Deterministic evidence curation for retrieved local-document chunks."""

from __future__ import annotations

import math
import re
from typing import Any
import logging

from config import ENABLE_CROSS_ENCODER, MIN_EVIDENCE_SCORE, RERANKER_MODEL
from . import retriever_worker, writer

logger = logging.getLogger(__name__)
_cross_encoder: Any | None = None
_cross_encoder_failed = False


class CuratorAnswerGenerationError(RuntimeError):
    """Raised when the grounded answer writer cannot generate an answer."""


def has_sufficient_evidence(chunks: list[dict[str, Any]]) -> bool:
    """Require at least one retained passage to meet the relevance threshold."""
    return any(
        float(chunk.get("rerank_score", 0.0)) >= MIN_EVIDENCE_SCORE
        for chunk in chunks
    )


def _calculation_evidence(calculation: Any | None) -> dict[str, Any] | None:
    if calculation is None:
        return None
    return {
        "operation": calculation.operation,
        "column": calculation.column,
        "input_values": [str(value) for value in calculation.values],
        "result": str(calculation.value),
        "source_count": calculation.source_count,
    }


def build_citation_metadata(
    citations: list[int],
    answer: str,
    context_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map internal citation numbers to document metadata and supporting claims."""
    metadata: list[dict[str, Any]] = []
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    for citation in dict.fromkeys(citations):
        if citation < 1 or citation > len(context_blocks):
            continue
        chunk = context_blocks[citation - 1]
        source = str(chunk.get("source_path") or chunk.get("title") or "Unknown document")
        document = source.replace("\\", "/").rsplit("/", 1)[-1]
        citation_pattern = re.compile(rf"[\[【]{citation}(?:\s*[,;，][^\]】]*)?[\]】]")
        supporting_claim = next(
            (
                citation_pattern.sub("", sentence).strip().rstrip(" .;")
                for sentence in sentences
                if citation_pattern.search(sentence)
            ),
            "",
        )
        item: dict[str, Any] = {
            "citation": citation,
            "document": document,
            "page": chunk.get("page_number"),
            "chunk_id": chunk.get("chunk_id"),
            "supporting_claim": supporting_claim,
        }
        metadata.append(item)
    return metadata


def curate_and_answer(
    question: str,
    chunks: list[dict[str, Any]],
    *,
    intent: str = "factual",
    request_intent: str | None = None,
    output_format: str = "short_answer",
    revision_feedback: str | None = None,
    history: list[dict[str, Any]] | None = None,
    calculation: Any | None = None,
    visual_evidence: list[dict[str, Any]] | None = None,
    retrieval_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rerank evidence, generate a grounded answer, and attach provenance."""
    table_request = intent == "table" or request_intent == "numeric"
    top_k = 12 if table_request else 8
    ranked_chunks, confidence = rerank_chunks(question, chunks, top_k=top_k)
    context_blocks = (
        expand_chunks(
            ranked_chunks,
            max_chars=20000 if table_request else 12000,
            max_chunks=12 if table_request else 8,
        )
        if ranked_chunks
        else []
    )
    evidence_sufficient = has_sufficient_evidence(ranked_chunks)
    status = retrieval_status or {}
    limitations = list(dict.fromkeys(
        str(item).strip()
        for item in status.get("limitations", [])
        if str(item).strip()
    ))

    if not evidence_sufficient:
        if not limitations:
            limitations.append("Retrieved passages did not meet the evidence threshold.")
        return {
            "answer": retriever_worker.NO_EVIDENCE_ANSWER,
            "raw_answer": retriever_worker.NO_EVIDENCE_ANSWER,
            "draft": retriever_worker.NO_EVIDENCE_ANSWER,
            "citations": [],
            "citation_metadata": [],
            "context_blocks": context_blocks,
            "answerable": False,
            "evidence_sufficient": False,
            "limitations": limitations,
            "rerank_confidence": confidence,
            "decision_log": [{
                "stage": "curator",
                "decision": "abstain",
                "reason": "insufficient_evidence",
            }],
        }

    retrieval: dict[str, Any] = {
        "chunks": context_blocks,
        "ranked_chunks": ranked_chunks,
        "context_blocks": context_blocks,
        "intent": request_intent or intent,
        "rerank_confidence": confidence,
        "history": history or [],
        "visual_evidence": visual_evidence or [],
    }
    calculated_evidence = _calculation_evidence(calculation)
    if calculated_evidence is not None:
        retrieval["calculated_evidence"] = calculated_evidence

    try:
        generated = writer.answer_from_retrieval(
            question,
            retrieval,
            revision_feedback=revision_feedback,
            output_format=output_format,
            return_details=True,
        )
    except Exception as exc:
        raise CuratorAnswerGenerationError("Grounded answer generation failed.") from exc

    if isinstance(generated, str):
        raw_answer = generated
        answer = generated
        citations = writer.extract_citations(raw_answer)
    else:
        raw_answer = str(generated.get("raw_answer") or generated.get("answer") or "")
        answer = str(generated.get("answer") or raw_answer)
        citations = list(generated.get("citations") or writer.extract_citations(raw_answer))
    context_blocks = retrieval.get("context_blocks") or context_blocks
    if not raw_answer.strip():
        raise CuratorAnswerGenerationError("The answer writer returned an empty response.")

    return {
        "answer": answer,
        "raw_answer": raw_answer,
        "draft": raw_answer,
        "citations": citations,
        "citation_metadata": build_citation_metadata(citations, raw_answer, context_blocks),
        "context_blocks": context_blocks,
        "answerable": raw_answer != retriever_worker.NO_EVIDENCE_ANSWER,
        "evidence_sufficient": evidence_sufficient and bool(status.get("sufficient", True)),
        "limitations": limitations,
        "rerank_confidence": confidence,
        "decision_log": [{
            "stage": "curator",
            "decision": "answer",
            "context_chunks": len(context_blocks),
            "evidence_sufficient": evidence_sufficient,
        }],
    }


def _get_cross_encoder() -> Any | None:
    """Load the optional cross-encoder once, falling back on load failure."""
    global _cross_encoder, _cross_encoder_failed
    if not ENABLE_CROSS_ENCODER or _cross_encoder_failed:
        return None
    if _cross_encoder is not None:
        return _cross_encoder
    try:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(RERANKER_MODEL)
    except Exception as exc:
        _cross_encoder_failed = True
        logger.warning("Cross-encoder unavailable; using hybrid reranking: %s", exc)
    return _cross_encoder


def _dedupe_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            continue
        score = float(
            chunk.get("hybrid_score")
            if chunk.get("hybrid_score") is not None
            else 1.0 - float(chunk.get("distance", 1.0))
        )
        previous = best.get(chunk_id)
        if previous is None or score > float(previous.get("_merge_score", -1.0)):
            item = dict(chunk)
            item["_merge_score"] = score
            best[chunk_id] = item
    return [
        {key: value for key, value in chunk.items() if key != "_merge_score"}
        for chunk in best.values()
    ]


def rerank_chunks(
    question: str,
    chunks: list[dict[str, Any]],
    *,
    top_k: int = 8,
) -> tuple[list[dict[str, Any]], str]:
    """Rerank retrieved chunks without making an LLM call."""
    deduped = _dedupe_chunks(chunks)
    if not deduped:
        return [], "none"
    encoder = _get_cross_encoder()
    if encoder is None:
        return retriever_worker.rerank_chunks(question, deduped, top_k=top_k)
    try:
        scores = encoder.predict([(question, chunk["text"]) for chunk in deduped])
        scored = []
        for chunk, score in zip(deduped, scores):
            item = dict(chunk)
            raw_score = float(score)
            probability = (
                1.0 / (1.0 + math.exp(-raw_score))
                if raw_score >= 0
                else math.exp(raw_score) / (1.0 + math.exp(raw_score))
            )
            item["cross_encoder_score"] = round(raw_score, 6)
            item["rerank_score"] = round(probability, 6)
            scored.append(item)
        scored.sort(key=lambda item: item["rerank_score"], reverse=True)
        ranked = [
            item for item in scored
            if item["rerank_score"] >= MIN_EVIDENCE_SCORE
        ][:top_k]
        confidence = "high" if ranked and ranked[0]["rerank_score"] >= 0.45 else "low"
        return ranked, confidence if ranked else "none"
    except Exception as exc:
        logger.warning("Cross-encoder reranking failed; using hybrid reranking: %s", exc)
        return retriever_worker.rerank_chunks(question, deduped, top_k=top_k)


def expand_chunks(
    ranked_chunks: list[dict[str, Any]],
    *,
    window: int = 1,
    max_chars: int = 12000,
    max_chunks: int = 8,
) -> list[dict[str, Any]]:
    """Add adjacent chunks while preserving the existing evidence ordering."""
    return retriever_worker.expand_chunks(
        ranked_chunks,
        window=window,
        max_chars=max_chars,
        max_chunks=max_chunks,
    )


get_neighbor_chunks = retriever_worker.get_neighbor_chunks
