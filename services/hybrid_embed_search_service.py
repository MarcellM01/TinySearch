from __future__ import annotations

import asyncio
import inspect
import math
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from rank_bm25 import BM25Okapi

EmbeddingVector = Sequence[float]
EmbeddingFn = Callable[
    [list[str]],
    Awaitable[Sequence[EmbeddingVector]] | Sequence[EmbeddingVector],
]


def _tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_./:#-]+", text.lower())


def tokenize_for_retrieval(text: str) -> list[str]:
    """Same tokenization as BM25 / hybrid chunk ranking (for dedupe and scoring)."""
    return _tokenize_for_bm25(text)


def _chunk_text(chunk: dict[str, Any]) -> str:
    return str(chunk.get("text") or "")


def _cosine_similarity(left: EmbeddingVector, right: EmbeddingVector) -> float:
    if len(left) != len(right) or not left:
        return 0.0

    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        dot += float(left_value) * float(right_value)
        left_norm += float(left_value) * float(left_value)
        right_norm += float(right_value) * float(right_value)

    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0

    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def _rank_by_score(scores: Sequence[float]) -> dict[int, int]:
    return {
        idx: rank
        for rank, idx in enumerate(
            sorted(range(len(scores)), key=lambda item: scores[item], reverse=True),
            start=1,
        )
    }


def _source_url(chunk: dict[str, Any]) -> str:
    return str(chunk.get("source_url") or "")


def _select_dense_candidate_indices(
    bm25_scores: Sequence[float],
    chunks: Sequence[dict[str, Any]],
    *,
    per_source: int | None,
    max_total: int | None,
) -> list[int]:
    if per_source is None or per_source <= 0:
        return list(range(len(chunks)))
    if max_total is not None and max_total <= 0:
        return list(range(len(chunks)))
    if max_total is not None and len(chunks) <= max_total:
        return list(range(len(chunks)))

    grouped: dict[str, list[int]] = {}
    for idx, chunk in enumerate(chunks):
        grouped.setdefault(_source_url(chunk), []).append(idx)

    selected: set[int] = set()
    for indices in grouped.values():
        ranked_source = sorted(
            indices,
            key=lambda idx: (float(bm25_scores[idx]), -idx),
            reverse=True,
        )
        selected.update(ranked_source[:per_source])

    selected_indices = sorted(
        selected,
        key=lambda idx: (float(bm25_scores[idx]), -idx),
        reverse=True,
    )
    if max_total is not None:
        selected_indices = selected_indices[:max_total]
    return sorted(selected_indices)


async def _call_embedder(embedder: EmbeddingFn, inputs: list[str]) -> list[list[float]]:
    embeddings = embedder(inputs)
    if inspect.isawaitable(embeddings):
        embeddings = await embeddings
    return [list(vector) for vector in embeddings]


async def _embed_inputs(
    inputs: list[str],
    *,
    embedder: EmbeddingFn,
    semaphore: "Any",
    timeout_seconds: float,
    max_timeout_retries: int,
) -> list[list[float]]:
    async def call_embedder() -> list[list[float]]:
        if semaphore is None:
            return await asyncio.wait_for(
                _call_embedder(embedder, inputs),
                timeout=timeout_seconds,
            )
        async with semaphore:
            return await asyncio.wait_for(
                _call_embedder(embedder, inputs),
                timeout=timeout_seconds,
            )

    attempts = max(0, max_timeout_retries) + 1
    for attempt in range(1, attempts + 1):
        try:
            return await call_embedder()
        except TimeoutError:
            if attempt >= attempts:
                raise

    raise RuntimeError("unreachable embedding retry state")


async def _embed_query_and_document_chunks(
    query: str,
    chunk_texts: list[str],
    *,
    dense_query_prefix: str,
    dense_document_prefix: str,
    embedder: EmbeddingFn,
    semaphore: Any,
    timeout_seconds: float,
    max_timeout_retries: int,
    document_embed_batch_size: int | None,
) -> tuple[list[float], list[list[float]]]:
    query_inputs = [f"{dense_query_prefix}{query}"]
    query_embeddings = await _embed_inputs(
        query_inputs,
        embedder=embedder,
        semaphore=semaphore,
        timeout_seconds=timeout_seconds,
        max_timeout_retries=max_timeout_retries,
    )
    if len(query_embeddings) != 1:
        raise ValueError(
            f"embedder returned {len(query_embeddings)} embeddings for 1 query input"
        )
    query_embedding = query_embeddings[0]

    if not chunk_texts:
        return query_embedding, []

    doc_embeddings: list[list[float]] = []
    batch_size = document_embed_batch_size
    if batch_size is None or batch_size <= 0:
        doc_inputs = [f"{dense_document_prefix}{text}" for text in chunk_texts]
        doc_embeddings = await _embed_inputs(
            doc_inputs,
            embedder=embedder,
            semaphore=semaphore,
            timeout_seconds=timeout_seconds,
            max_timeout_retries=max_timeout_retries,
        )
    else:
        for start in range(0, len(chunk_texts), batch_size):
            batch_texts = chunk_texts[start : start + batch_size]
            doc_inputs = [f"{dense_document_prefix}{text}" for text in batch_texts]
            batch_emb = await _embed_inputs(
                doc_inputs,
                embedder=embedder,
                semaphore=semaphore,
                timeout_seconds=timeout_seconds,
                max_timeout_retries=max_timeout_retries,
            )
            doc_embeddings.extend(batch_emb)

    if len(doc_embeddings) != len(chunk_texts):
        raise ValueError(
            f"embedder returned {len(doc_embeddings)} document embeddings for "
            f"{len(chunk_texts)} chunks"
        )

    return query_embedding, doc_embeddings


async def rank_chunks_hybrid(
    query: str,
    chunks: Sequence[dict[str, Any]],
    *,
    embedder: EmbeddingFn | None = None,
    top_k: int | None = None,
    rrf_similarity_cutoff: float | None = None,
    hybrid_similarity_cutoff: float | None = None,
    dense_weight: float = 0.5,
    dense_query_prefix: str = "task: search result | query: ",
    dense_document_prefix: str = "title: none | text: ",
    dense_document_embed_batch_size: int | None = 32,
    dense_bm25_prefilter_per_source: int | None = None,
    dense_bm25_prefilter_max_total: int | None = None,
    rrf_k: int = 60,
    semaphore: "Any" = None,
    timeout_seconds: float = 60.0,
    max_timeout_retries: int = 2,
) -> list[dict[str, Any]]:
    """
    Rank text chunks for a query using weighted BM25 + dense RRF.

    ``dense_weight`` controls the dense side, while BM25 gets ``1 - dense_weight``.
    Dense inputs are prefixed separately for asymmetric retrieval embedding models.
    ``dense_document_embed_batch_size`` embeds the query once, then documents in
    sub-batches (default 32). Use ``None`` or ``<= 0`` to embed all documents in
    one call (legacy behavior, larger peak memory).
    ``dense_bm25_prefilter_per_source`` and ``dense_bm25_prefilter_max_total``
    limit dense document embedding to a BM25-selected shortlist while preserving
    per-source representation. Use ``None`` or ``<= 0`` to disable prefiltering.
    The returned ``rrf_similarity`` is normalized to 0..1, which makes cutoffs
    easier to reason about. ``hybrid_similarity`` is kept as a compatibility alias.
    """
    if not query or not chunks:
        return []
    if rrf_k < 0:
        raise ValueError("rrf_k must be >= 0")
    if dense_weight <= 0.0 or dense_weight > 1.0:
        raise ValueError("dense_weight must be greater than 0 and at most 1")
    if embedder is None:
        raise ValueError("embedder is required")

    chunk_list = list(chunks)
    chunk_texts = [_chunk_text(chunk) for chunk in chunk_list]
    sparse_weight = 1.0 - dense_weight

    bm25_scores = [0.0 for _ in chunk_list]
    bm25_ranks = {idx: 1 for idx in range(len(chunk_list))}
    if sparse_weight > 0.0:
        corpus = [_tokenize_for_bm25(text) for text in chunk_texts]
        query_tokens = _tokenize_for_bm25(query)
        bm25_scores = (
            [float(score) for score in BM25Okapi(corpus).get_scores(query_tokens)]
            if query_tokens and any(corpus)
            else [0.0 for _ in chunk_list]
        )
        bm25_ranks = _rank_by_score(bm25_scores)

    dense_scores = [0.0 for _ in chunk_list]
    dense_ranks = {idx: len(chunk_list) + 1 for idx in range(len(chunk_list))}
    dense_candidate_indices = list(range(len(chunk_list)))
    if dense_weight > 0.0:
        dense_candidate_indices = _select_dense_candidate_indices(
            bm25_scores,
            chunk_list,
            per_source=dense_bm25_prefilter_per_source,
            max_total=dense_bm25_prefilter_max_total,
        )
        dense_candidate_texts = [chunk_texts[idx] for idx in dense_candidate_indices]
        query_embedding, doc_embeddings = await _embed_query_and_document_chunks(
            query,
            dense_candidate_texts,
            dense_query_prefix=dense_query_prefix,
            dense_document_prefix=dense_document_prefix,
            embedder=embedder,
            semaphore=semaphore,
            timeout_seconds=timeout_seconds,
            max_timeout_retries=max_timeout_retries,
            document_embed_batch_size=dense_document_embed_batch_size,
        )
        candidate_dense_scores = [
            _cosine_similarity(query_embedding, chunk_embedding)
            for chunk_embedding in doc_embeddings
        ]
        for idx, score in zip(dense_candidate_indices, candidate_dense_scores, strict=True):
            dense_scores[idx] = score
        candidate_dense_ranks = _rank_by_score(candidate_dense_scores)
        dense_ranks = {
            idx: candidate_dense_ranks[candidate_idx]
            for candidate_idx, idx in enumerate(dense_candidate_indices)
        }
        fallback_rank = len(dense_candidate_indices) + 1
        for idx in range(len(chunk_list)):
            dense_ranks.setdefault(idx, fallback_rank)

    cutoff = rrf_similarity_cutoff
    if cutoff is None:
        cutoff = hybrid_similarity_cutoff
    max_rrf_score = (sparse_weight + dense_weight) / (rrf_k + 1)

    ranked: list[dict[str, Any]] = []
    dense_candidate_set = set(dense_candidate_indices)
    for idx, chunk in enumerate(chunk_list):
        bm25_rank = bm25_ranks[idx]
        dense_rank = dense_ranks[idx]
        rrf_score = (
            (sparse_weight / (rrf_k + bm25_rank) if sparse_weight > 0.0 else 0.0)
            + (dense_weight / (rrf_k + dense_rank) if dense_weight > 0.0 else 0.0)
        )
        rrf_similarity = rrf_score / max_rrf_score if max_rrf_score else 0.0

        if cutoff is not None and rrf_similarity < cutoff:
            continue

        ranked.append(
            {
                **chunk,
                "bm25_score": float(bm25_scores[idx]),
                "bm25_rank": bm25_rank,
                "dense_score": float(dense_scores[idx]),
                "dense_rank": dense_rank,
                "dense_candidate": idx in dense_candidate_set,
                "rrf_score": float(rrf_score),
                "rrf_similarity": float(rrf_similarity),
                "hybrid_similarity": float(rrf_similarity),
            }
        )

    ranked.sort(
        key=lambda item: (
            item["rrf_score"],
            item["dense_score"],
            item["bm25_score"],
        ),
        reverse=True,
    )

    if top_k is not None:
        ranked = ranked[: max(0, top_k)]

    return ranked
