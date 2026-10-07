"""Reciprocal Rank Fusion (RRF) algorithm for combining lexical and semantic rankings."""

from typing import Dict, List, Optional, Tuple
from omnisleuth.core.logging import get_logger
from omnisleuth.rag.models import DocumentChunk

logger = get_logger("rag.fusion")


def reciprocal_rank_fusion(
    bm25_results: List[Tuple[float, DocumentChunk]],
    dense_results: List[Tuple[float, DocumentChunk]],
    k: int = 60,
    top_k: Optional[int] = None,
) -> List[DocumentChunk]:
    """Combine ranked candidates from BM25 and Dense vector search using Reciprocal Rank Fusion.

    Args:
        bm25_results: List of (score, DocumentChunk) ranked from BM25.
        dense_results: List of (score, DocumentChunk) ranked from dense vector search.
        k: Smoothing constant for RRF (default 60).
        top_k: Maximum number of fused candidates to return.

    Returns:
        List of DocumentChunk instances with populated rank, bm25_rank, dense_rank, and rrf_score.
    """
    fused_scores: Dict[str, float] = {}
    chunk_map: Dict[str, DocumentChunk] = {}
    bm25_ranks: Dict[str, int] = {}
    bm25_scores: Dict[str, float] = {}
    dense_ranks: Dict[str, int] = {}
    dense_scores: Dict[str, float] = {}

    # 1. Process BM25 ranked list
    for rank_idx, (score, chunk) in enumerate(bm25_results, start=1):
        c_id = chunk.chunk_id
        chunk_map[c_id] = chunk
        bm25_ranks[c_id] = rank_idx
        bm25_scores[c_id] = score
        rrf_contrib = 1.0 / (k + rank_idx)
        fused_scores[c_id] = fused_scores.get(c_id, 0.0) + rrf_contrib

    # 2. Process Dense ranked list
    for rank_idx, (score, chunk) in enumerate(dense_results, start=1):
        c_id = chunk.chunk_id
        if c_id not in chunk_map:
            chunk_map[c_id] = chunk
        dense_ranks[c_id] = rank_idx
        dense_scores[c_id] = score
        rrf_contrib = 1.0 / (k + rank_idx)
        fused_scores[c_id] = fused_scores.get(c_id, 0.0) + rrf_contrib

    if not fused_scores:
        return []

    # 3. Sort chunks by fused RRF score descending, breaking ties by chunk_id
    sorted_items = sorted(
        fused_scores.items(),
        key=lambda item: (item[1], dense_scores.get(item[0], 0.0), item[0]),
        reverse=True,
    )

    if top_k is not None:
        sorted_items = sorted_items[:top_k]

    fused_chunks: List[DocumentChunk] = []
    max_rrf = sorted_items[0][1] if sorted_items else 1.0

    for final_rank, (c_id, rrf_val) in enumerate(sorted_items, start=1):
        base_chunk = chunk_map[c_id]
        chunk_copy = base_chunk.model_copy()
        chunk_copy.rank = final_rank
        chunk_copy.rrf_score = round(rrf_val, 6)
        chunk_copy.bm25_rank = bm25_ranks.get(c_id)
        chunk_copy.bm25_score = bm25_scores.get(c_id)
        chunk_copy.dense_rank = dense_ranks.get(c_id)
        chunk_copy.dense_score = dense_scores.get(c_id)

        # Scale similarity score relative to top RRF candidate
        normalized_sim = round(min(1.0, rrf_val / (max_rrf or 1.0)), 4)
        chunk_copy.similarity_score = normalized_sim

        fused_chunks.append(chunk_copy)

    logger.debug(f"Fused {len(bm25_results)} BM25 and {len(dense_results)} Dense results into {len(fused_chunks)} candidates.")
    return fused_chunks
