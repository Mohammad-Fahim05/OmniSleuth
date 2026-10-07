"""Formal evaluation metrics for contextual RAG retrieval strategies."""

import math
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from omnisleuth.core.logging import get_logger
from omnisleuth.rag.benchmark_dataset import (
    RETRIEVAL_BENCHMARK_CASES,
    RetrievalBenchmarkCase,
)
from omnisleuth.rag.document_store import DocumentStore
from omnisleuth.rag.models import RetrievalMode

logger = get_logger("rag.evaluator")


def compute_recall_at_k(retrieved_chunk_ids: List[str], relevant_chunk_ids: List[str], k: int) -> float:
    """Compute binary Recall@K: 1.0 if at least one relevant chunk is within top-k, else 0.0."""
    if not relevant_chunk_ids:
        return 1.0
    top_k_retrieved = set(retrieved_chunk_ids[:k])
    for rel_id in relevant_chunk_ids:
        if rel_id in top_k_retrieved:
            return 1.0
    return 0.0


def compute_mrr(retrieved_chunk_ids: List[str], relevant_chunk_ids: List[str]) -> float:
    """Compute Mean Reciprocal Rank (MRR) for an individual query."""
    if not relevant_chunk_ids:
        return 1.0
    rel_set = set(relevant_chunk_ids)
    for rank_idx, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in rel_set:
            return 1.0 / rank_idx
    return 0.0


def compute_ndcg_at_k(retrieved_chunk_ids: List[str], relevant_chunk_ids: List[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain (nDCG@K) using standard binary relevance."""
    if not relevant_chunk_ids:
        return 1.0
    rel_set = set(relevant_chunk_ids)

    dcg = 0.0
    for idx, chunk_id in enumerate(retrieved_chunk_ids[:k], start=1):
        rel = 1.0 if chunk_id in rel_set else 0.0
        dcg += rel / math.log2(idx + 1.0)

    # Ideal DCG with all relevant items at the top
    idcg = 0.0
    num_ideal = min(len(relevant_chunk_ids), k)
    for idx in range(1, num_ideal + 1):
        idcg += 1.0 / math.log2(idx + 1.0)

    if idcg == 0.0:
        return 0.0
    return round(dcg / idcg, 4)


class StrategyEvaluationResult(BaseModel):
    """Aggregate benchmark results for a specific retrieval strategy."""

    mode: RetrievalMode
    total_queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    mean_latency_ms: float
    p95_latency_ms: float


class RetrievalBenchmarkReport(BaseModel):
    """Full comparative benchmark report across all 4 retrieval modes."""

    strategies: Dict[str, StrategyEvaluationResult]
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


class RetrievalEvaluator:
    """Evaluates and compares BM25, Dense, Hybrid, and Reranked retrieval pipelines."""

    def __init__(
        self,
        document_store: Optional[DocumentStore] = None,
        cases: Optional[List[RetrievalBenchmarkCase]] = None,
    ):
        self.doc_store = document_store or DocumentStore()
        self.cases = cases or RETRIEVAL_BENCHMARK_CASES

    def evaluate_mode(
        self,
        mode: RetrievalMode,
        top_k: int = 5,
    ) -> StrategyEvaluationResult:
        """Run benchmark evaluation for a single retrieval mode."""
        r1_list: List[float] = []
        r3_list: List[float] = []
        r5_list: List[float] = []
        mrr_list: List[float] = []
        ndcg5_list: List[float] = []
        latencies: List[float] = []

        for case in self.cases:
            res = self.doc_store.retrieve(
                query=case.query,
                mode=mode,
                top_k=top_k,
                min_score=0.0,
            )
            latencies.append(res.latency_ms)
            retrieved_ids = [c.chunk_id for c in res.chunks]

            r1_list.append(compute_recall_at_k(retrieved_ids, case.relevant_chunk_ids, k=1))
            r3_list.append(compute_recall_at_k(retrieved_ids, case.relevant_chunk_ids, k=3))
            r5_list.append(compute_recall_at_k(retrieved_ids, case.relevant_chunk_ids, k=5))
            mrr_list.append(compute_mrr(retrieved_ids, case.relevant_chunk_ids))
            ndcg5_list.append(compute_ndcg_at_k(retrieved_ids, case.relevant_chunk_ids, k=5))

        n = len(self.cases) if self.cases else 1
        latencies_sorted = sorted(latencies)
        p95_idx = int(0.95 * len(latencies_sorted))
        p95_lat = latencies_sorted[min(p95_idx, len(latencies_sorted) - 1)] if latencies_sorted else 0.0

        return StrategyEvaluationResult(
            mode=mode,
            total_queries=len(self.cases),
            recall_at_1=round(sum(r1_list) / n, 4),
            recall_at_3=round(sum(r3_list) / n, 4),
            recall_at_5=round(sum(r5_list) / n, 4),
            mrr=round(sum(mrr_list) / n, 4),
            ndcg_at_5=round(sum(ndcg5_list) / n, 4),
            mean_latency_ms=round(sum(latencies) / n, 2),
            p95_latency_ms=round(p95_lat, 2),
        )

    def run_comparative_benchmark(self, top_k: int = 5) -> RetrievalBenchmarkReport:
        """Run full evaluation comparing BM25, DENSE, HYBRID, and RERANKED strategies."""
        logger.info(f"Running comparative retrieval benchmark across {len(self.cases)} cases...")
        results = {}

        for mode in [RetrievalMode.BM25, RetrievalMode.DENSE, RetrievalMode.HYBRID, RetrievalMode.RERANKED]:
            eval_res = self.evaluate_mode(mode=mode, top_k=top_k)
            results[mode.value] = eval_res
            logger.info(
                f"[{mode.value.upper()}] Recall@5={eval_res.recall_at_5:.2f}, "
                f"MRR={eval_res.mrr:.4f}, nDCG@5={eval_res.ndcg_at_5:.4f}, Latency={eval_res.mean_latency_ms:.1f}ms"
            )

        return RetrievalBenchmarkReport(strategies=results)
