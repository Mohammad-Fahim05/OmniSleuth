"""Contextual RAG document ingestion and retrieval module."""

from omnisleuth.rag.bm25 import BM25Retriever
from omnisleuth.rag.dense_embedder import DenseEmbedder
from omnisleuth.rag.document_store import DocumentStore
from omnisleuth.rag.evaluator import (
    RetrievalBenchmarkReport,
    RetrievalEvaluator,
    StrategyEvaluationResult,
    compute_mrr,
    compute_ndcg_at_k,
    compute_recall_at_k,
)
from omnisleuth.rag.faiss_index import FAISSIndex
from omnisleuth.rag.fusion import reciprocal_rank_fusion
from omnisleuth.rag.models import (
    DocumentChunk,
    DocumentMetadata,
    DocumentType,
    RetrievalMode,
    RetrievalResult,
)
from omnisleuth.rag.reranker import SemanticReranker
from omnisleuth.rag.seed_docs import SEED_DOCUMENTS

__all__ = [
    "DocumentStore",
    "DocumentChunk",
    "DocumentMetadata",
    "DocumentType",
    "RetrievalMode",
    "RetrievalResult",
    "SEED_DOCUMENTS",
    "BM25Retriever",
    "DenseEmbedder",
    "FAISSIndex",
    "reciprocal_rank_fusion",
    "SemanticReranker",
    "RetrievalEvaluator",
    "RetrievalBenchmarkReport",
    "StrategyEvaluationResult",
    "compute_recall_at_k",
    "compute_mrr",
    "compute_ndcg_at_k",
]
