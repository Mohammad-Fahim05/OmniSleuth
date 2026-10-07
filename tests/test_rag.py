"""Unit and integration tests for Phase 9 & Phase 10 - Contextual & Hybrid RAG Subsystem."""

from pathlib import Path
import numpy as np
import pytest
from omnisleuth.agents.models import AgentRole, AgentTask
from omnisleuth.agents.retriever import RetrievalAgent
from omnisleuth.evidence.store import EvidenceStore
from omnisleuth.rag import (
    BM25Retriever,
    DenseEmbedder,
    DocumentChunk,
    DocumentStore,
    DocumentType,
    FAISSIndex,
    RetrievalBenchmarkReport,
    RetrievalEvaluator,
    RetrievalMode,
    RetrievalResult,
    SemanticReranker,
    compute_mrr,
    compute_ndcg_at_k,
    compute_recall_at_k,
    reciprocal_rank_fusion,
)
from omnisleuth.rag.benchmark_dataset import RETRIEVAL_BENCHMARK_CASES


# =========================================================================
# Phase 9 Baseline Tests (Preserved & Backward Compatible)
# =========================================================================

def test_seed_documents_loading():
    """Verify default knowledge documents are loaded on initialization."""
    store = DocumentStore()
    assert len(store.documents) == 4
    assert "doc_data_dict" in store.documents
    assert "doc_kpi_defs" in store.documents
    assert "doc_q1_report" in store.documents
    assert "doc_biz_rules" in store.documents
    assert len(store.chunks) >= 4


def test_retrieve_operational_context_for_march_decline():
    """Test retrieval of Q1 operational supplier outage for March Electronics decline."""
    store = DocumentStore()
    query = "March electronics supplier outage North region inventory stockout"
    result = store.retrieve(query=query, top_k=2)

    assert result.has_relevant_evidence is True
    assert len(result.chunks) > 0
    top_chunk = result.chunks[0]
    assert "supplier shutdown" in top_chunk.content or "grid outages" in top_chunk.content
    assert top_chunk.doc_type == DocumentType.QUARTERLY_REPORT
    assert "Q1 2024 Quarterly Business Review" in top_chunk.citation


def test_retrieve_kpi_definition():
    """Test retrieval of metric definition for AOV."""
    store = DocumentStore()
    query = "Average Order Value AOV definition formula"
    result = store.retrieve(query=query, doc_type_filter=DocumentType.KPI_DEFINITION)

    assert result.has_relevant_evidence is True
    top_chunk = result.chunks[0]
    assert "Average Order Value" in top_chunk.content
    assert "Revenue / Order Volume" in top_chunk.content


def test_retrieve_irrelevant_query():
    """Test that unrelated query explicitly reports lack of evidence."""
    store = DocumentStore()
    query = "Quantum entanglement teleportation particle astrophysics"
    result = store.retrieve(query=query, mode=RetrievalMode.BM25, min_score=0.2)

    assert result.has_relevant_evidence is False
    assert len(result.chunks) == 0
    assert "No relevant contextual document evidence found" in result.retrieval_notes


def test_add_custom_markdown_document(tmp_path: Path):
    """Test ingesting custom policy document."""
    store = DocumentStore()
    doc_text = """# Regional Shipping Policy 2024
## Winter Weather Protocols
During extreme winter weather storms in January and February, North regional transit hubs may operate at reduced capacity.
"""
    meta = store.add_document_text(
        doc_id="doc_shipping_policy",
        title="Regional Shipping Policy",
        doc_type=DocumentType.COMPANY_POLICY,
        text=doc_text,
    )
    assert meta.doc_id == "doc_shipping_policy"
    assert meta.total_chunks >= 1

    res = store.retrieve("winter weather transit hub shipping", mode=RetrievalMode.BM25)
    assert res.has_relevant_evidence is True
    assert "Winter Weather Protocols" in res.chunks[0].content


# =========================================================================
# Phase 10 - BM25, Dense, FAISS, RRF, Reranker, Evaluation Tests
# =========================================================================

def test_bm25_search_scoring_and_ranking():
    """Test BM25 indexer tokenization, IDF calculations, and ranking."""
    bm25 = BM25Retriever()
    chunks = [
        DocumentChunk(
            chunk_id="chk_1",
            doc_id="d1",
            doc_title="Doc 1",
            doc_type=DocumentType.KPI_DEFINITION,
            content="Gross revenue is calculated by summing all customer order transaction values.",
        ),
        DocumentChunk(
            chunk_id="chk_2",
            doc_id="d2",
            doc_title="Doc 2",
            doc_type=DocumentType.QUARTERLY_REPORT,
            content="Microchip supplier facility shutdown caused severe logistics disruption in March.",
        ),
    ]
    bm25.index_chunks(chunks)

    results = bm25.search("revenue order transaction", top_k=2)
    assert len(results) >= 1
    assert results[0][1].chunk_id == "chk_1"
    assert results[0][1].bm25_score is not None
    assert results[0][0] > 0.0


def test_dense_embedder_and_faiss_index():
    """Test DenseEmbedder embedding dimensions, normalization, and FAISS top-k retrieval."""
    embedder = DenseEmbedder()
    texts = [
        "Financial metrics and KPI calculations for quarterly reports.",
        "Operational supplier shutdown causing component stockouts.",
    ]
    embeddings = embedder.embed_documents(texts)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == embedder.dimension

    # Verify L2 normalization
    norms = np.linalg.norm(embeddings, axis=1)
    for n in norms:
        assert np.isclose(n, 1.0, atol=1e-3)

    chunks = [
        DocumentChunk(chunk_id="c1", doc_id="d1", doc_title="T1", doc_type=DocumentType.KPI_DEFINITION, content=texts[0]),
        DocumentChunk(chunk_id="c2", doc_id="d2", doc_title="T2", doc_type=DocumentType.QUARTERLY_REPORT, content=texts[1]),
    ]
    index = FAISSIndex(dimension=embedder.dimension)
    index.rebuild(chunks, embeddings)

    q_emb = embedder.embed_query("supplier factory shutdown stockout")
    matches = index.search(q_emb, top_k=1)
    assert len(matches) == 1
    assert matches[0][1].chunk_id == "c2"
    assert matches[0][1].dense_score is not None


def test_reciprocal_rank_fusion():
    """Test RRF candidate merging, duplicate handling, and score computation."""
    c1 = DocumentChunk(chunk_id="chk_A", doc_id="d1", doc_title="A", doc_type=DocumentType.GENERAL, content="A")
    c2 = DocumentChunk(chunk_id="chk_B", doc_id="d2", doc_title="B", doc_type=DocumentType.GENERAL, content="B")
    c3 = DocumentChunk(chunk_id="chk_C", doc_id="d3", doc_title="C", doc_type=DocumentType.GENERAL, content="C")

    # BM25 ranking: A (rank 1), B (rank 2)
    bm25_list = [(3.5, c1), (2.1, c2)]
    # Dense ranking: B (rank 1), C (rank 2), A (rank 3)
    dense_list = [(0.95, c2), (0.80, c3), (0.75, c1)]

    fused = reciprocal_rank_fusion(bm25_list, dense_list, k=60, top_k=3)

    assert len(fused) == 3
    # B was rank 2 in BM25 and rank 1 in Dense -> RRF = 1/62 + 1/61 = ~0.0325
    # A was rank 1 in BM25 and rank 3 in Dense -> RRF = 1/61 + 1/63 = ~0.0322
    # B should be top fused rank
    assert fused[0].chunk_id == "chk_B"
    assert fused[0].rank == 1
    assert fused[0].rrf_score is not None
    assert fused[0].bm25_rank == 2
    assert fused[0].dense_rank == 1


def test_semantic_reranker():
    """Test cross-encoder reranking produces deterministic ranking and sets rerank_score."""
    reranker = SemanticReranker()
    candidates = [
        DocumentChunk(
            chunk_id="c_weak",
            doc_id="d1",
            doc_title="Weak",
            doc_type=DocumentType.GENERAL,
            content="General unrelated information regarding warehouse employee shift schedules.",
            rrf_score=0.030,
        ),
        DocumentChunk(
            chunk_id="c_strong",
            doc_id="d2",
            doc_title="Strong",
            doc_type=DocumentType.KPI_DEFINITION,
            content="Average Order Value is calculated as total recognized revenue divided by order count.",
            rrf_score=0.025,
        ),
    ]

    reranked = reranker.rerank(query="What is the calculation formula for Average Order Value AOV?", candidates=candidates)
    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c_strong"
    assert reranked[0].rank == 1
    assert reranked[0].rerank_score is not None
    assert reranked[0].similarity_score > reranked[1].similarity_score


def test_all_four_retrieval_modes():
    """Test document store executing in BM25, DENSE, HYBRID, and RERANKED modes."""
    store = DocumentStore()
    query = "March tier-1 supplier shutdown grid outage North stockouts"

    # 1. BM25 mode
    res_bm25 = store.retrieve(query=query, mode=RetrievalMode.BM25, top_k=2)
    assert res_bm25.mode == RetrievalMode.BM25
    assert len(res_bm25.chunks) > 0
    assert res_bm25.chunks[0].retrieval_mode == RetrievalMode.BM25
    assert res_bm25.chunks[0].bm25_score is not None

    # 2. DENSE mode
    res_dense = store.retrieve(query=query, mode=RetrievalMode.DENSE, top_k=2)
    assert res_dense.mode == RetrievalMode.DENSE
    assert len(res_dense.chunks) > 0
    assert res_dense.chunks[0].retrieval_mode == RetrievalMode.DENSE
    assert res_dense.chunks[0].dense_score is not None

    # 3. HYBRID mode
    res_hybrid = store.retrieve(query=query, mode=RetrievalMode.HYBRID, top_k=2)
    assert res_hybrid.mode == RetrievalMode.HYBRID
    assert len(res_hybrid.chunks) > 0
    assert res_hybrid.chunks[0].retrieval_mode == RetrievalMode.HYBRID
    assert res_hybrid.chunks[0].rrf_score is not None

    # 4. RERANKED mode
    res_reranked = store.retrieve(query=query, mode=RetrievalMode.RERANKED, top_k=2)
    assert res_reranked.mode == RetrievalMode.RERANKED
    assert len(res_reranked.chunks) > 0
    assert res_reranked.chunks[0].retrieval_mode == RetrievalMode.RERANKED
    assert res_reranked.chunks[0].rerank_score is not None


def test_retrieval_evaluation_metrics():
    """Test computation of Recall@K, MRR, and nDCG@K metrics."""
    gt = ["doc_q1_report_chunk_01"]

    # Perfect ranking: relevant item at rank 1
    retrieved_perfect = ["doc_q1_report_chunk_01", "other_chunk_02", "other_chunk_03"]
    assert compute_recall_at_k(retrieved_perfect, gt, k=1) == 1.0
    assert compute_recall_at_k(retrieved_perfect, gt, k=3) == 1.0
    assert compute_mrr(retrieved_perfect, gt) == 1.0
    assert compute_ndcg_at_k(retrieved_perfect, gt, k=3) == 1.0

    # Relevant item at rank 2
    retrieved_rank2 = ["other_chunk_01", "doc_q1_report_chunk_01", "other_chunk_03"]
    assert compute_recall_at_k(retrieved_rank2, gt, k=1) == 0.0
    assert compute_recall_at_k(retrieved_rank2, gt, k=2) == 1.0
    assert compute_mrr(retrieved_rank2, gt) == 0.5
    assert 0.0 < compute_ndcg_at_k(retrieved_rank2, gt, k=3) < 1.0

    # Not found in top 3
    retrieved_miss = ["other_01", "other_02", "other_03"]
    assert compute_recall_at_k(retrieved_miss, gt, k=3) == 0.0
    assert compute_mrr(retrieved_miss, gt) == 0.0
    assert compute_ndcg_at_k(retrieved_miss, gt, k=3) == 0.0


def test_comparative_retrieval_benchmark_suite():
    """Run full benchmark evaluating all 4 strategies against ground truth cases."""
    store = DocumentStore()
    evaluator = RetrievalEvaluator(document_store=store, cases=RETRIEVAL_BENCHMARK_CASES)
    report = evaluator.run_comparative_benchmark(top_k=5)

    assert isinstance(report, RetrievalBenchmarkReport)
    assert len(report.strategies) == 4
    for mode_key in ["bm25", "dense", "hybrid", "reranked"]:
        assert mode_key in report.strategies
        strat = report.strategies[mode_key]
        assert strat.total_queries == len(RETRIEVAL_BENCHMARK_CASES)
        assert 0.0 <= strat.recall_at_5 <= 1.0
        assert 0.0 <= strat.mrr <= 1.0
        assert 0.0 <= strat.ndcg_at_5 <= 1.0
        assert strat.mean_latency_ms >= 0.0


def test_retrieval_agent_with_hybrid_mode():
    """Test RetrievalAgent integration with hybrid mode and provenance recording."""
    store = DocumentStore()
    agent = RetrievalAgent(document_store=store)
    ev_store = EvidenceStore(dataset_id="test_ds", dataset_version="v1")

    task = AgentTask(
        task_id="task_retrieval_test",
        assigned_to=AgentRole.RETRIEVAL,
        description="Retrieve contextual explanation for March sales decline" if hasattr(AgentTask, "description") else None,
        rationale="Retrieve operational context for March sales decline",
        parameters={"query": "March tier-1 electronics supplier outage", "top_k": 2, "mode": "hybrid"},
    )

    completed_task = agent.execute_task(task, ev_store)
    assert completed_task.status == "completed"
    assert len(completed_task.evidence_ids) > 0

    ev_items = ev_store.list_evidence()
    assert len(ev_items) >= 1
    top_ev = ev_items[0]
    assert top_ev.content["retrieval_mode"] == "hybrid"
    assert top_ev.content["rank"] is not None
    assert top_ev.location is not None
