# OmniSleuth — Frozen Scope Gap Report & Architecture Audit

**Date:** 2026-10-08  
**Audit Target:** `D:\OmniSleuth`  
**Test Suite Status:** 63 passed, 0 failed, 2 warnings (pytest 8.3.4, Python 3.13.5)

---

## 1. Phase-by-Phase Audit Matrix

| Phase | Requirement | Status | Existing Implementation | Missing Work | Tests | Priority |
|---|---|---|---|---|---|---|
| 0 | Foundation | COMPLETE | `pyproject.toml`, `core/config.py`, `core/logging.py`, `core/exceptions.py`, `cli.py` | None | `test_foundation.py` (4 tests passing) | P0 |
| 1 | Dataset Ingestion | COMPLETE | `ingestion/csv_loader.py`, `schema_detector.py`, `models.py` (CSV/Excel, SHA-256, type inference) | None | `test_ingestion.py` (9 tests passing) | P0 |
| 2 | Profiling & Data Quality | COMPLETE | `profiling/profiler.py`, `models.py` (Numeric, Categorical, Datetime, Quality Scorecard) | None | `test_profiling.py` (4 tests passing) | P0 |
| 3 | Deterministic Analytics Engine | COMPLETE | `analytics/engine.py`, `models.py` (Aggregations, Trends, Segments, Contributions, Decomposition, Charts) | None | `test_analytics.py` (7 tests passing) | P0 |
| 4 | ML Anomaly Detection | COMPLETE | `ml/detectors.py`, `models.py` (IQR, Isolation Forest, synthetic injection & metrics) | None | `test_ml.py` (3 tests passing) | P0 |
| 5 | Tool Layer | COMPLETE | `tools/definitions.py`, `base.py`, `__init__.py` (8 typed allowlisted tools with Pydantic I/O & audit tracking) | None | `test_tools.py` (9 tests passing) | P0 |
| 6 | Evidence & Provenance | COMPLETE | `evidence/models.py`, `store.py` (`ToolExecution`, `EvidenceItem`, `Claim`, `Hypothesis`, provenance graph) | None | `test_evidence.py` (3 tests passing) | P0 |
| 7 | Programmatic Claim Validation | COMPLETE | `evidence/validator.py` (Deterministic number tolerance matching, causal phrase calibration) | None | `test_validator.py` (4 tests passing) | P0 |
| 8 | Basic LangGraph Workflow | COMPLETE | `workflow/graph.py`, `nodes.py`, `state.py` (7-step state machine with bounded depth/retries) | None | `test_workflow.py` (3 tests passing) | P0 |
| 9 | Basic RAG | COMPLETE | `rag/document_store.py`, `seed_docs.py`, `models.py` (Chunking, BM25-style keyword search, citation metadata) | None | `test_rag.py` (13 tests passing) | P1 |
| 10 | Hybrid Retrieval + Reranking | COMPLETE | `rag/bm25.py`, `rag/dense_embedder.py`, `rag/faiss_index.py`, `rag/fusion.py`, `rag/reranker.py`, `rag/evaluator.py` (BM25, Dense FAISS, RRF, Cross-Encoder reranker, Recall@k, MRR, nDCG) | None | `test_rag.py` (13 tests passing) | P0 |
| 11 | Supervisor / Planner | COMPLETE | `agents/supervisor.py`, `models.py` (Structured JSON investigation plans, validation checks) | None | `test_agents.py::test_supervisor_planning` | P0 |
| 12 | Hypothesis Investigation | COMPLETE | `agents/investigator.py` (Multi-hypothesis generation, tool execution, status classification) | None | `test_agents.py` | P0 |
| 13 | Bounded Multi-Agent Components | COMPLETE | `agents/orchestrator.py`, `analyst.py`, `retriever.py`, `supervisor.py`, `investigator.py` | None | `test_agents.py` | P0 |
| 14 | Discovery Engine + Investigation Tree | COMPLETE | `discovery/scanner.py`, `models.py` (Autonomous multidimensional anomaly/trend scan, ranking, tree) | None | `test_discovery.py` (2 tests passing) | P0 |
| 15 | Guardrails | PARTIAL | Programmatic causal phrase filters in `validator.py`, bounding rules in `supervisor.py` | Formal standalone input guardrails (prompt injection, off-topic, arbitrary code rejection) and output schema guardrails | Needed: `test_guardrails.py` | P1 |
| 16 | Persistence | COMPLETE | `persistence/db.py`, `repository.py`, `serializer.py` (SQLite schema with full audit tables and restart persistence) | None | `test_persistence.py` (4 tests passing) | P0 |
| 17 | Streamlit UI | COMPLETE | `ui/app.py` (5 tabs: Upload, Profile, Ask & Trace, Discover & Tree, Benchmark Evaluation) | None | `test_ui.py` (1 test passing) | P0 |
| 18 | Evaluation | COMPLETE | `evaluation/benchmark_suite.py`, `evaluator.py`, `dataset_generator.py` (30+ questions, real latency/accuracy metrics) | Add retrieval evaluation metrics (Recall@k, MRR, nDCG) from Phase 10 | `test_evaluation.py` (2 tests passing) | P1 |
| 19 | LLM Gateway | PARTIAL | Direct OpenAI client calls with deterministic local fallback | Formal LLM Gateway module with token counting, cost tracking, retry backoff, latency tracing, and semantic cache | Needed: `test_gateway.py` | P2 |
| 20 | Docker + Cloud Deployment | COMPLETE | `Dockerfile` (optimized multi-stage build, healthcheck, container entrypoint), Cloud Run readiness | None | Container test verification | P2 |
| 21 | Documentation & Viva Defense | PARTIAL | `docs/interview_guide.md` is strong; `README.md` is basic | Expand `README.md` with complete architecture diagrams, data flow diagrams, evaluation benchmarks table, and viva defense Q&A | None | P1 |

---

## 2. Summary Status Categorization

### Already Complete (15 Phases)
- **Phase 0 (Foundation)**: Virtual env, packaging (`pyproject.toml`), logging, configuration (`pydantic-settings`), CLI entrypoint.
- **Phase 1 (Dataset Ingestion)**: CSV & Excel loader, encoding detection, SHA-256 hashing, schema detector.
- **Phase 2 (Profiling & Data Quality)**: Numeric/Categorical/Datetime profiling, data quality warnings, deterministic scorecard.
- **Phase 3 (Deterministic Analytics Engine)**: Aggregations, monthly/weekly trends, segment breakdowns, percentage contributions, Orders × AOV metric decomposition, Plotly chart generation.
- **Phase 4 (ML Anomaly Detection)**: IQR baseline, Isolation Forest detector with feature scaling/imputation, synthetic anomaly evaluation.
- **Phase 5 (Tool Layer)**: 8 strictly allowlisted, typed Pydantic tools with execution logging and provenance IDs.
- **Phase 6 (Evidence & Provenance)**: Immutable `ToolExecution`, `EvidenceItem`, `Claim`, `Hypothesis`, and `InvestigationTreeNode` provenance models.
- **Phase 7 (Programmatic Claim Validation)**: Tolerance-based numerical validation against evidence records, causal phrase detection & auto-calibration.
- **Phase 8 (Basic LangGraph Workflow)**: Compiled state graph with bounded execution constraints.
- **Phase 9 (Basic RAG)**: Seed business documents, markdown/PDF ingestion, token chunking, keyword retrieval with metadata citations.
- **Phase 11 (Supervisor / Planner)**: Structured investigation planning, task decomposition, pre-execution plan validation.
- **Phase 12 (Hypothesis Investigation)**: Systematic multi-hypothesis evaluation (`supported`, `rejected`, `inconclusive`).
- **Phase 13 (Bounded Multi-Agent Components)**: Orchestrated supervisor, analyst, retriever, investigator, and validator.
- **Phase 14 (Discovery Engine + Investigation Tree)**: Autonomous multidimensional scanner and interactive investigation trees.
- **Phase 16 (Persistence)**: SQLite database storing datasets, versions, investigations, tool executions, and claims.
- **Phase 17 (Streamlit UI)**: Professional 5-view analytical UI showcasing the complete investigation process.
- **Phase 20 (Docker + Deployment)**: Dockerfile with health checks and server config.

### Partially Complete / In Progress (5 Phases)
- **Phase 10 (Hybrid Retrieval + Reranking)**: Current RAG is term-frequency keyword based; missing dense embedding search, reciprocal rank fusion (RRF), semantic reranking, and formal retrieval metrics (Recall@k, MRR, nDCG).
- **Phase 15 (Guardrails)**: Causal phrasing rules and bounding limits are in place, but need a dedicated input guardrail module (prompt injection, jailbreak, out-of-scope intent classifier).
- **Phase 18 (Evaluation)**: Core benchmark runner exists; needs integration of dense/hybrid retrieval evaluation benchmarks.
- **Phase 19 (LLM Gateway)**: OpenAI + deterministic fallback exists; needs structured token/cost tracking, timeout, retry, and latency tracing wrapper.
- **Phase 21 (Documentation & Viva)**: `docs/interview_guide.md` is high quality; `README.md` needs comprehensive technical documentation, architecture flowcharts, and benchmark results.

### Missing / Broken / Needs Verification
- **Missing**: True dense vector retrieval + Reciprocal Rank Fusion (Phase 10), dedicated Input Guardrail module (Phase 15), formal LLM Gateway with cost tracking (Phase 19).
- **Broken**: None. All 63 existing tests pass cleanly.
- **Needs Verification**: End-to-end investigation trace with hybrid dense + BM25 retrieval.

---

## 3. Recommended Next Phase

### **Phase 10 — Hybrid Retrieval + Reranking**

**Objective:** Upgrade `omnisleuth/rag/document_store.py` with:
1. Dual-channel retrieval: BM25 keyword search + Dense vector embeddings (Cosine / FAISS / Numpy embeddings).
2. Reciprocal Rank Fusion (RRF) candidate merging.
3. Semantic / Cross-Encoder reranking stage.
4. Formal evaluation suite computing Recall@k, MRR, and nDCG metrics.

**Files to be modified/added in Phase 10:**
- `omnisleuth/rag/document_store.py` (Add dense vector indexing, fusion, reranking)
- `omnisleuth/rag/models.py` (Add retrieval mode, hybrid metadata, rank scores)
- `omnisleuth/rag/hybrid_retriever.py` (Dedicated hybrid fusion and reranking module)
- `tests/test_rag.py` (Expand with hybrid retrieval, dense search, and reranking test cases)
