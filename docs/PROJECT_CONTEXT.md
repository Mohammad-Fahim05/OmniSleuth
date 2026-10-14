# OmniSleuth — Project Context & System Specification

## 1. Project Purpose & Motivation
OmniSleuth is an evidence-grounded, multi-agent data investigation system designed to analyze tabular business and operational datasets (e.g., sales, revenue, orders, transactions).

Generative Large Language Models (LLMs) applied directly to structured data fail in three critical ways:
1. **Mathematical Hallucination**: Autoregressive probability models produce rounding errors, fabricated percentages, and inconsistent aggregates.
2. **Unsupported Causal Speculation**: Conversational AI conflates correlation with causation, asserting ungrounded claims (e.g., asserting an outage "caused" a revenue drop without empirical proof).
3. **Security Vulnerabilities (RCE)**: Allowing LLMs to write and execute arbitrary Python code (`exec()`/`eval()`) introduces severe Remote Code Execution and stability risks.

OmniSleuth eliminates these issues by treating the LLM strictly as a **planner and natural language synthesizer**, delegating all mathematical, statistical, and retrieval tasks to **strictly typed, allowlisted deterministic Python tools**. Every factual claim is programmatically validated against an immutable evidence store before being presented to the user.

---

## 2. Core Architectural Guarantees (The 7 Invariants)
- **Rule 1: Deterministic Math**: The LLM *never* performs arithmetic. All aggregations, trends, deltas, and decompositions run via Pandas, NumPy, and scikit-learn.
- **Rule 2: Allowlisted Tools Only**: Exactly 8 typed Pydantic analytical tools. Arbitrary code execution is strictly rejected.
- **Rule 3: Qualitative Contextual RAG**: Hybrid document search provides domain definitions and operational reports—never numerical calculations.
- **Rule 4: Evidence Provenance**: Every factual claim is linked to a unique evidence ID (`ev_*`, `exec_*`, `doc_*`).
- **Rule 5: Programmatic Claim Validation**: A regex + tolerance-matching validator verifies extracted numbers against raw tool execution records ($\pm 1.0\%$ relative tolerance).
- **Rule 6: Calibrated Causal Phrasing**: Definite causal assertions (e.g., "caused by") are automatically rewritten to calibrated observational phrases (e.g., "associated with").
- **Rule 7: Bounded Execution**: Finite execution boundaries enforce depth $\le 3$, tool calls $\le 10$, and retries $\le 2$.

---

## 3. Implemented Capabilities & Scope
The codebase is structured around 21 development phases (Phases 0–20):
- **Data Ingestion & Integrity (Phase 2)**: [CSVLoader](file:///d:/OmniSleuth/omnisleuth/ingestion/csv_loader.py) with encoding detection (UTF-8, Latin-1) and SHA-256 version fingerprinting.
- **Profiling & Quality Engine (Phase 3)**: [DatasetProfiler](file:///d:/OmniSleuth/omnisleuth/profiling/profiler.py) calculating completeness scores, duplicate counts, and cardinality warnings.
- **Allowlisted Analytical Engine (Phases 4 & 13)**: 8 allowlisted tools in [definitions.py](file:///d:/OmniSleuth/omnisleuth/tools/definitions.py) covering aggregations, trends, segment comparisons, contributions, revenue decomposition, Isolation Forest / IQR anomaly detection, and Plotly charting.
- **Provenance & Claim Validation (Phases 5 & 7)**: [EvidenceStore](file:///d:/OmniSleuth/omnisleuth/evidence/store.py) and [ClaimValidator](file:///d:/OmniSleuth/omnisleuth/evidence/validator.py).
- **Multi-Agent Coordination (Phase 6)**: [MultiAgentOrchestrator](file:///d:/OmniSleuth/omnisleuth/agents/orchestrator.py) coordinating Supervisor, Analyst, Retriever, and Investigator roles.
- **Workflow State Machine (Phase 12)**: [LangGraph StateGraph](file:///d:/OmniSleuth/omnisleuth/workflow/graph.py) with deterministic fallback graph when LangGraph is uninstalled.
- **Hybrid RAG & Reranking (Phase 10)**: BM25 sparse search + Dense FAISS/NumPy embeddings + Reciprocal Rank Fusion (RRF) + Cross-Encoder reranker.
- **Multi-Layer Guardrails (Phases 15 & 17)**: [InputGuardrailEngine](file:///d:/OmniSleuth/omnisleuth/guardrails/input_guardrails.py) (prompt injection, jailbreak, arbitrary code, off-topic) and [OutputGuardrailEngine](file:///d:/OmniSleuth/omnisleuth/guardrails/output_guardrails.py) (secret redaction, unsupported claim flagging, causal calibration).
- **Empirical Evaluation Harness (Phase 18)**: [OmniSleuthEvaluator](file:///d:/OmniSleuth/omnisleuth/evaluation/evaluator.py) testing 40 benchmark questions across 13 metrics against 2 baselines.
- **LLM Gateway (Phase 19)**: [LLMGateway](file:///d:/OmniSleuth/omnisleuth/llm/gateway.py) with provider abstraction (OpenAI, Azure, Deterministic Offline Fallback), bounded exponential retry, dataset-isolated semantic caching, and request tracing (`llm_req_*`).
- **Containerization & CI/CD (Phase 20)**: Hardened non-root [Dockerfile](file:///d:/OmniSleuth/Dockerfile) (appuser UID 10001), [docker-compose.yml](file:///d:/OmniSleuth/docker-compose.yml), and GitHub Actions [ci.yml](file:///d:/OmniSleuth/.github/workflows/ci.yml).
- **User Interfaces**: Interactive Streamlit workspace in [omnisleuth/ui/app.py](file:///d:/OmniSleuth/omnisleuth/ui/app.py) and terminal CLI in [omnisleuth/cli.py](file:///d:/OmniSleuth/omnisleuth/cli.py).

---

## 4. Technology Stack & Key Dependencies
- **Core Runtime**: Python 3.10+ (tested on Python 3.10, 3.11, 3.12).
- **Data & Computing**: `pandas>=2.1.0`, `numpy>=1.26.0`, `scikit-learn>=1.4.0`, `openpyxl>=3.1.2`, `plotly>=5.19.0`.
- **Validation & Settings**: `pydantic>=2.6.0`, `pydantic-settings>=2.2.0`, `python-dotenv>=1.0.0`.
- **Testing**: `pytest>=8.0.0`, `pytest-cov>=4.1.0`.
- **Optional Orchestration & RAG**: `langchain>=0.2.0`, `langgraph>=0.1.0`, `chromadb>=0.4.22`, `pypdf>=4.0.0` (core components fall back gracefully if not present).
- **UI Framework**: `streamlit>=1.32.0`.
- **Persistence**: SQLite (native `sqlite3` driver) with JSON serialization.

---

## 5. Architectural Boundaries & Known Constraints
1. **100% Offline-First Reproducibility**: The test suite and core workflows execute with zero external network connectivity and zero paid API keys, utilizing built-in deterministic fallbacks.
2. **No Monolithic Remote Code Execution**: Code execution is prohibited; all analytical operations are mapped to predefined functions.
3. **Tabular Scope**: Optimized for single-node tabular data up to ~1,000,000 rows. Distributed cluster computing (Spark, Dask, BigQuery) is out of scope.
4. **No FastAPI Backend Currently Present**: The Streamlit frontend interacts directly with Python agent classes in-process. There is no independent FastAPI HTTP service layer in the existing repository.
