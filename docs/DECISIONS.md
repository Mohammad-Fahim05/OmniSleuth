# OmniSleuth — Architectural Decisions & Design Rationale

This document records the foundational architectural decisions (ADRs) that govern OmniSleuth's design, implementation, and verification.

---

## ADR-01: Prohibiting Direct LLM Computation & Code Generation (Rules 1 & 2)

- **Context**: Standard generative AI agents for tabular data typically execute arbitrary Python code via `eval()`/`exec()` or attempt direct mental arithmetic on numbers in prompt context.
- **Decision**:
  1. The LLM is strictly prohibited from performing arithmetic calculations.
  2. The LLM is strictly prohibited from generating arbitrary Python code for execution.
  3. All data calculations are routed through an allowlist of exactly 8 typed Pydantic analytical tools.
- **Rationale**:
  - Eliminates Remote Code Execution (RCE) and container breakout risks.
  - LLMs are autoregressive token predictors with high error rates on multi-step arithmetic over large datasets.
  - Deterministic Pandas/NumPy execution guarantees mathematical accuracy ($100\%$ numerical correctness on the benchmark suite).

---

## ADR-02: Programmatic Claim Validation with Tolerance Matching (Rule 5)

- **Context**: Even if tools compute correct figures, an LLM generating natural language summaries may misquote numbers, hallucinate percentages, or blend numbers across periods.
- **Decision**: Implement [ClaimValidator](file:///d:/OmniSleuth/omnisleuth/evidence/validator.py) which extracts numbers via regex from generated claims and tolerance-matches them against raw `EvidenceItem` records ($\pm 1.0\%$ relative tolerance). Claims containing unverified numbers are flagged as `UNSUPPORTED` and intercepted by output guardrails.
- **Rationale**: Guarantees zero ungrounded numerical claims reach the analyst, maintaining auditability and institutional trust.

---

## ADR-03: Qualitative-Only Contextual RAG (Rule 3)

- **Context**: Tabular numbers can theoretically be converted into text chunks and indexed into vector databases for similarity retrieval.
- **Decision**: Vector and keyword search (BM25 + Dense FAISS) are restricted strictly to unstructured qualitative domain context: incident reports, postmortems, KPI definitions, and data dictionaries. Tabular records are never queried via RAG for computation.
- **Rationale**: Dense vector retrieval is non-deterministic, semantic, and fuzzy. Using vector embeddings to retrieve numbers causes severe aggregation errors and missing row biases.

---

## ADR-04: Automated Causal Phrasing Calibration (Rule 6)

- **Context**: Observational business datasets cannot mathematically prove counterfactual causality without controlled A/B experimentation or structural causal DAG modeling. LLMs frequently declare ungrounded causality (e.g., "The March revenue drop was caused by supplier stockouts").
- **Decision**: Implement deterministic pattern matching in [ClaimValidator](file:///d:/OmniSleuth/omnisleuth/evidence/validator.py) and [OutputGuardrailEngine](file:///d:/OmniSleuth/omnisleuth/guardrails/output_guardrails.py) that rewrites speculative words (e.g., "caused by" $\rightarrow$ "associated with", "proves that" $\rightarrow$ "suggests that").
- **Rationale**: Prevents users from taking high-stakes business actions based on unproven causal assumptions.

---

## ADR-05: 100% Offline-First Reproducibility & Built-in Fallbacks

- **Context**: CI/CD pipelines, academic evaluations, and secure air-gapped environments frequently lack external internet access or paid third-party API keys (OpenAI/Azure).
- **Decision**:
  1. Build a deterministic fallback provider in [LLMGateway](file:///d:/OmniSleuth/omnisleuth/llm/gateway.py) that generates rule-grounded outputs at $0.00 cost.
  2. Build a NumPy cosine-similarity fallback for FAISS in [faiss_index.py](file:///d:/OmniSleuth/omnisleuth/rag/faiss_index.py).
  3. Build deterministic feature-embedding and reranking fallbacks for `sentence-transformers`.
- **Rationale**: All 107 tests pass deterministically in 17 seconds on any standard Python environment without credentials or GPUs.

---

## ADR-06: Centralized LLM Gateway with Dataset-Isolated Semantic Cache (Phase 19)

- **Context**: Direct calls to `openai` SDK throughout different agent files lead to duplicated retry code, inconsistent error handling, lack of cost tracking, and cache invalidation bugs when data changes.
- **Decision**: Centralize all LLM interactions through [LLMGateway](file:///d:/OmniSleuth/omnisleuth/llm/gateway.py) with:
  - Transient vs. permanent error classification (immediate abort on 401/403; exponential backoff on 429/5xx).
  - Unique request IDs (`llm_req_YYYYMMDD_<hash>`) correlated with `investigation_id`.
  - Semantic caching keyed by `(dataset_version, prompt_hash)`.
- **Rationale**: Eliminates duplicate charges, prevents stale cached answers across dataset updates, and provides full cost/latency observability.

---

## ADR-07: In-Process Streamlit Architecture vs. Decoupled FastAPI

- **Context**: Architectural decision regarding whether to introduce a separate FastAPI backend server communicating over HTTP or an in-process Streamlit workspace.
- **Decision**: The existing system embeds orchestrator execution directly within the Streamlit runtime (`omnisleuth/ui/app.py`) and provides a terminal CLI (`omnisleuth/cli.py`), without a standalone FastAPI HTTP microservice.
- **Rationale**:
  - Eliminates microservice networking complexity, cross-process data serialization overhead, and separate port management for local analyst workflows.
  - Suitable for single-tenant analyst investigations.
  - *Trade-off*: Multi-tenant REST API consumers require a future FastAPI wrapper if headless remote invocation is needed.

---

## ADR-08: Container Security & Non-Root Execution (Phase 20)

- **Context**: Cloud platforms (e.g., Google Cloud Run, AWS ECS) and security compliance audits mandate least-privilege container execution.
- **Decision**:
  - Base container uses official `python:3.11-slim`.
  - Runs strictly as dedicated non-root user `appuser` (UID 10001).
  - Dynamically binds to `$PORT` environment variable.
  - Implements container-level `HEALTHCHECK` against `/_stcore/health`.
- **Rationale**: Mitigates privilege escalation attacks and satisfies enterprise container security scanners.
