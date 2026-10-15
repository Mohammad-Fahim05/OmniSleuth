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

---

## ADR-09: OpenRouter as an OpenAI-Compatible Optional LLM Provider (Phase C1)

- **Context**: Phase C1 needs an LLM that can map arbitrary natural language questions to a structured analysis plan. Introducing a bespoke OpenRouter client would duplicate the provider, retry, usage-metering, tracing, and caching layers already built for OpenAI and Azure.
- **Decision**:
  - `OpenRouterProvider` reuses the existing `openai` SDK with only `base_url` changed, following the existing `BaseProvider` pattern. No new dependency is introduced.
  - OpenRouter is selected purely through configuration: `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`). The model is never hard-coded in planner logic, and no model is assumed free or available.
  - When OpenRouter is selected without a key, the provider raises a permanent `ConfigurationError` **before** any network call, and the gateway activates the existing deterministic offline fallback.
- **Rationale**: Keeps one provider abstraction, one retry policy, one cache, and one trace pipeline; makes OpenRouter an optional add-on that cannot break offline operation.

---

## ADR-10: LLM Plans Are Types, Not Text — Planning Stays Separated From Execution (Phase C1)

- **Context**: An LLM-backed planner introduces two risks: arbitrary code execution (Rule 1/2) and an LLM performing arithmetic on data it never correctly observed. A third, subtler risk is that a deterministic fallback response could be mistaken for a genuine model plan.
- **Decision**:
  - `LLMPlanner` produces a **structured `AnalysisPlan` only**. It receives a `SchemaContext` (never a DataFrame), so it structurally cannot read raw rows or compute results; all numbers come from `PlanExecutor` and the deterministic analytics engine.
  - The prompt contains only a bounded, sanitized profile (column names/types, numeric bounds, date ranges, cardinality, and at most 10 distinct values for low-cardinality columns) plus the allowlisted vocabulary and the plan JSON schema. Raw rows and high-cardinality cell values are excluded by construction, with further prompt budgets (profile, question, total prompt, reply tokens).
  - Output handling is defensive: optional Markdown fences are stripped, the JSON object is sliced out, the payload is checked for executable content, and the plan is constructed by `AnalysisPlan` with `extra="forbid"` (no such field exists to carry code, SQL, or shell commands).
  - Every plan then passes through the **same** `PlanValidator` used by the rule-based planner. One bounded repair attempt is allowed, using only sanitized validation errors; a still-invalid plan raises `LLMPlanningError`.
  - Planner identity is recorded as `planner="llm"` on the plan; `LLMPlanner.plan_with_fallback()` delegates to `RulePlanner`, which reports `planner="deterministic"` so a deterministic plan is never presented as an LLM one.
  - `LLMGateway.is_fallback_response()` explicitly identifies offline fallback / error responses, and the planner refuses to parse them as a model plan.
  - Error messages and traces are sanitized through `omnisleuth/llm/errors.py`: API keys, Authorization headers, and other secrets are redacted before logging or raising, and failures are returned as actionable categories (`rate_limit`, `server_error`, `network_error`, `timeout`, `authentication_error`, `invalid_request_error`, `configuration_error`).
- **Rationale**: The LLM only chooses *what* to analyze; deterministic code always decides *what the answer is*. Security depends on validation, not on prompt compliance.
