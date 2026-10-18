# OmniSleuth — Technical Interview & Viva Defense Guide

This guide provides precise, senior-engineering answers to ten critical architecture and system design questions about **OmniSleuth**, grounded strictly in the current implementation.

---

## 1. What problem does OmniSleuth solve?

**Problem:** Standard conversational AI applications and generative BI agents fail on business data in three fundamental ways:
1. **Mathematical Hallucination:** LLMs are autoregressive token predictors, not calculators. Directly prompting an LLM to calculate revenue deltas, profit margins, or percent contributions on large tabular datasets yields subtle rounding errors and completely fabricated numbers.
2. **Arbitrary Code Execution (RCE) Vulnerabilities:** Naive systems that bypass mathematical hallucination by letting the LLM generate and run raw Python scripts (`eval()` / `exec()`) expose the host system to remote code execution, syntax failures, and non-deterministic logic.
3. **Unsubstantiated Causal Claims:** Language models routinely conflate correlation with causation, asserting that a metric changed "because of" an arbitrary feature without empirical hypothesis testing.

**Solution:** OmniSleuth decouples **reasoning and planning** (handled by the LLM) from **computation and grounding** (handled by deterministic Python/pandas math and allowlisted Pydantic analytical tools). An immutable Evidence Store and a programmatic claim validation engine guarantee that every factual statement presented to the user is grounded in verifiable data.

---

## 2. Why use deterministic analytics alongside an LLM?

**Rationale:** The LLM's strength is understanding ambiguous user queries, decomposing high-level questions into analytical sub-tasks, and translating numbers into clear executive language. Conversely, deterministic Python code excels at vectorized aggregation, floating-point precision, reproducibility, and exact statistical modeling.

By pairing them:
- The LLM acts purely as a **semantic planner** (emitting typed JSON execution plans) and a **narrator**.
- The deterministic analytics engine (`omnisleuth/analytics/engine.py` and `omnisleuth/tools/`) executes all calculations using pandas and scikit-learn.
- If the LLM produces a plan with nonexistent columns or illegal operations, the deterministic `PlanValidator` intercepts and rejects it before execution.

---

## 3. How does a natural-language question become an executed analysis?

**Workflow:**
1. **Input Guardrail:** The question passes through `InputGuardrailEngine` to detect prompt injection, jailbreak attempts, or code execution requests.
2. **Intent & Routing:** `AnalysisPipeline` inspects query structure:
   - Descriptive queries (e.g., *"What was total revenue in March?"*) route to metric aggregations.
   - Diagnostic queries (e.g., *"Why did revenue decline in March?"*) route to `deep_investigation` hypothesis testing.
3. **Plan Generation:**
   - *Deterministic Mode (Default):* `RuleBasedPlanner` extracts metrics, dimensions, and filters using schema context and fuzzy column matching.
   - *LLM-Assisted Mode (Optional):* `LLMPlanner` issues a structured JSON schema prompt to OpenRouter / OpenAI.
4. **Plan Validation:** `PlanValidator` verifies that every planned tool belongs to the 8 allowlisted tools and that every column name exists in the active dataset schema.
5. **Execution:** `PlanExecutor` invokes allowlisted tools sequentially, streaming intermediate tables and recording all results into the `EvidenceStore`.
6. **Grounding & Presentation:** Extracted numbers are verified against evidence, and the UI renders the answer card, execution plan, hypotheses, and allowlisted Plotly charts.

---

## 4. How are evidence and claims grounded?

**Implementation:** Located in [`omnisleuth/evidence/store.py`](file:///d:/OmniSleuth/omnisleuth/evidence/store.py) and [`omnisleuth/evidence/grounding.py`](file:///d:/OmniSleuth/omnisleuth/evidence/grounding.py):
1. **Evidence Minting:** When an allowlisted tool executes, its typed output is hashed and saved as an `EvidenceItem` with source parameters, timestamp, and raw computed numerical values.
2. **Claim Extraction:** The system extracts numerical claims, currency values, and percentages from natural language answers using regular expressions.
3. **Tolerance Matching:** Each extracted number is compared against numerical values in the active `EvidenceStore` using relative and absolute tolerance thresholds ($|a - b| \le \max(\text{rel\_tol} \times \max(|a|, |b|), \text{abs\_tol})$).
4. **Metadata Filtering:** System metadata (such as "80% confidence" or "model confidence: 90%") is filtered to prevent false matching, while genuine analytical figures (such as "100% of revenue in Q4") are strictly validated.
5. **Enforcement:** If any extracted number fails to match stored evidence, the claim is marked `UNSUPPORTED` and highlighted with a warning badge in the UI.

---

## 5. How does the system prevent arbitrary code execution?

**Implementation:**
- **No `eval()` or `exec()`:** The codebase contains no dynamic Python interpreters or code-evaluating primitives.
- **Closed Tool Allowlist:** The system can only invoke 8 predefined Pydantic tools (`ProfileDatasetTool`, `AggregateMetricTool`, `CalculateTrendTool`, `CompareSegmentsTool`, `DecomposeMetricTool`, `DetectAnomaliesTool`, `SearchContextTool`, `SynthesizeFindingsTool`).
- **Strict Parameter Schemas:** Each tool's inputs are strictly parsed into Pydantic models. Unrecognized arguments or injected Python snippets fail validation.
- **Schema Grounding:** Target column names are checked against dataset headers, preventing SQL injection or arbitrary object attribute access.

---

## 6. How are uploaded datasets validated and stored?

**Implementation:** Located in [`omnisleuth/ingestion/validator.py`](file:///d:/OmniSleuth/omnisleuth/ingestion/validator.py) and [`omnisleuth/ingestion/storage.py`](file:///d:/OmniSleuth/omnisleuth/ingestion/storage.py):
- **Size Limit:** 50 MB hard ceiling enforced during streaming.
- **Binary Signature Inspection:** Inspects leading magic bytes to reject disguised executables (Windows PE `MZ`, Linux ELF `\x7fELF`), images, PDFs, and compressed archives uploaded with `.csv` extensions.
- **Excel OpenXML Protection:** Inspects ZIP container structure and calls `zipfile.testzip()` to detect corrupted members and decompression-bomb inflation ratios.
- **DoS Prevention:** Maximum 1,000 columns enforced to prevent memory exhaustion.
- **Safe Storage:** Uses `safe_save_upload()` to stream files in 64 KB chunks with `os.O_CREAT | os.O_EXCL`, UUID-based non-colliding names, and strict path containment within `data/uploads/` to defeat path traversal attacks (`../`).

---

## 7. How does session isolation work?

**Implementation:** Located in [`omnisleuth/persistence/repository.py`](file:///d:/OmniSleuth/omnisleuth/persistence/repository.py):
- Each browser session generates an opaque identifier (`sess_<uuid>`).
- All datasets and investigations stored in SQLite have an indexed `session_id` column.
- `Repository` methods (`list_datasets`, `get_dataset`, `delete_dataset`, `get_investigation`) require `session_id` and enforce ownership filtering.
- If a user requests a dataset ID belonging to another session, the repository returns an error and logs a security warning. Deletions and activations across session boundaries are strictly blocked.

---

## 8. How is benchmark performance evaluated?

**Implementation:** Located in [`omnisleuth/evaluation/evaluator.py`](file:///d:/OmniSleuth/omnisleuth/evaluation/evaluator.py):
- **Benchmark Suite:** 40 curated business questions (`BENCHMARK_QUESTIONS`) covering aggregations, period-over-period trends, segment breakdowns, and anomaly detection on synthetic sales data with planted anomalies.
- **Empirical Metrics:**
  - *Numerical Correctness (%):* Ground-truth mathematical accuracy (100.0% achieved).
  - *Tool Selection Accuracy (%):* Selection of appropriate allowlisted tools (100.0% achieved).
  - *Grounded Claim Coverage (%):* Percentage of factual claims grounded in evidence (100.0% achieved).
  - *Unsupported Claim Rate (%):* 0.0% unverified claims.
  - *P95 Latency:* ~92 ms for deterministic rule-based planning.
- **Comparative Baselines:** Compared against a Direct Tool baseline (no dynamic planning) and a Fixed Pipeline baseline. Zero fabricated or hard-coded baseline values are displayed.

---

## 9. What are the current limitations?

**Honest Technical Trade-offs:**
1. **Single-Instance Deployment:** SQLite with WAL mode and `PRAGMA busy_timeout = 30000;` is robust for single-node deployments, but does not support horizontal multi-replica clustering.
2. **Pseudonymous Session vs. Real Identity:** Sessions are identified by browser UUIDs rather than cryptographic JWT tokens or OAuth2/OIDC user logins.
3. **HTTP / TLS Boundary:** The standalone Streamlit container does not terminate TLS or emit production security headers (HSTS, CSP, nosniff); an external reverse proxy is required.
4. **Host Docker Daemon Blocker:** While Docker configuration files (`Dockerfile`, `.dockerignore`, `docker-compose.yml`) are verified, actual container execution on the Windows development host was blocked by Docker Desktop daemon service permissions requiring administrator elevation.
5. **Tool Scope:** OmniSleuth supports tabular data analysis via its 8 allowlisted tools. It does not perform unconstrained multi-hop SQL joins across dozens of arbitrary external databases.

---

## 10. How would you evolve the system for authenticated multi-tenant cloud deployment?

**Architecture Evolution Plan:**
1. **Reverse Proxy & Ingress:** Place Nginx, Caddy, or an AWS Application Load Balancer in front of the container to terminate TLS 1.3, configure HSTS, and handle Streamlit WebSocket upgrades (`/_stcore/stream`).
2. **Identity & RBAC:** Integrate an OAuth2 / OpenID Connect provider (Okta, Azure AD, Cognito) to authenticate users and pass signed JWTs containing tenant and role claims.
3. **Managed Database:** Migrate repository persistence from SQLite to managed PostgreSQL (Amazon RDS or Cloud SQL), utilizing Row-Level Security (RLS) for cryptographic tenant isolation.
4. **Cloud Object Storage:** Replace local filesystem storage (`data/uploads/`) with Amazon S3 or Google Cloud Storage, using pre-signed URLs, SSE-KMS encryption, and bucket retention policies.
5. **Horizontal Autoscaling:** Deploy the stateless container service onto AWS ECS Fargate or Google Cloud Run with autoscaling based on concurrent HTTP/WebSocket connection volume.
