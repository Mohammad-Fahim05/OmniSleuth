# OmniSleuth — Master Viva Defense & Technical Presentation Guide

This guide prepares you to explain and defend every architectural decision, mathematical formulation, and empirical metric of **OmniSleuth** during major project evaluations, technical thesis defenses, and AI/ML system design interviews.

---

## 1. High-Impact Project Pitches

### The 2-Minute Elevator Pitch (Executive Overview)
> *"OmniSleuth is an evidence-grounded multi-agent system designed to solve the critical problems of mathematical hallucination, arbitrary code execution risks, and uncalibrated causal claims when LLMs investigate business data.
>
> Instead of letting an LLM write raw Python code or compute metrics directly, OmniSleuth enforces seven strict design rules:
> 1. All calculations run deterministically via 8 allowlisted Pydantic analytical tools;
> 2. Qualitative business context is retrieved via hybrid BM25 and dense vector RAG with cross-encoder reranking;
> 3. An immutable Evidence Store tracks every tool execution;
> 4. A programmatic Claim Validator verifies every number extracted from generated claims against stored evidence before presenting results to the user;
> 5. Causal language is automatically calibrated;
> 6. Orchestration is bounded by a 7-step LangGraph finite state machine; and
> 7. A production LLM Gateway provides retries, tracing, cost logging, and 100% deterministic offline fallback.
>
> Evaluated on a 40-question benchmark with planted anomalies, OmniSleuth achieves **100% numerical correctness**, **0% unsupported claims**, and an anomaly detection $F_1$ score of **0.902** across 105 passing automated tests."*

---

### The 5-Minute Technical Defense Pitch (In-Depth Architecture)
> *"When business analysts investigate structured datasets (such as a 15% revenue decline in Q1), standard generative AI approaches fail in three fundamental ways:
> First, autoregressive LLMs cannot reliably perform multi-stage arithmetic across large tables, producing fabricated numbers.
> Second, unconstrained code execution (`eval()`/`exec()`) introduces Remote Code Execution (RCE) vulnerabilities and unpredictable syntax failures.
> Third, naive conversational bots assert unsupported causal claims without empirical validation.
>
> OmniSleuth solves this through a modular, bounded multi-agent architecture:
>
> **1. Ingestion & Profiling**: We compute SHA-256 integrity fingerprints and generate automated data quality scorecards.
>
> **2. Multi-Layer Guardrails**: Deterministic regex and pattern matching intercept prompt injections, jailbreaks, and arbitrary code requests before execution begins.
>
> **3. Multi-Agent Orchestration**: We utilize LangGraph to compile a 7-step state graph where specialized agents perform distinct roles:
> - `SupervisorAgent` plans and decomposes questions into typed sub-goals;
> - `DataAnalystAgent` selects from 8 allowlisted, typed analytical tools (aggregations, trends, contributions, Order $\times$ AOV decomposition, and Isolation Forest anomaly detection);
> - `RetrievalAgent` executes hybrid search (BM25 keyword + FAISS dense embeddings + Reciprocal Rank Fusion + Cross-Encoder reranker) to find qualitative operational incident reports;
> - `InvestigationAgent` systematically generates and tests competing hypotheses, marking them as Supported, Rejected, or Inconclusive.
>
> **4. Programmatic Claim Validation**: Our validation engine extracts numbers from natural language statements and performs tolerance matching against raw `EvidenceItem` records. Any unsupported number is immediately flagged or redacted.
>
> **5. LLM Gateway & Offline Fallback**: Our gateway handles exponential retries, request tracing with unique `llm_req_*` IDs, token and USD cost tracking, and dataset-isolated caching, while providing an offline deterministic fallback that allows all 105 tests to pass without API keys.
>
> In our empirical benchmark across 40 questions, OmniSleuth outperformed unconstrained baselines by achieving **100% numerical correctness**, **100% evidence coverage**, **Recall@5 of 1.000 (MRR 0.95)**, and **p50 execution latency of 33.52 ms**."*

---

## 2. Fifteen Deep Architectural Rationales ("Why" Justifications)

### 1. Why multiple specialized agents instead of a single prompt?
- **Separation of Concerns**: A single monolithic LLM prompt attempting to plan, select tools, run statistics, search qualitative documents, and validate output suffers from context dilution and high error rates.
- **Role Specialization**:
  - `SupervisorAgent`: Decomposes high-level questions into structured JSON plans.
  - `DataAnalystAgent`: Focuses strictly on typed analytical tool parameters.
  - `RetrievalAgent`: Specializes in hybrid document search.
  - `InvestigationAgent`: Focuses on hypothesis verification and evidence matrix compilation.
- **Traceability**: Emits discrete, auditable state objects per step.

### 2. Why LangGraph instead of unconstrained loops (AutoGPT/ReAct)?
- **Finite State Machine Guarantees**: LangGraph enforces deterministic state transitions (`supervisor_plan` $\rightarrow$ `analyst_execute` $\rightarrow$ `rag_retrieve` $\rightarrow$ `investigate_hypotheses` $\rightarrow$ `validate_claims` $\rightarrow$ `apply_guardrails` $\rightarrow$ `END`).
- **Hard Bounded Limits**: Strict bounds prevent runaway executions: Depth $\le 3$, Tool Calls $\le 10$, Retries $\le 2$.
- **State Checkpointing**: The typed `InvestigationState` can be serialized and resumed from SQLite.

### 3. Why NOT allow the LLM to write and execute arbitrary Python code?
- **Remote Code Execution (RCE) Vulnerability**: Code execution engines (`exec()`, `eval()`) create security attack vectors.
- **Non-Deterministic Failures**: LLM-generated Python code frequently encounters `KeyError`, type mismatches, and incorrect pandas index alignments.
- **Allowlisted Typed Tools (Rule 2)**: Governing all data access through 8 Pydantic-validated tools guarantees 100% deterministic, bug-free, and safe calculation.

### 4. How are hallucinated calculations completely eliminated?
- **Rule 1 (Deterministic Python Execution)**: The LLM is never prompted to compute mathematical values. All sums, means, deltas, percentage contributions, and decompositions are calculated by Pandas, NumPy, and scikit-learn.
- **Rule 5 (Programmatic Claim Validation)**: The `ClaimValidator` extracts every number in synthesized text and verifies that it exists in the active `EvidenceStore` within a $\pm 1.0\%$ relative tolerance.

### 5. Why Hybrid RAG (BM25 + Dense FAISS + RRF + Cross-Encoder)?
- **Dual-Channel Complementarity**: BM25 excels at exact keyword matches (part numbers, incident codes, date strings), while Dense FAISS captures semantic conceptual matches (operational disruptions, supplier delays).
- **Reciprocal Rank Fusion (RRF)**: Merges sparse and dense ranks robustly without sensitive score normalization.
- **Cross-Encoder Reranker**: Performs deep token-level cross-attention to rank the top-5 most relevant chunks, achieving `Recall@5 = 1.000` and `MRR = 0.950`.

### 6. Why Isolation Forest for Anomaly Detection?
- **Multivariate Interaction**: Unlike univariate IQR which evaluates one column in isolation, Isolation Forest isolates anomalous transactions across multi-dimensional feature spaces (e.g., unusual combination of low units and high discount).
- **Subsampling Efficiency**: Operates in $O(n \log n)$ time, making it lightweight for real-time investigation.
- **Empirical Ground Truth**: Achieved $F_1 = 0.902$ on our benchmark dataset with 515 rows and 25 injected anomalies.

### 7. Why an immutable Evidence Store (`evi_*`)?
- **Verifiable Provenance**: Every factual claim generated by the system is explicitly bound to a hash-addressed `EvidenceItem` containing the source tool, raw execution data, dataset version hash, and timestamp.
- **Audit Replay**: Enables compliance auditors to inspect exactly which database rows and tool runs justified a given business insight.

### 8. Why calibrate causal language?
- **Correlation vs. Causation (Rule 6)**: Observational business data can prove that *Electronics sales dropped 60% and accounted for 73% of the total regional decline*, but cannot prove definitive real-world causation without randomized controlled trials.
- **Automated Rewriting**: OmniSleuth automatically detects and rewrites speculative terms ("caused by", "due entirely to") into calibrated phrasing ("associated with", "contributed to").

### 9. Why SQLite with the Repository Pattern?
- **Zero-Configuration & Portability**: Embedded serverless database with zero network latency, ensuring complete offline reproducibility.
- **ACID Integrity**: Full transaction support across dataset versions, investigations, tool executions, and validated claims.

### 10. Why a centralized LLM Gateway?
- **Provider Abstraction**: Seamlessly switches between OpenAI, Azure OpenAI, and offline Deterministic Fallback without modifying agent code.
- **Resilience**: Implements bounded exponential backoff retries on transient 429/5xx errors while immediately aborting on permanent 401/403 errors.
- **Audit & Cost Tracing**: Generates unique `llm_req_*` request IDs and calculates exact token and USD cost metrics ($0.00 in offline mode).
- **Semantic Caching**: Dataset-isolated query caching that preserves claim validation guarantees.

### 11. Why 100% Offline-First Architecture?
- **Enterprise Air-Gapped Readiness**: Financial and healthcare enterprises frequently operate in air-gapped environments where external API calls to public LLMs are strictly prohibited.
- **Reproducible Academic Evaluation**: Guarantees that all 105 tests and evaluation benchmarks execute reliably without rate limits, network outages, or API billing.

### 12. Why Orders $\times$ AOV Metric Decomposition?
- **Mathematical Root Cause Analysis**: A revenue shift $\Delta R$ can arise from order volume changes or basket size changes. The `decompose_metric` tool applies exact finite-difference decomposition:
  $$\Delta \text{Revenue} = (\Delta \text{Orders} \times \text{AOV}_0) + (\Delta \text{AOV} \times \text{Orders}_1)$$
  This instantly isolates whether a sales drop was driven by traffic or pricing.

### 13. Why Dual-Layer Guardrails (Input + Output)?
- **Defense in Depth**: Input guardrails filter prompt injections, jailbreaks, and out-of-scope queries before invoking agents. Output guardrails inspect agent responses for secret leaks (API keys), unverified claims, and uncalibrated causal phrasing before returning data to the user.

### 14. Why Docker with Non-Root Security (UID 10001)?
- **Container Hardening**: Running containers as `root` introduces privilege escalation risks in multi-tenant cloud environments. Creating a dedicated `appuser` (UID 10001) adheres to CIS Docker benchmarks.
- **Cloud Run & ECS Compatibility**: Supports dynamic `$PORT` environment variable injection and non-root execution out of the box.

### 15. Why GitHub Actions CI Matrix?
- **Multi-Version Verification**: Automatically tests across Python 3.10, 3.11, and 3.12, executing flake8 linting, full 105-test pytest suites, code coverage export, and Docker container build verification on every pull request.

---

## 3. Top 20 Technical Viva Questions & Exemplary Answers

#### Q1: What is the main innovation of OmniSleuth compared to standard Text-to-SQL?
> **Answer:** Text-to-SQL converts natural language into a single relational query. However, complex business investigations require multi-step analytical planning, temporal trend analysis, metric decomposition, ML anomaly detection, and qualitative context retrieval from business policy documents. OmniSleuth orchestrates all these capabilities through specialized multi-agent roles while programmatically validating every numerical claim against immutable evidence records.

#### Q2: How do you guarantee 0% mathematical hallucination in LLM outputs?
> **Answer:** We enforce Rule 1: The LLM is never asked to compute numbers. All mathematical calculations are computed by deterministic Python tools (`aggregate_metric`, `calculate_trend`, etc.). Furthermore, the `ClaimValidator` scans every sentence in the final output, extracts numerical entities via regex, and verifies that they match raw values in the `EvidenceStore` within a $\pm 1.0\%$ relative tolerance. If a number does not match, it is flagged as `UNSUPPORTED` and redacted.

#### Q3: How does the system handle ambiguous questions like "What happened recently?"
> **Answer:** The `SupervisorAgent` detects ambiguous scope and invokes the `Autonomous Discovery Engine` (`omnisleuth.discovery`). The scanner automatically computes high-level trend shifts, runs multivariate anomaly detection, calculates segment contributions, and constructs an interactive Investigation Tree displaying the most statistically significant findings.

#### Q4: What happens if an external LLM API experiences rate limits or goes down?
> **Answer:** The `LLMGateway` classifies errors into transient (429, 5xx, timeouts) and permanent (401, 403). For transient errors, it performs bounded exponential backoff retries with jitter. If retries are exhausted or no API key is provided, it automatically activates the `DeterministicFallbackProvider`, synthesizing rule-grounded, verifiable responses offline with $0.00 cost.

#### Q5: Explain the difference between Baseline 1, Baseline 2, and OmniSleuth in your evaluation.
> **Answer:**
> - **Baseline 1 (Direct Analytical Tools)** executes individual tools in isolation without agent planning, RAG, claim validation, or guardrails (resulting in 0% validated claims).
> - **Baseline 2 (Fixed Pipeline)** runs a static sequence of tools with evidence tracking, but cannot dynamically adapt to diverse questions.
> - **OmniSleuth (Full Multi-Agent System)** utilizes dynamic supervisor planning, allowlisted tools, hybrid RAG, programmatic claim validation, and guardrails, achieving 100% numerical correctness and 91.88% investigation completeness.

#### Q6: How does your Hybrid RAG combine BM25 and Dense FAISS?
> **Answer:** We compute both sparse BM25 scores and dense FAISS cosine similarity scores for a given query. Rather than attempting to normalize disparate score distributions, we apply **Reciprocal Rank Fusion (RRF)**:
> $$RRF(d) = \sum_{m \in \{BM25, Dense\}} \frac{1}{60 + \text{rank}_m(d)}$$
> The top fused candidates are then passed through a Cross-Encoder semantic reranker, achieving Recall@5 of 1.000 and MRR of 0.950.

#### Q7: What metric decomposition formula is implemented?
> **Answer:** We implement exact finite-difference revenue decomposition:
> $$\Delta \text{Revenue} = (\Delta \text{Orders} \times \text{AOV}_0) + (\Delta \text{AOV} \times \text{Orders}_1)$$
> where $\Delta \text{Orders} = \text{Orders}_1 - \text{Orders}_0$ and $\Delta \text{AOV} = \text{AOV}_1 - \text{AOV}_0$. This separates volume effects from price effects with mathematical exactness.

#### Q8: How does the system defend against Prompt Injection?
> **Answer:** The `InputGuardrail` checks incoming queries against compiled deterministic regular expressions targeting instruction overrides (e.g., "ignore previous instructions"), system prompt leaks, role reassignment, and arbitrary code execution requests (`eval()`, `exec()`, `os.system()`). Adversarial requests are rejected with a standardized safety violation before invoking any agent.

#### Q9: Why did you choose Isolation Forest over standard z-score or IQR?
> **Answer:** Z-score and IQR are univariate methods that assume independent, linear distributions. Real-world sales data contains multi-dimensional anomalies (e.g., a normal transaction amount that is anomalous given a specific region, product tier, and discount combination). Isolation Forest isolates points in high-dimensional space in $O(n \log n)$ time without parametric distribution assumptions.

#### Q10: What is the purpose of SHA-256 dataset hashing in your ingestion engine?
> **Answer:** Datasets in production can be modified or re-uploaded. By computing a deterministic SHA-256 hash of the raw dataset content upon ingestion, every `EvidenceItem` and cached LLM request is permanently bound to a specific dataset version (`dataset_version`), preventing stale or cross-dataset evidence corruption.

#### Q11: How many automated tests exist, and what do they cover?
> **Answer:** There are **105 automated unit and integration tests** passing with 0 failures across 17 test modules, covering ingestion, profiling, analytical tools, evidence store, claim validation, LangGraph state machine, hybrid RAG, ML anomaly detection, persistence, UI imports, guardrails, LLM Gateway, and container deployment.

#### Q12: How does OmniSleuth handle insufficient evidence?
> **Answer:** When analytical tools return empty datasets or hypotheses lack supporting data, OmniSleuth does not invent plausible answers. The `InvestigationAgent` explicitly marks the hypothesis as `INCONCLUSIVE`, documents the evidence gap, and returns a bounded report advising what data is missing.

#### Q13: What are the exact latency numbers observed in your benchmark?
> **Answer:** In our 40-question benchmark run (`exp_20261007_201806`), OmniSleuth achieved a median execution latency (p50) of **33.52 ms** and a 95th percentile latency (p95) of **49.85 ms** in local offline execution.

#### Q14: How does your Semantic Cache work without breaking claim validation?
> **Answer:** The cache is partitioned by the SHA-256 hash of the active dataset version, the prompt text, and tool parameters. Because cached responses contain verified claim structures linked to immutable evidence IDs for that exact dataset version, cache hits preserve full claim validation guarantees.

#### Q15: Explain why you use a non-root user in Docker.
> **Answer:** In container security, running as `root` allows potential container-escape vulnerabilities to gain root access on the host VM. We create and enforce a dedicated non-root user `appuser` with UID `10001`, adhering to CIS security benchmarks and Google Cloud Run / AWS ECS best practices.

#### Q16: Can OmniSleuth be deployed on Google Cloud Run?
> **Answer:** Yes. The Dockerfile is specifically configured with dynamic `$PORT` environment variable expansion (`${PORT:-8501}`), non-root execution, container health checks at `/_stcore/health`, and binding to `0.0.0.0`.

#### Q17: What is the difference between Faithfulness and Answer Relevance in your evaluation?
> **Answer:**
> - **Faithfulness (100.0%)** measures whether every claim made in the synthesis is mathematically supported by the retrieved evidence without hallucination.
> - **Answer Relevance (74.25%)** measures whether the response directly addresses the specific question asked using ground-truth keyword recall.

#### Q18: What are the main limitations of your system?
> **Answer:**
> 1. It is not a causal discovery engine; it measures observational statistical association and contribution percentages.
> 2. Tabular processing runs in-memory, making it suitable for datasets up to several hundred thousand rows rather than distributed petabyte-scale clusters.
> 3. Plotly charts are emitted as structured JSON/HTML data artifacts rather than parsed visually by Vision-Language Models.

#### Q19: Why do you avoid claiming causality in business reports?
> **Answer:** Observational data cannot prove real-world causality without randomized A/B experiments. Stating that *"Marketing campaign X caused the 20% sales spike"* is statistically unfounded if seasonal demand or competitor stockouts occurred simultaneously. OmniSleuth rewrites such assertions to calibrated phrasing like *"Marketing campaign X was associated with an increase of 20%"*.

#### Q20: What future improvements are planned for OmniSleuth?
> **Answer:** Future enhancements include:
> 1. Integrating formal constraint-based causal discovery algorithms (e.g., PC Algorithm);
> 2. Connecting in-memory analytical query engines like DuckDB and Apache Arrow for gigabyte-scale tables; and
> 3. Adding live WebSocket token streaming to the Streamlit UI.

---

## 4. Summary Defense Checklist

Before stepping into your presentation or viva defense, ensure you have:
- [x] Verified that all **105 tests pass cleanly** (`python -m pytest -v`).
- [x] Tested the local Streamlit workspace (`streamlit run omnisleuth/ui/app.py`).
- [x] Memorized the **2-minute elevator pitch**.
- [x] Reviewed the benchmark results from [`omnisleuth/evaluation/results/exp_20261007_201127.json`](file:///d:/OmniSleuth/omnisleuth/evaluation/results/exp_20261007_201127.json) (100% numerical correctness, 0% unsupported claims, $F_1 = 0.902$).
- [x] Reviewed the 7 core design rules.
