# OmniSleuth — Interview & Presentation Master Defense Guide

This guide prepares you to explain and defend every architectural decision of **OmniSleuth** in B.Tech project evaluations, technical presentations, and AI/ML system design interviews.

---

### 1. Why multiple agents instead of a single prompt?
- **Separation of Concerns**: A single LLM prompt trying to plan, calculate, search documents, and validate results suffers from context dilution and high failure rates.
- **Role Specialization**:
  - `SupervisorAgent`: Focuses solely on goal decomposition and bounding limits.
  - `DataAnalystAgent`: Specializes in typed analytical tool selection.
  - `RetrievalAgent`: Handles contextual knowledge retrieval.
  - `InvestigationAgent`: Generates and systematically tests hypotheses against stored evidence.
- **Auditability**: Having distinct agents producing typed state records ensures full traceability.

---

### 2. Why LangGraph?
- **State Machine Control**: Unlike unbounded autonomous loops (e.g. naive AutoGPT), LangGraph enforces explicit state transitions, cycle prevention, and bounded execution paths.
- **Typed Shared State**: Nodes communicate strictly through typed schemas (`InvestigationState`), preventing silent state corruption.
- **Deterministic Checkpointing & Resumption**: Enables pausing, human-in-the-loop review, and audit replay.

---

### 3. Why RAG?
- **Domain Context without Number Crunching**: Sales data alone cannot explain *why* an event happened (e.g., a 3-week tier-1 microchip supplier factory outage in March).
- **KPI & Schema Grounding**: Ingests data dictionaries and business formulas so agents use consistent metric definitions.
- **Rule 3 Enforcement**: RAG is strictly restricted to qualitative context; numerical calculations are always computed deterministically.

---

### 4. Why NOT let the LLM write arbitrary Python code?
- **Security Vulnerability**: Arbitrary code execution (`eval()` / `exec()`) introduces Remote Code Execution (RCE) and data corruption risks.
- **Non-Deterministic Edge Cases**: LLM-generated code frequently hallucinates variable names, produces pandas syntax errors, or performs incorrect statistical approximations.
- **Allowlisted Tools (Rule 2)**: Restricting the LLM to pre-validated, typed tools (with Pydantic schemas) guarantees 100% deterministic, bug-free, and safe calculation.

---

### 5. How are hallucinated numbers prevented?
- **Rule 1 (Deterministic Execution)**: Numerical metrics (sums, trends, deltas, segment contributions) are computed via Pandas, NumPy, and scikit-learn.
- **Rule 5 (Programmatic Claim Validation)**: The `ClaimValidator` extracts every number from generated claims and checks them against raw tool outputs in the `EvidenceStore`. Any number not matching within tolerance is marked `UNSUPPORTED` and flagged.

---

### 6. Why Isolation Forest for Anomaly Detection?
- **Multivariate Capability**: While IQR works well for univariate outliers, Isolation Forest isolates anomalies across complex multidimensional feature interactions without assuming normal distribution.
- **Sub-sampling Efficiency**: Isolation Forest operates in $O(n \log n)$ time, making it lightweight and real-time.
- **Clear Scoring**: Produces normalized anomaly scores that rank extreme transactions.

---

### 7. Why SQLite instead of PostgreSQL or MongoDB?
- **Zero-Configuration & Portability**: Embedded serverless database with zero network latency, ideal for single-node deployments and reproducible research.
- **Full ACID Transactions & Foreign Keys**: Guarantees integrity across dataset versions, investigations, tool executions, and evidence items.

---

### 8. Why deterministic tools?
- Financial and business analytics require absolute mathematical precision. An LLM cannot be trusted to accurately multiply large floating-point numbers or aggregate 100,000 rows.

---

### 9. Why calibrate language and avoid claiming causality?
- **Correlation vs Causation (Rule 6)**: Observational data can prove that *Electronics sales dropped 60% and contributed 73% to total decline*, but cannot prove definite causality without randomized controlled experiments.
- **Calibrated Contributor Phrasing**: OmniSleuth automatically detects and rewrites "caused by" into "associated with" or "contributed to".

---

### 10. What happens when evidence is insufficient?
- **Bounded Stopping & Inconclusive Status**: When tools return empty or ambiguous results, OmniSleuth explicitly marks claims as `INCONCLUSIVE` and reports evidence gaps rather than inventing explanations.

---

### 11. How is the system evaluated?
- **Synthetic Ground-Truth Benchmarking**: Tested against 30–50 benchmark questions with planted phenomena (e.g. March North Electronics drop, May South Furniture surge, outlier spikes).
- **Objective Metrics**:
  - *Numerical Correctness Rate*
  - *Tool Selection Accuracy*
  - *Evidence Coverage*
  - *Unsupported Claim Rate*
  - *Retrieval Relevance*
  - *p50 / p95 Execution Latency*
