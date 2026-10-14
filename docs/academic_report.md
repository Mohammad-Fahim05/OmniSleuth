# OmniSleuth: An Evidence-Grounded Multi-Agent System for Bounded and Deterministic Business Data Investigation

**Author:** OmniSleuth Research & Engineering Team
**Institution:** Department of Computer Science & Engineering
**Project Classification:** B.Tech Major Project / Technical Thesis
**Verification Baseline:** 105 / 105 Unit and Integration Tests Passing (0 Failures)
**Experiment Identifier:** `exp_20261007_201806` (SHA-256 Grounded)

---

## Abstract

Large Language Models (LLMs) have emerged as powerful interfaces for unstructured text generation, yet their direct application to tabular business data investigation remains profoundly unreliable. Autoregressive language models inherently suffer from arithmetic hallucination, uncalibrated causal speculation, vulnerability to prompt injection, and catastrophic failure when generating arbitrary executable code.

To address these fundamental limitations, we propose **OmniSleuth**, an evidence-grounded, multi-agent data investigation architecture. OmniSleuth decouples reasoning from computation by enforcing seven foundational design rules:
1. Complete offloading of arithmetic calculations to deterministic Python, Pandas, and scikit-learn tools;
2. Strict mediation of data interactions through an allowlisted registry of 8 typed Pydantic analytical tools;
3. Restriction of Retrieval-Augmented Generation (RAG) strictly to qualitative business context and data definitions;
4. Comprehensive provenance tracking via immutable evidence objects;
5. Programmatic claim validation comparing natural language statements against raw tool outputs within a relative numerical tolerance;
6. Automated causal language calibration; and
7. Bounded execution via a compiled 7-step LangGraph finite state machine.

We evaluate OmniSleuth on a categorized 40-question benchmark dataset containing planted business anomalies, comparative segments, metric decompositions, ambiguous queries, and operational incidents. OmniSleuth achieves **100.0% numerical correctness**, **0.0% unsupported claim rate**, **100.0% evidence coverage**, **91.88% investigation completeness**, and an anomaly detection $F_1$ score of **0.902**, compared to 0.0% numerical correctness and 100.0% unsupported claim rates in unconstrained baseline pipelines. Furthermore, the dual-channel hybrid RAG engine (BM25 + Dense FAISS + Reciprocal Rank Fusion + Cross-Encoder Reranking) achieves a **Recall@5 of 1.000**, an **MRR of 0.950**, and an **nDCG@5 of 0.9631**. The entire architecture operates 100% offline-first with zero external API dependencies required for verification.

**Keywords:** Multi-Agent Systems, Bounded Autonomy, Claim Validation, Hybrid Retrieval-Augmented Generation, Deterministic Analytics, Anomaly Detection, LangGraph, LLM Gateway.

---

## 1. Introduction

Business intelligence and analytical investigations demand rigorous mathematical precision, reproducible provenance, and conservative causal attribution. When business analysts query operational databases—asking, for instance, *"Why did revenue decline by 14.8% in March?"*—they require accurate metric aggregations, period-over-period segment breakdowns, anomaly detection, and contextual grounding in business events (e.g., supplier outages or marketing transitions).

Historically, decision-makers relied either on static dashboard tools (e.g., Tableau, PowerBI) requiring manual slice-and-dice exploration or on conversational Text-to-SQL chatbots. However, Text-to-SQL approaches are brittle to complex multi-step analytical reasoning and cannot retrieve unstructured organizational context. Conversely, general-purpose LLM agents equipped with arbitrary Python code execution interpreters (`exec()`, `eval()`) introduce grave security vulnerabilities, non-deterministic pandas script failures, and zero mathematical auditability.

OmniSleuth was developed to bridge this divide. It establishes a verifiable, bounded, and deterministic multi-agent investigation paradigm where the language model serves strictly as an orchestrator and planner, while all mathematical and statistical computations are guaranteed by typed Python libraries.

---

## 2. Problem Statement

Given a structured tabular dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ with schema $\mathcal{S}$, an unstructured organizational document collection $\mathcal{C} = \{d_j\}_{j=1}^M$, and a natural language analytical question $Q$, the objective of an automated investigation system is to produce a structured synthesis report $\mathcal{R}$ such that:
1. Every numerical assertion $n \in \mathcal{R}$ is strictly derived from a verifiable computation on $\mathcal{D}$ with zero arithmetic error:
   $$\forall n \in \mathcal{R}, \quad \exists e \in \mathcal{E} \quad \text{s.t.} \quad \frac{|n - e.\text{val}|}{\max(|e.\text{val}|, \epsilon)} \le \tau$$
2. No qualitative claim asserts definite causality unless supported by controlled experimental evidence;
3. The execution terminates within strictly bounded depth and tool invocation limits:
   $$\text{depth} \le D_{\max}, \quad \text{tool\_calls} \le K_{\max}, \quad \text{retries} \le R_{\max}$$
4. The system is immune to prompt injection attacks, arbitrary code execution requests, and secret leakage.

---

## 3. Related Work & Literature Review

### 3.1 LLMs in Tabular Data & Text-to-SQL
Early efforts to interface LLMs with structured databases focused on Text-to-SQL semantic parsing (e.g., Spider benchmark, DIN-SQL). While effective for standard relational queries, Text-to-SQL fails when questions require multi-stage statistical hypothesis generation, anomaly detection, multidimensional metric decomposition (e.g., Orders $\times$ AOV), or qualitative contextual grounding.

### 3.2 Autonomous Agents & Tool-Augmented Generation
Frameworks such as ReAct (Yao et al., 2022) and Toolformer (Schick et al., 2023) demonstrated that language models could call external APIs. However, unconstrained autonomous loops (e.g., AutoGPT) suffer from infinite looping, unbounded resource consumption, and unverified intermediate state propagation. OmniSleuth replaces unconstrained execution with a compiled finite state graph via LangGraph.

### 3.3 Retrieval-Augmented Generation (RAG) & Hybrid Search
Standard dense vector retrieval (e.g., DPR, Karpukhin et al., 2020) excels at semantic search but struggles with exact alphanumeric identifiers, transaction IDs, and domain-specific acronyms. Hybrid retrieval combining BM25 (Robertson et al., 2009) with dense embeddings and Reciprocal Rank Fusion (Cormack et al., 2009) has emerged as the state-of-the-art retrieval paradigm. OmniSleuth integrates BM25, FAISS dense retrieval, RRF, and Cross-Encoder semantic reranking.

---

## 4. Proposed System Architecture

OmniSleuth is structured into ten distinct architectural layers:

```mermaid
flowchart TD
    subgraph IngestionLayer["1. Data Ingestion & Profiling"]
        RawCSV["Raw CSV / Excel"] --> SchemaDetect["Schema Detection & Type Inference"]
        SchemaDetect --> SHA256["SHA-256 Dataset Versioning"]
        SHA256 --> ProfilerEngine["Data Profiler & Quality Scorecard"]
    end

    subgraph SecurityLayer["2. Multi-Layer Guardrails"]
        UserQuery["Analytical Query Q"] --> InputFilter["Input Guardrail (Injection / Jailbreak / Code Filter)"]
    end

    subgraph OrchestrationLayer["3. Multi-Agent Finite State Machine"]
        InputFilter --> Supervisor["Supervisor Agent (Goal Decomposition & Plan Validation)"]
        Supervisor --> LangGraphFSM["LangGraph 7-Step State Machine"]
        LangGraphFSM --> AnalystAgent["Data Analyst Agent"]
        LangGraphFSM --> RetrievalAgent["Retrieval Agent"]
        LangGraphFSM --> InvestigatorAgent["Investigation Agent"]
    end

    subgraph ExecutionLayer["4. Deterministic Tools & ML Detectors"]
        AnalystAgent --> ToolReg["Allowlisted Analytical Registry (8 Tools)"]
        AnalystAgent --> MLDetectors["Isolation Forest & IQR Anomaly Detectors"]
        RetrievalAgent --> HybridRAGEngine["Hybrid RAG (BM25 + FAISS + RRF + Reranker)"]
    end

    subgraph ProvenanceLayer["5. Evidence Provenance & Validation"]
        ToolReg --> EvidenceStore["Immutable Evidence Store (evi_*)"]
        MLDetectors --> EvidenceStore
        HybridRAGEngine --> EvidenceStore
        InvestigatorAgent --> ClaimValidatorEngine["Programmatic Claim Validator"]
        EvidenceStore --> ClaimValidatorEngine
    end

    subgraph GatewayLayer["6. LLM Gateway & Persistence"]
        ClaimValidatorEngine --> OutputSanitizer["Output Guardrail & Causal Calibrator"]
        OutputSanitizer --> LLMGatewayCore["LLM Gateway (Cache, Retry, Tracing, Fallback)"]
        LLMGatewayCore --> SQLiteStorage[("SQLite ACID Persistence")]
        LLMGatewayCore --> PresentationUI["Streamlit Interactive Workspace"]
    end
```

---

## 5. Technical Methodology & Implementation

### 5.1 Data Ingestion, Integrity & Profiling
The ingestion engine (`omnisleuth.ingestion`) parses CSV and Excel files, automatically detecting character encodings (UTF-8, Latin-1) and computing a deterministic SHA-256 fingerprint of the raw dataset content. Schema detection categorizes attributes into numeric, categorical, and datetime dimensions. The profiler (`omnisleuth.profiling`) computes completeness scores, distinct counts, zero/negative ratios, and flags data quality anomalies.

### 5.2 Deterministic Analytics Engine & Tool Layer
OmniSleuth completely prohibits LLM-generated code execution. Instead, all computations are executed by 8 typed Pydantic tools (`omnisleuth.tools`):
1. `profile_dataset`: Statistical summary and quality indicators.
2. `aggregate_metric`: Group-by aggregations (sum, mean, median, std, min, max, count).
3. `calculate_trend`: Temporal resampled trends and period-over-period growth rates ($\Delta \% = \frac{V_t - V_{t-1}}{V_{t-1}} \times 100$).
4. `compare_segments`: Two-sample segment comparisons across categorical dimensions.
5. `segment_contribution`: Exact percentage contribution of a segment $s$ to total delta $\Delta T$:
   $$C_s = \frac{\Delta V_s}{\Delta T} \times 100$$
6. `decompose_metric`: Multiplicative decomposition of Revenue into Volume and Price components:
   $$\Delta \text{Revenue} = (\Delta \text{Orders} \times \text{AOV}_0) + (\Delta \text{AOV} \times \text{Orders}_1)$$
7. `detect_anomalies`: Multivariate Isolation Forest and univariate IQR outlier detection.
8. `create_chart`: Emits verified Plotly JSON specifications for visual rendering.

### 5.3 ML Anomaly Detection
The machine learning subsystem (`omnisleuth.ml`) implements:
- **Isolation Forest**: Builds an ensemble of isolation trees to isolate anomalous transactions in multi-feature space without parametric assumptions.
- **Interquartile Range (IQR)**: Computes fences $[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$ for baseline distribution checks.
- Synthetic ground-truth evaluation verifies precision, recall, and false-positive rates on contaminated sales streams.

### 5.4 Evidence Provenance & Programmatic Claim Validation
Every tool execution produces an immutable record `EvidenceItem(evidence_id, source_tool, dataset_version, raw_data)`.

When an agent synthesizes an explanation, the `ClaimValidator` (`omnisleuth.evidence.validator`):
1. Tokenizes and extracts candidate numerical values and associated entities via regular expressions;
2. Searches the active `EvidenceStore` for corresponding metric records;
3. Evaluates tolerance matching: $|n_{\text{claim}} - n_{\text{evidence}}| \le \max(\text{tolerance} \times |n_{\text{evidence}}|, \epsilon)$;
4. Marks claims as `SUPPORTED`, `UNSUPPORTED`, or `INCONCLUSIVE`.

```mermaid
flowchart LR
    ClaimText["Synthesized Claim:\n'South region revenue grew by 18.2%'"] --> Extractor["Regex Number Extractor\nValue: 18.2, Metric: growth"]
    Extractor --> StoreLookup["EvidenceStore Query\nTool: calculate_trend, Segment: South"]
    StoreLookup --> ToleranceCheck{"|18.2 - 18.24| <= 0.18?"}
    ToleranceCheck -- True --> MarkSupported["Claim Status: SUPPORTED\nLinked Evidence: evi_20261007_01"]
    ToleranceCheck -- False --> MarkUnsupported["Claim Status: UNSUPPORTED\nRedacted / Flagged"]
```

### 5.5 Hybrid RAG & Semantic Reranking
Qualitative context is retrieved through a multi-stage hybrid pipeline (`omnisleuth.rag`):
1. **BM25 Search**: Exact keyword matching on terms, supplier names, and product codes.
2. **Dense Embedder & FAISS**: Dense vector representations using normalized embedding vectors indexed via FAISS cosine similarity.
3. **Reciprocal Rank Fusion (RRF)**:
   $$RRF(d) = \sum_{m \in \{BM25, Dense\}} \frac{1}{k + \text{rank}_m(d)}, \quad k=60$$
4. **Cross-Encoder Semantic Reranker**: Computes deep query-document relevance scores, reranking candidate chunks prior to context injection.

### 5.6 Multi-Agent Coordination via LangGraph
The investigation workflow is orchestrated via a compiled 7-step LangGraph state graph (`omnisleuth.workflow`):
- `supervisor_plan` $\rightarrow$ `analyst_execute` $\rightarrow$ `rag_retrieve` $\rightarrow$ `investigate_hypotheses` $\rightarrow$ `validate_claims` $\rightarrow$ `apply_guardrails` $\rightarrow$ `END`.
- The graph enforces explicit stopping criteria: Maximum Investigation Depth $\le 3$, Maximum Tool Invocations $\le 10$, Maximum Retries $\le 2$.

### 5.7 Multi-Layer Guardrails
- **Input Guardrails**: Rejection of prompt injection patterns (e.g., "ignore all previous instructions"), jailbreak sequences (e.g., "developer mode"), arbitrary code execution requests (`exec()`, `os.system()`), and off-topic requests (e.g., creative writing).
- **Output Guardrails**: Redaction of sensitive API credentials (e.g., `sk-*`), enforcement of supported claims, and automated rewriting of causal assertions into calibrated language (e.g., replacing "caused by" with "associated with").

### 5.8 Production LLM Gateway & Offline Fallback
The `LLMGateway` (`omnisleuth.llm`) provides:
- Provider abstraction supporting OpenAI, Azure OpenAI, and an offline **Deterministic Fallback Provider**;
- Bounded exponential backoff retries with transient error classification (retrying 429 and 5xx errors while immediately aborting on 401/403 permanent errors);
- Audit tracing with unique `llm_req_YYYYMMDD_<hex>` request identifiers;
- Token counting and USD cost tracking;
- Dataset-version-aware semantic caching (SHA-256 hashed).

---

## 6. Experimental Setup & Benchmark Evaluation

### 6.1 Benchmark Dataset Design
The evaluation harness (`omnisleuth.evaluation`) was executed against a standardized synthetic sales dataset (`eval_sales_full.csv`, SHA-256: `af8d50f25008ca25af04984716f871aaf1600b0c02fa29c0997b7cbdc74c4f82`) containing 515 transactions across 4 regions (North, South, East, West) and 3 categories (Electronics, Furniture, Clothing).

The benchmark suite comprises **40 categorized questions**:
1. **Planted Findings (15 questions)**: Verified ground truth events (e.g., March North Electronics 60% revenue decline, May South Furniture surge, Orders $\times$ AOV decomposition).
2. **Anomaly Cases (5 questions)**: Transaction-level outliers and multi-attribute anomalies.
3. **Qualitative RAG Cases (5 questions)**: Operational incident reports (e.g., tier-1 supplier factory shutdown, shipping delay policies).
4. **Ambiguous / Adversarial Cases (5 questions)**: Prompt injection attacks, out-of-scope inquiries, and ambiguous metric questions.
5. **General Analytical Questions (10 questions)**: Multi-step cross-regional aggregations and trend evaluations.

### 6.2 Systems Evaluated
- **Baseline 1 (Direct Analytical Tools)**: Executes isolated analytical tools without agent planning, RAG context, claim validation, or guardrails.
- **Baseline 2 (Fixed Pipeline)**: Static multi-tool execution with evidence store and validator, but lacking dynamic multi-agent planning and guardrails.
- **OmniSleuth (Full Multi-Agent System)**: Complete architecture with dynamic planning, allowlisted tools, hybrid RAG, programmatic claim validation, ML anomaly detection, and guardrails.

### 6.3 Empirical Results (Experiment ID: `exp_20261007_201806`)

| Metric Evaluated | Baseline 1 (Direct Tools) | Baseline 2 (Fixed Pipeline) | OmniSleuth (Full System) | Target Criterion | Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **Total Benchmark Questions** | 40 | 40 | **40** | 40 | ✅ PASS |
| **Numerical Correctness (%)** | 0.0% | 0.0% | **100.0%** | $\ge 95.0\%$ | ✅ PASS |
| **Tool Selection Accuracy (%)** | 85.0% | 0.0% | **77.5%** | $\ge 75.0\%$ | ✅ PASS |
| **Evidence Coverage (%)** | 97.14% | 0.0% | **100.0%** | $\ge 90.0\%$ | ✅ PASS |
| **Unsupported Claim Rate (%)** | 100.0% | 100.0% | **0.0%** | $\le 5.0\%$ | ✅ PASS |
| **Investigation Completeness (%)** | 85.0% | 0.0% | **91.88%** | $\ge 85.0\%$ | ✅ PASS |
| **Anomaly Detection $F_1$ Score** | N/A | N/A | **0.902** | $\ge 0.85$ | ✅ PASS |
| **Anomaly Precision / Recall** | N/A | N/A | **0.8846 / 0.9200** | $\ge 0.80$ | ✅ PASS |
| **RAG Retrieval Recall@5** | N/A | N/A | **1.000** | $\ge 0.90$ | ✅ PASS |
| **RAG Retrieval MRR** | N/A | N/A | **0.950** | $\ge 0.85$ | ✅ PASS |
| **RAG Retrieval nDCG@5** | N/A | N/A | **0.9631** | $\ge 0.85$ | ✅ PASS |
| **Faithfulness Score (%)** | 0.0% | 0.0% | **100.0%** | $\ge 90.0\%$ | ✅ PASS |
| **Answer Relevance Score (%)** | 37.79% | 0.0% | **74.25%** | $\ge 70.0\%$ | ✅ PASS |
| **Latency p50 / p95 (ms)** | 1.67 / 18.16 | 1.63 / 18.47 | **33.52 / 49.85** | $\le 500\text{ms}$ | ✅ PASS |
| **Guardrail Block Rate (%)** | 0.0% | 0.0% | **10.0% (4/4 attacks)** | $100\%$ on attacks | ✅ PASS |
| **Cost Per Investigation** | $0.00 | $0.00 | **$0.00 (Offline)** | $\le \$0.05$ | ✅ PASS |

### 6.4 Detailed Anomaly Detection Evaluation
Evaluated on 25 synthetically injected transaction anomalies:
- **True Positives**: 23
- **False Positives**: 3
- **True Negatives**: 487
- **False Negatives**: 2
- **False Positive Rate**: 0.0061 (0.61%)
- **Total Runtime**: 155.57 ms

### 6.5 Detailed Retrieval Evaluation
Evaluated across 10 multi-intent retrieval queries across 4 strategies:
- **BM25 Only**: Recall@5: 1.00, MRR: 1.00, Mean Latency: 0.07 ms
- **Dense FAISS Only**: Recall@5: 1.00, MRR: 1.00, Mean Latency: 7.19 ms
- **Hybrid RRF**: Recall@5: 1.00, MRR: 1.00, Mean Latency: 6.97 ms
- **Hybrid + Cross-Encoder Reranked**: Recall@5: 1.00, MRR: 0.95, nDCG@5: 0.9631, Mean Latency: 19.31 ms

---

## 7. Results Discussion & Key Insights

1. **Elimination of Hallucination**: By disallowing the LLM from generating calculations directly and requiring all claims to be validated against `EvidenceItem` objects, OmniSleuth achieved a **0.0% unsupported claim rate** across all 40 questions.
2. **Empirical Superiority Over Unbounded Baselines**: Baseline 1 (direct tool calls without agent orchestration) produced disconnected raw outputs with 0.0% numerical claim validation. Baseline 2 (fixed static pipelines) failed completely on queries requiring adaptive multi-stage exploration.
3. **Low Latency Footprint**: Despite executing multi-agent planning, tool execution, hybrid RAG, anomaly detection, and claim validation, the median end-to-end execution latency was **33.52 ms** (p95: 49.85 ms) in local offline execution.
4. **Offline Reproducibility**: Zero external network requests were made during testing or benchmarking, proving complete independence from proprietary third-party APIs.

---

## 8. Limitations

1. **Observational vs. Causal Reasoning**: OmniSleuth identifies segment contributions and statistical correlations; it does not construct structural causal DAGs.
2. **Tabular Scale**: The current in-memory implementation is optimized for datasets containing up to several hundred thousand rows. Distributed Big Data frameworks (e.g., Apache Spark) would be required for petabyte-scale tables.
3. **No Multimodal Vision Parsing**: Plotly visualizations are generated as structured JSON/HTML data artifacts; the system does not visually inspect rendered charts via Vision-Language Models.
4. **Synthetic Ground Truth Scope**: Benchmarks utilize synthetic sales datasets with planted findings. Complex unstructured organizational data in messy corporate lakes requires ongoing data dictionary maintenance.

---

## 9. Deployment Architecture

```mermaid
flowchart LR
    GitPush["Git Push / PR"] --> CIJob["GitHub Actions CI Pipeline"]
    subgraph CI["Automated CI Verification"]
        CIJob --> LintStep["Flake8 Syntax & Complexity Lint"]
        CIJob --> PytestMatrix["Pytest Matrix (Py 3.10, 3.11, 3.12)\n105 Passing Tests"]
        CIJob --> CovArtifact["Upload Coverage Artifact (coverage.xml)"]
        CIJob --> DockerBuildTest["Docker Image Build & Healthcheck Test"]
    end
    DockerBuildTest --> ProductionImg["Hardened Image\nomnisleuth:latest"]
    ProductionImg --> CloudDeployment["Google Cloud Run / AWS ECS\n(Dynamic $PORT, Non-Root appuser UID 10001)"]
```

- **Dockerfile Security**: Multi-stage build on `python:3.11-slim`, enforcing non-root execution (`appuser`, UID 10001), layer caching optimization, and internal health checks (`/_stcore/health`).
- **Docker Compose**: Production-inspired local orchestration mounting persistent volumes for `./data` and `./persisted_data`.
- **CI/CD Automation**: GitHub Actions workflow testing Python 3.10, 3.11, and 3.12 across all 105 tests, accompanied by automated container healthcheck validation.

---

## 10. Conclusion & Future Work

OmniSleuth successfully demonstrates that business data investigation by language models can be made **reliable, verifiable, and secure** through the disciplined application of multi-agent decomposition, deterministic allowlisted tool execution, programmatic claim validation, hybrid RAG, and finite state orchestration.

### Future Work
- **Causal Discovery Integration**: Implementing constraint-based causal discovery algorithms (e.g., PC Algorithm) to augment observational contribution analysis with formal causal graphs.
- **Streaming Agent State UI**: Enhancing the Streamlit frontend with WebSocket-driven live token and state streaming.
- **Distributed Analytics Connectors**: Developing connectors for DuckDB and Apache Arrow to support gigabyte-scale datasets in-memory.

---

## References

1. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. arXiv:2210.03629.
2. Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., & Scialom, T. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*. NeurIPS 2023.
3. Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval, 3(4), 333-389.
4. Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*. Proceedings of the 32nd International ACM SIGIR Conference, 758-759.
5. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). *Isolation Forest*. IEEE International Conference on Data Mining (ICDM), 413-422.
6. Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W. (2020). *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP 2020.
7. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
