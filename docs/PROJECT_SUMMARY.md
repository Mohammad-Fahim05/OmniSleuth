# OmniSleuth — Project Presentation & Executive Summary

This document provides a concise portfolio summary, interview pitch, implemented feature matrix, and live demo script for **OmniSleuth**.

---

## 1. Resume-Ready Project Description

**OmniSleuth — Evidence-Grounded Multi-Agent Data Investigation System**
*Technologies: Python, Streamlit, Pandas, Scikit-Learn, Plotly, SQLite, Pydantic, OpenRouter/OpenAI API, Docker, Pytest*

- Architected a verifiable analytical investigation platform that decouples LLM semantic planning from deterministic Python computation, eliminating mathematical hallucinations and Remote Code Execution (RCE) risks.
- Implemented an immutable Evidence Store and automated numerical claim validation engine that programmatically extracts figures from natural language responses and validates them against empirical calculations within strict tolerances.
- Developed a 6-tab responsive analytics workspace in Streamlit (Investigation, Autonomous Discovery, Data Quality Profiling, ML Anomaly Explorer, Cryptographic Provenance, and Benchmark Evaluation) with centralized WCAG-accessible design tokens.
- Engineered multi-layered upload security controls, defeating disguised executables, decompression bombs, path traversal attacks, and column-count denial-of-service attempts.
- Established an ACID persistence layer in SQLite with automated additive migrations, busy-timeout concurrency locks (30s), and strict session-isolated repository boundaries.
- Built a 40-question empirical evaluation harness achieving 100% numerical correctness, 100% evidence coverage, and 0% unsupported claims across 509 automated tests.

---

## 2. 60–90 Second Elevator Pitch

> *"When organizations try using generative AI for business intelligence, they immediately run into three major blockers: the LLM hallucinates calculations, unconstrained code execution creates severe security risks, and models make causal claims without empirical justification.*
>
> *I built OmniSleuth to solve this by fundamentally separating planning from computation. In OmniSleuth, the language model is strictly a planner and narrator—it never performs arithmetic or runs arbitrary code.*
>
> *All computations are executed by eight allowlisted, typed Pydantic tools running deterministic Python and pandas math. Every calculation mints an immutable evidence record. A programmatic claim validation engine extracts numbers, currency values, and percentages from the answer and verifies each against stored evidence before presenting it to the user. If a claim cannot be proven from data, it is flagged.*
>
> *The platform features a six-tab analytics workspace in Streamlit with autonomous anomaly detection, cryptographic provenance tracking, and an empirical evaluation harness. Across 509 automated tests, OmniSleuth delivers 100% numerical accuracy, zero unsupported claims, and complete offline reproducibility without requiring external API keys."*

---

## 3. Implemented Features vs. Future Work

| Domain | Implemented in Current Codebase (Verified) | Future Roadmap / Production Scope |
|---|---|---|
| **Query Planning** | • Deterministic rule-based planner with schema fuzzy matching<br/>• LLM-assisted structured JSON planner (OpenRouter/OpenAI)<br/>• Schema and tool allowlist validation before execution | • Multi-turn conversational query memory<br/>• Self-correcting plan re-planning loops |
| **Analytics Engine** | • 8 allowlisted Pydantic analytical tools<br/>• Pure pandas aggregations, multi-period trends, segment comparisons<br/>• Order volume $\times$ AOV metric decomposition | • Direct SQL execution on live data warehouses (Snowflake, BigQuery)<br/>• Geospatial analysis |
| **Machine Learning** | • Isolation Forest anomaly detector (scikit-learn)<br/>• Interquartile Range (IQR) anomaly detector<br/>• Non-causality statistical notices and Box plots | • Time-series forecasting (Prophet, ARIMA)<br/>• Supervised churn/classification modeling |
| **Evidence & Provenance** | • Immutable Evidence Store with execution tracking<br/>• Programmatic regex number/currency/percent extraction<br/>• Relative and absolute mathematical tolerance matching<br/>• System confidence metadata filtering | • Exportable cryptographically signed PDF audit certificates<br/>• Block-chain ledger provenance anchoring |
| **Security & Ingestion** | • 50 MB upload limits and 1,000 column DoS prevention<br/>• MIME & binary magic byte inspection (ELF, PE, PNG, PDF)<br/>• OpenXML Excel validation & ZIP-bomb ratio protection<br/>• Atomic file streaming and path traversal prevention | • Virus scanning daemon (ClamAV integration)<br/>• Client-side file pre-encryption |
| **User Interface** | • 6-tab Streamlit workspace with custom CSS design tokens<br/>• Responsive layouts across 1440px, 1200px, 1024px, 768px<br/>• Unified Plotly layout theme with accessible colorway | • Dark/light theme user toggle<br/>• Custom dashboard creation and widget pinning |
| **Persistence & Auth** | • SQLite with WAL mode & 30s busy timeout<br/>• Additive non-destructive schema migrations<br/>• Pseudonymous session-isolated repository queries | • PostgreSQL migration for multi-pod clustering<br/>• Enterprise OAuth2 / OIDC authentication (Okta, Azure AD)<br/>• Role-Based Access Control (RBAC) |
| **Deployment** | • Standardized `Dockerfile` with locked Python 3.11-slim base<br/>• Non-root `appuser` (UID 10001)<br/>• `.dockerignore` excluding secrets, local DBs, and audit artifacts<br/>• Internal healthcheck (`/_stcore/health`) | • Production Nginx/Caddy reverse proxy with TLS 1.3<br/>• Automated CI/CD container publishing pipeline<br/>• Cloud Run / ECS Fargate infrastructure-as-code |

---

## 4. Suggested Live-Demo Sequence (5-Minute Walkthrough)

This demo uses the built-in deterministic retail sales benchmark dataset and requires no external API keys or network connection.

### Step 1: Launch and Dataset Loading (1 minute)
1. Start the app: `streamlit run omnisleuth/ui/app.py`.
2. Notice the clean brand header and empty state: *"No Dataset Loaded"*.
3. In the sidebar, select **Sample Sales Dataset** and click **Load Sample Sales Dataset**.
4. Point out the top metadata pills: `sample_sales_2024.csv`, `1,000 rows`, `12 columns`, and the system status: `Deterministic only (Offline)`.

### Step 2: Natural Language Investigation (1.5 minutes)
1. On the **Investigation** tab, click the suggested question: *"Why did revenue decline in March?"* (or type it into the box).
2. Click **Investigate**.
3. Walk through the results:
   - **Answer Hero Card:** Shows the grounded answer (e.g. *-23.19% revenue decline driven by electronics unit volume drop*).
   - **Execution Plan:** Show that the pipeline deterministically executed `CalculateTrendTool` and `CompareSegmentsTool`.
   - **Hypothesis Testing:** Point out the competing hypotheses marked as `Supported` or `Rejected`.
   - **Grounded Claims Table:** Highlight the green `Verified` badges proving that every numerical claim was mathematically matched against tool evidence.

### Step 3: Autonomous Discovery & Data Quality (1 minute)
1. Switch to the **Discovery** tab and click **Run Discovery Scan**.
2. Show the ranked discovery cards, highlighting the significance badges (e.g. *Electronics category concentration: 71.8% of total revenue*), recommended follow-up questions, and collapsible raw evidence JSON payloads.
3. Switch to the **Data Quality** tab to show the 4 KPI cards (Completeness %, Duplicate Rows, Quality Alerts, Columns Profiled) and the structured numeric/categorical profiling tables.

### Step 4: Anomaly Explorer & Provenance (1.5 minutes)
1. Switch to **Anomaly Explorer**. Select `amount` and click **Detect Anomalies**.
2. Show the 3 anomaly KPI cards, the statistical non-causality warning banner, the interactive Plotly Box Plot, and the flagged outlier transaction table.
3. Switch to **Evidence & Provenance**.
4. Walk through the 5-stage Cryptographic Provenance Chain:
   `Dataset Ingestion` ➔ `Schema Validation` ➔ `Analytics Tool Execution` ➔ `Claim Extraction` ➔ `Grounded Answer`
5. Show the Session Audit Log and active Evidence Store items, proving that every calculation is fully traceable and reproducible.
