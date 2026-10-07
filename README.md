# OmniSleuth 🔍

**An Evidence-Grounded Multi-Agent Data Investigation System**

OmniSleuth is a bounded AI/ML system that investigates structured business datasets (such as sales and operational data). Rather than acting like a naive CSV chatbot that hallucinates calculations and makes unsupported causal claims, OmniSleuth enforces strict deterministic computation, allowlisted analytical tools, programmatic evidence validation, and bounded multi-agent orchestration.

---

## Core Architecture Principles

1. **Deterministic calculations**: The LLM never computes numbers. All aggregations, trends, contributions, and metric decompositions run via Python, Pandas, and scikit-learn.
2. **Allowlisted tools**: Typed Pydantic input/output schemas with validation, error handling, and execution logging. Arbitrary code generation is strictly disallowed.
3. **Contextual RAG**: Document retrieval provides definitions, business rules, and qualitative context—never numerical calculations.
4. **Evidence grounding**: Every factual and numerical claim references verifiable evidence IDs (dataset rows, tool results, ML detections, document citations).
5. **Programmatic claim validation**: Python validation logic compares claims against execution artifacts to reject or flag unsupported statements.
6. **No unsupported causality**: Distinguishes between correlation/contribution and definite causality with calibrated language.
7. **Bounded agentic behavior**: Enforces explicit limits (depth $\le 3$, tool calls $\le 10$, retries $\le 2$) and graceful termination under insufficient evidence.

---

## Project Structure

```text
omnisleuth/
├── core/             # Configuration, logging, base exceptions
├── ingestion/        # CSV loading, validation, schema detection, hashing
├── profiling/        # Profiling and data quality reporting
├── analytics/        # Deterministic computation & decomposition
├── ml/               # Anomaly detection (IQR, Isolation Forest)
├── tools/            # Allowlisted, typed analytical tools
├── evidence/         # Provenance tracking, claim validation
├── rag/              # Document ingestion, vector store, contextual RAG
├── agents/           # Supervisor, Analyst, Retriever, Investigator
├── persistence/      # SQLite persistence layer
└── ui/               # Streamlit interactive dashboard
tests/                # Comprehensive unit and integration test suite
```

---

## Getting Started

### 1. Installation

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configuration

```bash
cp .env.example .env
```

### 3. Run Tests

```bash
pytest
```
