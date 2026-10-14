# OmniSleuth — Prioritized Next Steps & Engineering Roadmap

This roadmap outlines the prioritized engineering actions for the next development iteration, derived directly from empirical repository inspection and verified test results.

---

## 1. Immediate Next Steps Post T-002

### 1. TASK T-003: Natural Language Planner and Intent Classifier [PARTIAL]
- **Phase A (Schema & Validator)**: [VERIFIED] COMPLETE. `AnalysisPlan`, `SchemaContext`, `Planner` Protocol, and `PlanValidator` implemented in `omnisleuth/nlq/`. 15 unit tests added in `tests/test_nlq_validator.py`. Full test suite: 179/179 tests passing.
- **Phase B (Deterministic Planner & Executor)**: [VERIFIED] COMPLETE. `RulePlanner` and `PlanExecutor` implemented in `omnisleuth/nlq/`, with 42 tests in `tests/test_nlq_executor.py` (32 parametrized analytical questions against independent pandas ground truth, honest-failure cases, injection/adversarial resistance, SQLite persistence and claim validation) plus 10 tests in `tests/test_nlq_column_resolution.py` covering literal column-name resolution for group-by and contribution questions. Full test suite: 179/179 tests passing. Three defects were fixed while verifying this phase: hardcoded month-end day `31` for period comparisons (`2024-02-31`, `2024-04-31`), the contribution branch not resolving a categorically-named column without a grouping preposition, and `ValidationResult` being read via a nonexistent `validation_notes` field instead of `validation_explanation`.
- **Phase C (Optional LLM Planner via OpenRouter)**: [NOT STARTED].
- **Phase D (Routing, Charts, UI Integration, Docs)**: [NOT STARTED].

**Canonical test total for this iteration: 179 tests (179 passed, 0 failed, 0 warnings), executed with the project `.venv` (Python 3.13.5, `python -m pytest`, no ignores).**

### 2. Stage and Commit Stabilized Work *(Requires User Approval)* [PARTIAL]
- **Context**: Branch `stabilize/ui-slice` holds uncommitted pre-existing work plus T-002 UI stabilization fixes and 5 new automated tests. All 179 tests pass.
- **Action**: Review and commit all verified changes upon user approval.
- **Relevant Files**: See `git status --short`.

### 3. Verify Local Docker Build with Active Docker Desktop Daemon [PARTIAL]
- **Context**: Dockerfile and compose configurations pass static validation (`tests/test_deployment.py`).
- **Action**: Start Docker Desktop engine on Windows host and run live container smoke tests.

---

## 2. Technical Debt & Subsystem Verification Status

| Priority | Task Description | Target File(s) | Status |
|:---:|---|---|:---:|
| **High** | End-to-end UI slice & history reopen without LLM key | `omnisleuth/ui/app.py`, `omnisleuth/persistence/repository.py` | [VERIFIED] (T-002) |
| **High** | Populate SQLite `claims` table during save | `omnisleuth/agents/orchestrator.py` | [VERIFIED] (T-002) |
| **High** | Real AppTest suite (5 new tests, 179 total) | `tests/test_ui_flow.py` | [VERIFIED] (T-002) |
| **Medium** | NL planner and intent classifier for arbitrary queries | `omnisleuth/nlq/rule_planner.py`, `omnisleuth/nlq/executor.py` | [VERIFIED] (T-003 Phases A+B) |
| **Medium** | Unify `MultiAgentOrchestrator` & `CompiledStateGraph` state formats | `omnisleuth/agents/orchestrator.py`, `omnisleuth/workflow/graph.py` | [PARTIAL] |
| **Low** | Expand Ingestion Formats (Parquet) | `omnisleuth/ingestion/` | [PARTIAL] |
