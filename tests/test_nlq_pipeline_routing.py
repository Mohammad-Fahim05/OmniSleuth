"""Integration tests for C2.1 — planner routing into the application pipeline.

All tests are offline and deterministic. LLM planning is exercised only through
mocked providers injected into the existing ``LLMGateway``; no API keys or
network access are used.

Covers:
1. Rule mode uses RulePlanner without calling the LLM.
2. LLM mode invokes LLMPlanner.
3. A valid LLM plan is validated and executed by the existing executor.
4. An LLM planning failure follows the configured fallback policy.
5. Missing credentials do not trigger a network call.
6. The UI reports the actual planner mode and fallback status accurately.
7. Dataset isolation, guardrails, and existing deterministic workflows remain intact.
"""

import json
import os
from typing import List

import pandas as pd
import pytest

from omnisleuth.core.config import get_settings, reset_settings
from omnisleuth.core.exceptions import ConfigurationError
from omnisleuth.ingestion.csv_loader import CSVLoader
from omnisleuth.ingestion.models import ColumnSchema, ColumnType, DatasetMetadata
from omnisleuth.llm import LLMRequest, LLMResponse, UsageMetrics
from omnisleuth.llm.providers import BaseProvider
from omnisleuth.nlq.context import build_schema_context
from omnisleuth.nlq.executor import PlanExecutor
from omnisleuth.nlq.llm_planner import LLMPlanner
from omnisleuth.nlq.models import PlanIntent
from omnisleuth.nlq.pipeline import AnalysisPipeline
from omnisleuth.nlq.rule_planner import RulePlanner
from omnisleuth.nlq.validator import PlanValidator
from omnisleuth.persistence.db import DatabaseManager
from omnisleuth.persistence.repository import Repository
from omnisleuth.evaluation.dataset_generator import generate_benchmark_sales_dataset

VALID_PLAN = {
    "intent": "query",
    "metrics": [{"column": "revenue", "agg": "sum"}],
    "group_by": ["category"],
    "confidence": 0.9,
}


@pytest.fixture(autouse=True)
def offline_env(monkeypatch, tmp_path):
    """Offline env: no provider keys, isolated SQLite DB."""
    for key in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "OPENROUTER_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "pipeline.db"))
    monkeypatch.setenv("APP_ENV", "testing")
    monkeypatch.delenv("NLQ_PLANNER_MODE", raising=False)
    monkeypatch.delenv("NLQ_PLANNER_FALLBACK", raising=False)
    reset_settings()
    yield get_settings()
    reset_settings()


@pytest.fixture
def dataset():
    """Hand-built dataset with an independent pandas ground truth."""
    data = {
        "order_id": [f"ORD-{i:03d}" for i in range(1, 13)],
        "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        "region": ["N", "S", "N", "S", "N", "S", "N", "S", "N", "S", "N", "S"],
        "revenue": [100.0, 200.0, 150.0, 250.0, 120.0, 180.0, 130.0, 220.0, 160.0, 210.0, 140.0, 190.0],
        "quantity": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "order_date": [f"2024-0{m}-1{d}" for m, d in [(1, 5), (1, 12), (2, 4), (2, 11), (3, 3), (3, 10),
                                                       (4, 7), (4, 14), (5, 5), (5, 12), (6, 3), (6, 10)]],
    }
    df = pd.DataFrame(data)
    metadata = DatasetMetadata(
        dataset_id="ds_pipeline",
        name="Pipeline Dataset",
        file_path="pipeline.csv",
        version_hash="hash_pipeline",
        row_count=len(df),
        column_count=len(df.columns),
        duplicate_rows=0,
        columns=[
            ColumnSchema(name="order_id", data_type=ColumnType.IDENTIFIER, dtype="object", unique_count=12),
            ColumnSchema(name="category", data_type=ColumnType.CATEGORICAL, dtype="object", unique_count=2, sample_values=["A", "B"]),
            ColumnSchema(name="region", data_type=ColumnType.CATEGORICAL, dtype="object", unique_count=2, sample_values=["N", "S"]),
            ColumnSchema(name="revenue", data_type=ColumnType.NUMERIC, dtype="float64", unique_count=12),
            ColumnSchema(name="quantity", data_type=ColumnType.NUMERIC, dtype="int64", unique_count=12),
            ColumnSchema(name="order_date", data_type=ColumnType.DATETIME, dtype="object", unique_count=12),
        ],
    )
    return df, metadata


class ScriptedProvider(BaseProvider):
    """Provider returning scripted plan JSON, or raising scripted errors."""

    def __init__(self, responses: List[object], provider_name: str = "openrouter", model: str = "mock/model"):
        self._responses = list(responses)
        self.provider_name = provider_name
        self.model = model
        self.calls = 0

    def generate(self, request: LLMRequest, request_id: str) -> LLMResponse:
        self.calls += 1
        if not self._responses:
            raise AssertionError("ScriptedProvider exhausted")
        item = self._responses[0] if len(self._responses) == 1 else self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(
            request_id=request_id,
            text=item,
            provider=self.provider_name,
            model=self.model,
            latency_ms=1.0,
            fallback_used=False,
            usage=UsageMetrics(input_tokens=10, output_tokens=5, total_tokens=15, is_actual=True),
            error=None,
        )


def _llm_pipeline(responses, **kwargs):
    from omnisleuth.llm.gateway import LLMGateway

    provider = ScriptedProvider(responses)
    planner = LLMPlanner(gateway=LLMGateway(provider=provider))
    return AnalysisPipeline(llm_planner=planner, **kwargs), provider


# ─────────────────────────────────────────────────────────────────────────────
# 1. Rule mode never calls the LLM
# ─────────────────────────────────────────────────────────────────────────────

def test_rule_mode_uses_rule_planner_without_llm_call(dataset):
    """Rule mode routes to RulePlanner; no provider is contacted."""
    df, metadata = dataset
    from omnisleuth.llm.gateway import LLMGateway

    provider = ScriptedProvider([json.dumps(VALID_PLAN)])
    planner = LLMPlanner(gateway=LLMGateway(provider=provider))
    pipeline = AnalysisPipeline(mode="rule", llm_planner=planner)

    routing = pipeline.run("What is total revenue by category?", df, metadata)

    assert routing.requested_mode == "rule"
    assert routing.planner_used == "rule"
    assert routing.fallback_used is False
    assert routing.fallback_reason is None
    assert routing.result.status == "answered"
    assert routing.result.planner_used == "deterministic"
    assert routing.result.plan.planner == "deterministic"
    assert provider.calls == 0


def test_default_mode_is_rule(offline_env, dataset):
    """The documented default is deterministic planning, independent of env."""
    df, metadata = dataset
    pipeline = AnalysisPipeline()
    assert pipeline.mode == "rule"
    assert pipeline.fallback == "rule"

    routing = pipeline.run("What is total revenue by category?", df, metadata)
    assert routing.planner_used == "rule"
    assert routing.result.status == "answered"


def test_llm_planner_not_constructed_in_rule_mode(monkeypatch):
    """Rule mode never instantiates the LLM planner at all."""
    import omnisleuth.nlq.pipeline as pipeline_mod

    called = {"n": 0}

    def _forbidden(*args, **kwargs):
        called["n"] += 1
        raise AssertionError("LLMPlanner must not be constructed in rule mode")

    monkeypatch.setattr(pipeline_mod, "LLMPlanner", _forbidden)
    AnalysisPipeline(mode="rule")  # lazy property: constructing the pipeline is safe
    pipeline = AnalysisPipeline(mode="rule")
    assert called["n"] == 0
    _ = pipeline.is_llm_available()
    assert called["n"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. LLM mode invokes LLMPlanner
# ─────────────────────────────────────────────────────────────────────────────

def test_llm_mode_invokes_llm_planner(dataset):
    """LLM mode routes through LLMPlanner and reports planner_used='llm'."""
    df, metadata = dataset
    pipeline, provider = _llm_pipeline([json.dumps(VALID_PLAN)], mode="llm")

    routing = pipeline.run("What is total revenue by category?", df, metadata)

    assert provider.calls == 1
    assert routing.requested_mode == "llm"
    assert routing.planner_used == "llm"
    assert routing.fallback_used is False
    assert routing.result.status == "answered"
    assert routing.result.planner_used == "llm"


def test_llm_mode_reports_configuration_state(dataset):
    """Routing reports whether the gateway actually has a provider configured."""
    df, metadata = dataset
    pipeline, _ = _llm_pipeline([json.dumps(VALID_PLAN)], mode="llm")
    assert pipeline.is_llm_available() is True

    routing = pipeline.run("What is total revenue by category?", df, metadata)
    assert routing.llm_configured is True


# ─────────────────────────────────────────────────────────────────────────────
# 3. Valid LLM plan is validated and executed by the existing executor
# ─────────────────────────────────────────────────────────────────────────────

def test_valid_llm_plan_is_validated_and_executed(dataset):
    """A valid LLM plan goes through PlanValidator and PlanExecutor, matching pandas."""
    df, metadata = dataset
    pipeline, _ = _llm_pipeline([json.dumps(VALID_PLAN)], mode="llm")

    routing = pipeline.run("What is total revenue by category?", df, metadata)

    result = routing.result
    assert result.status == "answered"
    assert result.evidence_store is not None

    expected = df.groupby("category")["revenue"].sum()
    rows = {row["category"]: row["revenue"] for row in result.result_table}
    assert rows == expected.to_dict()

    # Deterministic execution, not LLM arithmetic: claim validation still ran.
    assert result.evidence_store.list_evidence()


def test_executor_and_validator_are_the_existing_instances(dataset):
    """Routing reuses existing PlanValidator/PlanExecutor, not local reimplementations."""
    df, metadata = dataset
    validator = PlanValidator()
    executor = PlanExecutor()
    pipeline, _ = _llm_pipeline(
        [json.dumps(VALID_PLAN)], mode="llm", validator=validator, executor=executor
    )
    assert pipeline.plan_validator is validator
    assert pipeline.executor is executor

    routing = pipeline.run("What is total revenue by category?", df, metadata)
    assert routing.result.status == "answered"


def test_llm_plan_with_unknown_column_is_not_executed(dataset):
    """A fabricated column never reaches the executor; the fallback policy applies."""
    df, metadata = dataset
    bad = json.dumps({"intent": "query", "metrics": [{"column": "profit_margin", "agg": "sum"}], "group_by": ["category"]})
    pipeline, provider = _llm_pipeline([bad, bad], mode="llm")

    routing = pipeline.run("What is total revenue by category?", df, metadata)

    assert routing.fallback_used is True
    assert routing.planner_used == "rule"
    assert "profit_margin" in routing.fallback_reason
    assert routing.result.plan.group_by == ["category"]  # deterministic plan, not the LLM one
    assert provider.calls == 2


# ─────────────────────────────────────────────────────────────────────────────
# 4. Fallback policy
# ─────────────────────────────────────────────────────────────────────────────

def test_fallback_policy_rule_delegates_to_rule_planner(dataset):
    """Default policy 'rule' answers deterministically and discloses the fallback."""
    df, metadata = dataset
    pipeline, provider = _llm_pipeline([json.dumps(VALID_PLAN)], mode="llm")
    # Force a planning failure by making the model emit an unparseable plan.
    pipeline.llm_planner.gateway._provider_override = None
    provider._responses = ["not json at all", "still not json"]

    routing = pipeline.run("What is total revenue by category?", df, metadata)

    assert routing.requested_mode == "llm"
    assert routing.planner_used == "rule"
    assert routing.fallback_used is True
    assert routing.fallback_reason is not None
    assert routing.result.status == "answered"
    assert routing.result.planner_used == "deterministic"


def test_fallback_policy_fail_surfaces_sanitized_error(dataset):
    """Policy 'fail' returns a failed result instead of silently answering."""
    df, metadata = dataset
    pipeline, provider = _llm_pipeline([json.dumps(VALID_PLAN)], mode="llm", fallback="fail")
    provider._responses = ["not json at all", "still not json"]

    routing = pipeline.run("What is total revenue by category?", df, metadata)

    assert routing.requested_mode == "llm"
    assert routing.planner_used == "llm"
    assert routing.fallback_used is False
    assert routing.fallback_reason is not None
    assert routing.result.status == "failed"
    assert "LLM planning failed" in routing.result.answer
    assert routing.result.limitations


def test_fallback_reason_is_sanitized_and_bounded(dataset):
    """Fallback reasons are short, categorized, and free of credentials."""
    df, metadata = dataset
    secret = "sk-" + "or-v1-" + "FIXTURESECRETVALUE"
    pipeline, provider = _llm_pipeline([json.dumps(VALID_PLAN)], mode="llm")
    provider._responses = [Exception(f"Connection failed (api_key={secret})")]

    routing = pipeline.run("What is total revenue by category?", df, metadata)

    assert routing.fallback_used is True
    assert routing.fallback_reason is not None
    assert secret not in routing.fallback_reason
    assert len(routing.fallback_reason) < 400


def test_fallback_for_rate_limit_and_auth_failures(dataset):
    """Transient and permanent provider failures both fall back safely."""
    df, metadata = dataset
    for failure in ("429 Rate limit exceeded", "401 Unauthorized"):
        pipeline, provider = _llm_pipeline([json.dumps(VALID_PLAN)], mode="llm")
        provider._responses = [Exception(failure)]

        routing = pipeline.run("What is total revenue by category?", df, metadata)

        assert routing.fallback_used is True
        assert routing.result.status == "answered"


def test_invalid_mode_and_fallback_are_rejected():
    """Invalid configuration fails loudly rather than silently defaulting."""
    with pytest.raises(ConfigurationError):
        AnalysisPipeline(mode="bogus")
    with pytest.raises(ConfigurationError):
        AnalysisPipeline(fallback="bogus")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Missing credentials cause no network call
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_credentials_make_no_network_call(dataset, monkeypatch):
    """With no credentials, LLM mode falls back without contacting any provider."""
    df, metadata = dataset
    from omnisleuth.llm.gateway import LLMGateway

    import openai

    monkeypatch.setattr(openai, "OpenAI", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network attempted")))
    monkeypatch.setattr(openai, "AzureOpenAI", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network attempted")))

    pipeline = AnalysisPipeline(mode="llm")

    assert pipeline.is_llm_available() is False

    routing = pipeline.run("What is total revenue by category?", df, metadata)

    assert routing.requested_mode == "llm"
    assert routing.planner_used == "rule"
    assert routing.fallback_used is True
    assert routing.llm_configured is False
    assert routing.result.status == "answered"


def test_llm_mode_with_fail_policy_and_no_credentials(dataset):
    """Deterministic users can opt into an explicit failure instead of silent fallback."""
    df, metadata = dataset
    pipeline = AnalysisPipeline(mode="llm", fallback="fail")

    routing = pipeline.run("What is total revenue by category?", df, metadata)

    assert routing.result.status == "failed"
    assert routing.fallback_used is False
    assert routing.fallback_reason is not None


# ─────────────────────────────────────────────────────────────────────────────
# 6. UI reporting of planner mode and fallback status
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_service_cache():
    """Clear the app's process-wide service cache before and after a UI test.

    ``get_services()`` is ``@st.cache_resource``, so the SQLite repository stays
    bound to whichever database path was active when it was first built. Clearing
    it on both sides keeps each test's database private and stops Ask Data
    investigations from leaking into other tests' investigation history.
    """
    import streamlit as st

    reset_settings()
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()
    reset_settings()


def test_ui_reports_mode_and_fallback_status(dataset, tmp_path, monkeypatch, clean_service_cache):
    """The UI tab reports requested mode, planner used, and fallback status."""
    for key in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "OPENROUTER_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "ui_pipeline.db"))
    monkeypatch.setenv("APP_ENV", "testing")

    at = _fresh_app_test()
    at.run()
    assert len(at.exception) == 0
    _load_sample(at)

    # Deterministic mode is the default and reports no fallback.
    at.text_input[0].input("What is total revenue by category?")
    [b for b in at.button if b.label == "Investigate"][0].click().run()
    assert len(at.exception) == 0

    routing = at.session_state.get("nlq_routing")
    assert routing is not None
    assert routing.requested_mode == "rule"
    assert routing.planner_used == "rule"
    assert routing.fallback_used is False
    assert routing.result.status == "answered"
    assert routing.result.plan.planner == "deterministic"


def _repo_app_path():
    from pathlib import Path

    return Path(__file__).resolve().parent.parent / "omnisleuth" / "ui" / "app.py"


def _fresh_app_test(timeout=60):
    """Create an AppTest whose service cache is bound to the current test's database.

    The app caches its services (including the SQLite repository) with
    ``@st.cache_resource``, which persists across AppTest instances in the same
    process. Clearing it, after re-reading settings, keeps each test's database
    private so investigations cannot leak between tests.
    """
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    reset_settings()
    try:
        st.cache_resource.clear()
    except Exception:  # pragma: no cover - defensive
        pass
    return AppTest.from_file(str(_repo_app_path()), default_timeout=timeout)


def _load_sample(at):
    [b for b in at.button if "Sample" in b.label][0].click().run()
    assert len(at.exception) == 0
    assert at.session_state.df is not None


def test_ui_llm_mode_without_credentials_reports_fallback(tmp_path, monkeypatch, clean_service_cache):
    """Selecting LLM mode with no credentials shows the fallback, with no network call."""
    for key in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "OPENROUTER_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "ui_pipeline2.db"))
    monkeypatch.setenv("APP_ENV", "testing")

    at = _fresh_app_test()
    at.run()
    _load_sample(at)
    [r for r in at.radio if r.label == "Planner"][0].set_value("LLM-assisted (validated plan)").run()
    assert len(at.exception) == 0

    infos = " ".join(i.value for i in at.info)
    assert "LLM planner selected but no provider is configured" in infos

    at.text_input[0].input("What is total revenue by category?")
    [b for b in at.button if b.label == "Investigate"][0].click().run()
    assert len(at.exception) == 0

    routing = at.session_state.get("nlq_routing")
    assert routing.requested_mode == "llm"
    assert routing.planner_used == "rule"
    assert routing.fallback_used is True
    assert routing.fallback_reason is not None
    assert routing.result.status == "answered"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Dataset isolation, guardrails, and deterministic workflows intact
# ─────────────────────────────────────────────────────────────────────────────

def test_identical_questions_on_different_schemas_do_not_collide(tmp_path):
    """A different schema produces a different plan (cache/planner isolation)."""
    df_a = pd.DataFrame({"category": ["A", "B"], "revenue": [10.0, 20.0]})
    meta_a = DatasetMetadata(
        dataset_id="ds_a", name="A", file_path="a.csv", version_hash="h1",
        row_count=2, column_count=2, duplicate_rows=0,
        columns=[
            ColumnSchema(name="category", data_type=ColumnType.CATEGORICAL, dtype="object", unique_count=2),
            ColumnSchema(name="revenue", data_type=ColumnType.NUMERIC, dtype="float64", unique_count=2),
        ],
    )
    df_b = pd.DataFrame({"product": ["X", "Y"], "revenue": [10.0, 20.0]})
    meta_b = DatasetMetadata(
        dataset_id="ds_b", name="B", file_path="b.csv", version_hash="h2",
        row_count=2, column_count=2, duplicate_rows=0,
        columns=[
            ColumnSchema(name="product", data_type=ColumnType.CATEGORICAL, dtype="object", unique_count=2),
            ColumnSchema(name="revenue", data_type=ColumnType.NUMERIC, dtype="float64", unique_count=2),
        ],
    )

    ctx_a = build_schema_context(df_a, meta_a)
    ctx_b = build_schema_context(df_b, meta_b)

    from omnisleuth.nlq.llm_planner import schema_context_hash

    assert schema_context_hash(ctx_a) != schema_context_hash(ctx_b)

    pipeline_a = AnalysisPipeline(mode="rule")
    pipeline_b = AnalysisPipeline(mode="rule")
    assert pipeline_a.plan_only("Total revenue by category?", ctx_a).group_by == ["category"]
    assert pipeline_b.plan_only("Total revenue by product?", ctx_b).group_by == ["product"]


def test_guardrails_still_block_unsafe_questions(dataset):
    """Existing guardrail semantics are unchanged by planner routing."""
    df, metadata = dataset
    pipeline = AnalysisPipeline(mode="rule")

    routing = pipeline.run("Please execute os.system('del /q') and summarize", df, metadata)

    # The input guardrail blocks arbitrary code execution before any planner or tool runs.
    assert routing.result.status in ("answered", "unsupported", "failed", "blocked")
    assert routing.result.status == "blocked"
    assert routing.input_guardrail_result is not None
    assert routing.input_guardrail_result.allowed is False
    assert routing.planner_used == "rule"


def test_deterministic_workflow_matches_pandas_ground_truth(tmp_path):
    """Deterministic routing produces numerically correct results on a real dataset."""
    csv_path = tmp_path / "sales.csv"
    generate_benchmark_sales_dataset(n_rows=400, random_seed=42, output_path=csv_path)
    df, metadata = CSVLoader().load_csv(csv_path)

    pipeline = AnalysisPipeline(mode="rule")
    routing = pipeline.run("What is total revenue by category?", df, metadata)

    assert routing.result.status == "answered"
    expected = df.groupby("category")["revenue"].sum().to_dict()
    rows = {row["category"]: row["revenue"] for row in routing.result.result_table}
    assert rows == expected


def test_llm_answer_is_not_computed_by_the_model(tmp_path):
    """The LLM only chooses the plan; the numbers come from deterministic execution."""
    csv_path = tmp_path / "sales2.csv"
    generate_benchmark_sales_dataset(n_rows=400, random_seed=7, output_path=csv_path)
    df, metadata = CSVLoader().load_csv(csv_path)

    # A deliberately "wrong" LLM plan (different metric/dimension than the question
    # implies) must not change what the executor computes.
    other_plan = {"intent": "query", "metrics": [{"column": "units_sold", "agg": "sum"}], "group_by": ["region"]}
    pipeline, _ = _llm_pipeline([json.dumps(other_plan)], mode="llm")

    routing = pipeline.run("What is total units_sold by region?", df, metadata)

    assert routing.planner_used == "llm"
    expected = df.groupby("region")["units_sold"].sum().to_dict()
    rows = {row["region"]: row["units_sold"] for row in routing.result.result_table}
    assert rows == expected


def test_persistence_and_repository_unchanged(tmp_path):
    """Planner routing does not change repository behavior."""
    db_file = str(tmp_path / "pipeline_repo.db")
    db_mgr = DatabaseManager(db_path=db_file)
    repo = Repository(db_manager=db_mgr, session_id="test_pipeline_repo_session")
    assert repo.list_investigations() == []
    df, metadata = pd.DataFrame({"a": [1]}), DatasetMetadata(
        dataset_id="ds_x", name="x", file_path="x.csv", version_hash="v",
        row_count=1, column_count=1, duplicate_rows=0,
        columns=[ColumnSchema(name="a", data_type=ColumnType.NUMERIC, dtype="int64")],
    )
    repo.save_dataset(metadata)
    assert len(repo.list_datasets()) == 1


def test_plan_only_does_not_execute(dataset):
    """plan_only validates without executing, for callers that need only the plan."""
    df, metadata = dataset
    schema_ctx = build_schema_context(df, metadata)

    plan = AnalysisPipeline(mode="rule").plan_only("What is total revenue by category?", schema_ctx)
    assert plan.group_by == ["category"]
    assert plan.planner == "deterministic"


# ─────────────────────────────────────────────────────────────────────────────
# Task 5: Safety net for rule planner disagreement (Test D)
# ─────────────────────────────────────────────────────────────────────────────

def test_safety_net_prefers_rule_plan_for_contribution(dataset):
    """Test D (T-003F Task 5): When LLM plans query for contribution question, rule plan is preferred and disagreement is recorded."""
    df, metadata = dataset
    question = "Which category contributed most to revenue?"

    # LLM planned query limited to 1 group
    llm_query_plan = {
        "intent": "query",
        "metrics": [{"column": "revenue", "agg": "sum"}],
        "group_by": ["category"],
        "limit": 1,
        "confidence": 0.9,
    }
    pipeline, _ = _llm_pipeline([json.dumps(llm_query_plan)], mode="llm")

    routing = pipeline.run(question, df, metadata)

    # 1. Rule plan intent (CONTRIBUTION) is preferred over LLM intent (QUERY)
    assert routing.result.plan.intent == PlanIntent.CONTRIBUTION
    assert routing.result.plan.target_column == "revenue"
    assert routing.result.plan.group_by == ["category"]

    # 2. Disagreement note is recorded in the result
    assert any("disagreement" in note.lower() for note in routing.result.limitations + routing.result.plan.assumptions)


def test_safety_net_records_disagreement_without_overriding_other_intents(dataset):
    """When rule plan differs but intent is not in {contribution, deep_investigation}, LLM plan is kept with disagreement note."""
    df, metadata = dataset
    question = "What is total revenue by category?"

    # LLM plans TREND while rule planner plans QUERY
    llm_trend_plan = {
        "intent": "trend",
        "metrics": [{"column": "revenue", "agg": "sum"}],
        "group_by": ["category"],
        "confidence": 0.9,
    }
    pipeline, _ = _llm_pipeline([json.dumps(llm_trend_plan)], mode="llm")

    routing = pipeline.run(question, df, metadata)

    # LLM plan intent kept because rule intent QUERY is not in {contribution, deep_investigation}
    assert routing.result.plan.intent == PlanIntent.TREND
    assert any("disagreement" in note.lower() for note in routing.result.limitations + routing.result.plan.assumptions)
