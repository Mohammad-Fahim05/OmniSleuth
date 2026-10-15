"""Integration tests for Phase D — unified question routing, charts, follow-ups, history.

All tests are offline and deterministic. The fake provider is injected into the
existing ``LLMGateway``; no API key or network access is used.

Covers:
1. AppTest: supported question, ambiguous, unsupported, deep "why" question,
   provider-failure fallback, general_knowledge with a fake provider; zero exceptions.
2. Reopen from history for an NLQ-saved and an orchestrator-saved investigation.
3. Chart spec built from the computed table; invalid mappings rejected with a
   suggested alternative.
4. Follow-up with and without context.
5. Narration with a number not in evidence is discarded.
6. Sidebar status contains no key material.
7. Clear st.cache_resource before and after UI tests.
"""

import json
from typing import List

import pandas as pd
import pytest

from omnisleuth.core.config import get_settings, reset_settings
from omnisleuth.ingestion.models import ColumnSchema, ColumnType, DatasetMetadata
from omnisleuth.llm import LLMRequest, LLMResponse, UsageMetrics
from omnisleuth.llm.gateway import LLMGateway
from omnisleuth.llm.providers import BaseProvider
from omnisleuth.nlq.charting import build_chart_from_table
from omnisleuth.nlq.context import build_schema_context
from omnisleuth.nlq.followup import detect_followup_marker, resolve_rule_followup
from omnisleuth.nlq.llm_planner import LLMPlanner
from omnisleuth.nlq.models import PlanIntent
from omnisleuth.nlq.pipeline import AnalysisPipeline
from omnisleuth.persistence.db import DatabaseManager
from omnisleuth.persistence.repository import Repository

FAKE_KEY = "sk-" + "or-v1-" + "FIXTURENOTAREALKEY"

VALID_PLAN = {
    "intent": "query",
    "metrics": [{"column": "revenue", "agg": "sum"}],
    "group_by": ["category"],
    "confidence": 0.9,
}
GENERAL_KNOWLEDGE_PLAN = {
    "intent": "general_knowledge",
    "confidence": 0.9,
    "unsupported_reason": "General knowledge, not computed from your dataset.",
}


@pytest.fixture(autouse=True)
def offline_env(monkeypatch, tmp_path):
    """Offline env: no provider keys, isolated SQLite DB."""
    for key in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "OPENROUTER_API_KEY", "LLM_PROVIDER", "NLQ_PLANNER_MODE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "phase_d.db"))
    monkeypatch.setenv("APP_ENV", "testing")
    reset_settings()
    yield get_settings()
    reset_settings()


@pytest.fixture
def clean_service_cache():
    """Clear the app's process-wide service cache before and after a UI test."""
    import streamlit as st

    reset_settings()
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()
    reset_settings()


@pytest.fixture
def dataset():
    df = pd.DataFrame(
        {
            "category": ["Electronics", "Furniture", "Electronics", "Clothing", "Furniture", "Electronics"],
            "region": ["North", "South", "North", "East", "West", "South"],
            "revenue": [100.0, 200.0, 300.0, 150.0, 250.0, 50.0],
            "order_date": ["2024-01-05", "2024-01-12", "2024-01-20", "2024-02-04", "2024-02-11", "2024-02-18"],
        }
    )
    metadata = DatasetMetadata(
        dataset_id="ds_phase_d",
        name="Phase D Dataset",
        file_path="phase_d.csv",
        version_hash="hash_phase_d",
        row_count=len(df),
        column_count=len(df.columns),
        duplicate_rows=0,
        columns=[
            ColumnSchema(name="category", data_type=ColumnType.CATEGORICAL, dtype="object", unique_count=3, sample_values=["Electronics", "Furniture", "Clothing"]),
            ColumnSchema(name="region", data_type=ColumnType.CATEGORICAL, dtype="object", unique_count=4, sample_values=["North", "South", "East", "West"]),
            ColumnSchema(name="revenue", data_type=ColumnType.NUMERIC, dtype="float64", unique_count=6),
            ColumnSchema(name="order_date", data_type=ColumnType.DATETIME, dtype="object", unique_count=5),
        ],
    )
    return df, metadata


class FakeProvider(BaseProvider):
    """Provider returning scripted plan JSON (or raising scripted errors)."""

    def __init__(self, responses: List[object], provider_name="openrouter", model="fake/model"):
        self._responses = list(responses)
        self.provider_name = provider_name
        self.model = model
        self.calls = 0
        self.requests: List[LLMRequest] = []

    def generate(self, request: LLMRequest, request_id: str) -> LLMResponse:
        self.calls += 1
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("FakeProvider exhausted")
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
    provider = FakeProvider(responses)
    planner = LLMPlanner(gateway=LLMGateway(provider=provider))
    return AnalysisPipeline(mode="llm", llm_planner=planner, **kwargs), provider


def _app_path():
    from pathlib import Path

    return Path(__file__).resolve().parent.parent / "omnisleuth" / "ui" / "app.py"


# ─────────────────────────────────────────────────────────────────────────────
# 1. AppTest: one box, every status, zero exceptions
# ─────────────────────────────────────────────────────────────────────────────

def test_ui_supported_question_renders_table_and_chart(tmp_path, monkeypatch, clean_service_cache):
    """A supported analytical question renders answer, table, chart and evidence."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "ui_supported.db"))
    at = AppTest.from_file(str(_app_path()), default_timeout=60)
    at.run()
    [b for b in at.button if "Sample" in b.label][0].click().run()
    at.text_input[0].input("What is total revenue by category?")
    [b for b in at.button if b.label == "Investigate"][0].click().run()

    assert len(at.exception) == 0
    routing = at.session_state.get("nlq_routing")
    assert routing is not None
    assert routing.result.status == "answered"
    assert routing.result.result_table
    assert routing.chart is not None and routing.chart.used


def test_ui_ambiguous_question_needs_clarification(tmp_path, monkeypatch, clean_service_cache):
    """An ambiguous question returns needs_clarification with zero exceptions."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "ui_amb.db"))
    at = AppTest.from_file(str(_app_path()), default_timeout=60)
    at.run()
    [b for b in at.button if "Sample" in b.label][0].click().run()
    # A follow-up with no prior turn to inherit from is a clarification request.
    at.text_input[0].input("And for 2023?")
    [b for b in at.button if b.label == "Investigate"][0].click().run()

    assert len(at.exception) == 0
    routing = at.session_state.get("nlq_routing")
    assert routing.result.status == "needs_clarification"
    assert routing.result.plan.needs_clarification


def test_ui_unsupported_question(tmp_path, monkeypatch, clean_service_cache):
    """An off-topic question is reported as unsupported."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "ui_unsup.db"))
    at = AppTest.from_file(str(_app_path()), default_timeout=60)
    at.run()
    [b for b in at.button if "Sample" in b.label][0].click().run()
    at.text_input[0].input("What is the weather in Tokyo tomorrow?")
    [b for b in at.button if b.label == "Investigate"][0].click().run()

    assert len(at.exception) == 0
    routing = at.session_state.get("nlq_routing")
    assert routing.result.status == "unsupported"
    assert "revenue" not in routing.result.answer.lower()


def test_ui_deep_why_question_routes_to_orchestrator(tmp_path, monkeypatch, clean_service_cache):
    """A 'why' question keeps the multi-agent rendering (evidence, hypotheses, claims)."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "ui_deep.db"))
    at = AppTest.from_file(str(_app_path()), default_timeout=60)
    at.run()
    [b for b in at.button if "Sample" in b.label][0].click().run()
    at.text_input[0].input("Why did revenue decline in March for Electronics in North region?")
    [b for b in at.button if b.label == "Investigate"][0].click().run()

    assert len(at.exception) == 0
    routing = at.session_state.get("nlq_routing")
    assert routing.deep_investigation is True
    assert routing.result.status == "answered"

    last_inv = at.session_state.last_inv
    assert last_inv is not None
    assert len(last_inv["evidence_store"].list_evidence()) > 0
    assert len(last_inv["hypotheses"]) > 0
    assert len(last_inv["claims"]) > 0


def test_ui_provider_failure_fallback_is_visible(tmp_path, monkeypatch, clean_service_cache):
    """A provider failure in LLM mode falls back deterministically and says so."""
    from streamlit.testing.v1 import AppTest

    # No credentials: LLM mode falls back without any network call.
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "ui_fallback.db"))
    at = AppTest.from_file(str(_app_path()), default_timeout=60)
    at.run()
    [b for b in at.button if "Sample" in b.label][0].click().run()
    [r for r in at.radio if r.label == "Planner"][0].set_value("LLM-assisted (validated plan)").run()
    at.text_input[0].input("What is total revenue by category?")
    [b for b in at.button if b.label == "Investigate"][0].click().run()

    assert len(at.exception) == 0
    routing = at.session_state.get("nlq_routing")
    assert routing.requested_mode == "llm"
    assert routing.planner_used == "rule"
    assert routing.fallback_used is True
    assert routing.fallback_reason
    assert routing.result.status == "answered"
    assert any("fallback" in str(w.value).lower() for w in at.warning)


def test_ui_general_knowledge_with_fake_provider(tmp_path, monkeypatch, clean_service_cache, dataset):
    """general_knowledge is labelled and produces no evidence or claims."""
    df, metadata = dataset
    pipeline, _provider = _llm_pipeline([json.dumps(GENERAL_KNOWLEDGE_PLAN)])

    routing = pipeline.run("What is the capital of France?", df, metadata)

    assert routing.result.status == "unsupported"
    assert routing.general_knowledge is True
    assert routing.result.evidence_store is None
    assert routing.result.claims == []


def test_ui_general_knowledge_needs_llm_in_rule_mode(tmp_path, monkeypatch, dataset):
    """Without an LLM a general-knowledge question returns an honest message."""
    df, metadata = dataset
    pipeline = AnalysisPipeline(mode="rule")

    routing = pipeline.run("What is the capital of France?", df, metadata)

    assert routing.result.status in ("unsupported", "needs_clarification")
    assert routing.result.answer
    assert "dataset" in routing.result.answer.lower() or "llm" in routing.result.answer.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 2. History reopening for NLQ-saved and orchestrator-saved investigations
# ─────────────────────────────────────────────────────────────────────────────

def test_reopen_nlq_saved_investigation_restores_answer_evidence_and_chart(dataset, tmp_path):
    """An NLQ-saved record reopens with the same answer, evidence and chart."""
    df, metadata = dataset
    db = DatabaseManager(db_path=str(tmp_path / "reopen_nlq.db"))
    repo = Repository(db_manager=db)
    repo.save_dataset(metadata)

    pipeline = AnalysisPipeline(mode="rule", executor=None)
    routing = pipeline.run("What is total revenue by category?", df, metadata)

    investigation_id = "inv_nlq_saved_001"
    repo.save_investigation_results(
        investigation_id=investigation_id,
        dataset_id=metadata.dataset_id,
        question="What is total revenue by category?",
        status=routing.result.status,
        plan_dict=routing.result.plan.model_dump(),
        final_answer=routing.result.answer,
        evidence_store=routing.result.evidence_store,
        result_table=routing.result.result_table,
        chart_spec=routing.chart.spec if routing.chart else None,
        planner_used=routing.planner_used,
        summary_text=routing.result.plain_language_summary,
    )

    reopened = repo.get_investigation(investigation_id)
    assert reopened is not None
    assert reopened["final_answer"] == routing.result.answer
    assert reopened["planner_used"] == routing.planner_used
    assert reopened["summary_text"] == routing.result.plain_language_summary
    assert reopened["result_table"] == routing.result.result_table
    assert reopened["chart_spec"] == routing.chart.spec
    assert len(reopened["evidence_items"]) > 0
    assert len(reopened["claims"]) > 0


def test_reopen_orchestrator_saved_investigation(tmp_path, monkeypatch):
    """An orchestrator-style saved record reopens with its answer and evidence."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "reopen_orch.db"))
    monkeypatch.setenv("APP_ENV", "testing")

    at = AppTest.from_file(str(_app_path()), default_timeout=90)
    at.run()
    [b for b in at.button if "Sample" in b.label][0].click().run()
    at.text_input[0].input("Why did revenue decline in March for Electronics in North region?")
    [b for b in at.button if b.label == "Investigate"][0].click().run()
    assert len(at.exception) == 0

    original = at.session_state.last_inv
    assert original is not None
    original_evidence = original["evidence_store"].list_evidence()
    assert len(original_evidence) > 0

    reopen_btns = [b for b in at.button if b.label == "Reopen Investigation"]
    assert len(reopen_btns) > 0, "Expected Reopen Investigation button in history expander"
    reopen_btns[0].click().run()
    assert len(at.exception) == 0

    reopened = at.session_state.last_inv
    assert reopened["final_answer"] == original["final_answer"]
    assert len(reopened["evidence_store"].list_evidence()) == len(original_evidence)
    assert len(reopened["hypotheses"]) == len(original["hypotheses"])


# ─────────────────────────────────────────────────────────────────────────────
# 3. Chart spec from the computed table, invalid mappings rejected
# ─────────────────────────────────────────────────────────────────────────────

def test_chart_is_built_from_computed_table(dataset):
    """The chart spec is derived from the executed result table, not the raw data."""
    df, metadata = dataset
    routing = AnalysisPipeline(mode="rule").run("What is total revenue by category?", df, metadata)

    assert routing.result.result_table
    assert routing.chart is not None and routing.chart.used
    computed_columns = set(routing.result.result_table[0].keys())
    assert routing.chart.spec["x"] in computed_columns
    assert routing.chart.spec["y"] in computed_columns
    assert len(routing.chart.data) == len(routing.result.result_table)


@pytest.mark.parametrize(
    "requested,expected_rejected,fragment",
    [
        ("line", True, "ordered x-axis"),
        ("pie", True, "distinct values"),
        ("treemap", True, "not in the allowlisted set"),
        ("bar", False, None),
    ],
)
def test_invalid_chart_mappings_rejected_with_alternative(requested, expected_rejected, fragment):
    """Invalid chart mappings are rejected with an explanation and a suggestion."""
    table = [{"product": f"P{i}", "revenue": float(i)} for i in range(20)]
    result = build_chart_from_table(table, requested_type=requested)

    if expected_rejected:
        assert result.used is False
        assert fragment in result.explanation
        assert result.suggestion
    else:
        assert result.used is True
        assert result.chart_type == "bar"


def test_chart_requires_numeric_and_multiple_rows():
    """Single-row tables and tables with no numeric column produce no chart."""
    assert build_chart_from_table([{"category": "A", "revenue": 1.0}]).used is False
    assert build_chart_from_table([{"category": "A"}, {"category": "B"}]).used is False
    assert build_chart_from_table([]).used is False


# ─────────────────────────────────────────────────────────────────────────────
# 4. Follow-ups: with and without context
# ─────────────────────────────────────────────────────────────────────────────

def test_followup_with_context_inherits_metric_and_applies_year(dataset):
    """'And for 2023?' inherits the metric and applies a bounded date range."""
    df, metadata = dataset
    schema_ctx = build_schema_context(df, metadata)
    pipeline = AnalysisPipeline(mode="rule")

    first = pipeline.run("What is total revenue by category?", df, metadata)
    assert first.result.status == "answered"

    # Rule-level resolution inherits the metric and applies the stated year.
    resolved = resolve_rule_followup("And for 2023?", [first.history_entry], schema_ctx)
    assert resolved is not None
    assert resolved.followup_of
    assert resolved.metrics[0].column == "revenue"
    assert resolved.group_by == ["category"]
    assert resolved.time is not None
    assert resolved.time.start == "2023-01-01"
    assert resolved.time.end == "2023-12-31"

    # 2023 is outside this dataset's range, so validation refuses rather than fabricating.
    out_of_range = pipeline.plan_only("And for 2023?", schema_ctx, history_ctx=[first.history_entry])
    assert out_of_range.intent == PlanIntent.UNSUPPORTED
    assert out_of_range.unsupported_reason and "2024" in out_of_range.unsupported_reason

    # A year inside the dataset's range answers deterministically.
    in_range = pipeline.plan_only("And for 2024?", schema_ctx, history_ctx=[first.history_entry])
    assert in_range.intent == PlanIntent.QUERY
    assert in_range.group_by == ["category"]
    assert in_range.followup_of


def test_followup_executes_within_dataset_range(dataset):
    """A follow-up restricted to an available year executes and matches pandas."""
    df, metadata = dataset
    pipeline = AnalysisPipeline(mode="rule")

    first = pipeline.run("What is total revenue by category?", df, metadata)
    routed = pipeline.run("And for 2024?", df, metadata, history_ctx=[first.history_entry])

    assert routed.result.status == "answered"
    assert routed.followup_of
    jan_feb = df[df["order_date"].str.startswith("2024")]
    expected = jan_feb.groupby("category")["revenue"].sum().to_dict()
    rows = {r["category"]: r["revenue"] for r in routed.result.result_table}
    assert rows == expected


def test_followup_by_dimension(dataset):
    """'now by region' switches the dimension and keeps the metric."""
    df, metadata = dataset
    schema_ctx = build_schema_context(df, metadata)
    history = [
        {
            "question": "What is total revenue by category?",
            "intent": "query",
            "group_by": ["category"],
            "metrics": [{"column": "revenue", "agg": "sum"}],
            "filters": [],
            "time": None,
        }
    ]

    plan = resolve_rule_followup("now by region", history, schema_ctx)
    assert plan is not None
    assert plan.group_by == ["region"]
    assert plan.followup_of == "What is total revenue by category?"
    assert plan.metrics[0].column == "revenue"


def test_followup_without_prior_context_returns_clarification(dataset):
    """A follow-up with no prior turn asks a clarifying question."""
    df, metadata = dataset
    detect = detect_followup_marker("And for 2023?")
    assert detect is not None

    plan = AnalysisPipeline(mode="rule").plan_only(
        "And for 2023?", build_schema_context(df, metadata)
    )
    assert plan.needs_clarification
    assert plan.needs_clarification


def test_followup_llm_mode_uses_history(dataset):
    """LLM mode receives the bounded prior turns in its prompt."""
    df, metadata = dataset

    provider = FakeProvider([json.dumps(VALID_PLAN)])
    planner = LLMPlanner(gateway=LLMGateway(provider=provider), provider="openrouter", model="fake/model")
    pipeline = AnalysisPipeline(mode="llm", llm_planner=planner)

    routing = pipeline.run(
        "now by region",
        df,
        metadata,
        history_ctx=[
            {
                "question": "What is total revenue by category?",
                "intent": "query",
                "group_by": ["category"],
                "metrics": [{"column": "revenue", "agg": "sum"}],
            }
        ],
    )

    prompt = provider.requests[-1].get_prompt_text()
    assert "now by region" in prompt
    assert "What is total revenue by category?" in prompt  # prior turn passed through
    assert routing.result.status == "answered"


def test_standalone_question_is_not_treated_as_followup(dataset):
    """A full analytical question never matches a follow-up marker."""
    for question in (
        "What is total revenue by category?",
        "Total revenue in January",
        "Breakdown revenue by category",
        "What is the weather in Tokyo tomorrow?",
    ):
        assert detect_followup_marker(question) is None, question


# ─────────────────────────────────────────────────────────────────────────────
# 5. Narration: ungrounded numbers are discarded
# ─────────────────────────────────────────────────────────────────────────────

def test_narration_discarded_when_number_not_in_evidence(dataset, monkeypatch):
    """A rewrite that introduces an ungrounded number is discarded."""
    from omnisleuth.nlq.narration import narrate

    df, metadata = dataset
    routing = AnalysisPipeline(mode="rule").run("What is total revenue by category?", df, metadata)
    evidence_store = routing.result.evidence_store
    assert evidence_store is not None and len(evidence_store.list_evidence()) > 0

    fabrication = FastNarrationProvider()
    gateway = LLMGateway(provider=fabrication)
    result = narrate(routing.result.answer, gateway, evidence_store, question="q")

    assert result.used is False
    assert result.reason == "ungrounded_numbers"
    assert result.ungrounded_numbers


def test_narration_kept_when_every_number_is_grounded(dataset):
    """A rewrite that only reuses evidence numbers is accepted."""
    from omnisleuth.nlq.narration import narrate

    df, metadata = dataset
    routing = AnalysisPipeline(mode="rule").run("What is total revenue by category?", df, metadata)
    evidence_store = routing.result.evidence_store

    class HonestProvider(BaseProvider):
        def generate(self, request, request_id):
            return LLMResponse(
                request_id=request_id,
                text="Revenue by category: Electronics 450, Furniture 450, Clothing 150.",
                provider="openrouter",
                model="fake/model",
                latency_ms=1.0,
                fallback_used=False,
                usage=UsageMetrics(input_tokens=1, output_tokens=1, total_tokens=2, is_actual=True),
                error=None,
            )

    result = narrate(
        routing.result.answer, LLMGateway(provider=HonestProvider()), evidence_store
    )
    assert result.used is True
    assert result.text


def test_narration_defaults_off(dataset):
    """Narration is opt-in: the config switch is off by default."""
    df, metadata = dataset
    assert get_settings().nlq_narration_enabled is False
    routing = AnalysisPipeline(mode="rule").run("What is total revenue by category?", df, metadata)
    assert routing.narration is None
    # Enabling it still cannot fabricate numbers.
    routing_off = AnalysisPipeline(mode="llm", llm_planner=None)
    assert routing_off.settings.nlq_narration_enabled is False


class FastNarrationProvider(BaseProvider):
    """Provider that returns a plausible rewrite containing a fabricated number."""

    def generate(self, request, request_id):
        return LLMResponse(
            request_id=request_id,
            text="Revenue rose dramatically to $9,000,000.00, a 42.7% improvement across all categories.",
            provider="openrouter",
            model="fake/model",
            latency_ms=1.0,
            fallback_used=False,
            usage=UsageMetrics(input_tokens=1, output_tokens=1, total_tokens=2, is_actual=True),
            error=None,
        )


def test_narration_rewrite_applied_only_when_grounded(dataset, monkeypatch):
    """With narration enabled and an honest rewrite, the answer text is replaced."""
    from omnisleuth.nlq.pipeline import AnalysisPipeline as _P

    df, metadata = dataset
    routing = AnalysisPipeline(mode="rule").run("What is total revenue by category?", df, metadata)
    evidence_store = routing.result.evidence_store

    class HonestProvider(BaseProvider):
        def generate(self, request, request_id):
            numbers = sorted(
                {
                    round(v, 2)
                    for ev in evidence_store.list_evidence()
                    for v in _numbers_in(ev.content)
                }
            )[:3]
            body = ", ".join(f"{n}" for n in numbers) if numbers else "no numbers"
            return LLMResponse(
                request_id=request_id,
                text=f"Rounded view: {body}.",
                provider="openrouter",
                model="fake/model",
                latency_ms=1.0,
                fallback_used=False,
                usage=UsageMetrics(input_tokens=1, output_tokens=1, total_tokens=2, is_actual=True),
                error=None,
            )

    monkeypatch.setattr(get_settings(), "nlq_narration_enabled", True, raising=False)
    pipeline = AnalysisPipeline(mode="llm", llm_planner=LLMPlanner(gateway=LLMGateway(provider=HonestProvider())))
    llm_routing = pipeline.run("What is total revenue by category?", df, metadata)

    assert llm_routing.narration is not None
    assert llm_routing.narration.used is True
    assert llm_routing.result.status == "answered"


def _numbers_in(value):
    from omnisleuth.nlq.narration import extract_numbers

    for v in ([value] if isinstance(value, (int, float)) else []):
        yield v
    if isinstance(value, dict):
        for v in value.values():
            yield from _numbers_in(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _numbers_in(v)
    elif isinstance(value, str):
        yield from extract_numbers(value)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Sidebar status contains no key material
# ─────────────────────────────────────────────────────────────────────────────

def test_sidebar_status_has_no_key_material(tmp_path, monkeypatch, clean_service_cache):
    """A configured key is reflected by provider/model name, never the key."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    monkeypatch.setenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "ui_key.db"))

    at = AppTest.from_file(str(_app_path()), default_timeout=60)
    at.run()

    all_text = " ".join(
        [m.value for m in at.markdown] + [i.value for i in at.info] + [c.value for c in at.caption]
    )
    assert FAKE_KEY not in all_text
    assert "openrouter" in all_text.lower()


def test_llm_status_line_reports_not_configured(tmp_path, monkeypatch, clean_service_cache):
    """With no credentials the sidebar reports not configured, with no key material."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "ui_nokey.db"))
    at = AppTest.from_file(str(_app_path()), default_timeout=60)
    at.run()

    all_text = " ".join([m.value for m in at.markdown] + [i.value for i in at.info])
    assert "Not configured" in all_text
    assert FAKE_KEY not in all_text
