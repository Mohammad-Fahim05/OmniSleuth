"""Focused regression tests for documentation-query recognition, routing, and execution.

Tests:
1. RulePlanner recognizes definitional / explanatory questions before flexible query matching:
   - "explain what is revenue"
   - "Explain what revenue means in this dataset."
   - "what does revenue mean?"
2. RulePlanner preserves ordinary quantitative queries without regression:
   - "What is the average revenue by category?"
   - "What is total revenue by region?"
3. PlanExecutor handles PlanIntent.DOCUMENTATION_QUESTION:
   - Returns clear, dataset-grounded empirical characteristics when target column is present.
   - Provides a helpful general explanation when target column is missing/unmapped.
   - Never invents business definitions or claims metadata establishes the full meaning of a metric.
   - Evidence grounding and ClaimValidator verify all deterministic numbers.
4. Pipeline routing in both 'rule' and 'llm' modes:
   - No spurious disagreement note when LLM and Rule planners agree on documentation_question.
   - Quantitative analytical queries compute exact deterministic results.
"""

import json
import os
from typing import List
import pandas as pd
import pytest

from omnisleuth.core.config import get_settings, reset_settings
from omnisleuth.evidence.models import ClaimStatus
from omnisleuth.ingestion.models import ColumnSchema, ColumnType, DatasetMetadata
from omnisleuth.llm import LLMRequest, LLMResponse, UsageMetrics
from omnisleuth.llm.gateway import LLMGateway
from omnisleuth.llm.providers import BaseProvider
from omnisleuth.nlq.context import build_schema_context
from omnisleuth.nlq.executor import PlanExecutor
from omnisleuth.nlq.llm_planner import LLMPlanner
from omnisleuth.nlq.models import MetricAgg, PlanIntent
from omnisleuth.nlq.pipeline import AnalysisPipeline
from omnisleuth.nlq.rule_planner import RulePlanner


class ScriptedProvider(BaseProvider):
    """Deterministic provider returning scripted plan JSON."""

    def __init__(self, responses: List[str], provider_name: str = "openrouter", model: str = "mock/model"):
        self._responses = list(responses)
        self.provider_name = provider_name
        self.model = model
        self.calls = 0

    def generate(self, request: LLMRequest, request_id: str) -> LLMResponse:
        self.calls += 1
        if not self._responses:
            raise AssertionError("ScriptedProvider exhausted")
        item = self._responses[0] if len(self._responses) == 1 else self._responses.pop(0)
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


@pytest.fixture(autouse=True)
def offline_env(monkeypatch, tmp_path):
    """Offline test environment: isolated database, no remote API calls."""
    for key in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "OPENROUTER_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "doc_test.db"))
    monkeypatch.setenv("APP_ENV", "testing")
    monkeypatch.delenv("NLQ_PLANNER_MODE", raising=False)
    monkeypatch.delenv("NLQ_PLANNER_FALLBACK", raising=False)
    reset_settings()
    yield get_settings()
    reset_settings()


@pytest.fixture
def sales_dataset():
    """Create a fixed, deterministic DataFrame with known categories, regions, and revenue."""
    data = {
        "order_id": [f"ORD-{i:03d}" for i in range(1, 13)],
        "category": [
            "Clothing", "Electronics", "Furniture",
            "Clothing", "Electronics", "Furniture",
            "Clothing", "Electronics", "Furniture",
            "Clothing", "Electronics", "Furniture",
        ],
        "region": [
            "North", "South", "East", "West",
            "North", "South", "East", "West",
            "North", "South", "East", "West",
        ],
        "revenue": [
            120.0, 1500.0, 850.0,
            180.0, 1600.0, 900.0,
            200.0, 1400.0, 800.0,
            140.0, 1700.0, 950.0,
        ],
        "order_date": [f"2024-0{i%6 + 1}-15" for i in range(1, 13)],
    }
    df = pd.DataFrame(data)
    metadata = DatasetMetadata(
        dataset_id="ds_doc_test",
        name="Sales Test Dataset",
        file_path="sales_test.csv",
        version_hash="hash_doc_test",
        row_count=len(df),
        column_count=len(df.columns),
        duplicate_rows=0,
        columns=[
            ColumnSchema(name="order_id", data_type=ColumnType.IDENTIFIER, dtype="object", unique_count=12),
            ColumnSchema(name="category", data_type=ColumnType.CATEGORICAL, dtype="object", unique_count=3, sample_values=["Clothing", "Electronics", "Furniture"]),
            ColumnSchema(name="region", data_type=ColumnType.CATEGORICAL, dtype="object", unique_count=4, sample_values=["North", "South", "East", "West"]),
            ColumnSchema(name="revenue", data_type=ColumnType.NUMERIC, dtype="float64", unique_count=12),
            ColumnSchema(name="order_date", data_type=ColumnType.DATETIME, dtype="object", unique_count=6),
        ],
    )
    schema_ctx = build_schema_context(df, metadata)
    return df, metadata, schema_ctx


# ─────────────────────────────────────────────────────────────────────────────
# 1. RulePlanner recognition tests
# ─────────────────────────────────────────────────────────────────────────────

def test_rule_planner_recognizes_explain_what_is_revenue(sales_dataset):
    """'explain what is revenue' maps to DOCUMENTATION_QUESTION with target_column='revenue'."""
    _, _, schema_ctx = sales_dataset
    planner = RulePlanner()

    plan = planner.plan("explain what is revenue", schema_ctx)
    assert plan.intent == PlanIntent.DOCUMENTATION_QUESTION
    assert plan.target_column == "revenue"
    assert plan.confidence == 0.9


def test_rule_planner_recognizes_explain_what_revenue_means(sales_dataset):
    """'Explain what revenue means in this dataset.' maps to DOCUMENTATION_QUESTION."""
    _, _, schema_ctx = sales_dataset
    planner = RulePlanner()

    plan = planner.plan("Explain what revenue means in this dataset.", schema_ctx)
    assert plan.intent == PlanIntent.DOCUMENTATION_QUESTION
    assert plan.target_column == "revenue"


def test_rule_planner_recognizes_what_does_revenue_mean(sales_dataset):
    """'what does revenue mean?' maps to DOCUMENTATION_QUESTION."""
    _, _, schema_ctx = sales_dataset
    planner = RulePlanner()

    plan = planner.plan("what does revenue mean?", schema_ctx)
    assert plan.intent == PlanIntent.DOCUMENTATION_QUESTION
    assert plan.target_column == "revenue"


def test_rule_planner_recognizes_categorical_column_documentation(sales_dataset):
    """'what does region mean?' maps to DOCUMENTATION_QUESTION with target_column='region'."""
    _, _, schema_ctx = sales_dataset
    planner = RulePlanner()

    plan = planner.plan("what does region mean?", schema_ctx)
    assert plan.intent == PlanIntent.DOCUMENTATION_QUESTION
    assert plan.target_column == "region"


def test_rule_planner_recognizes_unmapped_glossary_term(sales_dataset):
    """'what is aov' maps to DOCUMENTATION_QUESTION with target_column=None when not in dataset."""
    _, _, schema_ctx = sales_dataset
    planner = RulePlanner()

    plan = planner.plan("what is aov", schema_ctx)
    assert plan.intent == PlanIntent.DOCUMENTATION_QUESTION
    assert plan.target_column is None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Ordinary quantitative queries preserve exact aggregation behavior
# ─────────────────────────────────────────────────────────────────────────────

def test_rule_planner_preserves_average_revenue_by_category(sales_dataset):
    """'What is the average revenue by category?' plans QUERY with mean(revenue) grouped by category."""
    _, _, schema_ctx = sales_dataset
    planner = RulePlanner()

    plan = planner.plan("What is the average revenue by category?", schema_ctx)
    assert plan.intent == PlanIntent.QUERY
    assert len(plan.metrics) == 1
    assert plan.metrics[0].column == "revenue"
    assert plan.metrics[0].agg == MetricAgg.MEAN
    assert plan.group_by == ["category"]


def test_rule_planner_preserves_total_revenue_by_region(sales_dataset):
    """'What is total revenue by region?' plans QUERY with sum(revenue) grouped by region."""
    _, _, schema_ctx = sales_dataset
    planner = RulePlanner()

    plan = planner.plan("What is total revenue by region?", schema_ctx)
    assert plan.intent == PlanIntent.QUERY
    assert len(plan.metrics) == 1
    assert plan.metrics[0].column == "revenue"
    assert plan.metrics[0].agg == MetricAgg.SUM
    assert plan.group_by == ["region"]


# ─────────────────────────────────────────────────────────────────────────────
# 3. PlanExecutor execution of DOCUMENTATION_QUESTION
# ─────────────────────────────────────────────────────────────────────────────

def test_executor_handles_documentation_question_with_target_column(sales_dataset):
    """Executor returns grounded distribution stats, clear caveat, and supported claims."""
    df, metadata, _ = sales_dataset
    executor = PlanExecutor()
    plan = RulePlanner().plan("explain what is revenue", build_schema_context(df, metadata))

    result = executor.execute(plan, df, metadata, "explain what is revenue")

    assert result.status == "answered"
    assert "revenue" in result.answer.lower()
    assert "Data Grounding Note" in result.answer
    assert "Suggested Analytical Questions" in result.answer
    assert len(result.result_table) >= 7

    # Check that structured table includes exact numeric characteristics
    props = {row["Property"]: row["Value"] for row in result.result_table}
    assert props["Column Name"] == "revenue"
    assert props["Inferred Data Type"] == "numeric"
    assert int(props["Total Rows"]) == len(df)

    # Confirm claims and evidence items are verified
    assert result.evidence_store is not None
    claims = result.evidence_store.list_claims()
    assert len(claims) == 1
    assert claims[0].status == ClaimStatus.SUPPORTED

    # Assert exact arithmetic match on the mean
    expected_mean = round(float(df["revenue"].mean()), 2)
    assert f"{expected_mean:,.2f}" in props["Mean Value"]


def test_executor_handles_documentation_question_without_target_column(sales_dataset):
    """Executor provides a helpful general explanation when target column is unmapped."""
    df, metadata, schema_ctx = sales_dataset
    executor = PlanExecutor()
    plan = RulePlanner().plan("what is aov", schema_ctx)

    result = executor.execute(plan, df, metadata, "what is aov")

    assert result.status == "answered"
    assert "Business definitions and metric formulas are not embedded in dataset metadata" in result.answer
    assert "Suggested Analytical Questions" in result.answer
    assert any("revenue" in row["Column"] for row in result.result_table)


# ─────────────────────────────────────────────────────────────────────────────
# 4. End-to-end Pipeline routing tests
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_rule_mode_answers_documentation_query(sales_dataset):
    """In 'rule' mode, pipeline executes documentation query and reports status='answered'."""
    df, metadata, _ = sales_dataset
    pipeline = AnalysisPipeline(mode="rule")

    routing = pipeline.run("Explain what revenue means in this dataset.", df, metadata)

    assert routing.result.status == "answered"
    assert routing.planner_used == "rule"
    assert routing.result.plan.intent == PlanIntent.DOCUMENTATION_QUESTION
    assert routing.result.plan.target_column == "revenue"
    # No spurious disagreement notes in limitations
    assert not any("disagreement note" in lim.lower() for lim in routing.result.limitations)


def test_pipeline_llm_mode_no_disagreement_when_both_agree_on_documentation(sales_dataset):
    """In 'llm' mode, when LLM plans documentation_question, RulePlanner agrees without disagreement note."""
    df, metadata, _ = sales_dataset

    llm_doc_plan = {
        "intent": "documentation_question",
        "target_column": "revenue",
        "confidence": 0.95,
    }
    provider = ScriptedProvider([json.dumps(llm_doc_plan)])
    gateway = LLMGateway(provider=provider)
    llm_planner = LLMPlanner(gateway=gateway)
    pipeline = AnalysisPipeline(mode="llm", llm_planner=llm_planner)

    routing = pipeline.run("explain what is revenue", df, metadata)

    assert routing.result.status == "answered"
    assert routing.planner_used == "llm"
    assert routing.result.plan.intent == PlanIntent.DOCUMENTATION_QUESTION
    # Crucial assertion: Rule planner planned documentation_question, so NO disagreement limitation was added!
    assert not any("disagreement note" in lim.lower() for lim in routing.result.limitations)


def test_pipeline_analytical_average_revenue_by_category_truth(sales_dataset):
    """Pipeline computes exact numeric truth for 'What is the average revenue by category?'."""
    df, metadata, _ = sales_dataset
    pipeline = AnalysisPipeline(mode="rule")

    routing = pipeline.run("What is the average revenue by category?", df, metadata)

    assert routing.result.status == "answered"
    assert routing.result.plan.intent == PlanIntent.QUERY
    assert routing.result.result_table is not None

    # Compare against independent pandas computation
    expected_means = df.groupby("category")["revenue"].mean().to_dict()
    for row in routing.result.result_table:
        cat = row["category"]
        actual_mean = row["revenue"]
        assert pytest.approx(actual_mean, rel=1e-3) == expected_means[cat]


def test_pipeline_analytical_total_revenue_by_region_truth(sales_dataset):
    """Pipeline computes exact numeric truth for 'What is total revenue by region?'."""
    df, metadata, _ = sales_dataset
    pipeline = AnalysisPipeline(mode="rule")

    routing = pipeline.run("What is total revenue by region?", df, metadata)

    assert routing.result.status == "answered"
    assert routing.result.plan.intent == PlanIntent.QUERY
    assert routing.result.result_table is not None

    # Compare against independent pandas computation
    expected_sums = df.groupby("region")["revenue"].sum().to_dict()
    for row in routing.result.result_table:
        reg = row["region"]
        actual_sum = row["revenue"]
        assert pytest.approx(actual_sum, rel=1e-3) == expected_sums[reg]


def test_pipeline_llm_mode_safety_net_overrides_general_knowledge_for_dataset_column(sales_dataset):
    """When LLM plans general_knowledge for 'explain what is revenue', safety net routes to rule plan because 'revenue' is a dataset column."""
    df, metadata, _ = sales_dataset

    llm_gk_plan = {
        "intent": "general_knowledge",
        "confidence": 1.0,
    }
    provider = ScriptedProvider([json.dumps(llm_gk_plan)])
    gateway = LLMGateway(provider=provider)
    llm_planner = LLMPlanner(gateway=gateway)
    pipeline = AnalysisPipeline(mode="llm", llm_planner=llm_planner)

    routing = pipeline.run("explain what is revenue", df, metadata)

    # 1. Status is answered, NOT unsupported
    assert routing.result.status == "answered"
    # 2. Safety net substituted rule plan intent DOCUMENTATION_QUESTION
    assert routing.result.plan.intent == PlanIntent.DOCUMENTATION_QUESTION
    assert routing.result.plan.target_column == "revenue"
    # 3. Disagreement note recorded for transparency
    assert any("disagreement" in note.lower() for note in routing.result.limitations + routing.result.plan.assumptions)


def test_pipeline_llm_mode_unrelated_general_knowledge_not_overridden(sales_dataset):
    """Truly unrelated general knowledge questions remain general_knowledge without being forced into dataset analysis."""
    df, metadata, _ = sales_dataset

    llm_gk_plan = {
        "intent": "general_knowledge",
        "confidence": 1.0,
    }
    provider = ScriptedProvider([json.dumps(llm_gk_plan)])
    gateway = LLMGateway(provider=provider)
    llm_planner = LLMPlanner(gateway=gateway)
    pipeline = AnalysisPipeline(mode="llm", llm_planner=llm_planner)

    routing = pipeline.run("what is the capital of france?", df, metadata)

    # Unrelated general knowledge stays general_knowledge and unsupported
    assert routing.result.plan.intent == PlanIntent.GENERAL_KNOWLEDGE
    assert routing.result.status == "unsupported"


def test_llm_planner_prompt_includes_documentation_rule(sales_dataset):
    """LLMPlanner prompt contains explicit rule for documentation_question on dataset columns."""
    _, _, schema_ctx = sales_dataset
    planner = LLMPlanner()
    messages = planner._build_prompt("explain what is revenue", schema_ctx)
    user_prompt = messages[1]["content"]
    assert "documentation_question" in user_prompt
    assert "explain what is revenue" in user_prompt

