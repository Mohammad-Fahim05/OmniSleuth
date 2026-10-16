# T-004 regression tests

"""Regression tests for strict tool-selection accuracy in Phase 18."""

from types import SimpleNamespace

import pandas as pd
import pytest

from omnisleuth.evaluation import OmniSleuthEvaluator
from omnisleuth.evaluation.benchmark_suite import BenchmarkQuestion
from omnisleuth.evaluation import evaluator as evaluator_module
from omnisleuth.ingestion.models import DatasetMetadata

def make_question(question_id, expected_tools):
    return BenchmarkQuestion(
        id=question_id,
        category="analytical",
        question=f"Test question {question_id}",
        expected_tools=expected_tools,
        expected_ground_truth_keywords=[],
        should_succeed=True,
    )


def make_metadata():
    return DatasetMetadata(
        dataset_id="test-dataset",
        name="Test dataset",
        file_path="test.csv",
        version_hash="test-version",
        row_count=2,
        column_count=2,
    )


def make_task(tool_name):
    return SimpleNamespace(tool_name=tool_name)


class StubOrchestrator:
    def __init__(self, executed_by_question):
        self.executed_by_question = executed_by_question

    def run_investigation(self, question, df, metadata):
        tool_names = self.executed_by_question[question]
        return {
            "status": "completed",
            "executed_tasks": [make_task(name) for name in tool_names],
            "evidence_store": None,
            "validation_results": [],
            "final_answer": "",
        }


@pytest.fixture
def dataframe():
    return pd.DataFrame({"value": [10, 20], "group": ["A", "B"]})


@pytest.mark.parametrize(
    ("expected", "executed", "correct"),
    [
        (["aggregate_metric"], ["aggregate_metric"], True),
        (["aggregate_metric"], [], False),
        (
            ["aggregate_metric", "compare_segments"],
            ["aggregate_metric", "compare_segments"],
            True,
        ),
        (
            ["aggregate_metric", "compare_segments"],
            ["aggregate_metric"],
            False,
        ),
        (
            ["aggregate_metric"],
            ["aggregate_metric", "create_chart"],
            True,
        ),
        (
            ["aggregate_metric", "aggregate_metric"],
            ["aggregate_metric"],
            True,
        ),
    ],
)
def test_omnisleuth_requires_all_distinct_expected_tools(
    dataframe, expected, executed, correct
):
    question = make_question("q1", expected)
    evaluator = OmniSleuthEvaluator(
        orchestrator=StubOrchestrator({question.question: executed})
    )

    result = evaluator.evaluate_omnisleuth(
        dataframe, make_metadata(), questions=[question]
    )

    assert result.per_question_results[0].tool_selection_correct is correct
    assert result.tool_selection_accuracy_pct == (100.0 if correct else 0.0)


def test_omnisleuth_excludes_empty_expected_tools_from_denominator(dataframe):
    questions = [
        make_question("q1", ["aggregate_metric"]),
        make_question("q2", []),
    ]
    executed = {
        questions[0].question: ["aggregate_metric"],
        questions[1].question: [],
    }
    evaluator = OmniSleuthEvaluator(
        orchestrator=StubOrchestrator(executed)
    )

    result = evaluator.evaluate_omnisleuth(
        dataframe, make_metadata(), questions=questions
    )

    assert result.tool_selection_accuracy_pct == 100.0


def test_omnisleuth_returns_zero_when_no_questions_have_expected_tools(dataframe):
    question = make_question("q1", [])
    evaluator = OmniSleuthEvaluator(
        orchestrator=StubOrchestrator({question.question: []})
    )

    result = evaluator.evaluate_omnisleuth(
        dataframe, make_metadata(), questions=[question]
    )

    assert result.tool_selection_accuracy_pct == 0.0


def test_omnisleuth_execution_exception_counts_as_incorrect(dataframe):
    question = make_question("q1", ["aggregate_metric"])

    class RaisingOrchestrator:
        def run_investigation(self, question_text, df, metadata):
            raise RuntimeError("simulated failure")

    evaluator = OmniSleuthEvaluator(orchestrator=RaisingOrchestrator())

    result = evaluator.evaluate_omnisleuth(
        dataframe, make_metadata(), questions=[question]
    )

    assert result.per_question_results[0].tool_selection_correct is False
    assert result.tool_selection_accuracy_pct == 0.0


# ─── Helper-Level Tests ────────────────────────────────────────────────────────


def test_helper_one_expected_tool_executed_correctly():
    assert (
        evaluator_module._tool_selection_correct(
            ["aggregate_metric"], ["aggregate_metric"]
        )
        is True
    )


def test_helper_one_expected_tool_missing():
    assert evaluator_module._tool_selection_correct(["aggregate_metric"], []) is False


def test_helper_two_expected_tools_both_executed():
    assert (
        evaluator_module._tool_selection_correct(
            ["aggregate_metric", "compare_segments"],
            ["aggregate_metric", "compare_segments"],
        )
        is True
    )


def test_helper_two_expected_tools_one_executed():
    assert (
        evaluator_module._tool_selection_correct(
            ["aggregate_metric", "compare_segments"], ["aggregate_metric"]
        )
        is False
    )


def test_helper_extra_unexpected_tools_allowed():
    assert (
        evaluator_module._tool_selection_correct(
            ["aggregate_metric"], ["aggregate_metric", "create_chart"]
        )
        is True
    )


def test_helper_duplicate_expected_tools_treated_as_one():
    assert (
        evaluator_module._tool_selection_correct(
            ["aggregate_metric", "aggregate_metric"], ["aggregate_metric"]
        )
        is True
    )


def test_helper_empty_expected_tools_returns_none():
    assert evaluator_module._tool_selection_correct([], ["aggregate_metric"]) is None
    assert evaluator_module._tool_selection_correct([], []) is None


def test_helper_empty_expected_tools_excluded_from_aggregate_accuracy():
    score = evaluator_module._tool_selection_accuracy(
        expected_tools=[["aggregate_metric"], []],
        executed_tools=[["aggregate_metric"], []],
    )
    assert score == 100.0


def test_helper_aggregate_accuracy_returns_zero_when_no_eligible():
    score = evaluator_module._tool_selection_accuracy(
        expected_tools=[[], []],
        executed_tools=[["aggregate_metric"], []],
    )
    assert score == 0.0


def test_helper_mixed_eligible_and_ineligible_uses_only_eligible_denominator():
    # 2 eligible (1 correct, 1 incorrect) + 1 ineligible -> 1/2 = 50.0%
    score = evaluator_module._tool_selection_accuracy(
        expected_tools=[["aggregate_metric"], ["compare_segments"], []],
        executed_tools=[["aggregate_metric"], [], ["something"]],
    )
    assert score == 50.0


def test_shared_accuracy_policy_handles_duplicates_and_empty_expected_tools():
    score = evaluator_module._tool_selection_accuracy(
        expected_tools=[["aggregate_metric", "aggregate_metric"], []],
        executed_tools=[["aggregate_metric"], []],
    )

    assert score == 100.0


def test_shared_accuracy_policy_returns_zero_without_eligible_questions():
    score = evaluator_module._tool_selection_accuracy(
        expected_tools=[[], []],
        executed_tools=[[], []],
    )

    assert score == 0.0


# ─── Baseline Integration Tests ───────────────────────────────────────────────


def make_baseline_metadata():
    return DatasetMetadata(
        dataset_id="test-dataset",
        name="Test dataset",
        file_path="test.csv",
        version_hash="test-version",
        row_count=2,
        column_count=3,
        numeric_columns=["revenue"],
        categorical_columns=["region"],
        datetime_columns=["order_date"],
        column_types={
            "revenue": "numeric",
            "region": "categorical",
            "order_date": "datetime",
        },
    )


@pytest.fixture
def baseline_dataframe():
    return pd.DataFrame(
        {
            "revenue": [100.0, 200.0],
            "region": ["North", "South"],
            "order_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )


def test_baseline_direct_tools_single_tool_correct(baseline_dataframe):
    question = make_question("q1", ["aggregate_metric"])
    evaluator = OmniSleuthEvaluator()

    result = evaluator.evaluate_baseline_direct_tools(
        baseline_dataframe, make_baseline_metadata(), questions=[question]
    )

    assert result.per_question_results[0].tool_selection_correct is True
    assert result.tool_selection_accuracy_pct == 100.0


def test_baseline_direct_tools_subset_executed_is_incorrect(baseline_dataframe):
    # Baseline 1 attempts only the first expected tool
    question = make_question("q1", ["aggregate_metric", "compare_segments"])
    evaluator = OmniSleuthEvaluator()

    result = evaluator.evaluate_baseline_direct_tools(
        baseline_dataframe, make_baseline_metadata(), questions=[question]
    )

    assert result.per_question_results[0].tool_selection_correct is False
    assert result.per_question_results[0].tools_executed == ["aggregate_metric"]
    assert result.tool_selection_accuracy_pct == 0.0


def test_baseline_direct_tools_excludes_empty_expected_tools(baseline_dataframe):
    questions = [
        make_question("q1", ["aggregate_metric"]),
        BenchmarkQuestion(
            id="q2",
            category="ambiguous_invalid",
            question="Invalid question",
            expected_tools=[],
            expected_ground_truth_keywords=[],
            should_succeed=False,
        ),
    ]
    evaluator = OmniSleuthEvaluator()

    result = evaluator.evaluate_baseline_direct_tools(
        baseline_dataframe, make_baseline_metadata(), questions=questions
    )

    assert result.tool_selection_accuracy_pct == 100.0


def test_baseline_direct_tools_returns_zero_when_no_eligible(baseline_dataframe):
    question = BenchmarkQuestion(
        id="q1",
        category="ambiguous_invalid",
        question="Invalid question",
        expected_tools=[],
        expected_ground_truth_keywords=[],
        should_succeed=False,
    )
    evaluator = OmniSleuthEvaluator()

    result = evaluator.evaluate_baseline_direct_tools(
        baseline_dataframe, make_baseline_metadata(), questions=[question]
    )

    assert result.tool_selection_accuracy_pct == 0.0


def test_baseline_direct_tools_execution_failure_is_incorrect_and_in_denominator(
    baseline_dataframe, monkeypatch
):
    from omnisleuth.tools.definitions import AggregateMetricTool

    question = make_question("q1", ["aggregate_metric"])
    evaluator = OmniSleuthEvaluator()

    def failing_run(*args, **kwargs):
        raise RuntimeError("simulated tool failure")

    monkeypatch.setattr(AggregateMetricTool, "run", failing_run)

    result = evaluator.evaluate_baseline_direct_tools(
        baseline_dataframe, make_baseline_metadata(), questions=[question]
    )

    assert result.per_question_results[0].tool_selection_correct is False
    assert result.per_question_results[0].status == "error"
    assert result.tool_selection_accuracy_pct == 0.0


def test_baseline_fixed_pipeline_two_tools_both_executed(
    baseline_dataframe, monkeypatch
):
    from omnisleuth.evidence.store import EvidenceStore

    monkeypatch.setattr(EvidenceStore, "add_evidence", lambda self, *a, **kw: None)

    question = make_question("q1", ["aggregate_metric", "compare_segments"])
    evaluator = OmniSleuthEvaluator()

    result = evaluator.evaluate_baseline_fixed_pipeline(
        baseline_dataframe, make_baseline_metadata(), questions=[question]
    )

    assert result.per_question_results[0].tool_selection_correct is True
    assert set(result.per_question_results[0].tools_executed) == {
        "aggregate_metric",
        "compare_segments",
    }
    assert result.tool_selection_accuracy_pct == 100.0


def test_baseline_fixed_pipeline_subset_executed_is_incorrect(
    baseline_dataframe, monkeypatch
):
    # Baseline 2 executes up to two expected tools; expecting three means only subset executes
    from omnisleuth.evidence.store import EvidenceStore

    monkeypatch.setattr(EvidenceStore, "add_evidence", lambda self, *a, **kw: None)

    question = make_question(
        "q1", ["aggregate_metric", "compare_segments", "calculate_trend"]
    )
    evaluator = OmniSleuthEvaluator()

    result = evaluator.evaluate_baseline_fixed_pipeline(
        baseline_dataframe, make_baseline_metadata(), questions=[question]
    )

    assert result.per_question_results[0].tool_selection_correct is False
    assert len(result.per_question_results[0].tools_executed) == 2
    assert result.tool_selection_accuracy_pct == 0.0


def test_baseline_fixed_pipeline_excludes_empty_expected_tools(
    baseline_dataframe, monkeypatch
):
    from omnisleuth.evidence.store import EvidenceStore

    monkeypatch.setattr(EvidenceStore, "add_evidence", lambda self, *a, **kw: None)

    questions = [
        make_question("q1", ["aggregate_metric"]),
        BenchmarkQuestion(
            id="q2",
            category="ambiguous_invalid",
            question="Invalid question",
            expected_tools=[],
            expected_ground_truth_keywords=[],
            should_succeed=False,
        ),
    ]
    evaluator = OmniSleuthEvaluator()

    result = evaluator.evaluate_baseline_fixed_pipeline(
        baseline_dataframe, make_baseline_metadata(), questions=questions
    )

    assert result.tool_selection_accuracy_pct == 100.0


def test_baseline_fixed_pipeline_returns_zero_when_no_eligible(baseline_dataframe):
    question = BenchmarkQuestion(
        id="q1",
        category="ambiguous_invalid",
        question="Invalid question",
        expected_tools=[],
        expected_ground_truth_keywords=[],
        should_succeed=False,
    )
    evaluator = OmniSleuthEvaluator()

    result = evaluator.evaluate_baseline_fixed_pipeline(
        baseline_dataframe, make_baseline_metadata(), questions=[question]
    )

    assert result.tool_selection_accuracy_pct == 0.0


def test_baseline_fixed_pipeline_execution_failure_is_incorrect_and_in_denominator(
    baseline_dataframe,
):
    # Without monkeypatching, store.add_evidence raises exception during execution
    question = make_question("q1", ["aggregate_metric"])
    evaluator = OmniSleuthEvaluator()

    result = evaluator.evaluate_baseline_fixed_pipeline(
        baseline_dataframe, make_baseline_metadata(), questions=[question]
    )

    assert result.per_question_results[0].tool_selection_correct is False
    assert result.per_question_results[0].status == "error"
    assert result.tool_selection_accuracy_pct == 0.0
