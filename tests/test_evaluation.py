"""Unit tests for Phase 18 - Benchmark Evaluation and Baselines Harness."""

from pathlib import Path
import pytest
import pandas as pd

from omnisleuth.evaluation import (
    BENCHMARK_QUESTIONS,
    OmniSleuthEvaluator,
    generate_benchmark_sales_dataset,
)
from omnisleuth.ingestion import CSVLoader


def test_synthetic_dataset_generator(tmp_path: Path):
    """Test synthetic benchmark sales dataset generator."""
    csv_path = tmp_path / "benchmark_sales.csv"
    df = generate_benchmark_sales_dataset(n_rows=200, random_seed=42, output_path=csv_path)

    assert len(df) > 200  # Includes duplicates and synthetic noise
    assert "order_id" in df.columns
    assert "revenue" in df.columns
    assert "region" in df.columns
    assert "category" in df.columns
    assert csv_path.exists()


def test_benchmark_suite_categories():
    """Verify that all 40 categorized benchmark questions exist."""
    assert len(BENCHMARK_QUESTIONS) == 40

    categories = {q.category for q in BENCHMARK_QUESTIONS}
    assert "planted_finding" in categories
    assert "anomaly_detection" in categories
    assert "rag_context" in categories
    assert "ambiguous_invalid" in categories
    assert "analytical" in categories

    planted = [q for q in BENCHMARK_QUESTIONS if q.category == "planted_finding"]
    assert len(planted) == 15

    anomaly = [q for q in BENCHMARK_QUESTIONS if q.category == "anomaly_detection"]
    assert len(anomaly) == 5

    rag = [q for q in BENCHMARK_QUESTIONS if q.category == "rag_context"]
    assert len(rag) == 5

    ambig = [q for q in BENCHMARK_QUESTIONS if q.category == "ambiguous_invalid"]
    assert len(ambig) == 5

    analytical = [q for q in BENCHMARK_QUESTIONS if q.category == "analytical"]
    assert len(analytical) == 10


def test_evaluator_benchmark_run(tmp_path: Path):
    """Test running OmniSleuth evaluator on a subset of benchmark questions."""
    csv_path = tmp_path / "eval_sales.csv"
    generate_benchmark_sales_dataset(n_rows=150, random_seed=42, output_path=csv_path)

    loader = CSVLoader()
    df, metadata = loader.load_csv(csv_path)

    evaluator = OmniSleuthEvaluator()
    subset = BENCHMARK_QUESTIONS[:3]
    result = evaluator.evaluate_omnisleuth(df, metadata, questions=subset)

    assert result.total_questions == 3
    assert result.tool_selection_accuracy_pct >= 80.0
    assert result.evidence_coverage_pct == 100.0
    assert result.unsupported_claim_rate_pct == 0.0
    assert result.latency_p50_ms > 0.0
    assert len(result.per_question_results) == 3


def test_evaluator_baselines(tmp_path: Path):
    """Test running Baseline 1 and Baseline 2 on benchmark questions."""
    csv_path = tmp_path / "eval_sales.csv"
    generate_benchmark_sales_dataset(n_rows=150, random_seed=42, output_path=csv_path)

    loader = CSVLoader()
    df, metadata = loader.load_csv(csv_path)

    evaluator = OmniSleuthEvaluator()
    subset = BENCHMARK_QUESTIONS[:3]

    # Baseline 1
    b1_res = evaluator.evaluate_baseline_direct_tools(df, metadata, questions=subset)
    assert b1_res.total_questions == 3
    assert b1_res.numerical_correctness_pct == 0.0
    assert len(b1_res.per_question_results) == 3

    # Baseline 2
    b2_res = evaluator.evaluate_baseline_fixed_pipeline(df, metadata, questions=subset)
    assert b2_res.total_questions == 3
    assert len(b2_res.per_question_results) == 3


def test_run_full_benchmark_and_persistence(tmp_path: Path):
    """Test full benchmark runner execution and JSON persistence."""
    csv_path = tmp_path / "eval_sales.csv"
    generate_benchmark_sales_dataset(n_rows=100, random_seed=42, output_path=csv_path)

    loader = CSVLoader()
    df, metadata = loader.load_csv(csv_path)

    evaluator = OmniSleuthEvaluator()
    subset = BENCHMARK_QUESTIONS[:2]

    exp = evaluator.run_full_benchmark(
        df, metadata, questions=subset, output_dir=tmp_path
    )

    assert exp.experiment_id.startswith("exp_")
    assert "omnisleuth" in exp.systems
    assert "baseline_1_direct_tools" in exp.systems
    assert "baseline_2_fixed_pipeline" in exp.systems

    json_files = list(tmp_path.glob("exp_*.json"))
    assert len(json_files) == 1
    assert json_files[0].exists()
