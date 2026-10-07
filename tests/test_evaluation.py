"""Unit tests for Phase 14 - Benchmark Evaluation."""

from pathlib import Path
import pytest
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

    assert len(df) > 200  # Includes duplicates
    assert "order_id" in df.columns
    assert "revenue" in df.columns
    assert "region" in df.columns
    assert "category" in df.columns
    assert csv_path.exists()


def test_evaluator_benchmark_run(tmp_path: Path):
    """Test running OmniSleuth evaluator on a subset of benchmark questions."""
    csv_path = tmp_path / "eval_sales.csv"
    generate_benchmark_sales_dataset(n_rows=150, random_seed=42, output_path=csv_path)

    loader = CSVLoader()
    df, metadata = loader.load_csv(csv_path)

    evaluator = OmniSleuthEvaluator()
    # Test on top 3 questions
    subset = BENCHMARK_QUESTIONS[:3]
    result = evaluator.evaluate_omnisleuth(df, metadata, questions=subset)

    assert result.total_questions == 3
    assert result.tool_selection_accuracy_pct >= 80.0
    assert result.evidence_coverage_pct == 100.0
    assert result.unsupported_claim_rate_pct == 0.0
    assert result.latency_p50_ms > 0.0
