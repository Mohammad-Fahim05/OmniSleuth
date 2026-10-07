"""Unit tests for Phase 2 - Dataset Profiling and Quality Assessment."""

from pathlib import Path
import pandas as pd
import pytest
from omnisleuth.ingestion import CSVLoader
from omnisleuth.profiling import DatasetProfiler, WarningSeverity


@pytest.fixture
def profiling_sample_csv(tmp_path: Path) -> Path:
    """Create a sample CSV with specific known distributions and quality flaws."""
    csv_file = tmp_path / "profiling_data.csv"
    csv_content = """transaction_id,timestamp,segment,amount,discount,constant_col,sparse_col
TX-001,2024-01-01,Enterprise,100.0,0.1,10.0,
TX-002,2024-01-02,SMB,200.0,0.0,10.0,
TX-003,2024-01-03,SMB,300.0,-5.0,10.0,
TX-004,2024-01-04,Enterprise,400.0,0.2,10.0,
TX-005,2024-01-05,Consumer,500.0,0.0,10.0,Val1
TX-005,2024-01-05,Consumer,500.0,0.0,10.0,Val1
"""
    csv_file.write_text(csv_content, encoding="utf-8")
    return csv_file


def test_numeric_profiling(profiling_sample_csv: Path):
    """Test numeric statistical summaries."""
    loader = CSVLoader()
    df, metadata = loader.load_csv(profiling_sample_csv)
    profiler = DatasetProfiler()
    report = profiler.generate_report(df, metadata)

    assert "amount" in report.numeric_summaries
    amt_sum = report.numeric_summaries["amount"]
    assert amt_sum.count == 6
    assert amt_sum.min == 100.0
    assert amt_sum.max == 500.0
    assert amt_sum.mean == pytest.approx(333.3333, abs=1e-2)
    assert amt_sum.median == 350.0
    assert amt_sum.iqr == pytest.approx(250.0, abs=1e-1)

    # Check discount negative / zero count
    disc_sum = report.numeric_summaries["discount"]
    assert disc_sum.negative_count == 1
    assert disc_sum.zero_count == 3


def test_categorical_profiling(profiling_sample_csv: Path):
    """Test categorical distribution and frequency calculations."""
    loader = CSVLoader()
    df, metadata = loader.load_csv(profiling_sample_csv)
    profiler = DatasetProfiler()
    report = profiler.generate_report(df, metadata)

    assert "segment" in report.categorical_summaries
    seg_sum = report.categorical_summaries["segment"]
    assert seg_sum.cardinality == 3
    # Enterprise: 2, SMB: 2, Consumer: 2 (total 6)
    assert len(seg_sum.top_frequencies) == 3
    for freq in seg_sum.top_frequencies:
        assert freq.count == 2
        assert freq.percentage == pytest.approx(33.33, abs=0.1)


def test_datetime_profiling(profiling_sample_csv: Path):
    """Test datetime range and span."""
    loader = CSVLoader()
    df, metadata = loader.load_csv(profiling_sample_csv)
    profiler = DatasetProfiler()
    report = profiler.generate_report(df, metadata)

    assert "timestamp" in report.datetime_summaries
    dt_sum = report.datetime_summaries["timestamp"]
    assert "2024-01-01" in dt_sum.min_date
    assert "2024-01-05" in dt_sum.max_date
    assert dt_sum.timespan_days == 4.0


def test_quality_warnings_and_completeness(profiling_sample_csv: Path):
    """Test quality warnings for duplicates, zero-variance columns, and missing values."""
    loader = CSVLoader()
    df, metadata = loader.load_csv(profiling_sample_csv)
    profiler = DatasetProfiler()
    report = profiler.generate_report(df, metadata)

    # Duplicate warning
    dup_warnings = [w for w in report.warnings if w.rule == "duplicate_rows"]
    assert len(dup_warnings) == 1
    assert "1 duplicate rows" in dup_warnings[0].message

    # Zero-variance warning on constant_col
    var_warnings = [w for w in report.warnings if w.rule == "zero_variance"]
    assert len(var_warnings) == 1
    assert var_warnings[0].column_name == "constant_col"

    # High missing values on sparse_col (4 out of 6 missing = 66.7%)
    missing_warnings = [w for w in report.warnings if w.rule == "high_missing_values"]
    assert len(missing_warnings) == 1
    assert missing_warnings[0].column_name == "sparse_col"
    assert missing_warnings[0].severity == WarningSeverity.CRITICAL

    # Completeness score < 100%
    assert report.completeness_score < 100.0
    assert report.completeness_score > 80.0
