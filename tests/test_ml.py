"""Unit tests for Phase 4 - ML Anomaly Detection and Benchmark Evaluation."""

import numpy as np
import pandas as pd
import pytest
from omnisleuth.core.exceptions import ToolExecutionError
from omnisleuth.ml import (
    AnomalyMethod,
    IQRAnomalyDetector,
    IsolationForestDetector,
    evaluate_anomaly_detection,
    inject_synthetic_anomalies,
)


@pytest.fixture
def clean_distribution_df() -> pd.DataFrame:
    """Create a dataframe with normal distributions and planted outliers."""
    np.random.seed(42)
    n = 100
    rev = np.random.normal(loc=1000.0, scale=100.0, size=n)
    units = np.random.normal(loc=20.0, scale=3.0, size=n)

    # Plant 2 explicit outliers at index 0 and 1
    rev[0] = 5000.0  # Massive spike
    rev[1] = 50.0    # Massive drop

    return pd.DataFrame({
        "order_id": [f"ORD-{i}" for i in range(n)],
        "revenue": rev,
        "units": units,
    })


def test_iqr_detector(clean_distribution_df: pd.DataFrame):
    """Test IQR detector captures planted outliers."""
    detector = IQRAnomalyDetector(multiplier=1.5)
    result = detector.detect(clean_distribution_df, feature_columns=["revenue"])

    assert result.method == AnomalyMethod.IQR
    assert result.total_records == 100
    assert result.anomaly_count >= 2

    # Verify planted outliers are detected
    anom_indices = [a.row_index for a in result.anomalies]
    assert 0 in anom_indices
    assert 1 in anom_indices
    assert "revenue" in result.anomalies[0].flagged_columns
    assert "Statistical anomalies represent extreme values" in result.non_causal_disclaimer


def test_isolation_forest_detector(clean_distribution_df: pd.DataFrame):
    """Test Isolation Forest detector flags multivariate anomalies."""
    detector = IsolationForestDetector(contamination=0.04, random_state=42)
    result = detector.detect(clean_distribution_df, feature_columns=["revenue", "units"])

    assert result.method == AnomalyMethod.ISOLATION_FOREST
    assert result.anomaly_count > 0
    assert result.execution_time_ms > 0

    # Planted extreme spike should have top anomaly score
    top_anomaly = result.anomalies[0]
    assert top_anomaly.anomaly_score > 0.5


def test_synthetic_injection_and_evaluation():
    """Test synthetic anomaly injection and metrics computation."""
    np.random.seed(42)
    base_df = pd.DataFrame({
        "sales": np.random.normal(500, 50, 200),
        "traffic": np.random.normal(1000, 100, 200),
    })

    injected_df, ground_truth = inject_synthetic_anomalies(
        df=base_df,
        target_columns=["sales", "traffic"],
        anomaly_rate=0.05,
        magnitude_factor=5.0,
    )

    assert ground_truth.sum() == 10  # 5% of 200 = 10 anomalies

    # Run detector
    detector = IsolationForestDetector(contamination=0.06, random_state=42)
    res = detector.detect(injected_df, feature_columns=["sales", "traffic"])

    y_pred = np.zeros(len(injected_df), dtype=int)
    for a in res.anomalies:
        y_pred[a.row_index] = 1

    eval_metrics = evaluate_anomaly_detection(ground_truth, y_pred, runtime_ms=res.execution_time_ms)

    # With extreme 5-sigma outliers, recall and precision should be high
    assert eval_metrics.recall >= 0.7
    assert eval_metrics.precision >= 0.5
    assert eval_metrics.f1 >= 0.6
    assert eval_metrics.false_positive_rate < 0.1


# ─────────────────────────────────────────────────────────────────────────────
# T-006 Regression Tests: Anomaly Injection Casting and Column Types
# ─────────────────────────────────────────────────────────────────────────────

def test_synthetic_injection_integer_columns():
    """Verify that integer columns (int64, int32, uint32) accept anomalies without casting errors."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "order_id": [f"ORD-{i}" for i in range(n)],
        "qty_int64": pd.Series(np.random.randint(10, 50, size=n), dtype=np.int64),
        "qty_int32": pd.Series(np.random.randint(5, 25, size=n), dtype=np.int32),
        "qty_uint32": pd.Series(np.random.randint(1, 15, size=n), dtype=np.uint32),
    })

    injected, ground_truth = inject_synthetic_anomalies(
        df=df,
        target_columns=["qty_int64", "qty_int32", "qty_uint32"],
        anomaly_rate=0.05,
        magnitude_factor=4.0,
        random_state=42,
    )

    assert ground_truth.sum() == 5
    assert injected["qty_int64"].dtype == np.float64
    assert injected["qty_int32"].dtype == np.float64
    assert injected["qty_uint32"].dtype == np.float64
    # Non-target column retains original type and values
    assert injected["order_id"].dtype == df["order_id"].dtype
    assert (injected["order_id"] == df["order_id"]).all()


def test_synthetic_injection_nullable_integer_with_missing_values():
    """Verify that nullable integer (Int64) containing missing values (NA) works gracefully."""
    np.random.seed(42)
    n = 100
    vals = [float(i) if i % 10 != 0 else None for i in range(n)]
    df = pd.DataFrame({
        "units": pd.Series(vals, dtype="Int64"),
        "revenue": np.random.normal(500.0, 50.0, size=n),
    })

    injected, ground_truth = inject_synthetic_anomalies(
        df=df,
        target_columns=["units", "revenue"],
        anomaly_rate=0.05,
        magnitude_factor=4.0,
        random_state=42,
    )

    assert ground_truth.sum() == 5
    assert injected["units"].dtype == np.float64
    assert injected["revenue"].dtype == np.float64


def test_synthetic_injection_mixed_numeric_and_nonnumeric():
    """Verify that non-numeric columns (strings, dates, categories) are strictly preserved."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "sku": [f"SKU-{i}" for i in range(n)],
        "category": pd.Categorical(["Electronics" if i % 2 == 0 else "Apparel" for i in range(n)]),
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "quantity": np.random.randint(10, 50, size=n),
        "revenue": np.random.normal(100.0, 10.0, size=n),
    })

    injected, ground_truth = inject_synthetic_anomalies(
        df=df,
        target_columns=["quantity", "revenue"],
        anomaly_rate=0.06,
        magnitude_factor=4.0,
        random_state=42,
    )

    assert ground_truth.sum() == 3
    assert injected["sku"].dtype == df["sku"].dtype
    assert injected["category"].dtype == df["category"].dtype
    assert injected["timestamp"].dtype == df["timestamp"].dtype
    assert (injected["sku"] == df["sku"]).all()
    assert (injected["category"] == df["category"]).all()
    assert (injected["timestamp"] == df["timestamp"]).all()


def test_synthetic_injection_immutability():
    """Verify that the input DataFrame is never mutated in-place."""
    np.random.seed(42)
    df_orig = pd.DataFrame({
        "id": ["A", "B", "C", "D", "E"],
        "count": [10, 20, 30, 40, 50],
    })
    df_copy_before = df_orig.copy(deep=True)

    injected, ground_truth = inject_synthetic_anomalies(
        df=df_orig,
        target_columns=["count"],
        anomaly_rate=0.2,
        magnitude_factor=5.0,
        random_state=42,
    )

    # df_orig must remain identical to before
    pd.testing.assert_frame_equal(df_orig, df_copy_before)
    assert df_orig["count"].dtype == np.int64


def test_synthetic_injection_reproducibility():
    """Verify that random_state ensures deterministic reproducibility."""
    df = pd.DataFrame({
        "qty": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "rev": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0],
    })

    injected_1, gt_1 = inject_synthetic_anomalies(df, ["qty", "rev"], random_state=123)
    injected_2, gt_2 = inject_synthetic_anomalies(df, ["qty", "rev"], random_state=123)
    injected_3, gt_3 = inject_synthetic_anomalies(df, ["qty", "rev"], random_state=999)

    pd.testing.assert_frame_equal(injected_1, injected_2)
    assert np.array_equal(gt_1, gt_2)

    # Different seed should generate different anomaly positions
    assert not np.array_equal(gt_1, gt_3)


def test_synthetic_evaluation_with_integer_target_column():
    """Verify full detector evaluation pipeline succeeds on integer target columns."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "units": np.random.randint(10, 100, size=n),
        "price": np.random.normal(50.0, 5.0, size=n),
    })

    injected_df, ground_truth = inject_synthetic_anomalies(
        df=df,
        target_columns=["units", "price"],
        anomaly_rate=0.05,
        magnitude_factor=5.0,
        random_state=42,
    )

    detector = IsolationForestDetector(contamination=0.06, random_state=42)
    res = detector.detect(injected_df, feature_columns=["units", "price"])

    y_pred = np.zeros(len(injected_df), dtype=int)
    for a in res.anomalies:
        y_pred[a.row_index] = 1

    eval_metrics = evaluate_anomaly_detection(ground_truth, y_pred, runtime_ms=res.execution_time_ms)
    assert eval_metrics.recall >= 0.7
    assert eval_metrics.precision >= 0.5
    assert eval_metrics.f1 >= 0.6


def test_synthetic_evaluation_benchmark_sales_dataset():
    """Verify evaluation on benchmark sales dataset (containing revenue:float64 and units_sold:int64)."""
    import tempfile
    from pathlib import Path
    from omnisleuth.evaluation.dataset_generator import generate_benchmark_sales_dataset
    from omnisleuth.evaluation.evaluator import OmniSleuthEvaluator
    from omnisleuth.ingestion.csv_loader import CSVLoader

    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "sales.csv"
        generate_benchmark_sales_dataset(n_rows=200, random_seed=42, output_path=csv_path)
        df, meta = CSVLoader().load_csv(csv_path)

        assert df["units_sold"].dtype == np.int64
        assert df["revenue"].dtype == np.float64

        evaluator = OmniSleuthEvaluator()
        result = evaluator.evaluate_anomaly_detection(df, target_columns=["revenue", "units_sold"])

        assert result["total_injected"] > 0
        assert result["total_detected"] > 0
        assert result["precision"] > 0.5
        assert result["recall"] > 0.5
        assert result["f1"] > 0.5
        assert result["false_positive_rate"] < 0.1


def test_synthetic_injection_edge_cases():
    """Verify empty DataFrame and missing target columns handling."""
    empty_df = pd.DataFrame({"col": []})
    out_df, gt = inject_synthetic_anomalies(empty_df, ["col"])
    assert len(out_df) == 0
    assert len(gt) == 0

    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ToolExecutionError) as exc_info:
        inject_synthetic_anomalies(df, ["non_existent"])
    assert "Target columns not found" in str(exc_info.value)
