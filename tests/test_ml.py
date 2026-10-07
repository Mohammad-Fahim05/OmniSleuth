"""Unit tests for Phase 4 - ML Anomaly Detection and Benchmark Evaluation."""

import numpy as np
import pandas as pd
import pytest
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
