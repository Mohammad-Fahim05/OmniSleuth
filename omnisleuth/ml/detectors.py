"""ML and statistical anomaly detectors for structured datasets."""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from omnisleuth.core.exceptions import ToolExecutionError
from omnisleuth.core.logging import get_logger
from omnisleuth.ml.models import (
    AnomalyDetectionResult,
    AnomalyMethod,
    AnomalyRecord,
    EvaluationMetrics,
)

logger = get_logger("ml")


class IQRAnomalyDetector:
    """Univariate and multi-column IQR outlier detection."""

    def __init__(self, multiplier: float = 1.5):
        self.multiplier = multiplier

    def detect(self, df: pd.DataFrame, feature_columns: List[str]) -> AnomalyDetectionResult:
        """Flag rows exceeding Q3 + k*IQR or below Q1 - k*IQR for given numeric columns."""
        start_time = time.perf_counter()

        missing_cols = [c for c in feature_columns if c not in df.columns]
        if missing_cols:
            raise ToolExecutionError(f"Columns not found for IQR detection: {missing_cols}")

        num_df = df[feature_columns].apply(pd.to_numeric, errors="coerce")
        bounds = {}
        for col in feature_columns:
            s = num_df[col].dropna()
            if len(s) == 0:
                continue
            q25 = np.percentile(s, 25)
            q75 = np.percentile(s, 75)
            iqr = q75 - q25
            lower = q25 - (self.multiplier * iqr)
            upper = q75 + (self.multiplier * iqr)
            bounds[col] = (lower, upper, iqr, q25, q75)

        anomalies: List[AnomalyRecord] = []
        for idx, row in df.iterrows():
            flagged: List[str] = []
            max_score = 0.0
            reasons = []

            for col in feature_columns:
                if col not in bounds:
                    continue
                val = num_df.loc[idx, col]
                if pd.isna(val):
                    continue
                lower, upper, iqr, q25, q75 = bounds[col]
                if val < lower or val > upper:
                    flagged.append(col)
                    dev = (val - upper) if val > upper else (lower - val)
                    score = float(dev / iqr) if iqr > 0 else 1.0
                    max_score = max(max_score, score)
                    reasons.append(f"{col}={val} outside [{lower:.2f}, {upper:.2f}]")

            if flagged:
                norm_score = min(1.0, round(max_score / 3.0, 4))
                anomalies.append(
                    AnomalyRecord(
                        row_index=int(idx),
                        data=row.to_dict(),
                        anomaly_score=norm_score,
                        flagged_columns=flagged,
                        explanation="; ".join(reasons),
                    )
                )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
        total = len(df)
        pct = round((len(anomalies) / total * 100.0), 2) if total > 0 else 0.0

        return AnomalyDetectionResult(
            method=AnomalyMethod.IQR,
            feature_columns=feature_columns,
            total_records=total,
            anomaly_count=len(anomalies),
            anomaly_percentage=pct,
            execution_time_ms=elapsed_ms,
            anomalies=anomalies,
        )


class IsolationForestDetector:
    """Multivariate anomaly detection using scikit-learn's Isolation Forest."""

    def __init__(self, contamination: float = 0.05, random_state: int = 42, n_estimators: int = 100):
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators

    def detect(self, df: pd.DataFrame, feature_columns: List[str]) -> AnomalyDetectionResult:
        """Fit Isolation Forest on selected numeric features and detect anomalous instances."""
        start_time = time.perf_counter()

        missing_cols = [c for c in feature_columns if c not in df.columns]
        if missing_cols:
            raise ToolExecutionError(f"Columns not found for Isolation Forest: {missing_cols}")

        num_df = df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if len(num_df) < 5:
            raise ToolExecutionError(f"Insufficient data rows ({len(num_df)}) to fit Isolation Forest.")

        iso = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
        )
        preds = iso.fit_predict(num_df)
        raw_scores = iso.decision_function(num_df)  # lower score = more anomalous

        # Normalize score between 0 (normal) and 1 (highly anomalous)
        min_score, max_score = raw_scores.min(), raw_scores.max()
        score_span = max_score - min_score if max_score != min_score else 1.0

        anomalies: List[AnomalyRecord] = []
        for i, (pred, raw_sc) in enumerate(zip(preds, raw_scores)):
            if pred == -1:  # Anomalous
                orig_idx = df.index[i]
                norm_score = round(float(1.0 - ((raw_sc - min_score) / score_span)), 4)
                row_data = df.iloc[i].to_dict()

                anomalies.append(
                    AnomalyRecord(
                        row_index=int(orig_idx),
                        data=row_data,
                        anomaly_score=norm_score,
                        flagged_columns=feature_columns,
                        explanation=f"Multivariate outlier detected by Isolation Forest (score: {norm_score:.3f})",
                    )
                )

        # Sort anomalies by highest anomaly score
        anomalies = sorted(anomalies, key=lambda x: x.anomaly_score, reverse=True)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
        total = len(df)
        pct = round((len(anomalies) / total * 100.0), 2) if total > 0 else 0.0

        return AnomalyDetectionResult(
            method=AnomalyMethod.ISOLATION_FOREST,
            feature_columns=feature_columns,
            total_records=total,
            anomaly_count=len(anomalies),
            anomaly_percentage=pct,
            execution_time_ms=elapsed_ms,
            anomalies=anomalies,
        )


def evaluate_anomaly_detection(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    runtime_ms: float,
) -> EvaluationMetrics:
    """Compute precision, recall, F1, and FPR for anomaly evaluation."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = round(float(tp / (tp + fp)), 4) if (tp + fp) > 0 else 0.0
    recall = round(float(tp / (tp + fn)), 4) if (tp + fn) > 0 else 0.0
    f1 = round(float(2 * precision * recall / (precision + recall)), 4) if (precision + recall) > 0 else 0.0
    fpr = round(float(fp / (fp + tn)), 4) if (fp + tn) > 0 else 0.0

    return EvaluationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fpr,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        runtime_ms=runtime_ms,
    )


def inject_synthetic_anomalies(
    df: pd.DataFrame,
    target_columns: List[str],
    anomaly_rate: float = 0.05,
    magnitude_factor: float = 4.0,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Inject synthetic anomalies into numerical columns for benchmarking.

    Returns:
        Tuple of (df_with_anomalies, ground_truth_binary_array)
    """
    np.random.seed(random_state)
    df_copy = df.copy()
    n_rows = len(df_copy)
    n_anomalies = max(1, int(n_rows * anomaly_rate))

    ground_truth = np.zeros(n_rows, dtype=int)
    anomaly_indices = np.random.choice(n_rows, size=n_anomalies, replace=False)
    ground_truth[anomaly_indices] = 1

    for idx in anomaly_indices:
        for col in target_columns:
            std = float(df_copy[col].std()) if df_copy[col].std() > 0 else 1.0
            mean = float(df_copy[col].mean())
            # Inject extreme spike or drop
            multiplier = np.random.choice([-1, 1]) * magnitude_factor
            df_copy.loc[df_copy.index[idx], col] = round(mean + (multiplier * std), 2)

    return df_copy, ground_truth
