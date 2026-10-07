"""Data models for ML anomaly detection results and performance evaluation."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AnomalyMethod(str, Enum):
    """Supported anomaly detection algorithms."""

    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    ZSCORE = "zscore"


class AnomalyRecord(BaseModel):
    """An individual record flagged as anomalous."""

    row_index: int
    data: Dict[str, Any]
    anomaly_score: float = Field(description="Normalized anomaly score (higher means more anomalous)")
    flagged_columns: List[str] = Field(default_factory=list, description="Columns contributing to the anomaly")
    explanation: str = Field(description="Statistical description of why this row was flagged")


class AnomalyDetectionResult(BaseModel):
    """Result of anomaly detection execution."""

    method: AnomalyMethod
    feature_columns: List[str]
    total_records: int
    anomaly_count: int
    anomaly_percentage: float
    execution_time_ms: float
    anomalies: List[AnomalyRecord] = Field(default_factory=list)
    non_causal_disclaimer: str = (
        "Statistical anomalies represent extreme values or deviations from expected distributions; "
        "they do not imply direct causality."
    )


class EvaluationMetrics(BaseModel):
    """Standard evaluation metrics for benchmarking anomaly detection algorithms."""

    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    false_positive_rate: float = Field(ge=0.0, le=1.0)
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    runtime_ms: float
