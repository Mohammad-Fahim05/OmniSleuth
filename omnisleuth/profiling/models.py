"""Data models for dataset profiling and data quality assessment."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class WarningSeverity(str, Enum):
    """Severity classification for data quality warnings."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class QualityWarning(BaseModel):
    """Specific data quality issue or risk detected in the dataset."""

    column_name: Optional[str] = Field(default=None, description="Affected column if applicable")
    severity: WarningSeverity = Field(description="Severity level")
    rule: str = Field(description="Name of the quality rule that triggered")
    message: str = Field(description="Human-readable explanation of the issue")


class NumericSummary(BaseModel):
    """Statistical distribution summary for a numeric column."""

    column_name: str
    count: int
    missing_count: int
    mean: float
    std: float
    median: float
    min: float
    max: float
    q25: float
    q75: float
    iqr: float
    zero_count: int
    negative_count: int


class CategoryFrequency(BaseModel):
    """Frequency metrics for a specific category value."""

    value: str
    count: int
    percentage: float


class CategoricalSummary(BaseModel):
    """Distribution summary for a categorical column."""

    column_name: str
    cardinality: int
    missing_count: int
    top_frequencies: List[CategoryFrequency] = Field(default_factory=list)


class DatetimeSummary(BaseModel):
    """Temporal range summary for a datetime column."""

    column_name: str
    min_date: str
    max_date: str
    timespan_days: Optional[float] = None
    missing_count: int


class QualityReport(BaseModel):
    """Comprehensive, reproducible data quality and profiling report."""

    dataset_id: str
    version_hash: str
    row_count: int
    column_count: int
    duplicate_rows: int
    duplicate_percentage: float
    completeness_score: float = Field(ge=0.0, le=100.0, description="Overall non-null data completeness percentage")
    warnings: List[QualityWarning] = Field(default_factory=list)
    numeric_summaries: Dict[str, NumericSummary] = Field(default_factory=dict)
    categorical_summaries: Dict[str, CategoricalSummary] = Field(default_factory=dict)
    datetime_summaries: Dict[str, DatetimeSummary] = Field(default_factory=dict)
