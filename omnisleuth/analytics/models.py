"""Pydantic data models for deterministic analytics results."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AggregationFunction(str, Enum):
    """Supported deterministic aggregation functions."""

    SUM = "sum"
    MEAN = "mean"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"


class FilterCondition(BaseModel):
    """Deterministic filter condition applied to a dataframe."""

    column: str
    operator: str = Field(description="Operator: '==', '!=', '>', '>=', '<', '<=', 'in', 'between'")
    value: Any = Field(description="Filter target value, list of values, or tuple for between")


class AggregationResult(BaseModel):
    """Result of group-by aggregation."""

    metric_column: str
    agg_func: AggregationFunction
    group_by_columns: List[str]
    records: List[Dict[str, Any]]
    total_records: int
    grand_total: Optional[float] = None


class TrendPoint(BaseModel):
    """A single time period in a trend."""

    period: str
    value: float
    change_pct_from_prev: Optional[float] = None
    delta_from_prev: Optional[float] = None


class TrendResult(BaseModel):
    """Time-series trend calculation result."""

    date_column: str
    metric_column: str
    frequency: str = Field(description="'D' (daily), 'W' (weekly), 'M' (monthly), 'Q' (quarterly)")
    time_series: List[TrendPoint]
    start_period: str
    end_period: str
    start_value: float
    end_value: float
    overall_delta: float
    overall_change_pct: float


class SegmentRecord(BaseModel):
    """Metrics for a single segment category."""

    segment: str
    value: float
    share_percentage: float
    rank: int


class SegmentComparisonResult(BaseModel):
    """Comparison across segments of a categorical dimension."""

    segment_column: str
    metric_column: str
    agg_func: AggregationFunction
    total_metric_value: float
    segments: List[SegmentRecord]


class SegmentContributionItem(BaseModel):
    """Contribution of an individual segment to total change across two periods."""

    segment: str
    period_1_value: float
    period_2_value: float
    delta: float
    segment_change_pct: float
    contribution_to_total_delta_pct: float


class SegmentContributionResult(BaseModel):
    """Exact mathematical decomposition of total change across segments."""

    segment_column: str
    metric_column: str
    period_1_label: str
    period_2_label: str
    total_period_1: float
    total_period_2: float
    total_delta: float
    total_change_pct: float
    contributions: List[SegmentContributionItem]


class FactorEffect(BaseModel):
    """Contribution of a multiplicative or additive factor in metric decomposition."""

    factor_name: str
    period_1_value: float
    period_2_value: float
    delta: float
    change_pct: float
    estimated_impact_on_target: float
    impact_percentage: float


class MetricDecompositionResult(BaseModel):
    """Decomposition of a metric into constituent drivers (e.g. Revenue = Orders * AOV)."""

    target_metric: str
    decomposition_type: str = Field(description="'multiplicative' or 'additive'")
    period_1_label: str
    period_2_label: str
    target_period_1: float
    target_period_2: float
    target_delta: float
    target_change_pct: float
    factor_effects: List[FactorEffect]


class ChartType(str, Enum):
    """Supported chart types."""

    LINE = "line"
    BAR = "bar"
    WATERFALL = "waterfall"
    PIE = "pie"


class ChartDataSpec(BaseModel):
    """Declarative specification for chart visualization."""

    chart_type: ChartType
    title: str
    x_column: str
    y_column: str
    color_column: Optional[str] = None
    data: List[Dict[str, Any]]
    layout_options: Dict[str, Any] = Field(default_factory=dict)
