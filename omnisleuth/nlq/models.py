"""Pydantic data models for structured NLQ analysis plans and constraints."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PlanIntent(str, Enum):
    """Supported analytical intents for natural language questions."""

    DATASET_OVERVIEW = "dataset_overview"
    MISSING_VALUES = "missing_values"
    QUERY = "query"
    TREND = "trend"
    PERIOD_COMPARISON = "period_comparison"
    CORRELATION = "correlation"
    ANOMALY = "anomaly"
    CONTRIBUTION = "contribution"
    CHART_REQUEST = "chart_request"
    DEEP_INVESTIGATION = "deep_investigation"
    DOCUMENTATION_QUESTION = "documentation_question"
    GENERAL_KNOWLEDGE = "general_knowledge"
    FOLLOWUP = "followup"
    UNSUPPORTED = "unsupported"


class MetricAgg(str, Enum):
    """Strict allowlisted aggregation functions."""

    SUM = "sum"
    MEAN = "mean"
    COUNT = "count"
    NUNIQUE = "nunique"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"


class FilterOp(str, Enum):
    """Strict allowlisted filter operators."""

    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    BETWEEN = "between"
    CONTAINS = "contains"
    IS_NULL = "is_null"
    NOT_NULL = "not_null"


class SortDirection(str, Enum):
    """Sort direction."""

    ASC = "asc"
    DESC = "desc"


class TimeGrain(str, Enum):
    """Time-series aggregation granularity."""

    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class ChartType(str, Enum):
    """Allowlisted chart visualizations."""

    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    PIE = "pie"


class MetricSpec(BaseModel, extra="forbid"):
    """Specification of an aggregate metric to calculate."""

    column: str
    agg: MetricAgg = MetricAgg.SUM


class FilterSpec(BaseModel, extra="forbid"):
    """Specification of a boolean filter to apply."""

    column: str
    op: FilterOp
    value: Any = None


class SortSpec(BaseModel, extra="forbid"):
    """Specification of ordering."""

    by: str
    direction: SortDirection = SortDirection.DESC


class TimeSpec(BaseModel, extra="forbid"):
    """Specification of temporal grain and slicing."""

    date_column: str
    grain: TimeGrain = TimeGrain.MONTH
    start: Optional[str] = None
    end: Optional[str] = None


class ChartSpec(BaseModel, extra="forbid"):
    """Specification of visualization output."""

    type: ChartType
    x: str
    y: Optional[str] = None
    color: Optional[str] = None


class PeriodSpec(BaseModel, extra="forbid"):
    """Temporal period boundary specification."""

    start: str
    end: str
    label: Optional[str] = None


class AnalysisPlan(BaseModel, extra="forbid"):
    """Validated, allowlisted execution plan for analytical queries."""

    intent: PlanIntent
    metrics: List[MetricSpec] = Field(default_factory=list, max_length=5)
    group_by: List[str] = Field(default_factory=list, max_length=3)
    filters: List[FilterSpec] = Field(default_factory=list, max_length=8)
    sort: Optional[SortSpec] = None
    limit: Optional[int] = Field(default=None, ge=1, le=50)
    time: Optional[TimeSpec] = None
    chart: Optional[ChartSpec] = None
    period_a: Optional[PeriodSpec] = None
    period_b: Optional[PeriodSpec] = None
    target_column: Optional[str] = None
    followup_of: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    assumptions: List[str] = Field(default_factory=list)
    needs_clarification: Optional[str] = None
    unsupported_reason: Optional[str] = None
    planner: str = Field(default="deterministic", pattern="^(deterministic|llm)$")


class QueryResult(BaseModel):
    """Encapsulated execution outcome of an analytical plan."""

    status: str  # "answered", "needs_clarification", "unsupported", "failed"
    answer: str
    result_table: Optional[List[Dict[str, Any]]] = None
    chart_spec: Optional[Dict[str, Any]] = None
    evidence_store: Optional[Any] = None
    claims: List[Any] = Field(default_factory=list)
    plan: AnalysisPlan
    plain_language_summary: str
    planner_used: str = "deterministic"
    limitations: List[str] = Field(default_factory=list)
    investigation_id: Optional[str] = None
