"""Allowlisted tool implementations for OmniSleuth."""

from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, Field
from omnisleuth.analytics.engine import DeterministicAnalyticsEngine
from omnisleuth.analytics.models import (
    AggregationFunction,
    AggregationResult,
    ChartDataSpec,
    ChartType,
    FilterCondition,
    MetricDecompositionResult,
    SegmentComparisonResult,
    SegmentContributionResult,
    TrendResult,
)
from omnisleuth.ingestion.models import DatasetMetadata
from omnisleuth.ml.detectors import IQRAnomalyDetector, IsolationForestDetector
from omnisleuth.ml.models import AnomalyDetectionResult, AnomalyMethod
from omnisleuth.profiling.models import QualityReport
from omnisleuth.profiling.profiler import DatasetProfiler
from omnisleuth.tools.base import BaseTool, BaseToolInput, BaseToolOutput, ToolRegistry

# Singleton instances of deterministic engines
_analytics_engine = DeterministicAnalyticsEngine()
_profiler = DatasetProfiler()


# ==========================================
# 1. Profile Dataset Tool
# ==========================================
class ProfileDatasetInput(BaseToolInput):
    """Input parameters for dataset profiling."""


class ProfileDatasetOutput(BaseToolOutput):
    data: Optional[QualityReport] = None


class ProfileDatasetTool(BaseTool):
    name = "profile_dataset"
    description = "Generate a comprehensive statistical profile and data quality assessment report for the dataset."
    input_model = ProfileDatasetInput
    output_model = ProfileDatasetOutput

    def _execute(self, params: ProfileDatasetInput, df: pd.DataFrame, metadata: DatasetMetadata) -> QualityReport:
        return _profiler.generate_report(df, metadata)


# ==========================================
# 2. Aggregate Metric Tool
# ==========================================
class AggregateMetricInput(BaseToolInput):
    metric_column: str = Field(description="Numerical column to aggregate")
    agg_func: AggregationFunction = Field(default=AggregationFunction.SUM, description="Aggregation function (sum, mean, count, min, max, median)")
    group_by_columns: Optional[List[str]] = Field(default=None, description="Optional categorical columns to group by")
    filters: Optional[List[FilterCondition]] = Field(default=None, description="Optional filters to apply")


class AggregateMetricOutput(BaseToolOutput):
    data: Optional[AggregationResult] = None


class AggregateMetricTool(BaseTool):
    name = "aggregate_metric"
    description = "Perform deterministic group-by or total aggregation (sum, mean, count, min, max, median) on a numeric metric."
    input_model = AggregateMetricInput
    output_model = AggregateMetricOutput

    def _execute(self, params: AggregateMetricInput, df: pd.DataFrame, metadata: DatasetMetadata) -> AggregationResult:
        return _analytics_engine.aggregate(
            df=df,
            metric_column=params.metric_column,
            agg_func=params.agg_func,
            group_by_columns=params.group_by_columns,
            filters=params.filters,
        )


# ==========================================
# 3. Calculate Trend Tool
# ==========================================
class CalculateTrendInput(BaseToolInput):
    date_column: str = Field(description="Date/time column for time series")
    metric_column: str = Field(description="Numeric column to aggregate across periods")
    frequency: str = Field(default="M", description="Resampling frequency: 'D' (Daily), 'W' (Weekly), 'M' (Monthly), 'Q' (Quarterly)")
    agg_func: AggregationFunction = Field(default=AggregationFunction.SUM, description="Aggregation function")
    filters: Optional[List[FilterCondition]] = Field(default=None, description="Optional filters")


class CalculateTrendOutput(BaseToolOutput):
    data: Optional[TrendResult] = None


class CalculateTrendTool(BaseTool):
    name = "calculate_trend"
    description = "Calculate period-over-period time series trends, absolute deltas, and percentage changes across time."
    input_model = CalculateTrendInput
    output_model = CalculateTrendOutput

    def _execute(self, params: CalculateTrendInput, df: pd.DataFrame, metadata: DatasetMetadata) -> TrendResult:
        return _analytics_engine.calculate_trend(
            df=df,
            date_column=params.date_column,
            metric_column=params.metric_column,
            frequency=params.frequency,
            agg_func=params.agg_func,
            filters=params.filters,
        )


# ==========================================
# 4. Compare Segments Tool
# ==========================================
class CompareSegmentsInput(BaseToolInput):
    segment_column: str = Field(description="Categorical column to compare segments of (e.g. region, category)")
    metric_column: str = Field(description="Numeric metric to aggregate per segment")
    agg_func: AggregationFunction = Field(default=AggregationFunction.SUM, description="Aggregation function")
    filters: Optional[List[FilterCondition]] = Field(default=None, description="Optional filters")


class CompareSegmentsOutput(BaseToolOutput):
    data: Optional[SegmentComparisonResult] = None


class CompareSegmentsTool(BaseTool):
    name = "compare_segments"
    description = "Compare totals, percentage shares, and rankings across segments of a categorical dimension."
    input_model = CompareSegmentsInput
    output_model = CompareSegmentsOutput

    def _execute(self, params: CompareSegmentsInput, df: pd.DataFrame, metadata: DatasetMetadata) -> SegmentComparisonResult:
        return _analytics_engine.compare_segments(
            df=df,
            segment_column=params.segment_column,
            metric_column=params.metric_column,
            agg_func=params.agg_func,
            filters=params.filters,
        )


# ==========================================
# 5. Segment Contribution Tool
# ==========================================
class SegmentContributionInput(BaseToolInput):
    segment_column: str = Field(description="Categorical column whose segments contributed to total change")
    metric_column: str = Field(description="Target metric (e.g. revenue)")
    period_1_filters: List[FilterCondition] = Field(description="Filters defining baseline period 1")
    period_2_filters: List[FilterCondition] = Field(description="Filters defining comparison period 2")
    period_1_label: str = Field(default="Period 1", description="Display label for period 1")
    period_2_label: str = Field(default="Period 2", description="Display label for period 2")
    agg_func: AggregationFunction = Field(default=AggregationFunction.SUM, description="Aggregation function")


class SegmentContributionOutput(BaseToolOutput):
    data: Optional[SegmentContributionResult] = None


class SegmentContributionTool(BaseTool):
    name = "segment_contribution"
    description = "Calculate exact mathematical contribution percentage of each segment to the overall metric delta between two periods."
    input_model = SegmentContributionInput
    output_model = SegmentContributionOutput

    def _execute(self, params: SegmentContributionInput, df: pd.DataFrame, metadata: DatasetMetadata) -> SegmentContributionResult:
        return _analytics_engine.calculate_segment_contribution(
            df=df,
            segment_column=params.segment_column,
            metric_column=params.metric_column,
            period_1_filters=params.period_1_filters,
            period_2_filters=params.period_2_filters,
            period_1_label=params.period_1_label,
            period_2_label=params.period_2_label,
            agg_func=params.agg_func,
        )


# ==========================================
# 6. Decompose Metric Tool
# ==========================================
class DecomposeMetricInput(BaseToolInput):
    revenue_column: str = Field(description="Revenue or value metric column")
    order_id_column: str = Field(description="Order ID or transaction identifier column")
    period_1_filters: List[FilterCondition] = Field(description="Filters defining baseline period 1")
    period_2_filters: List[FilterCondition] = Field(description="Filters defining comparison period 2")
    period_1_label: str = Field(default="Period 1", description="Display label for period 1")
    period_2_label: str = Field(default="Period 2", description="Display label for period 2")


class DecomposeMetricOutput(BaseToolOutput):
    data: Optional[MetricDecompositionResult] = None


class DecomposeMetricTool(BaseTool):
    name = "decompose_metric"
    description = "Decompose changes in Revenue into Order Volume (Orders) and Average Order Value (AOV) driver effects."
    input_model = DecomposeMetricInput
    output_model = DecomposeMetricOutput

    def _execute(self, params: DecomposeMetricInput, df: pd.DataFrame, metadata: DatasetMetadata) -> MetricDecompositionResult:
        return _analytics_engine.decompose_revenue_orders_aov(
            df=df,
            revenue_column=params.revenue_column,
            order_id_column=params.order_id_column,
            period_1_filters=params.period_1_filters,
            period_2_filters=params.period_2_filters,
            period_1_label=params.period_1_label,
            period_2_label=params.period_2_label,
        )


# ==========================================
# 7. Detect Anomalies Tool
# ==========================================
class DetectAnomaliesInput(BaseToolInput):
    feature_columns: List[str] = Field(description="Numeric columns to analyze for anomalies")
    method: AnomalyMethod = Field(default=AnomalyMethod.ISOLATION_FOREST, description="Algorithm: 'iqr' or 'isolation_forest'")
    contamination: float = Field(default=0.05, ge=0.01, le=0.5, description="Expected proportion of anomalies for Isolation Forest")
    iqr_multiplier: float = Field(default=1.5, ge=1.0, le=5.0, description="Multiplier for IQR threshold")


class DetectAnomaliesOutput(BaseToolOutput):
    data: Optional[AnomalyDetectionResult] = None


class DetectAnomaliesTool(BaseTool):
    name = "detect_anomalies"
    description = "Detect statistical anomalies using univariate IQR or multivariate Isolation Forest."
    input_model = DetectAnomaliesInput
    output_model = DetectAnomaliesOutput

    def _execute(self, params: DetectAnomaliesInput, df: pd.DataFrame, metadata: DatasetMetadata) -> AnomalyDetectionResult:
        if params.method == AnomalyMethod.IQR:
            detector = IQRAnomalyDetector(multiplier=params.iqr_multiplier)
            return detector.detect(df, params.feature_columns)
        else:
            detector = IsolationForestDetector(contamination=params.contamination)
            return detector.detect(df, params.feature_columns)


# ==========================================
# 8. Create Chart Tool
# ==========================================
class CreateChartInput(BaseToolInput):
    chart_type: ChartType = Field(description="Chart type: 'line', 'bar', 'waterfall', 'pie'")
    title: str = Field(description="Chart title")
    data: List[Dict[str, Any]] = Field(description="List of records to visualize")
    x_column: str = Field(description="Field name for X axis")
    y_column: str = Field(description="Field name for Y axis")
    color_column: Optional[str] = Field(default=None, description="Optional field name for color grouping")
    layout_options: Optional[Dict[str, Any]] = Field(default=None, description="Custom layout options")


class CreateChartOutput(BaseToolOutput):
    data: Optional[ChartDataSpec] = None


class CreateChartTool(BaseTool):
    name = "create_chart"
    description = "Generate declarative chart visualization specifications (line, bar, waterfall, pie)."
    input_model = CreateChartInput
    output_model = CreateChartOutput

    def _execute(self, params: CreateChartInput, df: pd.DataFrame, metadata: DatasetMetadata) -> ChartDataSpec:
        return _analytics_engine.generate_chart_spec(
            chart_type=params.chart_type,
            title=params.title,
            data=params.data,
            x_column=params.x_column,
            y_column=params.y_column,
            color_column=params.color_column,
            layout_options=params.layout_options,
        )


def get_default_tool_registry() -> ToolRegistry:
    """Create and return a ToolRegistry pre-populated with all 8 allowlisted tools."""
    registry = ToolRegistry()
    registry.register(ProfileDatasetTool())
    registry.register(AggregateMetricTool())
    registry.register(CalculateTrendTool())
    registry.register(CompareSegmentsTool())
    registry.register(SegmentContributionTool())
    registry.register(DecomposeMetricTool())
    registry.register(DetectAnomaliesTool())
    registry.register(CreateChartTool())
    return registry
