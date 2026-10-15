"""Natural Language Query (NLQ) planning, validation, and execution module."""

from omnisleuth.nlq.charting import ChartBuildResult, build_chart_from_table, infer_example_questions
from omnisleuth.nlq.context import ColumnContext, SchemaContext, build_schema_context
from omnisleuth.nlq.executor import PlanExecutor
from omnisleuth.nlq.followup import (
    detect_followup_marker,
    format_history_for_prompt,
    resolve_rule_followup,
    summarize_plan,
    summarize_result,
)
from omnisleuth.nlq.llm_planner import (
    LLMPlanningError,
    LLMPlanner,
    build_dataset_profile,
    build_plan_json_schema,
    schema_context_hash,
)
from omnisleuth.nlq.models import (
    AnalysisPlan,
    ChartSpec,
    ChartType,
    FilterOp,
    FilterSpec,
    MetricAgg,
    MetricSpec,
    PeriodSpec,
    PlanIntent,
    QueryResult,
    SortDirection,
    SortSpec,
    TimeGrain,
    TimeSpec,
)
from omnisleuth.nlq.narration import NarrationResult, narrate
from omnisleuth.nlq.planner import Planner
from omnisleuth.nlq.pipeline import AnalysisPipeline, PlanRoutingResult
from omnisleuth.nlq.rule_planner import RulePlanner
from omnisleuth.nlq.validator import PlanValidationError, PlanValidator

__all__ = [
    "AnalysisPlan",
    "AnalysisPipeline",
    "ChartSpec",
    "ChartBuildResult",
    "ChartType",
    "build_chart_from_table",
    "infer_example_questions",
    "detect_followup_marker",
    "resolve_rule_followup",
    "summarize_plan",
    "summarize_result",
    "NarrationResult",
    "narrate",
    "ColumnContext",
    "FilterOp",
    "FilterSpec",
    "MetricAgg",
    "MetricSpec",
    "PeriodSpec",
    "PlanExecutor",
    "PlanIntent",
    "PlanValidationError",
    "PlanValidator",
    "Planner",
    "PlanRoutingResult",
    "QueryResult",
    "RulePlanner",
    "LLMPlanner",
    "LLMPlanningError",
    "build_dataset_profile",
    "build_plan_json_schema",
    "schema_context_hash",
    "SchemaContext",
    "SortDirection",
    "SortSpec",
    "TimeGrain",
    "TimeSpec",
    "build_schema_context",
]
