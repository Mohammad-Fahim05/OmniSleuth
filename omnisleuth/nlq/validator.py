"""PlanValidator: Strict schema and semantic validation of analytical query plans."""

from typing import List, Optional, Set, Tuple
import pandas as pd
from omnisleuth.nlq.context import ColumnContext, SchemaContext
from omnisleuth.nlq.models import (
    AnalysisPlan,
    ChartType,
    FilterOp,
    MetricAgg,
    PlanIntent,
    TimeGrain,
    TimeSpec,
)


class PlanValidationError(Exception):
    """Raised when an AnalysisPlan fails structural or semantic validation."""

    def __init__(self, message: str, reason: str):
        super().__init__(message)
        self.reason = reason


class PlanValidator:
    """Validates AnalysisPlan against SchemaContext without hallucinating or defaulting."""

    def __init__(
        self,
        max_limit: int = 50,
        max_in_list_size: int = 50,
        max_group_by_cardinality: int = 30,
    ):
        self.max_limit = max_limit
        self.max_in_list_size = max_in_list_size
        self.max_group_by_cardinality = max_group_by_cardinality

    def validate(self, plan: AnalysisPlan, schema_ctx: SchemaContext) -> AnalysisPlan:
        """Validate an AnalysisPlan against SchemaContext.

        If validation succeeds, returns the valid plan.
        If validation fails, returns an unsupported or needs_clarification plan
        naming the problem and available columns. NEVER substitutes a default metric.
        """
        # If the plan already flagged needs_clarification, preserve it
        if plan.needs_clarification:
            return plan

        # If the plan is already unsupported or off-topic, preserve it
        if plan.intent in (PlanIntent.UNSUPPORTED, PlanIntent.GENERAL_KNOWLEDGE):
            if not plan.unsupported_reason and plan.intent == PlanIntent.UNSUPPORTED:
                plan.unsupported_reason = "Question cannot be mapped to allowlisted dataset operations."
            return plan

        # Non-data intents like documentation or deep_investigation pass validation
        if plan.intent in (PlanIntent.DOCUMENTATION_QUESTION, PlanIntent.DEEP_INVESTIGATION):
            return plan

        try:
            self._validate_columns_exist(plan, schema_ctx)
            self._validate_metric_aggregations(plan, schema_ctx)
            self._validate_group_by(plan, schema_ctx)
            self._validate_filters(plan, schema_ctx)
            self._validate_time_spec(plan, schema_ctx)
            self._validate_chart_compatibility(plan, schema_ctx)
            self._validate_limits(plan)
            return plan

        except PlanValidationError as exc:
            return AnalysisPlan(
                intent=PlanIntent.UNSUPPORTED,
                confidence=0.0,
                unsupported_reason=exc.reason,
                planner=plan.planner,
                assumptions=plan.assumptions,
            )

    def is_valid(self, plan: AnalysisPlan, schema_ctx: SchemaContext) -> Tuple[bool, Optional[str]]:
        """Quick boolean validation check."""
        try:
            self._validate_columns_exist(plan, schema_ctx)
            self._validate_metric_aggregations(plan, schema_ctx)
            self._validate_group_by(plan, schema_ctx)
            self._validate_filters(plan, schema_ctx)
            self._validate_time_spec(plan, schema_ctx)
            self._validate_chart_compatibility(plan, schema_ctx)
            self._validate_limits(plan)
            return True, None
        except PlanValidationError as exc:
            return False, exc.reason

    def _validate_columns_exist(self, plan: AnalysisPlan, schema_ctx: SchemaContext) -> None:
        """Ensure all referenced columns exist in the active schema."""
        all_cols = set(schema_ctx.columns.keys())
        available_str = ", ".join(sorted(all_cols)) if all_cols else "None"

        # Check metrics
        for m in plan.metrics:
            if m.column not in all_cols and m.column != "*":
                raise PlanValidationError(
                    f"Metric column '{m.column}' not in dataset",
                    f"Column '{m.column}' does not exist in the dataset. Available columns: {available_str}",
                )

        # Check group_by
        for g in plan.group_by:
            if g not in all_cols:
                raise PlanValidationError(
                    f"Group-by column '{g}' not in dataset",
                    f"Group-by column '{g}' does not exist in the dataset. Available columns: {available_str}",
                )

        # Check filters
        for f in plan.filters:
            if f.column not in all_cols:
                raise PlanValidationError(
                    f"Filter column '{f.column}' not in dataset",
                    f"Filter column '{f.column}' does not exist in the dataset. Available columns: {available_str}",
                )

        # Check sort
        if plan.sort:
            valid_sort_targets = all_cols.union(plan.group_by).union(
                {m.column for m in plan.metrics}
            )
            # Allow metric aliases (e.g. 'metric_0', 'value')
            valid_sort_targets.update({"metric_0", "metric_1", "metric_2", "value", "total", "count"})
            if plan.sort.by not in valid_sort_targets:
                raise PlanValidationError(
                    f"Sort column '{plan.sort.by}' not in dataset",
                    f"Sort column '{plan.sort.by}' does not exist. Available columns: {available_str}",
                )

        # Check time date_column
        if plan.time:
            if plan.time.date_column not in all_cols:
                raise PlanValidationError(
                    f"Date column '{plan.time.date_column}' not in dataset",
                    f"Date column '{plan.time.date_column}' does not exist. Available date columns: {', '.join(schema_ctx.date_columns) or 'None'}",
                )

        # Check target_column (contribution)
        if plan.target_column and plan.target_column not in all_cols:
            raise PlanValidationError(
                f"Target column '{plan.target_column}' not in dataset",
                f"Target column '{plan.target_column}' does not exist. Available columns: {available_str}",
            )

    def _validate_metric_aggregations(self, plan: AnalysisPlan, schema_ctx: SchemaContext) -> None:
        """Validate aggregation compatibility with underlying column data types."""
        numeric_str = ", ".join(schema_ctx.numeric_columns) if schema_ctx.numeric_columns else "None"

        for m in plan.metrics:
            if m.column == "*":
                if m.agg not in (MetricAgg.COUNT,):
                    raise PlanValidationError(
                        f"Cannot compute {m.agg.value} on wildcard '*'",
                        f"Wildcard '*' only supports 'count' aggregation.",
                    )
                continue

            col_ctx = schema_ctx.columns.get(m.column)
            if not col_ctx:
                continue

            # Mathematical aggregations require numeric data
            if m.agg in (MetricAgg.SUM, MetricAgg.MEAN, MetricAgg.MEDIAN):
                if col_ctx.data_type != "numeric":
                    raise PlanValidationError(
                        f"Cannot compute {m.agg.value} on non-numeric column '{m.column}'",
                        f"Cannot compute {m.agg.value} of non-numeric column '{m.column}' (detected type: {col_ctx.data_type}). "
                        f"Available numeric columns: {numeric_str}",
                    )

            # Min/max require ordered data (numeric or datetime)
            elif m.agg in (MetricAgg.MIN, MetricAgg.MAX):
                if col_ctx.data_type not in ("numeric", "datetime"):
                    raise PlanValidationError(
                        f"Cannot compute {m.agg.value} on non-ordered column '{m.column}'",
                        f"Cannot compute {m.agg.value} of non-ordered column '{m.column}' (detected type: {col_ctx.data_type}).",
                    )

    def _estimate_bucketed_date_cardinality(
        self,
        grain: TimeGrain,
        col_ctx: ColumnContext,
        schema_ctx: SchemaContext,
        time_spec: Optional[TimeSpec] = None,
    ) -> int:
        """Estimate the number of discrete time buckets for a date column."""
        min_d = (time_spec.start if time_spec and time_spec.start else None) or schema_ctx.min_date or col_ctx.min_value
        max_d = (time_spec.end if time_spec and time_spec.end else None) or schema_ctx.max_date or col_ctx.max_value

        if grain == TimeGrain.DAY:
            if min_d and max_d:
                try:
                    dt_min = pd.to_datetime(min_d)
                    dt_max = pd.to_datetime(max_d)
                    return max(1, (dt_max - dt_min).days + 1)
                except Exception:
                    pass
            return col_ctx.unique_count or 0

        if min_d and max_d:
            try:
                dt_min = pd.to_datetime(min_d)
                dt_max = pd.to_datetime(max_d)
                if grain == TimeGrain.MONTH:
                    return max(1, (dt_max.year - dt_min.year) * 12 + (dt_max.month - dt_min.month) + 1)
                elif grain == TimeGrain.QUARTER:
                    q_min = (dt_min.month - 1) // 3
                    q_max = (dt_max.month - 1) // 3
                    return max(1, (dt_max.year - dt_min.year) * 4 + (q_max - q_min) + 1)
                elif grain == TimeGrain.YEAR:
                    return max(1, dt_max.year - dt_min.year + 1)
                elif grain == TimeGrain.WEEK:
                    days = (dt_max - dt_min).days
                    return max(1, days // 7 + 1)
            except Exception:
                pass

        raw = col_ctx.unique_count or 0
        if grain == TimeGrain.MONTH:
            return max(1, (raw + 27) // 28)
        elif grain == TimeGrain.QUARTER:
            return max(1, (raw + 89) // 90)
        elif grain == TimeGrain.YEAR:
            return max(1, (raw + 364) // 365)
        elif grain == TimeGrain.WEEK:
            return max(1, (raw + 6) // 7)

        return raw

    def _validate_group_by(self, plan: AnalysisPlan, schema_ctx: SchemaContext) -> None:
        """Reject group-by columns marked as identifier or exceeding max cardinality."""
        for g in plan.group_by:
            col_ctx = schema_ctx.columns.get(g)
            if not col_ctx:
                continue

            if str(col_ctx.data_type).lower() == "identifier":
                raise PlanValidationError(
                    f"Group-by column '{g}' is an identifier",
                    f"Column '{g}' is marked as an identifier and cannot be used as a group-by dimension.",
                )

            is_date_col = (str(col_ctx.data_type).lower() == "datetime") or (g in schema_ctx.date_columns)

            if is_date_col and plan.intent == PlanIntent.TREND:
                # In trend plans, date columns are bucketed into time periods (e.g., months/quarters),
                # not grouped by raw timestamps. Estimate bucketed period cardinality.
                grain = plan.time.grain if (plan.time and plan.time.grain) else TimeGrain.MONTH
                cardinality = self._estimate_bucketed_date_cardinality(grain, col_ctx, schema_ctx, plan.time)
                # If plan has no time specification, populate it with standard monthly bucketing
                if plan.time is None:
                    plan.time = TimeSpec(date_column=g, grain=grain)
                elif not plan.time.date_column:
                    plan.time.date_column = g
            else:
                cardinality = col_ctx.unique_count or 0

            if cardinality > self.max_group_by_cardinality:
                raise PlanValidationError(
                    f"Group-by column '{g}' exceeds cardinality limit",
                    f"Group-by column '{g}' has {cardinality} distinct values, "
                    f"exceeding the maximum allowed cardinality of {self.max_group_by_cardinality}.",
                )

    def _validate_filters(self, plan: AnalysisPlan, schema_ctx: SchemaContext) -> None:
        """Validate filter operators, value types, and bounds."""
        for f in plan.filters:
            col_ctx = schema_ctx.columns.get(f.column)
            if not col_ctx:
                continue

            # Operator bounds check for IN
            if f.op == FilterOp.IN:
                if not isinstance(f.value, (list, tuple, set)):
                    raise PlanValidationError(
                        f"'in' operator requires a list",
                        f"Filter operator 'in' for '{f.column}' requires a list of values, got {type(f.value).__name__}.",
                    )
                if len(f.value) > self.max_in_list_size:
                    raise PlanValidationError(
                        f"'in' list exceeds maximum length",
                        f"Filter list for '{f.column}' exceeds maximum allowed size ({self.max_in_list_size}).",
                    )

            # Operator check for BETWEEN
            elif f.op == FilterOp.BETWEEN:
                if not isinstance(f.value, (list, tuple)) or len(f.value) != 2:
                    raise PlanValidationError(
                        f"'between' operator requires two bounds",
                        f"Filter operator 'between' for '{f.column}' requires exactly two boundary values [start, end].",
                    )

            # Type checking for numeric comparisons
            if col_ctx.data_type == "numeric":
                if f.op in (FilterOp.GT, FilterOp.GTE, FilterOp.LT, FilterOp.LTE, FilterOp.EQ, FilterOp.NE):
                    try:
                        float(f.value)
                    except (ValueError, TypeError):
                        raise PlanValidationError(
                            f"Numeric filter requires numeric value",
                            f"Filter on numeric column '{f.column}' requires a numerical value, got '{f.value}'.",
                        )
                elif f.op == FilterOp.CONTAINS:
                    raise PlanValidationError(
                        f"'contains' operator unsupported on numeric columns",
                        f"Operator 'contains' is for text columns, not numeric column '{f.column}'.",
                    )

    def _validate_time_spec(self, plan: AnalysisPlan, schema_ctx: SchemaContext) -> None:
        """Validate time specs, date parsing, and range boundaries."""
        if not plan.time:
            return

        col_ctx = schema_ctx.columns.get(plan.time.date_column)
        if col_ctx and col_ctx.data_type != "datetime":
            raise PlanValidationError(
                f"Time specification requires a datetime column",
                f"Column '{plan.time.date_column}' is not classified as datetime (type: {col_ctx.data_type}). "
                f"Available date columns: {', '.join(schema_ctx.date_columns) or 'None'}",
            )

        # Parse date boundaries if specified
        start_dt = None
        end_dt = None
        if plan.time.start:
            try:
                start_dt = pd.to_datetime(plan.time.start)
            except Exception:
                raise PlanValidationError(
                    f"Unparseable start date '{plan.time.start}'",
                    f"Could not parse start date '{plan.time.start}'. Use ISO format YYYY-MM-DD.",
                )

        if plan.time.end:
            try:
                end_dt = pd.to_datetime(plan.time.end)
            except Exception:
                raise PlanValidationError(
                    f"Unparseable end date '{plan.time.end}'",
                    f"Could not parse end date '{plan.time.end}'. Use ISO format YYYY-MM-DD.",
                )

        # Boundary checks against dataset min/max date
        if schema_ctx.min_date and schema_ctx.max_date:
            try:
                ds_min = pd.to_datetime(schema_ctx.min_date)
                ds_max = pd.to_datetime(schema_ctx.max_date)

                if start_dt and start_dt > ds_max:
                    raise PlanValidationError(
                        f"Start date {plan.time.start} is after dataset max date {schema_ctx.max_date}",
                        f"Requested date range starts at {plan.time.start}, which is completely outside the dataset's date range ({schema_ctx.min_date} to {schema_ctx.max_date}).",
                    )
                if end_dt and end_dt < ds_min:
                    raise PlanValidationError(
                        f"End date {plan.time.end} is before dataset min date {schema_ctx.min_date}",
                        f"Requested date range ends at {plan.time.end}, which is completely outside the dataset's date range ({schema_ctx.min_date} to {schema_ctx.max_date}).",
                    )
            except PlanValidationError:
                raise
            except Exception:
                pass

        # Ensure trend plans do not exceed maximum bucket cardinality
        if plan.intent == PlanIntent.TREND and plan.time:
            col_ctx = schema_ctx.columns.get(plan.time.date_column)
            if col_ctx:
                cardinality = self._estimate_bucketed_date_cardinality(plan.time.grain, col_ctx, schema_ctx, plan.time)
                if cardinality > self.max_group_by_cardinality:
                    raise PlanValidationError(
                        f"Trend time grain '{plan.time.grain.value}' exceeds cardinality limit",
                        f"Time grain '{plan.time.grain.value}' on column '{plan.time.date_column}' produces ~{cardinality} buckets, "
                        f"exceeding the maximum allowed cardinality of {self.max_group_by_cardinality}.",
                    )

        # Validate discrete comparison periods (period_a and period_b)
        for p_name, p_spec in [("period_a", plan.period_a), ("period_b", plan.period_b)]:
            if p_spec and schema_ctx.min_date and schema_ctx.max_date:
                try:
                    ds_min = pd.to_datetime(schema_ctx.min_date)
                    ds_max = pd.to_datetime(schema_ctx.max_date)
                    p_start = pd.to_datetime(p_spec.start) if p_spec.start else None
                    p_end = pd.to_datetime(p_spec.end) if p_spec.end else None
                    lbl = p_spec.label or p_name
                    if p_start and p_start > ds_max:
                        raise PlanValidationError(
                            f"Comparison period '{lbl}' start is after dataset range",
                            f"Requested comparison period '{lbl}' ({p_spec.start}) is after the dataset's date range ({schema_ctx.min_date} to {schema_ctx.max_date}).",
                        )
                    if p_end and p_end < ds_min:
                        raise PlanValidationError(
                            f"Comparison period '{lbl}' end is before dataset range",
                            f"Requested comparison period '{lbl}' ({p_spec.end}) is before the dataset's date range ({schema_ctx.min_date} to {schema_ctx.max_date}).",
                        )
                except PlanValidationError:
                    raise
                except Exception:
                    pass

    def _validate_chart_compatibility(self, plan: AnalysisPlan, schema_ctx: SchemaContext) -> None:
        """Validate chart specification against column types and cardinalities."""
        if not plan.chart:
            return

        c = plan.chart
        col_x = schema_ctx.columns.get(c.x)
        col_y = schema_ctx.columns.get(c.y) if c.y else None

        # Line chart requires ordered x-axis (datetime or numeric or temporal group)
        if c.type == ChartType.LINE:
            if col_x and col_x.data_type not in ("datetime", "numeric"):
                if c.x not in ("period", "date", "month", "quarter", "year"):
                    raise PlanValidationError(
                        f"Line chart requires an ordered x-axis",
                        f"Line chart requires an ordered x-axis (such as date or numeric column), but '{c.x}' is categorical.",
                    )

        # Scatter plot requires two numeric columns
        elif c.type == ChartType.SCATTER:
            if not c.y:
                raise PlanValidationError(
                    f"Scatter plot requires both x and y columns",
                    f"Scatter plot requires both x and y coordinates specified.",
                )
            if (col_x and col_x.data_type != "numeric") or (col_y and col_y.data_type != "numeric"):
                raise PlanValidationError(
                    f"Scatter plot requires numeric x and y",
                    f"Scatter plot requires two numeric columns, but got '{c.x}' ({col_x.data_type if col_x else 'unknown'}) "
                    f"and '{c.y}' ({col_y.data_type if col_y else 'unknown'}).",
                )

        # Pie chart requires low cardinality categories (<= 10)
        elif c.type == ChartType.PIE:
            if col_x and col_x.unique_count > 10 and (plan.limit is None or plan.limit > 10):
                raise PlanValidationError(
                    f"Pie chart unsuitable for high-cardinality column",
                    f"Pie chart is unsuitable for '{c.x}' because it has {col_x.unique_count} distinct categories (>10). "
                    f"Use a bar chart instead.",
                )

    def _validate_limits(self, plan: AnalysisPlan) -> None:
        """Validate limit bounds."""
        if plan.limit is not None:
            if plan.limit < 1 or plan.limit > self.max_limit:
                raise PlanValidationError(
                    f"Limit {plan.limit} out of bounds (1..{self.max_limit})",
                    f"Requested limit of {plan.limit} rows is outside allowlisted bounds (1 to {self.max_limit}).",
                )
