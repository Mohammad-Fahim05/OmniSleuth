"""Deterministic analytical computation engine using Pandas and NumPy."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from omnisleuth.analytics.models import (
    AggregationFunction,
    AggregationResult,
    ChartDataSpec,
    ChartType,
    FactorEffect,
    FilterCondition,
    MetricDecompositionResult,
    SegmentComparisonResult,
    SegmentContributionItem,
    SegmentContributionResult,
    SegmentRecord,
    TrendPoint,
    TrendResult,
)
from omnisleuth.core.exceptions import ToolExecutionError
from omnisleuth.core.logging import get_logger

logger = get_logger("analytics")


class DeterministicAnalyticsEngine:
    """Performs deterministic statistical and financial calculations on DataFrames."""

    def apply_filters(self, df: pd.DataFrame, filters: Optional[List[FilterCondition]] = None) -> pd.DataFrame:
        """Apply a sequence of deterministic filters to a DataFrame."""
        if not filters:
            return df

        filtered_df = df.copy()
        for f in filters:
            if f.column not in filtered_df.columns:
                raise ToolExecutionError(f"Filter column '{f.column}' not found in dataset columns: {list(filtered_df.columns)}")

            series = filtered_df[f.column]
            op = f.operator.strip().lower()

            try:
                if op in ["==", "eq", "="]:
                    filtered_df = filtered_df[series == f.value]
                elif op in ["!=", "ne"]:
                    filtered_df = filtered_df[series != f.value]
                elif op in [">", "gt"]:
                    filtered_df = filtered_df[series > f.value]
                elif op in [">=", "gte"]:
                    filtered_df = filtered_df[series >= f.value]
                elif op in ["<", "lt"]:
                    filtered_df = filtered_df[series < f.value]
                elif op in ["<=", "lte"]:
                    filtered_df = filtered_df[series <= f.value]
                elif op in ["in", "isin"]:
                    val_list = f.value if isinstance(f.value, list) else [f.value]
                    filtered_df = filtered_df[series.isin(val_list)]
                elif op in ["not in", "isnotin"]:
                    val_list = f.value if isinstance(f.value, list) else [f.value]
                    filtered_df = filtered_df[~series.isin(val_list)]
                elif op == "between":
                    if isinstance(f.value, (list, tuple)) and len(f.value) == 2:
                        filtered_df = filtered_df[series.between(f.value[0], f.value[1])]
                    else:
                        raise ToolExecutionError(f"Filter operator 'between' expects a 2-element sequence, got: {f.value}")
                else:
                    raise ToolExecutionError(f"Unsupported filter operator: '{f.operator}'")
            except Exception as e:
                raise ToolExecutionError(f"Error applying filter on '{f.column}' ({f.operator} {f.value}): {str(e)}")

        return filtered_df

    def aggregate(
        self,
        df: pd.DataFrame,
        metric_column: str,
        agg_func: AggregationFunction = AggregationFunction.SUM,
        group_by_columns: Optional[List[str]] = None,
        filters: Optional[List[FilterCondition]] = None,
    ) -> AggregationResult:
        """Perform deterministic group-by aggregation."""
        filtered_df = self.apply_filters(df, filters)
        if metric_column not in filtered_df.columns:
            raise ToolExecutionError(f"Metric column '{metric_column}' not found.")

        group_cols = group_by_columns or []
        for col in group_cols:
            if col not in filtered_df.columns:
                raise ToolExecutionError(f"Group-by column '{col}' not found.")

        func_map = {
            AggregationFunction.SUM: "sum",
            AggregationFunction.MEAN: "mean",
            AggregationFunction.COUNT: "count",
            AggregationFunction.MIN: "min",
            AggregationFunction.MAX: "max",
            AggregationFunction.MEDIAN: "median",
        }
        pd_func = func_map[agg_func]

        if not group_cols:
            # Overall single aggregate
            calc_val = getattr(pd.to_numeric(filtered_df[metric_column], errors="coerce"), pd_func)()
            val = round(float(calc_val), 4) if pd.notna(calc_val) else 0.0
            records = [{metric_column: val}]
            grand_total = val if agg_func in [AggregationFunction.SUM, AggregationFunction.COUNT] else None
        else:
            grouped = filtered_df.groupby(group_cols, dropna=False)[metric_column]
            agg_series = getattr(grouped, pd_func)().reset_index()
            # Clean numeric and round
            agg_series[metric_column] = agg_series[metric_column].apply(lambda x: round(float(x), 4) if pd.notna(x) else 0.0)
            records = agg_series.to_dict(orient="records")
            grand_total = round(float(pd.to_numeric(filtered_df[metric_column], errors="coerce").sum()), 4)

        return AggregationResult(
            metric_column=metric_column,
            agg_func=agg_func,
            group_by_columns=group_cols,
            records=records,
            total_records=len(records),
            grand_total=grand_total,
        )

    def calculate_trend(
        self,
        df: pd.DataFrame,
        date_column: str,
        metric_column: str,
        frequency: str = "M",  # 'D', 'W', 'M', 'Q'
        agg_func: AggregationFunction = AggregationFunction.SUM,
        filters: Optional[List[FilterCondition]] = None,
    ) -> TrendResult:
        """Calculate period-over-period time-series trends and percentage changes."""
        filtered_df = self.apply_filters(df, filters).copy()

        if date_column not in filtered_df.columns:
            raise ToolExecutionError(f"Date column '{date_column}' not found.")
        if metric_column not in filtered_df.columns:
            raise ToolExecutionError(f"Metric column '{metric_column}' not found.")

        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(filtered_df[date_column]):
            filtered_df[date_column] = pd.to_datetime(filtered_df[date_column], errors="coerce")

        filtered_df = filtered_df.dropna(subset=[date_column])
        if len(filtered_df) == 0:
            raise ToolExecutionError("No valid date records available after filtering.")

        filtered_df[metric_column] = pd.to_numeric(filtered_df[metric_column], errors="coerce").fillna(0.0)

        # Resample frequency mapping
        freq_upper = frequency.upper()
        resample_rule = "ME" if freq_upper == "M" else "QE" if freq_upper == "Q" else freq_upper

        grouped = filtered_df.set_index(date_column).resample(resample_rule)[metric_column]
        func_map = {
            AggregationFunction.SUM: "sum",
            AggregationFunction.MEAN: "mean",
            AggregationFunction.COUNT: "count",
            AggregationFunction.MIN: "min",
            AggregationFunction.MAX: "max",
            AggregationFunction.MEDIAN: "median",
        }
        resampled_series = getattr(grouped, func_map[agg_func])()

        trend_points: List[TrendPoint] = []
        prev_val: Optional[float] = None

        for dt_idx, val in resampled_series.items():
            period_label = dt_idx.strftime("%Y-%m-%d" if freq_upper in ["D", "W"] else "%Y-%m" if freq_upper == "M" else "%Y-Q%q")
            num_val = round(float(val), 4)

            delta = round(num_val - prev_val, 4) if prev_val is not None else None
            change_pct = (
                round(((num_val - prev_val) / abs(prev_val)) * 100.0, 2)
                if (prev_val is not None and prev_val != 0.0)
                else None
            )

            trend_points.append(
                TrendPoint(
                    period=period_label,
                    value=num_val,
                    delta_from_prev=delta,
                    change_pct_from_prev=change_pct,
                )
            )
            prev_val = num_val

        start_val = trend_points[0].value if trend_points else 0.0
        end_val = trend_points[-1].value if trend_points else 0.0
        overall_delta = round(end_val - start_val, 4)
        overall_pct = round(((end_val - start_val) / abs(start_val) * 100.0), 2) if start_val != 0.0 else 0.0

        return TrendResult(
            date_column=date_column,
            metric_column=metric_column,
            frequency=frequency,
            time_series=trend_points,
            start_period=trend_points[0].period if trend_points else "",
            end_period=trend_points[-1].period if trend_points else "",
            start_value=start_val,
            end_value=end_val,
            overall_delta=overall_delta,
            overall_change_pct=overall_pct,
        )

    def compare_segments(
        self,
        df: pd.DataFrame,
        segment_column: str,
        metric_column: str,
        agg_func: AggregationFunction = AggregationFunction.SUM,
        filters: Optional[List[FilterCondition]] = None,
    ) -> SegmentComparisonResult:
        """Compare metric totals and percentage shares across segments of a category."""
        filtered_df = self.apply_filters(df, filters)
        if segment_column not in filtered_df.columns:
            raise ToolExecutionError(f"Segment column '{segment_column}' not found.")
        if metric_column not in filtered_df.columns:
            raise ToolExecutionError(f"Metric column '{metric_column}' not found.")

        agg_res = self.aggregate(
            df=filtered_df,
            metric_column=metric_column,
            agg_func=agg_func,
            group_by_columns=[segment_column],
        )

        records = agg_res.records
        total_val = sum(r[metric_column] for r in records)

        # Sort descending by metric value
        records_sorted = sorted(records, key=lambda x: x[metric_column], reverse=True)

        segment_records: List[SegmentRecord] = []
        for rank, r in enumerate(records_sorted, start=1):
            val = r[metric_column]
            share = round((val / total_val * 100.0), 2) if total_val != 0 else 0.0
            segment_records.append(
                SegmentRecord(
                    segment=str(r[segment_column]),
                    value=val,
                    share_percentage=share,
                    rank=rank,
                )
            )

        return SegmentComparisonResult(
            segment_column=segment_column,
            metric_column=metric_column,
            agg_func=agg_func,
            total_metric_value=round(total_val, 4),
            segments=segment_records,
        )

    def calculate_segment_contribution(
        self,
        df: pd.DataFrame,
        segment_column: str,
        metric_column: str,
        period_1_filters: List[FilterCondition],
        period_2_filters: List[FilterCondition],
        period_1_label: str = "Period 1",
        period_2_label: str = "Period 2",
        agg_func: AggregationFunction = AggregationFunction.SUM,
    ) -> SegmentContributionResult:
        """Calculate mathematical contribution of each segment to the overall metric delta between two periods.

        Contribution % = (Delta_segment / Delta_total) * 100%
        """
        df_p1 = self.apply_filters(df, period_1_filters)
        df_p2 = self.apply_filters(df, period_2_filters)

        res_p1 = self.aggregate(df_p1, metric_column=metric_column, agg_func=agg_func, group_by_columns=[segment_column])
        res_p2 = self.aggregate(df_p2, metric_column=metric_column, agg_func=agg_func, group_by_columns=[segment_column])

        p1_map = {str(r[segment_column]): r[metric_column] for r in res_p1.records}
        p2_map = {str(r[segment_column]): r[metric_column] for r in res_p2.records}

        all_segments = sorted(list(set(p1_map.keys()).union(set(p2_map.keys()))))

        total_p1 = sum(p1_map.values())
        total_p2 = sum(p2_map.values())
        total_delta = round(total_p2 - total_p1, 4)
        total_change_pct = round(((total_p2 - total_p1) / abs(total_p1) * 100.0), 2) if total_p1 != 0 else 0.0

        contributions: List[SegmentContributionItem] = []
        for seg in all_segments:
            v1 = round(p1_map.get(seg, 0.0), 4)
            v2 = round(p2_map.get(seg, 0.0), 4)
            delta = round(v2 - v1, 4)
            seg_pct = round(((v2 - v1) / abs(v1) * 100.0), 2) if v1 != 0 else (100.0 if v2 > 0 else 0.0)

            # Mathematical contribution to total delta
            contrib_pct = round((delta / total_delta * 100.0), 2) if total_delta != 0 else 0.0

            contributions.append(
                SegmentContributionItem(
                    segment=seg,
                    period_1_value=v1,
                    period_2_value=v2,
                    delta=delta,
                    segment_change_pct=seg_pct,
                    contribution_to_total_delta_pct=contrib_pct,
                )
            )

        # Sort by largest absolute delta
        contributions = sorted(contributions, key=lambda x: abs(x.delta), reverse=True)

        return SegmentContributionResult(
            segment_column=segment_column,
            metric_column=metric_column,
            period_1_label=period_1_label,
            period_2_label=period_2_label,
            total_period_1=round(total_p1, 4),
            total_period_2=round(total_p2, 4),
            total_delta=total_delta,
            total_change_pct=total_change_pct,
            contributions=contributions,
        )

    def decompose_revenue_orders_aov(
        self,
        df: pd.DataFrame,
        revenue_column: str,
        order_id_column: str,
        period_1_filters: List[FilterCondition],
        period_2_filters: List[FilterCondition],
        period_1_label: str = "Period 1",
        period_2_label: str = "Period 2",
    ) -> MetricDecompositionResult:
        """Decompose Revenue change into Volume (Orders) and Price/Basket (Average Order Value) effects.

        Formula: Revenue = Orders * AOV
        Logarithmic Mean Weighting / Shapley decomposition:
        Delta_Rev = Volume_Effect + AOV_Effect
        """
        df_p1 = self.apply_filters(df, period_1_filters)
        df_p2 = self.apply_filters(df, period_2_filters)

        rev1 = float(df_p1[revenue_column].sum())
        orders1 = float(df_p1[order_id_column].nunique())
        aov1 = rev1 / orders1 if orders1 > 0 else 0.0

        rev2 = float(df_p2[revenue_column].sum())
        orders2 = float(df_p2[order_id_column].nunique())
        aov2 = rev2 / orders2 if orders2 > 0 else 0.0

        target_delta = round(rev2 - rev1, 4)
        target_pct = round(((rev2 - rev1) / abs(rev1) * 100.0), 2) if rev1 > 0 else 0.0

        # Shapley decomposition for Revenue = Orders * AOV
        # Impact of Orders = Delta_Orders * AOV_avg
        # Impact of AOV = Delta_AOV * Orders_avg
        orders_avg = (orders1 + orders2) / 2.0
        aov_avg = (aov1 + aov2) / 2.0

        orders_delta = orders2 - orders1
        aov_delta = aov2 - aov1

        impact_orders = round(orders_delta * aov_avg, 4)
        impact_aov = round(aov_delta * orders_avg, 4)

        # Exact percentage contribution to delta
        orders_pct_contrib = round((impact_orders / target_delta * 100.0), 2) if target_delta != 0 else 0.0
        aov_pct_contrib = round((impact_aov / target_delta * 100.0), 2) if target_delta != 0 else 0.0

        factor_effects = [
            FactorEffect(
                factor_name="Order Volume (Orders)",
                period_1_value=round(orders1, 2),
                period_2_value=round(orders2, 2),
                delta=round(orders_delta, 2),
                change_pct=round((orders_delta / orders1 * 100.0), 2) if orders1 > 0 else 0.0,
                estimated_impact_on_target=impact_orders,
                impact_percentage=orders_pct_contrib,
            ),
            FactorEffect(
                factor_name="Average Order Value (AOV)",
                period_1_value=round(aov1, 2),
                period_2_value=round(aov2, 2),
                delta=round(aov_delta, 2),
                change_pct=round((aov_delta / aov1 * 100.0), 2) if aov1 > 0 else 0.0,
                estimated_impact_on_target=impact_aov,
                impact_percentage=aov_pct_contrib,
            ),
        ]

        return MetricDecompositionResult(
            target_metric=revenue_column,
            decomposition_type="multiplicative",
            period_1_label=period_1_label,
            period_2_label=period_2_label,
            target_period_1=round(rev1, 4),
            target_period_2=round(rev2, 4),
            target_delta=target_delta,
            target_change_pct=target_pct,
            factor_effects=factor_effects,
        )

    def generate_chart_spec(
        self,
        chart_type: ChartType,
        title: str,
        data: List[Dict[str, Any]],
        x_column: str,
        y_column: str,
        color_column: Optional[str] = None,
        layout_options: Optional[Dict[str, Any]] = None,
    ) -> ChartDataSpec:
        """Create a declarative chart specification."""
        return ChartDataSpec(
            chart_type=chart_type,
            title=title,
            x_column=x_column,
            y_column=y_column,
            color_column=color_column,
            data=data,
            layout_options=layout_options or {},
        )
