"""Chart construction from computed result tables (Phase D).

Charts are always built from the **computed** table produced by ``PlanExecutor``,
never from raw dataset rows. Mappings that are unsuitable for the requested chart
type are rejected with an explanation and a suggested alternative, so the UI can
disclose why a chart is absent instead of silently drawing a misleading one.
"""

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from omnisleuth.nlq.context import SchemaContext
from omnisleuth.nlq.models import ChartType

MAX_CHART_ROWS = 50
MAX_CHART_CATEGORIES = 12

_PERIODISH = ("period", "month", "quarter", "year", "date", "week", "day")


class ChartBuildResult(BaseModel):
    """Outcome of building a chart from a computed table."""

    used: bool = False
    chart_type: Optional[str] = None
    spec: Optional[Dict[str, Any]] = None
    data: List[Dict[str, Any]] = Field(default_factory=list)
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    explanation: str = ""
    suggestion: Optional[str] = None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _column_kind(table: List[Dict[str, Any]], column: str) -> str:
    """Classify a column as 'numeric', 'period', or 'categorical' from the table."""
    values = [row.get(column) for row in table if row.get(column) is not None]
    if not values:
        return "empty"
    numeric = sum(1 for v in values if _is_number(v))
    if numeric == len(values):
        return "numeric"
    lowered = column.lower()
    if any(term in lowered for term in _PERIODISH):
        sample = str(values[0])
        if re.match(r"^\d{4}(-\d{2})?(-\d{2})?$", sample) or "Q" in sample:
            return "period"
    return "categorical"


def _ordered(column: str, kind: str) -> bool:
    return kind in ("numeric", "period")


def _suggest(kind: str, x_reason: str, y_reason: Optional[str]) -> Optional[str]:
    if "numeric" in (y_reason or ""):
        return "Use a bar or line chart, or provide a categorical column for x."
    if x_reason:
        return "Use a bar chart, which accepts categorical x-axes."
    return "Use a bar chart for categorical comparisons."


def build_chart_from_table(
    result_table: Optional[List[Dict[str, Any]]],
    requested_type: Optional[str] = None,
    title: Optional[str] = None,
) -> ChartBuildResult:
    """Build a chart specification from a computed result table.

    ``requested_type`` (if any) comes from the validated plan; when it is
    incompatible with the computed columns the request is rejected with an
    explanation and a suggested alternative.
    """
    table = list(result_table or [])
    if not table:
        return ChartBuildResult(used=False, explanation="No result table was produced, so there is nothing to chart.")

    rows = table[:MAX_CHART_ROWS]
    columns = list(rows[0].keys())

    x_col = next((c for c in columns if _column_kind(rows, c) == "categorical"), None)
    if x_col is None:
        x_col = next((c for c in columns if _column_kind(rows, c) == "period"), None)
    if x_col is None:
        x_col = next((c for c in columns if _is_number(rows[0].get(c))), columns[0])

    numeric_cols = [c for c in columns if _column_kind(rows, c) == "numeric"]
    y_col = numeric_cols[0] if numeric_cols else None

    kinds = {c: _column_kind(rows, c) for c in columns}
    x_kind = kinds.get(x_col, "categorical")

    if y_col is None:
        return ChartBuildResult(
            used=False,
            explanation="The computed table has no numeric column to plot on the y-axis.",
            suggestion="Add an aggregate metric (for example total revenue) to the question.",
            data=rows,
        )

    if len(rows) < 2:
        return ChartBuildResult(
            used=False,
            explanation="The computed table has a single row, so a chart would add no information.",
            suggestion="Group the result by a dimension to obtain multiple points.",
            data=rows,
        )

    chart_type = (requested_type or "").strip().lower() or None
    if chart_type is not None and chart_type not in {t.value for t in ChartType}:
        return ChartBuildResult(
            used=False,
            explanation=f"Chart type '{chart_type}' is not in the allowlisted set.",
            suggestion=f"Allowed chart types: {', '.join(t.value for t in ChartType)}.",
        )

    if chart_type == "line" and not _ordered(x_col, x_kind):
        return ChartBuildResult(
            used=False,
            explanation=(
                f"A line chart needs an ordered x-axis, but '{x_col}' is {x_kind}."
            ),
            suggestion=f"Use a bar chart with x='{x_col}', or plot a time series by month.",
        )
    if chart_type == "scatter":
        if len(numeric_cols) < 2:
            return ChartBuildResult(
                used=False,
                explanation="A scatter plot needs two numeric columns; the computed table has only one.",
                suggestion="Use a bar chart with a categorical x-axis instead.",
            )
        x_col, y_col = numeric_cols[0], numeric_cols[1]
    if chart_type == "pie":
        cardinality = len({str(r.get(x_col)) for r in rows})
        if cardinality > MAX_CHART_CATEGORIES:
            return ChartBuildResult(
                used=False,
                explanation=(
                    f"A pie chart is unsuitable for '{x_col}', which has {cardinality} distinct values "
                    f"(limit {MAX_CHART_CATEGORIES})."
                ),
                suggestion="Use a bar chart instead, or restrict the question to the top categories.",
            )

    if chart_type is None:
        chart_type = "line" if _ordered(x_col, x_kind) else "bar"

    spec = {
        "type": chart_type,
        "x": x_col,
        "y": y_col,
        "title": title or f"{y_col} by {x_col}",
        "rows": len(rows),
    }

    return ChartBuildResult(
        used=True,
        chart_type=chart_type,
        spec=spec,
        data=rows,
        x_column=x_col,
        y_column=y_col,
        explanation=f"Built a {chart_type} chart from the computed table ({x_col} vs {y_col}).",
    )


def infer_example_questions(schema_ctx: SchemaContext, max_questions: int = 5) -> List[str]:
    """Generate example questions from the active dataset's real columns."""
    numeric = list(schema_ctx.numeric_columns)
    categorical = list(schema_ctx.categorical_columns)
    date_cols = list(schema_ctx.date_columns)

    metric = next((c for c in numeric if c in ("revenue", "sales", "amount")), None) or (numeric[0] if numeric else None)
    dimension = next((c for c in categorical if c in ("category", "product", "segment")), None) or (categorical[0] if categorical else None)

    examples: List[str] = []
    if metric and dimension:
        examples.append(f"What is total {metric} by {dimension}?")
        examples.append(f"Which {dimension} contributed most to {metric}?")
    if metric:
        examples.append(f"What is the average {metric}?")
    if metric and date_cols:
        examples.append(f"What is the monthly {metric} trend?")
        examples.append(f"Compare {metric} in January vs February")
        if dimension:
            examples.append(f"Why did {metric} decline in March for the largest {dimension}?")
    if metric and dimension:
        examples.append(f"Show a bar chart of {metric} by {dimension}")

    # De-duplicate while preserving order, then bound.
    seen = set()
    unique: List[str] = []
    for q in examples:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique[:max_questions]
