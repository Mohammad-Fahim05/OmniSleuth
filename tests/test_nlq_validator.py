"""Unit tests for Phase A: NLQ AnalysisPlan models and PlanValidator."""

import pytest
from pydantic import ValidationError

from omnisleuth.nlq.context import ColumnContext, SchemaContext
from omnisleuth.nlq.models import (
    AnalysisPlan,
    ChartSpec,
    ChartType,
    FilterOp,
    FilterSpec,
    MetricAgg,
    MetricSpec,
    PlanIntent,
    SortDirection,
    SortSpec,
    TimeGrain,
    TimeSpec,
)
from omnisleuth.nlq.validator import PlanValidator


@pytest.fixture
def sample_schema_ctx() -> SchemaContext:
    """Fixture providing a realistic SchemaContext."""
    return SchemaContext(
        dataset_id="ds_test_sales",
        dataset_name="Test Sales 2024",
        row_count=1000,
        column_count=5,
        columns={
            "order_id": ColumnContext(name="order_id", data_type="identifier", dtype="object", unique_count=1000),
            "order_date": ColumnContext(name="order_date", data_type="datetime", dtype="datetime64[ns]", unique_count=365),
            "category": ColumnContext(name="category", data_type="categorical", dtype="object", unique_count=4, sample_values=["Electronics", "Furniture", "Clothing", "Home"]),
            "region": ColumnContext(name="region", data_type="categorical", dtype="object", unique_count=25),  # High cardinality
            "revenue": ColumnContext(name="revenue", data_type="numeric", dtype="float64", unique_count=950, min_value=10.0, max_value=5000.0),
        },
        date_columns=["order_date"],
        numeric_columns=["revenue"],
        categorical_columns=["category", "region"],
        min_date="2024-01-01",
        max_date="2024-12-31",
    )


def test_valid_plan_passes(sample_schema_ctx):
    """Verify that a well-formed query plan passes validation untouched."""
    validator = PlanValidator()
    plan = AnalysisPlan(
        intent=PlanIntent.QUERY,
        metrics=[MetricSpec(column="revenue", agg=MetricAgg.SUM)],
        group_by=["category"],
        filters=[FilterSpec(column="revenue", op=FilterOp.GT, value=100.0)],
        sort=SortSpec(by="revenue", direction=SortDirection.DESC),
        limit=10,
        chart=ChartSpec(type=ChartType.BAR, x="category", y="revenue"),
    )

    validated = validator.validate(plan, sample_schema_ctx)
    assert validated.intent == PlanIntent.QUERY
    assert validated.unsupported_reason is None
    assert len(validated.metrics) == 1
    assert validated.metrics[0].column == "revenue"


def test_bad_column_in_metric(sample_schema_ctx):
    """Assert that a non-existent column in metrics returns unsupported listing available columns."""
    validator = PlanValidator()
    plan = AnalysisPlan(
        intent=PlanIntent.QUERY,
        metrics=[MetricSpec(column="profit", agg=MetricAgg.SUM)],
    )

    validated = validator.validate(plan, sample_schema_ctx)
    assert validated.intent == PlanIntent.UNSUPPORTED
    assert "profit" in validated.unsupported_reason
    assert "revenue" in validated.unsupported_reason  # Lists available columns
    assert validated.confidence == 0.0


def test_bad_column_in_group_by(sample_schema_ctx):
    """Assert that a non-existent column in group_by returns unsupported."""
    validator = PlanValidator()
    plan = AnalysisPlan(
        intent=PlanIntent.QUERY,
        metrics=[MetricSpec(column="revenue", agg=MetricAgg.SUM)],
        group_by=["department"],
    )

    validated = validator.validate(plan, sample_schema_ctx)
    assert validated.intent == PlanIntent.UNSUPPORTED
    assert "department" in validated.unsupported_reason


def test_bad_column_in_filter(sample_schema_ctx):
    """Assert that a non-existent column in filter returns unsupported."""
    validator = PlanValidator()
    plan = AnalysisPlan(
        intent=PlanIntent.QUERY,
        metrics=[MetricSpec(column="revenue", agg=MetricAgg.SUM)],
        filters=[FilterSpec(column="customer_tier", op=FilterOp.EQ, value="Gold")],
    )

    validated = validator.validate(plan, sample_schema_ctx)
    assert validated.intent == PlanIntent.UNSUPPORTED
    assert "customer_tier" in validated.unsupported_reason


def test_bad_dtype_sum_of_text(sample_schema_ctx):
    """Assert that mathematical aggregation on non-numeric column is rejected."""
    validator = PlanValidator()
    plan = AnalysisPlan(
        intent=PlanIntent.QUERY,
        metrics=[MetricSpec(column="category", agg=MetricAgg.SUM)],
    )

    validated = validator.validate(plan, sample_schema_ctx)
    assert validated.intent == PlanIntent.UNSUPPORTED
    assert "Cannot compute sum of non-numeric column 'category'" in validated.unsupported_reason
    assert "revenue" in validated.unsupported_reason  # Recommends available numeric columns


def test_allowed_agg_count_and_nunique_on_text(sample_schema_ctx):
    """Assert that count and nunique are allowed on text/categorical columns."""
    validator = PlanValidator()
    plan = AnalysisPlan(
        intent=PlanIntent.QUERY,
        metrics=[
            MetricSpec(column="category", agg=MetricAgg.COUNT),
            MetricSpec(column="category", agg=MetricAgg.NUNIQUE),
        ],
    )

    validated = validator.validate(plan, sample_schema_ctx)
    assert validated.intent == PlanIntent.QUERY
    assert validated.unsupported_reason is None


def test_bad_filter_operator_contains_on_numeric(sample_schema_ctx):
    """Assert that text operators on numeric columns are rejected."""
    validator = PlanValidator()
    plan = AnalysisPlan(
        intent=PlanIntent.QUERY,
        metrics=[MetricSpec(column="revenue", agg=MetricAgg.SUM)],
        filters=[FilterSpec(column="revenue", op=FilterOp.CONTAINS, value="100")],
    )

    validated = validator.validate(plan, sample_schema_ctx)
    assert validated.intent == PlanIntent.UNSUPPORTED
    assert "contains" in validated.unsupported_reason


def test_bad_filter_in_list_size_limit(sample_schema_ctx):
    """Assert that unbounded 'in' lists are rejected."""
    validator = PlanValidator(max_in_list_size=10)
    plan = AnalysisPlan(
        intent=PlanIntent.QUERY,
        metrics=[MetricSpec(column="revenue", agg=MetricAgg.SUM)],
        filters=[FilterSpec(column="category", op=FilterOp.IN, value=[f"Cat_{i}" for i in range(15)])],
    )

    validated = validator.validate(plan, sample_schema_ctx)
    assert validated.intent == PlanIntent.UNSUPPORTED
    assert "exceeds maximum allowed size" in validated.unsupported_reason


def test_bad_filter_between_bounds(sample_schema_ctx):
    """Assert that 'between' operator requires exactly 2 bounds."""
    validator = PlanValidator()
    plan = AnalysisPlan(
        intent=PlanIntent.QUERY,
        metrics=[MetricSpec(column="revenue", agg=MetricAgg.SUM)],
        filters=[FilterSpec(column="revenue", op=FilterOp.BETWEEN, value=[10.0])],  # Only 1 item
    )

    validated = validator.validate(plan, sample_schema_ctx)
    assert validated.intent == PlanIntent.UNSUPPORTED
    assert "requires exactly two boundary values" in validated.unsupported_reason


def test_date_range_completely_outside_dataset(sample_schema_ctx):
    """Assert that date ranges completely outside dataset range are rejected."""
    validator = PlanValidator()
    plan = AnalysisPlan(
        intent=PlanIntent.TREND,
        metrics=[MetricSpec(column="revenue", agg=MetricAgg.SUM)],
        time=TimeSpec(
            date_column="order_date",
            grain=TimeGrain.MONTH,
            start="2020-01-01",
            end="2020-12-31",  # Dataset is only in 2024
        ),
    )

    validated = validator.validate(plan, sample_schema_ctx)
    assert validated.intent == PlanIntent.UNSUPPORTED
    assert "outside the dataset's" in validated.unsupported_reason


def test_limit_out_of_bounds(sample_schema_ctx):
    """Assert that limit > 50 is rejected by validator."""
    validator = PlanValidator(max_limit=50)
    # Testing validator check when limit is invalid
    with pytest.raises(ValidationError):
        # Pydantic schema constraint check
        AnalysisPlan(
            intent=PlanIntent.QUERY,
            metrics=[MetricSpec(column="revenue", agg=MetricAgg.SUM)],
            limit=100,
        )


def test_bad_chart_line_on_categorical_axis(sample_schema_ctx):
    """Assert that line charts with unordered categorical x-axis are rejected."""
    validator = PlanValidator()
    plan = AnalysisPlan(
        intent=PlanIntent.QUERY,
        metrics=[MetricSpec(column="revenue", agg=MetricAgg.SUM)],
        group_by=["category"],
        chart=ChartSpec(type=ChartType.LINE, x="category", y="revenue"),
    )

    validated = validator.validate(plan, sample_schema_ctx)
    assert validated.intent == PlanIntent.UNSUPPORTED
    assert "Line chart requires an ordered x-axis" in validated.unsupported_reason


def test_bad_chart_scatter_with_non_numeric(sample_schema_ctx):
    """Assert that scatter plots require numeric x and y axes."""
    validator = PlanValidator()
    plan = AnalysisPlan(
        intent=PlanIntent.QUERY,
        chart=ChartSpec(type=ChartType.SCATTER, x="category", y="revenue"),
    )

    validated = validator.validate(plan, sample_schema_ctx)
    assert validated.intent == PlanIntent.UNSUPPORTED
    assert "Scatter plot requires two numeric columns" in validated.unsupported_reason


def test_bad_chart_pie_on_high_cardinality(sample_schema_ctx):
    """Assert that pie charts on high-cardinality columns are rejected."""
    validator = PlanValidator()
    plan = AnalysisPlan(
        intent=PlanIntent.QUERY,
        metrics=[MetricSpec(column="revenue", agg=MetricAgg.SUM)],
        group_by=["region"],  # region has 25 unique values
        chart=ChartSpec(type=ChartType.PIE, x="region", y="revenue"),
    )

    validated = validator.validate(plan, sample_schema_ctx)
    assert validated.intent == PlanIntent.UNSUPPORTED
    assert "Pie chart is unsuitable" in validated.unsupported_reason


def test_unknown_fields_rejected():
    """Assert that unknown / arbitrary fields in AnalysisPlan and sub-models are rejected."""
    with pytest.raises(ValidationError):
        AnalysisPlan(
            intent=PlanIntent.QUERY,
            arbitrary_field="malicious_payload",  # Should fail extra="forbid"
        )

    with pytest.raises(ValidationError):
        MetricSpec(
            column="revenue",
            agg=MetricAgg.SUM,
            extra_param="unexpected",
        )

    with pytest.raises(ValidationError):
        FilterSpec(
            column="revenue",
            op=FilterOp.EQ,
            value=100,
            injected_sql="SELECT * FROM users",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test C: PlanValidator rejects identifier or high-cardinality group_by columns
# ─────────────────────────────────────────────────────────────────────────────

def test_reject_group_by_identifier_column(sample_schema_ctx):
    """Test C: PlanValidator rejects group_by on columns marked as identifier."""
    validator = PlanValidator()
    plan = AnalysisPlan(
        intent=PlanIntent.QUERY,
        metrics=[MetricSpec(column="revenue", agg=MetricAgg.MEAN)],
        group_by=["order_id"],
    )

    validated = validator.validate(plan, sample_schema_ctx)
    assert validated.intent == PlanIntent.UNSUPPORTED
    assert "order_id" in validated.unsupported_reason
    assert "identifier" in validated.unsupported_reason.lower()


def test_reject_group_by_exceeds_cardinality_limit(sample_schema_ctx):
    """Test C: PlanValidator rejects group_by on columns exceeding default cardinality limit (30)."""
    # Add a categorical column with 50 distinct values
    sample_schema_ctx.columns["city"] = ColumnContext(
        name="city",
        data_type="categorical",
        dtype="object",
        unique_count=50,
    )
    validator = PlanValidator()
    plan = AnalysisPlan(
        intent=PlanIntent.QUERY,
        metrics=[MetricSpec(column="revenue", agg=MetricAgg.SUM)],
        group_by=["city"],
    )

    validated = validator.validate(plan, sample_schema_ctx)
    assert validated.intent == PlanIntent.UNSUPPORTED
    assert "city" in validated.unsupported_reason
    assert "cardinality" in validated.unsupported_reason.lower() or "distinct" in validated.unsupported_reason.lower()
    assert "30" in validated.unsupported_reason


def test_configurable_group_by_cardinality(sample_schema_ctx):
    """Configurable max_group_by_cardinality allows tightening or loosening the limit."""
    # region has 25 unique values
    strict_validator = PlanValidator(max_group_by_cardinality=20)
    lenient_validator = PlanValidator(max_group_by_cardinality=50)

    plan = AnalysisPlan(
        intent=PlanIntent.QUERY,
        metrics=[MetricSpec(column="revenue", agg=MetricAgg.SUM)],
        group_by=["region"],
    )

    rejected = strict_validator.validate(plan, sample_schema_ctx)
    assert rejected.intent == PlanIntent.UNSUPPORTED
    assert "region" in rejected.unsupported_reason
    assert "20" in rejected.unsupported_reason

    accepted = lenient_validator.validate(plan, sample_schema_ctx)
    assert accepted.intent == PlanIntent.QUERY
    assert accepted.unsupported_reason is None


def test_legitimate_categorical_group_by_allowed(sample_schema_ctx):
    """Legitimate categorical columns with cardinality within limits remain valid."""
    validator = PlanValidator()
    plan = AnalysisPlan(
        intent=PlanIntent.QUERY,
        metrics=[MetricSpec(column="revenue", agg=MetricAgg.SUM)],
        group_by=["category"],
    )

    validated = validator.validate(plan, sample_schema_ctx)
    assert validated.intent == PlanIntent.QUERY
    assert validated.unsupported_reason is None


# ─────────────────────────────────────────────────────────────────────────────
# Defect A: Legitimate date grouping for trends
# ─────────────────────────────────────────────────────────────────────────────

def test_legitimate_monthly_trend_group_by_order_date_passes(sample_schema_ctx):
    """Defect A: A monthly trend plan grouping by order_date buckets into 12 months (<= 30) and passes."""
    validator = PlanValidator()
    plan = AnalysisPlan(
        intent=PlanIntent.TREND,
        metrics=[MetricSpec(column="revenue", agg=MetricAgg.SUM)],
        group_by=["order_date"],
        time=TimeSpec(date_column="order_date", grain=TimeGrain.MONTH),
    )

    validated = validator.validate(plan, sample_schema_ctx)
    assert validated.intent == PlanIntent.TREND
    assert validated.unsupported_reason is None


def test_legitimate_monthly_trend_without_explicit_timespec_passes(sample_schema_ctx):
    """Defect A: A trend plan with group_by=['order_date'] defaults to monthly bucketing and passes."""
    validator = PlanValidator()
    plan = AnalysisPlan(
        intent=PlanIntent.TREND,
        metrics=[MetricSpec(column="revenue", agg=MetricAgg.SUM)],
        group_by=["order_date"],
    )

    validated = validator.validate(plan, sample_schema_ctx)
    assert validated.intent == PlanIntent.TREND
    assert validated.unsupported_reason is None
    assert validated.time is not None
    assert validated.time.date_column == "order_date"
    assert validated.time.grain == TimeGrain.MONTH


def test_raw_date_grouping_in_query_plan_rejected(sample_schema_ctx):
    """Defect A: Raw-date grouping in a query plan (365 distinct dates > 30) is rejected."""
    validator = PlanValidator()
    plan = AnalysisPlan(
        intent=PlanIntent.QUERY,
        metrics=[MetricSpec(column="revenue", agg=MetricAgg.SUM)],
        group_by=["order_date"],
    )

    validated = validator.validate(plan, sample_schema_ctx)
    assert validated.intent == PlanIntent.UNSUPPORTED
    assert "order_date" in validated.unsupported_reason
    assert "cardinality" in validated.unsupported_reason.lower() or "distinct" in validated.unsupported_reason.lower()


def test_daily_trend_exceeding_cardinality_rejected(sample_schema_ctx):
    """Defect A: Daily trend plan produces 365 daily groups (> 30) and is rejected as unbounded raw grouping."""
    validator = PlanValidator()
    plan = AnalysisPlan(
        intent=PlanIntent.TREND,
        metrics=[MetricSpec(column="revenue", agg=MetricAgg.SUM)],
        group_by=["order_date"],
        time=TimeSpec(date_column="order_date", grain=TimeGrain.DAY),
    )

    validated = validator.validate(plan, sample_schema_ctx)
    assert validated.intent == PlanIntent.UNSUPPORTED
    assert "order_date" in validated.unsupported_reason
