"""Unit tests for Phase 3 - Deterministic Analytics Engine."""

from pathlib import Path
import pandas as pd
import pytest
from omnisleuth.analytics import (
    AggregationFunction,
    ChartType,
    DeterministicAnalyticsEngine,
    FilterCondition,
)


@pytest.fixture
def sales_df() -> pd.DataFrame:
    """Create sample sales dataframe for analytics testing."""
    data = {
        "order_id": ["ORD-1", "ORD-2", "ORD-3", "ORD-4", "ORD-5", "ORD-6", "ORD-7", "ORD-8"],
        "order_date": [
            "2024-01-10", "2024-01-20",
            "2024-02-05", "2024-02-25",
            "2024-03-01", "2024-03-15",
            "2024-04-10", "2024-04-20"
        ],
        "region": ["North", "South", "North", "South", "North", "South", "North", "South"],
        "category": ["Electronics", "Furniture", "Electronics", "Furniture", "Electronics", "Furniture", "Electronics", "Furniture"],
        "revenue": [1000.0, 500.0, 1200.0, 600.0, 400.0, 300.0, 1500.0, 700.0],
        "units": [10, 5, 12, 6, 4, 3, 15, 7],
    }
    df = pd.DataFrame(data)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def test_filters(sales_df: pd.DataFrame):
    """Test filter application with various operators."""
    engine = DeterministicAnalyticsEngine()

    # Equal
    f1 = [FilterCondition(column="region", operator="==", value="North")]
    res1 = engine.apply_filters(sales_df, f1)
    assert len(res1) == 4
    assert (res1["region"] == "North").all()

    # Greater than
    f2 = [FilterCondition(column="revenue", operator=">", value=1000.0)]
    res2 = engine.apply_filters(sales_df, f2)
    assert len(res2) == 2  # 1200, 1500

    # In list
    f3 = [FilterCondition(column="order_id", operator="in", value=["ORD-1", "ORD-3"])]
    res3 = engine.apply_filters(sales_df, f3)
    assert len(res3) == 2

    # Between
    f4 = [FilterCondition(column="revenue", operator="between", value=(500.0, 1000.0))]
    res4 = engine.apply_filters(sales_df, f4)
    assert len(res4) == 4  # 1000, 500, 600, 700


def test_aggregation(sales_df: pd.DataFrame):
    """Test group-by and overall aggregation."""
    engine = DeterministicAnalyticsEngine()

    # Grand total sum
    res_total = engine.aggregate(sales_df, metric_column="revenue", agg_func=AggregationFunction.SUM)
    assert res_total.grand_total == 6200.0

    # Group by category sum
    res_cat = engine.aggregate(
        sales_df,
        metric_column="revenue",
        agg_func=AggregationFunction.SUM,
        group_by_columns=["category"],
    )
    assert res_cat.total_records == 2
    cat_map = {r["category"]: r["revenue"] for r in res_cat.records}
    assert cat_map["Electronics"] == 4100.0  # 1000 + 1200 + 400 + 1500
    assert cat_map["Furniture"] == 2100.0    # 500 + 600 + 300 + 700


def test_calculate_trend(sales_df: pd.DataFrame):
    """Test monthly trend aggregation and period-over-period percentage change."""
    engine = DeterministicAnalyticsEngine()
    trend = engine.calculate_trend(
        df=sales_df,
        date_column="order_date",
        metric_column="revenue",
        frequency="M",
        agg_func=AggregationFunction.SUM,
    )

    # 4 months: 2024-01 (1500), 2024-02 (1800), 2024-03 (700), 2024-04 (2200)
    assert len(trend.time_series) == 4
    m1, m2, m3, m4 = trend.time_series
    assert m1.value == 1500.0
    assert m2.value == 1800.0
    assert m2.delta_from_prev == 300.0
    assert m2.change_pct_from_prev == 20.0

    # March drop from Feb: 1800 -> 700 = -1100 (-61.11%)
    assert m3.value == 700.0
    assert m3.delta_from_prev == -1100.0
    assert m3.change_pct_from_prev == pytest.approx(-61.11, abs=0.1)


def test_compare_segments(sales_df: pd.DataFrame):
    """Test segment comparison with ranking and share percentages."""
    engine = DeterministicAnalyticsEngine()
    comp = engine.compare_segments(
        df=sales_df,
        segment_column="category",
        metric_column="revenue",
    )

    assert comp.total_metric_value == 6200.0
    assert len(comp.segments) == 2
    top = comp.segments[0]
    assert top.segment == "Electronics"
    assert top.value == 4100.0
    assert top.share_percentage == pytest.approx(66.13, abs=0.1)
    assert top.rank == 1


def test_segment_contribution(sales_df: pd.DataFrame):
    """Test mathematical segment contribution between February and March."""
    engine = DeterministicAnalyticsEngine()

    feb_filter = [
        FilterCondition(column="order_date", operator=">=", value="2024-02-01"),
        FilterCondition(column="order_date", operator="<=", value="2024-02-29"),
    ]
    mar_filter = [
        FilterCondition(column="order_date", operator=">=", value="2024-03-01"),
        FilterCondition(column="order_date", operator="<=", value="2024-03-31"),
    ]

    res = engine.calculate_segment_contribution(
        df=sales_df,
        segment_column="category",
        metric_column="revenue",
        period_1_filters=feb_filter,
        period_2_filters=mar_filter,
        period_1_label="Feb 2024",
        period_2_label="Mar 2024",
    )

    # Feb total: 1800 (Elec: 1200, Furn: 600)
    # Mar total: 700 (Elec: 400, Furn: 300)
    # Delta: -1100
    assert res.total_period_1 == 1800.0
    assert res.total_period_2 == 700.0
    assert res.total_delta == -1100.0

    elec_contrib = next(c for c in res.contributions if c.segment == "Electronics")
    furn_contrib = next(c for c in res.contributions if c.segment == "Furniture")

    # Elec: 1200 -> 400 = -800 delta. Contrib % = -800 / -1100 = 72.73%
    assert elec_contrib.delta == -800.0
    assert elec_contrib.contribution_to_total_delta_pct == pytest.approx(72.73, abs=0.1)

    # Furn: 600 -> 300 = -300 delta. Contrib % = -300 / -1100 = 27.27%
    assert furn_contrib.delta == -300.0
    assert furn_contrib.contribution_to_total_delta_pct == pytest.approx(27.27, abs=0.1)


def test_metric_decomposition(sales_df: pd.DataFrame):
    """Test multiplicative decomposition of Revenue into Orders and AOV."""
    engine = DeterministicAnalyticsEngine()

    feb_filter = [
        FilterCondition(column="order_date", operator=">=", value="2024-02-01"),
        FilterCondition(column="order_date", operator="<=", value="2024-02-29"),
    ]
    mar_filter = [
        FilterCondition(column="order_date", operator=">=", value="2024-03-01"),
        FilterCondition(column="order_date", operator="<=", value="2024-03-31"),
    ]

    decomp = engine.decompose_revenue_orders_aov(
        df=sales_df,
        revenue_column="revenue",
        order_id_column="order_id",
        period_1_filters=feb_filter,
        period_2_filters=mar_filter,
        period_1_label="Feb",
        period_2_label="Mar",
    )

    assert decomp.target_delta == -1100.0
    assert len(decomp.factor_effects) == 2
    # In Feb: 2 orders, Rev 1800, AOV 900
    # In Mar: 2 orders, Rev 700, AOV 350
    # Volume delta = 0, AOV delta = -550
    vol_effect = decomp.factor_effects[0]
    aov_effect = decomp.factor_effects[1]

    assert vol_effect.delta == 0.0
    assert vol_effect.estimated_impact_on_target == 0.0
    assert aov_effect.delta == -550.0
    assert aov_effect.estimated_impact_on_target == -1100.0
    assert aov_effect.impact_percentage == 100.0


def test_chart_generation():
    """Test declarative chart specification generation."""
    engine = DeterministicAnalyticsEngine()
    spec = engine.generate_chart_spec(
        chart_type=ChartType.LINE,
        title="Monthly Revenue Trend",
        data=[{"month": "2024-01", "rev": 100}, {"month": "2024-02", "rev": 200}],
        x_column="month",
        y_column="rev",
    )
    assert spec.chart_type == ChartType.LINE
    assert spec.title == "Monthly Revenue Trend"
    assert len(spec.data) == 2
