"""Unit tests for Phase 5 - Allowlisted Tool Layer."""

from pathlib import Path
import pandas as pd
import pytest
from omnisleuth.analytics.models import AggregationFunction
from omnisleuth.core.exceptions import ToolExecutionError
from omnisleuth.ingestion import CSVLoader
from omnisleuth.tools import (
    BaseToolOutput,
    ToolRegistry,
    get_default_tool_registry,
)


@pytest.fixture
def sales_context(tmp_path: Path):
    """Load sample sales data and return (df, metadata, registry)."""
    csv_file = tmp_path / "sales.csv"
    csv_file.write_text(
        """order_id,order_date,category,region,revenue
ORD-1,2024-01-10,Electronics,North,1000.0
ORD-2,2024-01-20,Furniture,South,500.0
ORD-3,2024-02-05,Electronics,North,1200.0
ORD-4,2024-02-25,Furniture,South,600.0
ORD-5,2024-03-01,Electronics,North,400.0
ORD-6,2024-03-15,Furniture,South,300.0
""",
        encoding="utf-8",
    )
    loader = CSVLoader()
    df, metadata = loader.load_csv(csv_file)
    registry = get_default_tool_registry()
    return df, metadata, registry


def test_tool_registry_allowlist(sales_context):
    """Verify only allowlisted tools are accessible."""
    df, metadata, registry = sales_context
    assert len(registry.list_tools()) == 8

    # Unregistered tool raises error
    with pytest.raises(ToolExecutionError, match="not in allowlist"):
        registry.get("unapproved_arbitrary_code_runner")


def test_tool_profile_dataset(sales_context):
    """Test profile_dataset tool."""
    df, metadata, registry = sales_context
    tool = registry.get("profile_dataset")
    out = tool.run({}, df, metadata)

    assert out.success is True
    assert out.data is not None
    assert out.data.row_count == 6


def test_tool_aggregate_metric(sales_context):
    """Test aggregate_metric tool with validation and execution."""
    df, metadata, registry = sales_context
    tool = registry.get("aggregate_metric")

    params = {
        "metric_column": "revenue",
        "agg_func": "sum",
        "group_by_columns": ["category"],
    }
    out = tool.run(params, df, metadata)
    assert out.success is True
    assert out.data.grand_total == 4000.0


def test_tool_calculate_trend(sales_context):
    """Test calculate_trend tool."""
    df, metadata, registry = sales_context
    tool = registry.get("calculate_trend")

    params = {
        "date_column": "order_date",
        "metric_column": "revenue",
        "frequency": "M",
    }
    out = tool.run(params, df, metadata)
    assert out.success is True
    assert len(out.data.time_series) == 3


def test_tool_compare_segments(sales_context):
    """Test compare_segments tool."""
    df, metadata, registry = sales_context
    tool = registry.get("compare_segments")

    params = {
        "segment_column": "category",
        "metric_column": "revenue",
    }
    out = tool.run(params, df, metadata)
    assert out.success is True
    assert len(out.data.segments) == 2


def test_tool_segment_contribution(sales_context):
    """Test segment_contribution tool."""
    df, metadata, registry = sales_context
    tool = registry.get("segment_contribution")

    params = {
        "segment_column": "category",
        "metric_column": "revenue",
        "period_1_filters": [{"column": "order_date", "operator": "<=", "value": "2024-02-29"}],
        "period_2_filters": [{"column": "order_date", "operator": ">=", "value": "2024-03-01"}],
    }
    out = tool.run(params, df, metadata)
    assert out.success is True
    assert out.data.total_delta < 0


def test_tool_decompose_metric(sales_context):
    """Test decompose_metric tool."""
    df, metadata, registry = sales_context
    tool = registry.get("decompose_metric")

    params = {
        "revenue_column": "revenue",
        "order_id_column": "order_id",
        "period_1_filters": [{"column": "order_date", "operator": "<=", "value": "2024-02-29"}],
        "period_2_filters": [{"column": "order_date", "operator": ">=", "value": "2024-03-01"}],
    }
    out = tool.run(params, df, metadata)
    assert out.success is True
    assert len(out.data.factor_effects) == 2


def test_tool_detect_anomalies(sales_context):
    """Test detect_anomalies tool."""
    df, metadata, registry = sales_context
    tool = registry.get("detect_anomalies")

    params = {
        "feature_columns": ["revenue"],
        "method": "iqr",
    }
    out = tool.run(params, df, metadata)
    assert out.success is True
    assert out.data.total_records == 6


def test_tool_create_chart(sales_context):
    """Test create_chart tool."""
    df, metadata, registry = sales_context
    tool = registry.get("create_chart")

    params = {
        "chart_type": "bar",
        "title": "Category Revenue",
        "data": [{"category": "Electronics", "revenue": 2600}, {"category": "Furniture", "revenue": 1400}],
        "x_column": "category",
        "y_column": "revenue",
    }
    out = tool.run(params, df, metadata)
    assert out.success is True
    assert out.data.title == "Category Revenue"


def test_tool_error_handling(sales_context):
    """Test graceful error response when invalid column name is provided."""
    df, metadata, registry = sales_context
    tool = registry.get("aggregate_metric")

    params = {
        "metric_column": "non_existent_column_xyz",
    }
    out = tool.run(params, df, metadata)
    assert out.success is False
    assert "not found" in out.error
