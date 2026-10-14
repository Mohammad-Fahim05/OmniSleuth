"""Tests that a categorical column named literally in a question resolves.

Covers group-by questions (e.g. "Total revenue by product") and contribution
questions (e.g. "Which product contributed most to revenue?") for the column
names ``category``, ``region`` and ``product``. Every assertion compares the
planner/executor output against an independent pandas groupby calculation.
"""

import pandas as pd
import pytest

from omnisleuth.ingestion.models import ColumnSchema, ColumnType, DatasetMetadata
from omnisleuth.nlq.context import build_schema_context
from omnisleuth.nlq.executor import PlanExecutor
from omnisleuth.nlq.models import PlanIntent
from omnisleuth.nlq.rule_planner import RulePlanner
from omnisleuth.nlq.validator import PlanValidator

GROUP_BY_COLUMNS = ["category", "region", "product"]


@pytest.fixture
def resolution_dataset():
    """Hand-built DataFrame with three categorical columns and one metric."""
    data = {
        "order_id": [f"ORD-{i:03d}" for i in range(1, 25)],
        "order_date": [f"2024-01-{(i % 28) + 1:02d}" for i in range(1, 25)],
        "category": [
            "Electronics", "Furniture", "Electronics", "Clothing",
            "Furniture", "Electronics", "Clothing", "Furniture",
            "Electronics", "Clothing", "Furniture", "Electronics",
            "Clothing", "Furniture", "Electronics", "Clothing",
            "Furniture", "Electronics", "Clothing", "Furniture",
            "Electronics", "Clothing", "Furniture", "Electronics",
        ],
        "product": [
            "Laptop", "Desk", "Phone", "Shirt", "Chair", "Tablet", "Pants", "Sofa",
            "Monitor", "Jacket", "Shelf", "Keyboard", "Scarf", "Bed", "Mouse", "Hat",
            "Wardrobe", "Speaker", "Gloves", "Lamp", "Camera", "Shoes", "Bench", "Drone",
        ],
        "region": [
            "North", "South", "North", "East", "West", "South", "North", "East",
            "West", "North", "South", "North", "East", "West", "South",
            "North", "East", "West", "North", "South",
            "North", "East", "West", "South",
        ],
        "revenue": [
            500.0, 1200.0, 300.0, 150.0, 800.0, 950.0, 200.0, 1100.0,
            600.0, 250.0, 1300.0, 450.0, 180.0, 750.0, 850.0,
            220.0, 1050.0, 700.0, 190.0, 900.0,
            650.0, 320.0, 140.0, 430.0,
        ],
    }
    df = pd.DataFrame(data)

    metadata = DatasetMetadata(
        dataset_id="ds_col_resolution",
        name="Column Resolution Dataset",
        file_path="column_resolution.csv",
        version_hash="hash_col_resolution",
        row_count=len(df),
        column_count=len(df.columns),
        duplicate_rows=0,
        columns=[
            ColumnSchema(name="order_id", data_type=ColumnType.IDENTIFIER, dtype="object", unique_count=24),
            ColumnSchema(name="order_date", data_type=ColumnType.DATETIME, dtype="object", unique_count=28),
            ColumnSchema(name="category", data_type=ColumnType.CATEGORICAL, dtype="object", unique_count=3, sample_values=["Electronics", "Furniture", "Clothing"]),
            ColumnSchema(name="product", data_type=ColumnType.CATEGORICAL, dtype="object", unique_count=24, sample_values=["Laptop", "Desk", "Phone"]),
            ColumnSchema(name="region", data_type=ColumnType.CATEGORICAL, dtype="object", unique_count=4, sample_values=["North", "South", "East", "West"]),
            ColumnSchema(name="revenue", data_type=ColumnType.NUMERIC, dtype="float64", unique_count=24),
        ],
    )
    return df, metadata, build_schema_context(df, metadata)


def _plan(question, schema_ctx):
    """Plan + validate a question, returning the validated plan."""
    planner = RulePlanner()
    validator = PlanValidator()
    plan = planner.plan(question, schema_ctx)
    return validator.validate(plan, schema_ctx)


@pytest.mark.parametrize("col", GROUP_BY_COLUMNS)
def test_literally_named_column_resolves_in_group_by_question(resolution_dataset, col):
    """A column named with an explicit grouping preposition resolves as the group-by."""
    df, metadata, schema_ctx = resolution_dataset
    question = f"Total revenue by {col}"
    plan = _plan(question, schema_ctx)

    assert plan.intent == PlanIntent.QUERY
    assert plan.group_by == [col], f"{col} was not resolved for question: {question}"

    result = PlanExecutor().execute(plan, df, metadata, question)
    assert result.status == "answered"

    expected = df.groupby(col)["revenue"].sum()
    assert any(
        row.get(col) == expected.idxmax() and row.get("revenue") == pytest.approx(expected.max(), abs=0.01)
        for row in result.result_table
    )


@pytest.mark.parametrize("col", GROUP_BY_COLUMNS)
def test_literally_named_column_resolves_in_average_group_by_question(resolution_dataset, col):
    """Averaging questions also resolve the literally named group-by column."""
    df, metadata, schema_ctx = resolution_dataset
    question = f"Average revenue by {col}"
    plan = _plan(question, schema_ctx)

    assert plan.intent == PlanIntent.QUERY
    assert plan.group_by == [col], f"{col} was not resolved for question: {question}"

    result = PlanExecutor().execute(plan, df, metadata, question)
    assert result.status == "answered"

    expected = df.groupby(col)["revenue"].mean()
    assert any(
        row.get(col) == expected.idxmax() and row.get("revenue") == pytest.approx(expected.max(), abs=0.01)
        for row in result.result_table
    )


@pytest.mark.parametrize("col", GROUP_BY_COLUMNS)
def test_literally_named_column_resolves_in_contribution_question(resolution_dataset, col):
    """A column named in a contribution question (no grouping preposition) resolves."""
    df, metadata, schema_ctx = resolution_dataset
    question = f"Which {col} contributed most to revenue?"
    plan = _plan(question, schema_ctx)

    assert plan.intent == PlanIntent.CONTRIBUTION, f"{question} returned {plan.intent}"
    assert plan.target_column == "revenue"
    assert plan.group_by == [col], f"{col} was not resolved for question: {question}"

    result = PlanExecutor().execute(plan, df, metadata, question)
    assert result.status == "answered"

    grouped = df.groupby(col)["revenue"].sum()
    expected_top = grouped.idxmax()
    expected_share = round(float(grouped.max() / df["revenue"].sum() * 100.0), 2)

    ev_numbers = []
    top_segments = []
    for ev in result.evidence_store.list_evidence():
        for v in ev.content.values():
            if isinstance(v, (int, float)):
                ev_numbers.append(round(float(v), 2))
            elif isinstance(v, list):
                for row in v:
                    for rv in row.values():
                        if isinstance(rv, (int, float)):
                            ev_numbers.append(round(float(rv), 2))
        if "top_segment" in ev.content:
            top_segments.append(str(ev.content["top_segment"]))

    assert top_segments == [expected_top]
    assert any(pytest.approx(expected_share, abs=0.05) == ev_n for ev_n in ev_numbers)


def test_plural_column_name_resolves_in_group_by_question(resolution_dataset):
    """Plural forms of the column name resolve as the group-by column."""
    df, metadata, schema_ctx = resolution_dataset
    plan = _plan("Total revenue by products", schema_ctx)

    assert plan.intent == PlanIntent.QUERY
    assert plan.group_by == ["product"]

    result = PlanExecutor().execute(plan, df, metadata, "Total revenue by products")
    assert result.status == "answered"
    expected = df.groupby("product")["revenue"].sum()
    assert any(
        row.get("product") == expected.idxmax() and row.get("revenue") == pytest.approx(expected.max(), abs=0.01)
        for row in result.result_table
    )
