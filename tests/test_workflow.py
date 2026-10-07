"""Unit tests for Phase 8 - Basic LangGraph Workflow."""

from pathlib import Path
import pytest
from omnisleuth.evidence.models import ClaimStatus
from omnisleuth.ingestion import CSVLoader
from omnisleuth.workflow import build_investigation_graph, run_investigation


@pytest.fixture
def sales_data_bundle(tmp_path: Path):
    """Create sample sales dataset for workflow tests."""
    csv_file = tmp_path / "sales_workflow.csv"
    csv_file.write_text(
        """order_id,order_date,category,region,revenue
ORD-101,2024-01-10,Electronics,North,1500.0
ORD-102,2024-01-20,Furniture,South,500.0
ORD-103,2024-02-05,Electronics,North,1800.0
ORD-104,2024-02-25,Furniture,South,600.0
ORD-105,2024-03-01,Electronics,North,400.0
ORD-106,2024-03-15,Furniture,South,300.0
""",
        encoding="utf-8",
    )
    loader = CSVLoader()
    df, metadata = loader.load_csv(csv_file)
    return df, metadata


def test_build_investigation_graph():
    """Verify state graph compiles without errors."""
    graph = build_investigation_graph()
    assert graph is not None


def test_end_to_end_trend_investigation(sales_data_bundle):
    """Test full LangGraph investigation on a decline question."""
    df, metadata = sales_data_bundle
    question = "Why did revenue decline in March?"

    final_state = run_investigation(question=question, df=df, metadata=metadata)

    assert final_state["status"] == "completed"
    assert final_state["question_intent"] == "trend_investigation"
    assert final_state["target_metric"] == "revenue"
    assert final_state["plan_is_valid"] is True
    assert len(final_state["tool_executions"]) >= 3  # profile, trend, compare, anomalies

    # Check claims and validation
    assert len(final_state["claims"]) > 0
    assert len(final_state["validation_results"]) > 0
    for vr in final_state["validation_results"]:
        assert vr.status == ClaimStatus.SUPPORTED

    # Check evidence store
    store = final_state["evidence_store"]
    assert len(store.list_evidence()) >= 3

    # Check final answer text
    answer = final_state["final_answer"]
    assert "Investigation Summary" in answer
    assert "Evidence Provenance" in answer
    assert "Supported" in answer


def test_end_to_end_segment_comparison(sales_data_bundle):
    """Test segment comparison workflow."""
    df, metadata = sales_data_bundle
    question = "Compare revenue by region"

    final_state = run_investigation(question=question, df=df, metadata=metadata)

    assert final_state["status"] == "completed"
    assert final_state["question_intent"] == "segment_comparison"
    assert final_state["target_metric"] == "revenue"
    assert "revenue" in final_state["final_answer"]
