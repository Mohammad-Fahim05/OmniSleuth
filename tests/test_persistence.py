"""Unit tests for Phase 16 - SQLite Persistence."""

from pathlib import Path
import pytest
from omnisleuth.evidence.models import Claim, ClaimStatus, Hypothesis, HypothesisStatus
from omnisleuth.evidence.store import EvidenceStore
from omnisleuth.ingestion import CSVLoader
from omnisleuth.persistence import DatabaseManager, Repository
from omnisleuth.tools import get_default_tool_registry


@pytest.fixture
def repo(tmp_path: Path) -> Repository:
    """Create a temporary SQLite database and repository instance."""
    db_file = tmp_path / "test_storage.db"
    db_man = DatabaseManager(db_path=str(db_file))
    return Repository(db_manager=db_man)


def test_save_and_list_datasets(repo: Repository, tmp_path: Path):
    """Test saving and retrieving dataset descriptors."""
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("order_id,revenue\nORD-1,100\nORD-2,200\n", encoding="utf-8")

    loader = CSVLoader()
    df, metadata = loader.load_csv(csv_file)

    repo.save_dataset(metadata)
    datasets = repo.list_datasets()

    assert len(datasets) == 1
    assert datasets[0]["dataset_id"] == metadata.dataset_id
    assert datasets[0]["row_count"] == 2


def test_save_and_load_investigation(repo: Repository, tmp_path: Path):
    """Test saving investigation, tool executions, evidence, hypotheses, and claims."""
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("order_id,order_date,revenue\nORD-1,2024-01-01,100\n", encoding="utf-8")
    loader = CSVLoader()
    df, metadata = loader.load_csv(csv_file)
    repo.save_dataset(metadata)

    store = EvidenceStore(dataset_id=metadata.dataset_id, dataset_version=metadata.version_hash)
    registry = get_default_tool_registry()

    # Execute a tool and mint evidence
    tool = registry.get("profile_dataset")
    out = tool.run({}, df, metadata)
    store.record_tool_execution("profile_dataset", {}, out)

    # Add hypothesis
    hyp = Hypothesis(statement="Test Hypothesis Statement", status=HypothesisStatus.SUPPORTED, confidence=0.95)
    store.add_hypothesis(hyp)

    # Add claim
    claim = Claim(statement="Test Claim Statement", status=ClaimStatus.SUPPORTED)
    store.add_claim(claim)

    inv_id = "inv_test_001"
    repo.save_investigation_results(
        investigation_id=inv_id,
        dataset_id=metadata.dataset_id,
        question="What is the quality of the dataset?",
        status="completed",
        plan_dict={"steps": ["profile_dataset"]},
        final_answer="The dataset has 1 row and 3 columns.",
        evidence_store=store,
    )

    loaded = repo.get_investigation(inv_id)
    assert loaded is not None
    assert loaded["investigation_id"] == inv_id
    assert loaded["question"] == "What is the quality of the dataset?"
    assert len(loaded["hypotheses"]) == 1
    assert loaded["hypotheses"][0]["statement"] == "Test Hypothesis Statement"
    assert len(loaded["claims"]) == 1
    assert loaded["claims"][0]["statement"] == "Test Claim Statement"


def test_serializer_handles_timestamp_and_datetime():
    """Test safe conversion of Timestamp, Datetime, and NaT."""
    from datetime import date, datetime
    import numpy as np
    import pandas as pd
    from omnisleuth.persistence.serializer import safe_json_dumps, safe_json_loads, to_json_safe

    data = {
        "ts": pd.Timestamp("2024-03-15 10:30:00"),
        "nat": pd.NaT,
        "dt": datetime(2024, 1, 1, 12, 0),
        "d": date(2024, 5, 20),
        "pd_na": pd.NA,
        "np_int": np.int64(42),
        "np_float": np.float64(3.14159),
        "np_nan": np.nan,
        "np_bool": np.bool_(True),
        "arr": np.array([1, 2, 3]),
        "nested": {
            "sub_ts": pd.Timestamp("2024-06-01"),
            "sub_list": [pd.Timestamp("2024-07-01"), np.int32(100)],
        },
    }

    safe = to_json_safe(data)
    assert safe["ts"] == "2024-03-15T10:30:00"
    assert safe["nat"] is None
    assert safe["pd_na"] is None
    assert safe["np_nan"] is None
    assert safe["np_int"] == 42
    assert safe["np_bool"] is True
    assert safe["arr"] == [1, 2, 3]
    assert safe["nested"]["sub_ts"] == "2024-06-01T00:00:00"
    assert safe["nested"]["sub_list"][0] == "2024-07-01T00:00:00"

    # Verify safe_json_dumps and roundtrip
    dumped = safe_json_dumps(data)
    loaded = safe_json_loads(dumped)
    assert loaded["ts"] == "2024-03-15T10:30:00"
    assert loaded["np_int"] == 42
    assert loaded["nested"]["sub_list"][1] == 100


def test_save_investigation_with_timestamp_plan_and_evidence(repo: Repository, tmp_path: Path):
    """Regression test ensuring investigations with Timestamp in plan and evidence persist cleanly."""
    import numpy as np
    import pandas as pd

    from omnisleuth.ingestion.models import DatasetMetadata

    meta = DatasetMetadata(
        dataset_id="ds_test",
        name="Test Dataset",
        file_path="data/test.csv",
        version_hash="ver_test",
        row_count=100,
        column_count=5,
    )
    repo.save_dataset(meta)

    store = EvidenceStore(dataset_id="ds_test", dataset_version="ver_test")

    from omnisleuth.tools.base import BaseToolOutput

    # Simulate tool execution with timestamps in parameters and result
    tool_out = BaseToolOutput(
        tool_name="calculate_trend",
        success=True,
        data={
            "time_series": [
                {"period": pd.Timestamp("2024-01-01"), "value": np.float64(12500.50)},
                {"period": pd.Timestamp("2024-02-01"), "value": np.float64(14200.00)},
                {"period": pd.Timestamp("2024-03-01"), "value": np.float64(9800.25)},
            ],
            "overall_delta": np.float64(-4399.75),
            "missing_sample": pd.NaT,
        },
        execution_time_ms=12.5,
    )

    exec_rec = store.record_tool_execution(
        tool_name="calculate_trend",
        parameters={
            "date_range": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-31")],
            "interval": "monthly",
            "threshold": np.float64(0.05),
        },
        output=tool_out,
    )

    from omnisleuth.evidence.models import EvidenceItem, EvidenceType

    ev_item = EvidenceItem(
        evidence_type=EvidenceType.DETERMINISTIC_CALC,
        source_id=exec_rec.execution_id,
        location="Trend(order_date -> revenue)",
        content={
            "period": pd.Timestamp("2024-03-01"),
            "values": [np.float64(100.0), np.float64(200.0)],
            "missing": pd.NaT,
        },
        confidence=0.95,
        related_tool_execution_id=exec_rec.execution_id,
    )
    store.add_evidence(ev_item)

    plan_dict = {
        "objective": "Investigate revenue decline in March",
        "target_metric": "revenue",
        "tasks": [
            {
                "task_id": "task_01",
                "assigned_to": "data_analyst",
                "tool_name": "calculate_trend",
                "parameters": {"start_date": pd.Timestamp("2024-01-01")},
                "status": "completed",
                "result": {"peak_timestamp": pd.Timestamp("2024-02-01")},
            }
        ],
    }

    inv_id = "inv_regression_timestamp_001"

    # This MUST NOT raise TypeError: Object of type Timestamp is not JSON serializable
    repo.save_investigation_results(
        investigation_id=inv_id,
        dataset_id="ds_test",
        question="Why did revenue decline in March?",
        status="completed",
        plan_dict=plan_dict,
        final_answer="Revenue declined due to March supplier disruption.",
        evidence_store=store,
    )

    loaded = repo.get_investigation(inv_id)
    assert loaded is not None
    assert loaded["investigation_id"] == inv_id
    assert loaded["plan"]["tasks"][0]["parameters"]["start_date"] == "2024-01-01T00:00:00"
    assert len(loaded["evidence_items"]) > 0

