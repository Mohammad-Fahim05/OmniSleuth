import io
import os
from pathlib import Path
import sqlite3
from unittest.mock import patch, MagicMock
import pytest
from omnisleuth.core.exceptions import IngestionError
from omnisleuth.evidence.models import (
    Claim,
    ClaimStatus,
    EvidenceItem,
    EvidenceType,
    Hypothesis,
    HypothesisStatus,
)
from omnisleuth.evidence.store import EvidenceStore
from omnisleuth.ingestion import CSVLoader
from omnisleuth.ingestion.models import DatasetMetadata
from omnisleuth.ingestion.storage import safe_save_upload, safe_cleanup_file
from omnisleuth.persistence import DatabaseManager, Repository
from omnisleuth.tools import get_default_tool_registry


@pytest.fixture
def repo(tmp_path: Path) -> Repository:
    """Create a temporary SQLite database and repository instance."""
    db_file = tmp_path / "test_storage.db"
    db_man = DatabaseManager(db_path=str(db_file))
    return Repository(db_manager=db_man, session_id="test_session_default")


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


# ============================================================================
# T-005.3 Database and File Consistency Regression Tests
# ============================================================================


def test_successful_save_creates_valid_file_and_db_record(repo: Repository, tmp_path: Path):
    """T-005.3.1: Successful dataset save creates a valid file reference and usable database record."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    sample_file = upload_dir / "up_valid_001.csv"
    sample_file.write_text("col_a,col_b\n1,2\n3,4\n", encoding="utf-8")

    meta = DatasetMetadata(
        dataset_id="ds_success_001",
        name="Valid Dataset.csv",
        file_path=str(sample_file),
        version_hash="vhash_001",
        row_count=2,
        column_count=2,
    )

    repo.save_dataset(meta, verify_file=True, cleanup_on_failure=True)

    loaded = repo.get_dataset(meta.dataset_id, check_file=True)
    assert loaded is not None
    assert loaded["dataset_id"] == "ds_success_001"
    assert loaded["name"] == "Valid Dataset.csv"
    assert loaded["file_exists"] is True

    # Verify file is verified and returns valid Path
    verified_path = repo.verify_dataset_file(meta.dataset_id)
    assert verified_path.exists()
    assert verified_path.is_file()


def test_save_dataset_missing_file_rejected_when_verify_file_enabled(repo: Repository, tmp_path: Path):
    """T-005.3.1: Attempting to save a dataset pointing to non-existent file is rejected before insert."""
    missing_path = tmp_path / "uploads" / "does_not_exist.csv"

    meta = DatasetMetadata(
        dataset_id="ds_missing_001",
        name="Missing File.csv",
        file_path=str(missing_path),
        version_hash="vhash_missing",
        row_count=0,
        column_count=0,
    )

    with pytest.raises(IngestionError) as exc_info:
        repo.save_dataset(meta, verify_file=True, cleanup_on_failure=True)

    assert "does not exist on disk" in str(exc_info.value)
    # Ensure no database record was created
    assert repo.get_dataset(meta.dataset_id) is None


def test_database_insert_failure_rolls_back_transaction(repo: Repository, tmp_path: Path):
    """T-005.3.1: Database insert failure rolls back the database transaction."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    sample_file = upload_dir / "up_insert_fail.csv"
    sample_file.write_text("x,y\n1,2\n", encoding="utf-8")

    meta = DatasetMetadata(
        dataset_id="ds_insert_fail_001",
        name="Insert Fail.csv",
        file_path=str(sample_file),
        version_hash="vhash_fail",
        row_count=1,
        column_count=2,
    )

    real_get_conn = repo.db.get_connection

    class FailingInsertConnection:
        def __init__(self, real_conn):
            self._conn = real_conn

        def __enter__(self):
            self._conn.__enter__()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return self._conn.__exit__(exc_type, exc_val, exc_tb)

        def execute(self, sql, *args, **kwargs):
            if "INSERT OR REPLACE INTO dataset_versions" in str(sql):
                raise sqlite3.OperationalError("Simulated database failure during insert")
            return self._conn.execute(sql, *args, **kwargs)

        def commit(self):
            return self._conn.commit()

        def rollback(self):
            return self._conn.rollback()

        def __getattr__(self, name):
            return getattr(self._conn, name)

    with patch.object(repo.db, "get_connection", side_effect=lambda: FailingInsertConnection(real_get_conn())):
        with pytest.raises(IngestionError):
            repo.save_dataset(meta, verify_file=True, cleanup_on_failure=False)

    # Verify rollback: no record in datasets or dataset_versions
    with repo.db.get_connection() as conn:
        ds_row = conn.execute("SELECT * FROM datasets WHERE dataset_id = ?", (meta.dataset_id,)).fetchone()
        assert ds_row is None
        ver_row = conn.execute("SELECT * FROM dataset_versions WHERE dataset_id = ?", (meta.dataset_id,)).fetchone()
        assert ver_row is None


def test_database_commit_failure_does_not_leave_misleading_record(repo: Repository, tmp_path: Path):
    """T-005.3.1: Database commit failure rolls back and does not leave a misleading record."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    sample_file = upload_dir / "up_commit_fail.csv"
    sample_file.write_text("a,b\n5,6\n", encoding="utf-8")

    meta = DatasetMetadata(
        dataset_id="ds_commit_fail_001",
        name="Commit Fail.csv",
        file_path=str(sample_file),
        version_hash="vhash_commit",
        row_count=1,
        column_count=2,
    )

    real_get_conn = repo.db.get_connection

    class FailingCommitConnection:
        def __init__(self, real_conn):
            self._conn = real_conn

        def __enter__(self):
            self._conn.__enter__()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return self._conn.__exit__(exc_type, exc_val, exc_tb)

        def execute(self, sql, *args, **kwargs):
            return self._conn.execute(sql, *args, **kwargs)

        def commit(self):
            raise sqlite3.DatabaseError("Simulated commit crash")

        def rollback(self):
            return self._conn.rollback()

        def __getattr__(self, name):
            return getattr(self._conn, name)

    with patch.object(repo.db, "get_connection", side_effect=lambda: FailingCommitConnection(real_get_conn())):
        with pytest.raises(IngestionError):
            repo.save_dataset(meta, verify_file=True, cleanup_on_failure=True)

    # Verify record was not committed
    assert repo.get_dataset(meta.dataset_id) is None
    # And newly created file was cleaned up on failure
    assert not sample_file.exists()


def test_failed_database_persistence_cleans_up_only_new_upload_file(repo: Repository, tmp_path: Path):
    """T-005.3.3: Failed persistence cleans up only the new upload file and preserves unrelated files."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Existing pre-existing file and unrelated file
    existing_file = upload_dir / "up_existing_safe.csv"
    existing_file.write_text("existing,data\n1,2\n", encoding="utf-8")
    existing_meta = DatasetMetadata(
        dataset_id="ds_existing_001",
        name="Existing Dataset.csv",
        file_path=str(existing_file),
        version_hash="vhash_existing",
        row_count=1,
        column_count=2,
    )
    repo.save_dataset(existing_meta, verify_file=True, cleanup_on_failure=False)

    unrelated_file = upload_dir / "other_user_file.csv"
    unrelated_file.write_text("other,content\n9,9\n", encoding="utf-8")

    # New file for failed persistence
    new_fail_file = upload_dir / "up_new_fail.csv"
    new_fail_file.write_text("bad,data\n0,0\n", encoding="utf-8")
    fail_meta = DatasetMetadata(
        dataset_id="ds_fail_002",
        name="Fail Dataset.csv",
        file_path=str(new_fail_file),
        version_hash="vhash_fail2",
        row_count=1,
        column_count=2,
    )

    real_get_conn = repo.db.get_connection

    class FailingCommitConnection:
        def __init__(self, real_conn):
            self._conn = real_conn

        def __enter__(self):
            self._conn.__enter__()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return self._conn.__exit__(exc_type, exc_val, exc_tb)

        def execute(self, sql, *args, **kwargs):
            return self._conn.execute(sql, *args, **kwargs)

        def commit(self):
            raise sqlite3.OperationalError("Disk error")

        def rollback(self):
            return self._conn.rollback()

        def __getattr__(self, name):
            return getattr(self._conn, name)

    with patch.object(repo.db, "get_connection", side_effect=lambda: FailingCommitConnection(real_get_conn())):
        with pytest.raises(IngestionError):
            repo.save_dataset(fail_meta, verify_file=True, cleanup_on_failure=True)

    # New fail file must be cleaned up
    assert not new_fail_file.exists()

    # Pre-existing file and unrelated file MUST remain intact
    assert existing_file.exists()
    assert unrelated_file.exists()
    assert unrelated_file.read_text(encoding="utf-8") == "other,content\n9,9\n"

    # Pre-existing dataset in DB must remain intact
    existing_record = repo.get_dataset("ds_existing_001")
    assert existing_record is not None
    assert existing_record["name"] == "Existing Dataset.csv"


def test_safe_save_upload_cleans_up_partial_file_on_write_error(tmp_path: Path):
    """T-005.3.3: Write failure during upload streaming cleans up the partial file."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    class FailingStream(io.BytesIO):
        def __init__(self, data: bytes):
            super().__init__(data)
            self.calls = 0

        def read(self, size=-1):
            self.calls += 1
            if self.calls > 1:
                raise OSError("Simulated connection or disk failure mid-stream")
            return super().read(min(size, 1024))

    failing_input = FailingStream(b"A" * 50000)

    with pytest.raises(OSError, match="mid-stream"):
        safe_save_upload(
            file_obj=failing_input,
            raw_filename="interrupted.csv",
            upload_dir=upload_dir,
        )

    # Verify no partial or orphaned files remain in upload_dir
    created_files = list(upload_dir.iterdir())
    assert len(created_files) == 0


def test_missing_dataset_file_produces_controlled_error_without_leaking_paths(repo: Repository, tmp_path: Path):
    """T-005.3.2: Missing dataset file produces controlled domain error without exposing absolute path."""
    meta = DatasetMetadata(
        dataset_id="ds_missing_check",
        name="Ghost Dataset.csv",
        file_path=str(tmp_path / "uploads" / "ghost.csv"),
        version_hash="vhash_ghost",
        row_count=10,
        column_count=2,
    )
    # Save without verifying file to simulate a file that disappeared later
    repo.save_dataset(meta, verify_file=False)

    with pytest.raises(IngestionError) as exc_info:
        repo.verify_dataset_file(meta.dataset_id)

    err_msg = str(exc_info.value)
    # Must mention dataset name and controlled description
    assert "Ghost Dataset.csv" in err_msg
    assert "missing from storage" in err_msg
    # Must NOT expose raw server path
    assert str(tmp_path) not in err_msg


def test_unreadable_stored_dataset_fails_safely(repo: Repository, tmp_path: Path):
    """T-005.3.2: Unreadable stored dataset raises controlled IngestionError."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    real_file = upload_dir / "up_unreadable.csv"
    real_file.write_text("x,y\n1,2\n", encoding="utf-8")

    meta = DatasetMetadata(
        dataset_id="ds_unreadable_001",
        name="Unreadable Dataset.csv",
        file_path=str(real_file),
        version_hash="vhash_unread",
        row_count=1,
        column_count=2,
    )
    repo.save_dataset(meta, verify_file=True)

    with patch("os.access", return_value=False):
        with pytest.raises(IngestionError) as exc_info:
            repo.verify_dataset_file(meta.dataset_id)
        assert "not readable" in str(exc_info.value)
        assert "Unreadable Dataset.csv" in str(exc_info.value)


def test_dataset_listing_annotates_missing_file_without_deleting_record(repo: Repository, tmp_path: Path):
    """T-005.3.2: Listing datasets annotates file_exists without deleting database records."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    real_file = upload_dir / "up_present.csv"
    real_file.write_text("a,b\n1,2\n", encoding="utf-8")

    meta_present = DatasetMetadata(
        dataset_id="ds_present",
        name="Present Dataset.csv",
        file_path=str(real_file),
        version_hash="vhash_p",
        row_count=1,
        column_count=2,
    )
    repo.save_dataset(meta_present, verify_file=True)

    meta_missing = DatasetMetadata(
        dataset_id="ds_missing_listed",
        name="Vanished Dataset.csv",
        file_path=str(upload_dir / "vanished.csv"),
        version_hash="vhash_v",
        row_count=1,
        column_count=2,
    )
    repo.save_dataset(meta_missing, verify_file=False)

    datasets = repo.list_datasets(check_files=True)
    assert len(datasets) == 2

    by_id = {d["dataset_id"]: d for d in datasets}
    assert by_id["ds_present"]["file_exists"] is True
    assert by_id["ds_missing_listed"]["file_exists"] is False

    # Ensure database record for missing file is preserved, not silently deleted
    assert repo.get_dataset("ds_missing_listed", check_file=False) is not None


def test_repeated_save_does_not_create_duplicate_records(repo: Repository, tmp_path: Path):
    """T-005.3.1: Repeated save of same dataset_id updates record rather than duplicating."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    sample_file = upload_dir / "up_retry.csv"
    sample_file.write_text("col\n1\n", encoding="utf-8")

    meta1 = DatasetMetadata(
        dataset_id="ds_retry_001",
        name="Retry Dataset.csv",
        file_path=str(sample_file),
        version_hash="vhash_r1",
        row_count=1,
        column_count=1,
    )
    repo.save_dataset(meta1, verify_file=True)

    meta2 = DatasetMetadata(
        dataset_id="ds_retry_001",
        name="Retry Dataset Updated.csv",
        file_path=str(sample_file),
        version_hash="vhash_r2",
        row_count=5,
        column_count=1,
    )
    repo.save_dataset(meta2, verify_file=True)

    datasets = [d for d in repo.list_datasets() if d["dataset_id"] == "ds_retry_001"]
    assert len(datasets) == 1
    assert datasets[0]["name"] == "Retry Dataset Updated.csv"
    assert datasets[0]["row_count"] == 5


def test_delete_dataset_cleans_up_database_and_file(repo: Repository, tmp_path: Path):
    """T-005.3.1 & 3: delete_dataset removes DB record and safe upload file."""
    from omnisleuth.core.config import Settings

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    sample_file = upload_dir / "up_to_delete.csv"
    sample_file.write_text("x\n10\n", encoding="utf-8")

    meta = DatasetMetadata(
        dataset_id="ds_to_del",
        name="To Delete.csv",
        file_path=str(sample_file),
        version_hash="vhash_del",
        row_count=1,
        column_count=1,
    )
    repo.save_dataset(meta, verify_file=True)

    mock_settings = Settings(uploads_dir=str(upload_dir))
    with patch("omnisleuth.core.config.get_settings", return_value=mock_settings):
        deleted = repo.delete_dataset("ds_to_del", delete_file=True)
        assert deleted is True

    assert repo.get_dataset("ds_to_del") is None
    assert not sample_file.exists()

    # Second delete returns False gracefully
    assert repo.delete_dataset("ds_to_del") is False


# ============================================================================
# T-005.4 Session Ownership and Isolation Regression Tests
# ============================================================================


def test_session_isolation_dataset_listing(repo: Repository, tmp_path: Path):
    """T-005.4: Datasets are scoped to owner sessions and do not leak across sessions in listings."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_a = upload_dir / "up_sess_a.csv"
    file_a.write_text("a,b\n1,2\n", encoding="utf-8")
    file_b = upload_dir / "up_sess_b.csv"
    file_b.write_text("c,d\n3,4\n", encoding="utf-8")

    meta_a = DatasetMetadata(
        dataset_id="ds_sess_a",
        name="Dataset A.csv",
        file_path=str(file_a),
        version_hash="vhash_a",
        row_count=1,
        column_count=2,
        session_id="session_alpha",
    )
    repo.save_dataset(meta_a, verify_file=True, session_id="session_alpha")

    meta_b = DatasetMetadata(
        dataset_id="ds_sess_b",
        name="Dataset B.csv",
        file_path=str(file_b),
        version_hash="vhash_b",
        row_count=1,
        column_count=2,
        session_id="session_beta",
    )
    repo.save_dataset(meta_b, verify_file=True, session_id="session_beta")

    # Session Alpha listing
    list_alpha = repo.list_datasets(session_id="session_alpha")
    assert len(list_alpha) == 1
    assert list_alpha[0]["dataset_id"] == "ds_sess_a"

    # Session Beta listing
    list_beta = repo.list_datasets(session_id="session_beta")
    assert len(list_beta) == 1
    assert list_beta[0]["dataset_id"] == "ds_sess_b"

    # Foreign session listing
    list_gamma = repo.list_datasets(session_id="session_gamma")
    assert len(list_gamma) == 0


def test_session_isolation_dataset_retrieval_by_id(repo: Repository, tmp_path: Path):
    """T-005.4: Retrieving a dataset by ID enforces session ownership and returns None for foreign sessions."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_a = upload_dir / "up_retrieve_a.csv"
    file_a.write_text("x\n100\n", encoding="utf-8")

    meta = DatasetMetadata(
        dataset_id="ds_ret_a",
        name="Retrieve A.csv",
        file_path=str(file_a),
        version_hash="vhash_ret",
        row_count=1,
        column_count=1,
        session_id="session_owner",
    )
    repo.save_dataset(meta, verify_file=True, session_id="session_owner")

    # Owner session can access
    ds_owner = repo.get_dataset("ds_ret_a", session_id="session_owner")
    assert ds_owner is not None
    assert ds_owner["dataset_id"] == "ds_ret_a"

    # Foreign session gets None without leaking existence
    ds_foreign = repo.get_dataset("ds_ret_a", session_id="session_intruder")
    assert ds_foreign is None

    # verify_dataset_file raises controlled error for foreign session
    with pytest.raises(IngestionError) as exc_info:
        repo.verify_dataset_file("ds_ret_a", session_id="session_intruder")
    assert "not found" in str(exc_info.value).lower()


def test_session_isolation_foreign_investigation_creation_blocked(repo: Repository, tmp_path: Path):
    """T-005.4: Foreign session cannot start or persist an investigation using another session's dataset."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_a = upload_dir / "up_inv_block.csv"
    file_a.write_text("val\n5\n", encoding="utf-8")

    meta = DatasetMetadata(
        dataset_id="ds_inv_block",
        name="Inv Block.csv",
        file_path=str(file_a),
        version_hash="vhash_block",
        row_count=1,
        column_count=1,
        session_id="session_alice",
    )
    repo.save_dataset(meta, verify_file=True, session_id="session_alice")

    store = EvidenceStore(dataset_id="ds_inv_block", dataset_version="vhash_block")

    # Bob attempts to save investigation on Alice's dataset
    with pytest.raises(IngestionError) as exc_info:
        repo.save_investigation_results(
            investigation_id="inv_bob_attempt",
            dataset_id="ds_inv_block",
            question="What is total revenue?",
            status="completed",
            plan_dict={"tasks": []},
            final_answer="Answer",
            evidence_store=store,
            session_id="session_bob",
        )
    assert "Access denied" in str(exc_info.value)

    # Ensure no investigation was persisted
    assert repo.get_investigation("inv_bob_attempt") is None


def test_session_isolation_investigation_retrieval_and_listing(repo: Repository, tmp_path: Path):
    """T-005.4: Session B cannot retrieve or list Session A's investigation or evidence."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_a = upload_dir / "up_inv_list.csv"
    file_a.write_text("val\n10\n", encoding="utf-8")

    meta = DatasetMetadata(
        dataset_id="ds_inv_list",
        name="Inv List.csv",
        file_path=str(file_a),
        version_hash="vhash_list",
        row_count=1,
        column_count=1,
        session_id="session_alice",
    )
    repo.save_dataset(meta, verify_file=True, session_id="session_alice")

    store = EvidenceStore(dataset_id="ds_inv_list", dataset_version="vhash_list")
    ev = EvidenceItem(
        evidence_id="ev_secret_001",
        evidence_type=EvidenceType.DETERMINISTIC_CALC,
        source_id="calc",
        location="row_1",
        content={"kpi": "revenue_secret"},
        confidence=0.99,
    )
    store.add_evidence(ev)

    inv_id = "inv_alice_secret"
    repo.save_investigation_results(
        investigation_id=inv_id,
        dataset_id="ds_inv_list",
        question="Alice question?",
        status="completed",
        plan_dict={"tasks": []},
        final_answer="Alice answer",
        evidence_store=store,
        session_id="session_alice",
    )

    # Alice can list and retrieve
    alice_invs = repo.list_investigations(dataset_id="ds_inv_list", session_id="session_alice")
    assert len(alice_invs) == 1
    alice_loaded = repo.get_investigation(inv_id, session_id="session_alice")
    assert alice_loaded is not None
    assert len(alice_loaded["evidence_items"]) == 1

    # Bob cannot list or retrieve
    bob_invs = repo.list_investigations(dataset_id="ds_inv_list", session_id="session_bob")
    assert len(bob_invs) == 0
    bob_loaded = repo.get_investigation(inv_id, session_id="session_bob")
    assert bob_loaded is None


def test_session_isolation_mutual_coexistence(repo: Repository, tmp_path: Path):
    """T-005.4: Session A continues to access its dataset after Session B creates another dataset."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_a = upload_dir / "up_coexist_a.csv"
    file_a.write_text("a\n1\n", encoding="utf-8")
    file_b = upload_dir / "up_coexist_b.csv"
    file_b.write_text("b\n2\n", encoding="utf-8")

    meta_a = DatasetMetadata(
        dataset_id="ds_coexist_a",
        name="Coexist A.csv",
        file_path=str(file_a),
        version_hash="va",
        row_count=1,
        column_count=1,
        session_id="session_user_a",
    )
    repo.save_dataset(meta_a, verify_file=True, session_id="session_user_a")

    meta_b = DatasetMetadata(
        dataset_id="ds_coexist_b",
        name="Coexist B.csv",
        file_path=str(file_b),
        version_hash="vb",
        row_count=1,
        column_count=1,
        session_id="session_user_b",
    )
    repo.save_dataset(meta_b, verify_file=True, session_id="session_user_b")

    # Both sessions independently access their own datasets
    rec_a = repo.get_dataset("ds_coexist_a", session_id="session_user_a")
    assert rec_a is not None
    assert rec_a["name"] == "Coexist A.csv"

    rec_b = repo.get_dataset("ds_coexist_b", session_id="session_user_b")
    assert rec_b is not None
    assert rec_b["name"] == "Coexist B.csv"

    # Cross-access is completely segregated
    assert repo.get_dataset("ds_coexist_a", session_id="session_user_b") is None
    assert repo.get_dataset("ds_coexist_b", session_id="session_user_a") is None


def test_legacy_unassigned_records_hidden_from_session_queries(repo: Repository, tmp_path: Path):
    """T-005.4: Legacy records with NULL session_id are not silently exposed to ordinary session queries."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_legacy = upload_dir / "up_legacy.csv"
    file_legacy.write_text("legacy\n999\n", encoding="utf-8")

    # Save dataset with no session_id (simulates legacy unassigned record)
    meta_legacy = DatasetMetadata(
        dataset_id="ds_legacy_001",
        name="Legacy Unassigned.csv",
        file_path=str(file_legacy),
        version_hash="vhash_legacy",
        row_count=1,
        column_count=1,
        session_id=None,
    )
    repo.save_dataset(meta_legacy, verify_file=True, session_id=None)

    # Active session queries MUST NOT return legacy record
    active_listing = repo.list_datasets(session_id="session_active_client")
    assert len(active_listing) == 0

    active_lookup = repo.get_dataset("ds_legacy_001", session_id="session_active_client")
    assert active_lookup is None


def test_cross_session_dataset_overwrite_blocked(repo: Repository, tmp_path: Path):
    """T-005.4: A foreign session cannot overwrite an existing dataset owned by another session."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_a = upload_dir / "up_alice_orig.csv"
    file_a.write_text("alice\n1\n", encoding="utf-8")
    file_b = upload_dir / "up_bob_overwrite.csv"
    file_b.write_text("bob\n2\n", encoding="utf-8")

    meta_a = DatasetMetadata(
        dataset_id="ds_target_id",
        name="Alice Original.csv",
        file_path=str(file_a),
        version_hash="vhash_alice",
        row_count=1,
        column_count=1,
        session_id="session_alice",
    )
    repo.save_dataset(meta_a, verify_file=True, session_id="session_alice")

    meta_b = DatasetMetadata(
        dataset_id="ds_target_id",
        name="Bob Hijack.csv",
        file_path=str(file_b),
        version_hash="vhash_bob",
        row_count=1,
        column_count=1,
        session_id="session_bob",
    )
    with pytest.raises(IngestionError) as exc_info:
        repo.save_dataset(meta_b, verify_file=True, session_id="session_bob")
    assert "Access denied" in str(exc_info.value)

    # Alice's dataset remains unchanged
    alice_record = repo.get_dataset("ds_target_id", session_id="session_alice")
    assert alice_record is not None
    assert alice_record["name"] == "Alice Original.csv"


def test_cross_session_delete_blocked(repo: Repository, tmp_path: Path):
    """T-005.4: A foreign session cannot delete another session's dataset."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_a = upload_dir / "up_safe_delete.csv"
    file_a.write_text("val\n42\n", encoding="utf-8")

    meta = DatasetMetadata(
        dataset_id="ds_protect_del",
        name="Protected Delete.csv",
        file_path=str(file_a),
        version_hash="vhash_prot",
        row_count=1,
        column_count=1,
        session_id="session_owner",
    )
    repo.save_dataset(meta, verify_file=True, session_id="session_owner")

    # Foreign session attempts delete -> returns False
    res = repo.delete_dataset("ds_protect_del", delete_file=True, session_id="session_intruder")
    assert res is False

    # Dataset and file still exist intact
    assert repo.get_dataset("ds_protect_del", session_id="session_owner") is not None
    assert file_a.exists()


# ==============================================================================
# T-005.5: Restart Persistence and Rehydration Tests
# ==============================================================================

def test_restart_dataset_metadata_persists_across_reinit(tmp_path: Path):
    """Case 1: Dataset metadata remains available after database/repository reinitialization."""
    db_path = str(tmp_path / "restart_meta.db")
    upload_file = tmp_path / "data.csv"
    upload_file.write_text("item,qty,price\nwidget,10,25.5\ngadget,5,40.0\n", encoding="utf-8")

    # Initial session saves dataset
    loader = CSVLoader()
    df, meta = loader.load_csv(upload_file, dataset_name="Catalog", session_id="session_alpha")
    repo1 = Repository(DatabaseManager(db_path=db_path))
    repo1.save_dataset(meta, verify_file=True, session_id="session_alpha")

    # Simulate restart with fresh DB manager and Repository instance
    repo2 = Repository(DatabaseManager(db_path=db_path))
    recovered_meta = repo2.get_dataset_metadata(meta.dataset_id, session_id="session_alpha")

    assert recovered_meta is not None
    assert recovered_meta.dataset_id == meta.dataset_id
    assert recovered_meta.name == "Catalog"
    assert recovered_meta.row_count == 2
    assert recovered_meta.column_count == 3
    assert len(recovered_meta.columns) == 3
    col_names = [c.name for c in recovered_meta.columns]
    assert col_names == ["item", "qty", "price"]


def test_restart_authorized_dataset_contents_lazy_loaded(tmp_path: Path):
    """Case 2: Authorized dataset contents can be loaded lazily after restart."""
    db_path = str(tmp_path / "restart_lazy.db")
    upload_file = tmp_path / "orders.csv"
    upload_file.write_text("order_id,amount\nORD-101,150.0\nORD-102,275.5\n", encoding="utf-8")

    loader = CSVLoader()
    df_orig, meta = loader.load_csv(upload_file, dataset_name="Orders", session_id="session_alpha")
    repo1 = Repository(DatabaseManager(db_path=db_path))
    repo1.save_dataset(meta, verify_file=True, session_id="session_alpha")

    # Restart
    repo2 = Repository(DatabaseManager(db_path=db_path))
    loaded_df, loaded_meta = repo2.load_dataset_dataframe(meta.dataset_id, session_id="session_alpha")

    assert len(loaded_df) == 2
    assert list(loaded_df.columns) == ["order_id", "amount"]
    assert loaded_df["amount"].sum() == 425.5
    assert loaded_meta.dataset_id == meta.dataset_id


def test_listing_dataset_metadata_does_not_parse_file(tmp_path: Path):
    """Case 3: Listing dataset metadata does not read or parse the raw file on disk."""
    import pandas as pd
    from omnisleuth.ingestion.csv_loader import CSVLoader

    db_path = str(tmp_path / "no_parse.db")
    upload_file = tmp_path / "large_data.csv"
    upload_file.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    loader = CSVLoader()
    df, meta = loader.load_csv(upload_file, dataset_name="Large Data", session_id="session_alpha")
    repo1 = Repository(DatabaseManager(db_path=db_path))
    repo1.save_dataset(meta, verify_file=True, session_id="session_alpha")

    # Restart
    repo2 = Repository(DatabaseManager(db_path=db_path))

    with patch.object(pd, "read_csv") as mock_read_csv, patch.object(CSVLoader, "load_csv") as mock_loader:
        # Listing datasets
        ds_list = repo2.list_datasets(check_files=True, session_id="session_alpha")
        assert len(ds_list) == 1
        assert ds_list[0]["dataset_id"] == meta.dataset_id

        # Retrieving metadata descriptor
        rec_meta = repo2.get_dataset_metadata(meta.dataset_id, session_id="session_alpha")
        assert rec_meta is not None
        assert rec_meta.row_count == 2

        # Assert disk file was NOT parsed
        mock_read_csv.assert_not_called()
        mock_loader.assert_not_called()


def test_investigation_history_reconstruction_from_sqlite(tmp_path: Path):
    """Case 4: Investigation history is reconstructed from SQLite without rerunning tools."""
    db_path = str(tmp_path / "restart_inv.db")
    upload_file = tmp_path / "test.csv"
    upload_file.write_text("x,y\n10,20\n", encoding="utf-8")

    loader = CSVLoader()
    df, meta = loader.load_csv(upload_file, dataset_name="Test DS", session_id="session_alpha")
    repo1 = Repository(DatabaseManager(db_path=db_path))
    repo1.save_dataset(meta, verify_file=True, session_id="session_alpha")

    # Build investigation
    store = EvidenceStore(dataset_id=meta.dataset_id, dataset_version=meta.version_hash)
    store.add_hypothesis(Hypothesis(statement="Hyp 1", status=HypothesisStatus.SUPPORTED, confidence=0.88))
    store.add_claim(Claim(statement="Claim 1", status=ClaimStatus.SUPPORTED))

    inv_id = "inv_restart_001"
    repo1.save_investigation_results(
        investigation_id=inv_id,
        dataset_id=meta.dataset_id,
        question="What is the distribution of x?",
        status="completed",
        plan_dict={"tasks": ["summarize_columns"]},
        final_answer="x has a single value of 10.",
        evidence_store=store,
        session_id="session_alpha",
    )

    # Restart
    repo2 = Repository(DatabaseManager(db_path=db_path))
    inv_list = repo2.list_investigations(session_id="session_alpha")
    assert len(inv_list) == 1
    assert inv_list[0]["investigation_id"] == inv_id

    # Reconstruct investigation state
    reconstructed = repo2.reconstruct_investigation_state(inv_id, session_id="session_alpha")
    assert reconstructed is not None
    assert reconstructed["question"] == "What is the distribution of x?"
    assert reconstructed["status"] == "completed"
    assert reconstructed["executed_tasks"] == ["summarize_columns"]
    assert len(reconstructed["hypotheses"]) == 1
    assert reconstructed["hypotheses"][0].statement == "Hyp 1"
    assert len(reconstructed["claims"]) == 1
    assert reconstructed["claims"][0].statement == "Claim 1"


def test_stored_claims_and_evidence_remain_available_to_owner(tmp_path: Path):
    """Case 5: Stored claims and evidence remain available to the authorized session owner."""
    db_path = str(tmp_path / "restart_evidence.db")
    upload_file = tmp_path / "metrics.csv"
    upload_file.write_text("metric,val\nm1,100\n", encoding="utf-8")

    loader = CSVLoader()
    df, meta = loader.load_csv(upload_file, dataset_name="Metrics", session_id="session_alpha")
    repo1 = Repository(DatabaseManager(db_path=db_path))
    repo1.save_dataset(meta, verify_file=True, session_id="session_alpha")

    store = EvidenceStore(dataset_id=meta.dataset_id, dataset_version=meta.version_hash)
    ev = EvidenceItem(
        evidence_id="ev_001",
        evidence_type=EvidenceType.DETERMINISTIC_CALC,
        source_id="calc_tool",
        location="metric=m1",
        content={"result": 100, "formula": "sum"},
        confidence=0.99,
    )
    store.add_evidence(ev)
    clm = Claim(claim_id="clm_001", statement="Metric m1 equals 100", status=ClaimStatus.SUPPORTED)
    store.add_claim(clm)

    repo1.save_investigation_results(
        investigation_id="inv_owner_test",
        dataset_id=meta.dataset_id,
        question="Check m1",
        status="completed",
        plan_dict={},
        final_answer="Verified",
        evidence_store=store,
        session_id="session_alpha",
    )

    # Restart
    repo2 = Repository(DatabaseManager(db_path=db_path))
    state = repo2.reconstruct_investigation_state("inv_owner_test", session_id="session_alpha")
    assert state is not None
    ev_list = state["evidence_store"].list_evidence()
    assert len(ev_list) == 1
    assert ev_list[0].evidence_id == "ev_001"
    assert ev_list[0].content == {"result": 100, "formula": "sum"}
    assert ev_list[0].confidence == 0.99

    clm_list = state["claims"]
    assert len(clm_list) == 1
    assert clm_list[0].statement == "Metric m1 equals 100"
    assert clm_list[0].status == ClaimStatus.SUPPORTED


def test_unauthorized_session_cannot_retrieve_dataset_after_restart(tmp_path: Path):
    """Case 6: A new unauthorized session cannot retrieve another session's dataset after restart."""
    db_path = str(tmp_path / "restart_sec_ds.db")
    upload_file = tmp_path / "confidential.csv"
    upload_file.write_text("secret_id,secret_val\n1,classified\n", encoding="utf-8")

    loader = CSVLoader()
    df, meta = loader.load_csv(upload_file, dataset_name="Confidential", session_id="session_alice")
    repo1 = Repository(DatabaseManager(db_path=db_path))
    repo1.save_dataset(meta, verify_file=True, session_id="session_alice")

    # Restart with Bob's session
    repo2 = Repository(DatabaseManager(db_path=db_path))

    # Bob cannot list Alice's dataset
    assert len(repo2.list_datasets(session_id="session_bob")) == 0

    # Bob cannot fetch Alice's dataset record
    assert repo2.get_dataset(meta.dataset_id, session_id="session_bob") is None

    # Bob cannot fetch Alice's metadata
    assert repo2.get_dataset_metadata(meta.dataset_id, session_id="session_bob") is None

    # Bob cannot load Alice's dataframe
    with pytest.raises(IngestionError):
        repo2.load_dataset_dataframe(meta.dataset_id, session_id="session_bob")


def test_unauthorized_session_cannot_retrieve_investigation_after_restart(tmp_path: Path):
    """Case 7: A new unauthorized session cannot retrieve another session's investigation or evidence."""
    db_path = str(tmp_path / "restart_sec_inv.db")
    upload_file = tmp_path / "data.csv"
    upload_file.write_text("k,v\n1,2\n", encoding="utf-8")

    loader = CSVLoader()
    df, meta = loader.load_csv(upload_file, dataset_name="Data", session_id="session_alice")
    repo1 = Repository(DatabaseManager(db_path=db_path))
    repo1.save_dataset(meta, verify_file=True, session_id="session_alice")

    store = EvidenceStore(dataset_id=meta.dataset_id, dataset_version=meta.version_hash)
    store.add_claim(Claim(statement="Secret claim", status=ClaimStatus.SUPPORTED))

    repo1.save_investigation_results(
        investigation_id="inv_secret_alice",
        dataset_id=meta.dataset_id,
        question="Alice private investigation",
        status="completed",
        plan_dict={},
        final_answer="Classified answer",
        evidence_store=store,
        session_id="session_alice",
    )

    # Restart with Bob's session
    repo2 = Repository(DatabaseManager(db_path=db_path))

    # Bob cannot list Alice's investigations
    assert len(repo2.list_investigations(session_id="session_bob")) == 0

    # Bob cannot fetch Alice's investigation by ID
    assert repo2.get_investigation("inv_secret_alice", session_id="session_bob") is None

    # Bob cannot reconstruct Alice's investigation state
    assert repo2.reconstruct_investigation_state("inv_secret_alice", session_id="session_bob") is None


def test_missing_dataset_files_marked_unavailable_without_metadata_deletion(tmp_path: Path):
    """Case 8: Missing dataset files are detected and marked unavailable without deleting metadata."""
    db_path = str(tmp_path / "restart_missing.db")
    upload_file = tmp_path / "ephemeral.csv"
    upload_file.write_text("a,b\n1,2\n", encoding="utf-8")

    loader = CSVLoader()
    df, meta = loader.load_csv(upload_file, dataset_name="Ephemeral", session_id="session_alpha")
    repo1 = Repository(DatabaseManager(db_path=db_path))
    repo1.save_dataset(meta, verify_file=True, session_id="session_alpha")

    # File disappears on disk
    upload_file.unlink()
    assert not upload_file.exists()

    # Restart
    repo2 = Repository(DatabaseManager(db_path=db_path))

    # Listing still includes the dataset, but marked file_exists=False
    listing = repo2.list_datasets(check_files=True, session_id="session_alpha")
    assert len(listing) == 1
    assert listing[0]["dataset_id"] == meta.dataset_id
    assert listing[0]["file_exists"] is False

    # Metadata remains in SQLite
    rec_meta = repo2.get_dataset_metadata(meta.dataset_id, session_id="session_alpha")
    assert rec_meta is not None
    assert rec_meta.row_count == 1

    # Attempting to load DataFrame raises IngestionError without deleting record
    with pytest.raises(IngestionError) as exc_info:
        repo2.load_dataset_dataframe(meta.dataset_id, session_id="session_alpha")
    assert "missing from storage" in str(exc_info.value)

    # Verify record was NOT deleted
    assert repo2.get_dataset(meta.dataset_id, check_file=False, session_id="session_alpha") is not None


def test_corrupt_or_incomplete_persisted_records_handled_gracefully(tmp_path: Path):
    """Case 9: Corrupt or incomplete persisted records do not crash investigation rehydration."""
    db_path = str(tmp_path / "restart_corrupt.db")
    db_man = DatabaseManager(db_path=db_path)
    repo = Repository(db_man)

    inv_id = "inv_corrupt_test"
    with db_man.get_connection() as conn:
        # Insert parent dataset to satisfy foreign key
        conn.execute(
            """
            INSERT INTO datasets (dataset_id, name, file_path, version_hash, row_count, column_count, duplicate_rows, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("ds_dummy", "Dummy DS", "data/dummy.csv", "vdummy", 1, 1, 0, "sess_corrupt"),
        )
        # Insert investigation
        conn.execute(
            """
            INSERT INTO investigations (investigation_id, dataset_id, question, status, plan_json, final_answer, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (inv_id, "ds_dummy", "Corrupt test question", "completed", "invalid json {{{", "Corrupt answer", "sess_corrupt"),
        )
        # Insert corrupt evidence item with invalid evidence_type and malformed content_json
        conn.execute(
            """
            INSERT INTO evidence_items (evidence_id, investigation_id, evidence_type, source_id, content_json, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("ev_bad", inv_id, "COMPLETELY_INVALID_TYPE", "bad_tool", "{malformed_json", 0.5),
        )
        # Insert corrupt hypothesis with invalid status
        conn.execute(
            """
            INSERT INTO hypotheses (hypothesis_id, investigation_id, statement, status, confidence)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("hyp_bad", inv_id, "Bad hyp statement", "NON_EXISTENT_STATUS", 0.7),
        )
        # Insert corrupt claim with invalid status
        conn.execute(
            """
            INSERT INTO claims (claim_id, investigation_id, statement, status)
            VALUES (?, ?, ?, ?)
            """,
            ("clm_bad", inv_id, "Bad claim statement", "ANOTHER_BAD_STATUS"),
        )
        conn.commit()

    # Reconstruct should succeed defensively without raising exceptions
    reconstructed = repo.reconstruct_investigation_state(inv_id, session_id="sess_corrupt")
    assert reconstructed is not None
    assert reconstructed["question"] == "Corrupt test question"
    assert reconstructed["plan"] == {}
    assert reconstructed["executed_tasks"] == []

    # Corrupt items were either converted to safe defaults or skipped
    assert len(reconstructed["hypotheses"]) == 1
    assert reconstructed["hypotheses"][0].status == HypothesisStatus.INCONCLUSIVE

    assert len(reconstructed["claims"]) == 1
    assert reconstructed["claims"][0].status == ClaimStatus.INCONCLUSIVE

    ev_list = reconstructed["evidence_store"].list_evidence()
    assert len(ev_list) == 1
    assert ev_list[0].evidence_type == EvidenceType.STRUCTURED_DATA


def test_repeated_initialization_does_not_duplicate_history(tmp_path: Path):
    """Case 10: Repeated repository reinitialization and queries do not duplicate history."""
    db_path = str(tmp_path / "restart_idempotent.db")
    upload_file = tmp_path / "single.csv"
    upload_file.write_text("col\n1\n", encoding="utf-8")

    loader = CSVLoader()
    df, meta = loader.load_csv(upload_file, dataset_name="Single", session_id="session_dup")
    repo = Repository(DatabaseManager(db_path=db_path))
    repo.save_dataset(meta, verify_file=True, session_id="session_dup")

    store = EvidenceStore(dataset_id=meta.dataset_id, dataset_version=meta.version_hash)
    repo.save_investigation_results(
        investigation_id="inv_single",
        dataset_id=meta.dataset_id,
        question="Single question?",
        status="completed",
        plan_dict={},
        final_answer="Answer",
        evidence_store=store,
        session_id="session_dup",
    )

    # Reinitialize multiple times
    for _ in range(5):
        fresh_repo = Repository(DatabaseManager(db_path=db_path))
        datasets = fresh_repo.list_datasets(session_id="session_dup")
        assert len(datasets) == 1

        investigations = fresh_repo.list_investigations(session_id="session_dup")
        assert len(investigations) == 1


def test_legacy_unassigned_records_remain_inaccessible_after_restart(tmp_path: Path):
    """Case 11: Legacy unassigned records remain inaccessible to active sessions across restarts."""
    db_path = str(tmp_path / "restart_legacy.db")
    db_man = DatabaseManager(db_path=db_path)

    # Directly insert unassigned dataset and investigation records (session_id = NULL)
    with db_man.get_connection() as conn:
        conn.execute(
            """
            INSERT INTO datasets (dataset_id, name, file_path, version_hash, row_count, column_count, duplicate_rows, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            ("ds_legacy_reinit", "Legacy Reinit DS", "data/legacy.csv", "vlegacy", 10, 2, 0),
        )
        conn.execute(
            """
            INSERT INTO investigations (investigation_id, dataset_id, question, status, session_id)
            VALUES (?, ?, ?, ?, NULL)
            """,
            ("inv_legacy_reinit", "ds_legacy_reinit", "Legacy question?", "completed"),
        )
        conn.commit()

    # Reinitialize repository
    fresh_repo = Repository(DatabaseManager(db_path=db_path))

    # Active session MUST NOT see the legacy records
    active_session = "sess_active_user_123"
    assert len(fresh_repo.list_datasets(session_id=active_session)) == 0
    assert fresh_repo.get_dataset("ds_legacy_reinit", session_id=active_session) is None
    assert fresh_repo.get_dataset_metadata("ds_legacy_reinit", session_id=active_session) is None

    assert len(fresh_repo.list_investigations(session_id=active_session)) == 0
    assert fresh_repo.get_investigation("inv_legacy_reinit", session_id=active_session) is None
    assert fresh_repo.reconstruct_investigation_state("inv_legacy_reinit", session_id=active_session) is None


def test_legacy_database_migration_ordering_and_session_indices(tmp_path: Path):
    """Regression test: older SQLite databases without session_id migrate safely on startup."""
    legacy_db_file = tmp_path / "pre_t005_legacy.db"

    # 1. Create a pre-T005 database without session_id columns or session indices
    with sqlite3.connect(str(legacy_db_file)) as conn:
        conn.executescript(
            """
            CREATE TABLE datasets (
                dataset_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                version_hash TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                column_count INTEGER NOT NULL,
                duplicate_rows INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE investigations (
                investigation_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                question TEXT NOT NULL,
                status TEXT NOT NULL,
                plan_json TEXT,
                final_answer TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
            );
            """
        )
        conn.execute(
            """
            INSERT INTO datasets (dataset_id, name, file_path, version_hash, row_count, column_count, duplicate_rows)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("ds_pre_t005", "Pre-T005 Dataset", "data/pre.csv", "vpre", 50, 4, 0),
        )
        conn.execute(
            """
            INSERT INTO investigations (investigation_id, dataset_id, question, status, plan_json, final_answer)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("inv_pre_t005", "ds_pre_t005", "Legacy question?", "completed", "{}", "Legacy Answer"),
        )
        conn.commit()

        # Confirm columns do NOT have session_id yet
        cols_before_ds = {r[1] for r in conn.execute("PRAGMA table_info(datasets)").fetchall()}
        cols_before_inv = {r[1] for r in conn.execute("PRAGMA table_info(investigations)").fetchall()}
        assert "session_id" not in cols_before_ds
        assert "session_id" not in cols_before_inv

    # 2. Initialize DatabaseManager — this must migrate safely without OperationalError
    db_man = DatabaseManager(db_path=str(legacy_db_file))

    with db_man.get_connection() as conn:
        cols_after_ds = {r[1] for r in conn.execute("PRAGMA table_info(datasets)").fetchall()}
        cols_after_inv = {r[1] for r in conn.execute("PRAGMA table_info(investigations)").fetchall()}
        assert "session_id" in cols_after_ds
        assert "session_id" in cols_after_inv

        # 3. Existing records survived migration intact
        ds_row = conn.execute("SELECT dataset_id, name, session_id FROM datasets WHERE dataset_id = 'ds_pre_t005'").fetchone()
        assert ds_row is not None
        assert ds_row["dataset_id"] == "ds_pre_t005"
        assert ds_row["name"] == "Pre-T005 Dataset"
        assert ds_row["session_id"] is None

        inv_row = conn.execute("SELECT investigation_id, question, session_id FROM investigations WHERE investigation_id = 'inv_pre_t005'").fetchone()
        assert inv_row is not None
        assert inv_row["investigation_id"] == "inv_pre_t005"
        assert inv_row["question"] == "Legacy question?"
        assert inv_row["session_id"] is None

        # 4. Both session indexes were created
        indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()}
        assert "idx_datasets_session_id" in indexes
        assert "idx_investigations_session_id" in indexes

    # 5. Repeated initialization does not fail or duplicate columns
    for _ in range(3):
        reinit_man = DatabaseManager(db_path=str(legacy_db_file))
        with reinit_man.get_connection() as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(datasets)").fetchall()]
            assert cols.count("session_id") == 1

    # 6. Verify fresh database initializes with all tables, session_id, and indexes
    fresh_db_file = tmp_path / "fresh_new.db"
    fresh_man = DatabaseManager(db_path=str(fresh_db_file))
    with fresh_man.get_connection() as conn:
        cols_fresh_ds = {r[1] for r in conn.execute("PRAGMA table_info(datasets)").fetchall()}
        cols_fresh_inv = {r[1] for r in conn.execute("PRAGMA table_info(investigations)").fetchall()}
        assert "session_id" in cols_fresh_ds
        assert "session_id" in cols_fresh_inv

        fresh_indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()}
        assert "idx_datasets_session_id" in fresh_indexes
        assert "idx_investigations_session_id" in fresh_indexes


def test_sqlite_wal_journal_mode_configured_on_disk(tmp_path: Path):
    """Phase E: Disk databases configure SQLite WAL journal mode persistently."""
    db_file = tmp_path / "wal_verify.db"
    db_man = DatabaseManager(db_path=str(db_file))

    # 1. Check journal mode via DatabaseManager helper
    assert db_man.get_journal_mode() == "WAL"

    # 2. Open an independent raw connection directly to the file without running initialize_schema
    # WAL mode is persisted in the database header
    with sqlite3.connect(str(db_file)) as raw_conn:
        cur = raw_conn.execute("PRAGMA journal_mode;")
        row = cur.fetchone()
        assert str(row[0]).upper() == "WAL"


def test_sqlite_wal_mode_preserves_existing_data(tmp_path: Path):
    """Phase E: Converting an existing DELETE-journal database to WAL preserves existing data intact."""
    db_file = tmp_path / "existing_rollback.db"

    # 1. Create a database in DELETE mode and insert a record
    with sqlite3.connect(str(db_file)) as conn:
        conn.execute("PRAGMA journal_mode = DELETE;")
        conn.execute(
            """
            CREATE TABLE datasets (
                dataset_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                version_hash TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                column_count INTEGER NOT NULL,
                duplicate_rows INTEGER NOT NULL,
                session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.execute(
            """
            INSERT INTO datasets (dataset_id, name, file_path, version_hash, row_count, column_count, duplicate_rows, session_id)
            VALUES ('ds_survivor', 'Survivor Dataset', 'data/s.csv', 'vh123', 100, 5, 0, 'sess_test')
            """
        )
        conn.commit()

        cur = conn.execute("PRAGMA journal_mode;")
        assert str(cur.fetchone()[0]).upper() == "DELETE"

    # 2. Initialize DatabaseManager — safely upgrades journal_mode to WAL without dropping data
    db_man = DatabaseManager(db_path=str(db_file))
    assert db_man.get_journal_mode() == "WAL"

    # 3. Verify existing record survived intact
    with db_man.get_connection() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE dataset_id = 'ds_survivor'").fetchone()
        assert row is not None
        assert row["name"] == "Survivor Dataset"
        assert row["row_count"] == 100
        assert row["session_id"] == "sess_test"


def test_sqlite_wal_mode_idempotent_reinitialization(tmp_path: Path):
    """Phase E: Repeated initialization does not fail and preserves WAL mode and table integrity."""
    db_file = tmp_path / "idempotent_wal.db"

    for _ in range(5):
        db_man = DatabaseManager(db_path=str(db_file))
        assert db_man.get_journal_mode() == "WAL"

    # Tables and indices remain intact
    with db_man.get_connection() as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()}
        assert "datasets" in tables
        assert "investigations" in tables


def test_sqlite_memory_database_safe_fallback():
    """Phase E: In-memory databases handle WAL configuration safely without exceptions."""
    db_man = DatabaseManager(db_path=":memory:")
    mode = db_man.get_journal_mode()
    # In SQLite, in-memory databases return 'MEMORY' when journal_mode is queried
    assert mode in ("MEMORY", "OFF")


def test_sqlite_wal_concurrent_readers_and_writer(tmp_path: Path):
    """Phase E: Under WAL mode, concurrent readers do not block writers."""
    db_file = tmp_path / "concurrent_wal.db"
    db_man = DatabaseManager(db_path=str(db_file))

    # Populate initial data
    with db_man.get_connection() as conn:
        conn.execute(
            """
            INSERT INTO datasets (dataset_id, name, file_path, version_hash, row_count, column_count, duplicate_rows, session_id)
            VALUES ('ds_conc', 'Concurrent Dataset', 'data/c.csv', 'v_conc', 10, 2, 0, 'sess_conc')
            """
        )
        conn.commit()

    reader1 = db_man.get_connection()
    reader2 = db_man.get_connection()
    writer = db_man.get_connection()

    try:
        # Reader 1 reads
        r1_row = reader1.execute("SELECT name FROM datasets WHERE dataset_id = 'ds_conc'").fetchone()
        assert r1_row[0] == "Concurrent Dataset"

        # Writer performs an update while Reader 1 is still open
        writer.execute("UPDATE datasets SET row_count = 50 WHERE dataset_id = 'ds_conc'")
        writer.commit()

        # Reader 2 reads immediately
        r2_row = reader2.execute("SELECT row_count FROM datasets WHERE dataset_id = 'ds_conc'").fetchone()
        assert r2_row[0] == 50
    finally:
        reader1.close()
        reader2.close()
        writer.close()
