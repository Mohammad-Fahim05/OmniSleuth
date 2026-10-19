"""Regression tests for T-012 Phase B: Session Isolation, Parent-Gated Child Records, and Administrative Paths."""

from pathlib import Path
import pytest

from omnisleuth.core.exceptions import AccessDeniedError, IngestionError
from omnisleuth.evidence.models import (
    Claim,
    ClaimStatus,
    EvidenceItem,
    EvidenceType,
    Hypothesis,
    HypothesisStatus,
    ToolExecution,
)
from omnisleuth.evidence.store import EvidenceStore
from omnisleuth.ingestion.models import ColumnSchema, ColumnType, DatasetMetadata
from omnisleuth.persistence.db import DatabaseManager
from omnisleuth.persistence.repository import Repository


@pytest.fixture
def isolated_db(tmp_path: Path) -> DatabaseManager:
    """Create an isolated SQLite database manager for regression tests."""
    db_file = tmp_path / "test_session_iso.db"
    return DatabaseManager(db_path=str(db_file))


def _create_sample_csv(tmp_path: Path, filename: str = "data.csv") -> Path:
    f = tmp_path / filename
    f.write_text("id,metric,value\n1,alpha,10.5\n2,beta,20.0\n", encoding="utf-8")
    return f


def test_session_a_cannot_retrieve_list_update_delete_or_reconstruct_session_b_data(
    isolated_db: DatabaseManager, tmp_path: Path
):
    """Session A cannot retrieve, list, update, delete, or reconstruct Session B's dataset."""
    csv_a = _create_sample_csv(tmp_path, "alice.csv")
    repo = Repository(db_manager=isolated_db)

    # Session Alice saves a dataset
    meta_a = DatasetMetadata(
        dataset_id="ds_alice_001",
        name="Alice Data.csv",
        file_path=str(csv_a),
        version_hash="vhash_alice",
        row_count=2,
        column_count=3,
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.NUMERIC, dtype="int64"),
            ColumnSchema(name="metric", data_type=ColumnType.CATEGORICAL, dtype="object"),
            ColumnSchema(name="value", data_type=ColumnType.NUMERIC, dtype="float64"),
        ],
        session_id="session_alice",
    )
    repo.save_dataset(meta_a, verify_file=True, session_id="session_alice")

    # Bob's scoped repository
    repo_bob = repo.scoped("session_bob")

    # 1. Bob cannot list Alice's dataset
    bob_datasets = repo_bob.list_datasets()
    assert len(bob_datasets) == 0
    assert not any(d["dataset_id"] == "ds_alice_001" for d in bob_datasets)

    # 2. Bob cannot retrieve Alice's dataset record
    assert repo_bob.get_dataset("ds_alice_001") is None

    # 3. Bob cannot retrieve Alice's dataset metadata descriptor
    assert repo_bob.get_dataset_metadata("ds_alice_001") is None

    # 4. Bob cannot load Alice's dataset DataFrame
    with pytest.raises(IngestionError):
        repo_bob.load_dataset_dataframe("ds_alice_001")

    # 5. Bob cannot verify Alice's dataset file
    with pytest.raises(IngestionError):
        repo_bob.verify_dataset_file("ds_alice_001")

    # 6. Bob cannot overwrite Alice's dataset record
    csv_bob = _create_sample_csv(tmp_path, "bob_tamper.csv")
    meta_tamper = DatasetMetadata(
        dataset_id="ds_alice_001",
        name="Tampered By Bob.csv",
        file_path=str(csv_bob),
        version_hash="vhash_tamper",
        row_count=2,
        column_count=3,
        session_id="session_bob",
    )
    with pytest.raises(AccessDeniedError) as exc_info:
        repo_bob.save_dataset(meta_tamper, verify_file=True)
    assert "Access denied" in str(exc_info.value)

    # 7. Bob cannot delete Alice's dataset
    delete_result = repo_bob.delete_dataset("ds_alice_001", delete_file=True)
    assert delete_result is False

    # Verify Alice's dataset and file remain completely intact
    alice_check = repo.scoped("session_alice").get_dataset("ds_alice_001")
    assert alice_check is not None
    assert alice_check["name"] == "Alice Data.csv"
    assert csv_a.exists()


def test_session_a_cannot_retrieve_session_b_investigation_or_child_records(
    isolated_db: DatabaseManager, tmp_path: Path
):
    """Session A cannot retrieve Session B's investigation or child records (evidence, claims, hypotheses, executions)."""
    csv_a = _create_sample_csv(tmp_path, "alice_inv.csv")
    repo = Repository(db_manager=isolated_db)

    meta_a = DatasetMetadata(
        dataset_id="ds_alice_inv",
        name="Alice Inv Data.csv",
        file_path=str(csv_a),
        version_hash="vhash_alice_inv",
        row_count=2,
        column_count=3,
        session_id="session_alice",
    )
    repo.save_dataset(meta_a, verify_file=True, session_id="session_alice")

    # Build full investigation with evidence items, hypotheses, claims, and executions
    store = EvidenceStore(dataset_id="ds_alice_inv", dataset_version="vhash_alice_inv")
    store.add_evidence(
        EvidenceItem(
            evidence_id="ev_alice_secret",
            evidence_type=EvidenceType.STRUCTURED_DATA,
            source_id="sales_table",
            location="row=1",
            content={"confidential_profit": 99999},
            confidence=0.99,
            related_tool_execution_id="exec_alice_001",
        )
    )
    store.add_hypothesis(
        Hypothesis(
            hypothesis_id="hyp_alice_secret",
            statement="Secret hypothesis on margins",
            status=HypothesisStatus.SUPPORTED,
            confidence=0.95,
            findings_summary="Internal audit supported margin hypothesis",
        )
    )
    store.add_claim(
        Claim(
            claim_id="clm_alice_secret",
            statement="Confidential profit is 99999",
            status=ClaimStatus.SUPPORTED,
            validation_notes="Verified against ground truth",
        )
    )
    store.executions["exec_alice_001"] = ToolExecution(
        execution_id="exec_alice_001",
        dataset_id="ds_alice_inv",
        dataset_version="vhash_alice_inv",
        tool_name="margin_calculator",
        parameters={"year": 2024},
        result={"margin": 0.42},
        success=True,
        execution_time_ms=12.5,
    )

    inv_id = "inv_alice_confidential"
    repo.save_investigation_results(
        investigation_id=inv_id,
        dataset_id="ds_alice_inv",
        question="What are Alice's secret margins?",
        status="completed",
        plan_dict={"objective": "Margin analysis", "tasks": ["calc_margins"]},
        final_answer="Confidential margin report",
        evidence_store=store,
        session_id="session_alice",
    )

    # Bob attempts to access Alice's investigation
    repo_bob = repo.scoped("session_bob")

    # 1. Bob cannot list Alice's investigation
    bob_invs = repo_bob.list_investigations(dataset_id="ds_alice_inv")
    assert len(bob_invs) == 0
    assert repo_bob.list_investigations() == []

    # 2. Bob cannot retrieve Alice's investigation by ID
    assert repo_bob.get_investigation(inv_id) is None

    # 3. Bob cannot reconstruct Alice's investigation state
    assert repo_bob.reconstruct_investigation_state(inv_id) is None

    # 4. Bob cannot hijack or overwrite Alice's investigation
    with pytest.raises(AccessDeniedError) as exc_info:
        repo_bob.save_investigation_results(
            investigation_id=inv_id,
            dataset_id="ds_alice_inv",
            question="Bob hijack attempt",
            status="completed",
            plan_dict={},
            final_answer="Hijacked",
            evidence_store=EvidenceStore(dataset_id="ds_alice_inv", dataset_version="vhash_alice_inv"),
        )
    assert "Access denied" in str(exc_info.value)

    # 5. Verify Alice can still retrieve her full investigation and all child records
    alice_inv = repo.scoped("session_alice").get_investigation(inv_id)
    assert alice_inv is not None
    assert len(alice_inv["evidence_items"]) == 1
    assert alice_inv["evidence_items"][0]["evidence_id"] == "ev_alice_secret"
    assert len(alice_inv["hypotheses"]) == 1
    assert alice_inv["hypotheses"][0]["hypothesis_id"] == "hyp_alice_secret"
    assert len(alice_inv["claims"]) == 1
    assert alice_inv["claims"][0]["claim_id"] == "clm_alice_secret"
    assert len(alice_inv["tool_executions"]) == 1
    assert alice_inv["tool_executions"][0]["execution_id"] == "exec_alice_001"


def test_missing_session_identity_fails_closed(isolated_db: DatabaseManager, tmp_path: Path):
    """Missing session identity fails closed across all normal get/list/load/update/delete operations."""
    csv_file = _create_sample_csv(tmp_path, "missing_session.csv")
    repo = Repository(db_manager=isolated_db)

    # Seed one valid dataset under session_owner
    meta = DatasetMetadata(
        dataset_id="ds_seed_001",
        name="Seed Data.csv",
        file_path=str(csv_file),
        version_hash="vhash_seed",
        row_count=2,
        column_count=3,
        session_id="session_owner",
    )
    repo.save_dataset(meta, verify_file=True, session_id="session_owner")

    # Build unauthenticated / unscoped repository instance
    unscoped_repo = Repository(db_manager=isolated_db, session_id=None)

    # 1. list_datasets fails closed
    with pytest.raises(AccessDeniedError) as exc_info:
        unscoped_repo.list_datasets()
    assert "Valid session context is required" in str(exc_info.value)

    # 2. list_investigations fails closed
    with pytest.raises(AccessDeniedError) as exc_info:
        unscoped_repo.list_investigations()
    assert "Valid session context is required" in str(exc_info.value)

    # 3. get_dataset fails closed
    with pytest.raises(AccessDeniedError) as exc_info:
        unscoped_repo.get_dataset("ds_seed_001")
    assert "Valid session context is required" in str(exc_info.value)

    # 4. get_dataset_metadata fails closed
    with pytest.raises(AccessDeniedError) as exc_info:
        unscoped_repo.get_dataset_metadata("ds_seed_001")
    assert "Valid session context is required" in str(exc_info.value)

    # 5. load_dataset_dataframe fails closed
    with pytest.raises(AccessDeniedError) as exc_info:
        unscoped_repo.load_dataset_dataframe("ds_seed_001")
    assert "Valid session context is required" in str(exc_info.value)

    # 6. verify_dataset_file fails closed
    with pytest.raises(AccessDeniedError) as exc_info:
        unscoped_repo.verify_dataset_file("ds_seed_001")
    assert "Valid session context is required" in str(exc_info.value)

    # 7. get_investigation fails closed
    with pytest.raises(AccessDeniedError) as exc_info:
        unscoped_repo.get_investigation("inv_any")
    assert "Valid session context is required" in str(exc_info.value)

    # 8. reconstruct_investigation_state fails closed
    with pytest.raises(AccessDeniedError) as exc_info:
        unscoped_repo.reconstruct_investigation_state("inv_any")
    assert "Valid session context is required" in str(exc_info.value)

    # 9. delete_dataset fails closed by returning False without altering data
    assert unscoped_repo.delete_dataset("ds_seed_001") is False

    # 10. save_investigation_results fails closed
    store = EvidenceStore(dataset_id="ds_seed_001", dataset_version="vhash_seed")
    with pytest.raises(AccessDeniedError) as exc_info:
        unscoped_repo.save_investigation_results(
            investigation_id="inv_unscoped",
            dataset_id="ds_seed_001",
            question="Unscoped query",
            status="completed",
            plan_dict={},
            final_answer="None",
            evidence_store=store,
            session_id=None,
        )
    assert "Access denied" in str(exc_info.value)

    # 11. Empty string or whitespace session_id also fails closed
    empty_repo = Repository(db_manager=isolated_db, session_id="   ")
    with pytest.raises(AccessDeniedError):
        empty_repo.list_datasets()
    with pytest.raises(AccessDeniedError):
        empty_repo.get_dataset("ds_seed_001")


def test_unrestricted_session_id_resumption_cannot_expose_another_session():
    """Unrestricted session resumption text input is removed from UI to prevent cross-session impersonation."""
    app_py_path = Path("omnisleuth/ui/app.py")
    assert app_py_path.exists(), "app.py must exist"
    app_code = app_py_path.read_text(encoding="utf-8")

    # Verify unrestricted text input for arbitrary session ID is absent
    assert 'st.text_input("Resume session ID"' not in app_code, (
        "Vulnerability regression: 'Resume session ID' input must not be present in app.py"
    )
    assert 'key="reconnect_session_input"' not in app_code, (
        "reconnect_session_input key must be removed from app.py"
    )

    # Verify session isolation notice and browser refresh documentation are present
    assert "Session Isolation" in app_code
    assert "Browser Refresh Notice" in app_code
    assert "btn_start_new_session" in app_code


def test_existing_valid_workflows_still_work_with_explicit_session_context(
    isolated_db: DatabaseManager, tmp_path: Path
):
    """Existing valid workflows execute cleanly when explicit session context is provided."""
    csv_file = _create_sample_csv(tmp_path, "workflow.csv")
    session_id = "session_valid_client"
    repo = Repository(db_manager=isolated_db, session_id=session_id)

    # 1. Save dataset
    meta = DatasetMetadata(
        dataset_id="ds_wf_001",
        name="Workflow.csv",
        file_path=str(csv_file),
        version_hash="vhash_wf",
        row_count=2,
        column_count=3,
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.NUMERIC, dtype="int64"),
            ColumnSchema(name="metric", data_type=ColumnType.CATEGORICAL, dtype="object"),
            ColumnSchema(name="value", data_type=ColumnType.NUMERIC, dtype="float64"),
        ],
        session_id=session_id,
    )
    repo.save_dataset(meta, verify_file=True)

    # 2. List datasets
    listing = repo.list_datasets(check_files=True)
    assert len(listing) == 1
    assert listing[0]["dataset_id"] == "ds_wf_001"
    assert listing[0]["file_exists"] is True

    # 3. Load DataFrame
    df, loaded_meta = repo.load_dataset_dataframe("ds_wf_001")
    assert len(df) == 2
    assert loaded_meta.name == "Workflow.csv"

    # 4. Save investigation results with child records
    store = EvidenceStore(dataset_id="ds_wf_001", dataset_version="vhash_wf")
    store.add_evidence(
        EvidenceItem(
            evidence_id="ev_wf_01",
            evidence_type=EvidenceType.DETERMINISTIC_CALC,
            source_id="calc",
            content={"sum": 30.5},
            confidence=1.0,
        )
    )
    store.add_claim(Claim(claim_id="clm_wf_01", statement="Sum is 30.5", status=ClaimStatus.SUPPORTED))
    store.add_hypothesis(Hypothesis(hypothesis_id="hyp_wf_01", statement="Sum equals 30.5", status=HypothesisStatus.SUPPORTED))

    inv_id = "inv_wf_001"
    repo.save_investigation_results(
        investigation_id=inv_id,
        dataset_id="ds_wf_001",
        question="What is the total value?",
        status="completed",
        plan_dict={"tasks": ["sum_value"]},
        final_answer="Total value is 30.5",
        evidence_store=store,
    )

    # 5. List and retrieve investigation
    inv_list = repo.list_investigations()
    assert len(inv_list) == 1
    assert inv_list[0]["investigation_id"] == inv_id

    reconstructed = repo.reconstruct_investigation_state(inv_id)
    assert reconstructed is not None
    assert reconstructed["final_answer"] == "Total value is 30.5"
    assert len(reconstructed["evidence_store"].list_evidence()) == 1
    assert len(reconstructed["claims"]) == 1
    assert len(reconstructed["hypotheses"]) == 1

    # 6. Delete dataset cascades to investigations and child records
    assert repo.delete_dataset("ds_wf_001", delete_file=True) is True
    assert repo.get_dataset("ds_wf_001") is None
    assert repo.get_investigation(inv_id) is None
    assert len(repo.list_investigations()) == 0


def test_cli_and_internal_maintenance_callers_use_appropriate_access_path(
    isolated_db: DatabaseManager, tmp_path: Path
):
    """CLI callers use explicit session_id='cli_local', while internal maintenance callers use admin_* paths."""
    csv_file = _create_sample_csv(tmp_path, "admin_test.csv")
    repo = Repository(db_manager=isolated_db)

    # Seed datasets under two different user sessions
    meta1 = DatasetMetadata(
        dataset_id="ds_maint_01",
        name="User1.csv",
        file_path=str(csv_file),
        version_hash="v1",
        row_count=2,
        column_count=3,
        session_id="session_user_1",
    )
    repo.save_dataset(meta1, verify_file=True, session_id="session_user_1")

    meta2 = DatasetMetadata(
        dataset_id="ds_maint_02",
        name="User2.csv",
        file_path=str(csv_file),
        version_hash="v2",
        row_count=2,
        column_count=3,
        session_id="session_user_2",
    )
    repo.save_dataset(meta2, verify_file=True, session_id="session_user_2")

    store = EvidenceStore(dataset_id="ds_maint_01", dataset_version="v1")
    repo.save_investigation_results(
        investigation_id="inv_maint_01",
        dataset_id="ds_maint_01",
        question="Q1",
        status="completed",
        plan_dict={},
        final_answer="A1",
        evidence_store=store,
        session_id="session_user_1",
    )

    # 1. Normal user-facing methods cannot view cross-session records
    user1_repo = repo.scoped("session_user_1")
    assert len(user1_repo.list_datasets()) == 1
    assert user1_repo.get_dataset("ds_maint_02") is None

    # 2. Administrative maintenance paths access records globally for audit and maintenance
    admin_datasets = repo.admin_list_all_datasets()
    assert len(admin_datasets) == 2
    ds_ids = {d["dataset_id"] for d in admin_datasets}
    assert "ds_maint_01" in ds_ids
    assert "ds_maint_02" in ds_ids

    # 3. Administrative dataset lookup and metadata reconstruction
    admin_ds1 = repo.admin_get_dataset("ds_maint_01")
    assert admin_ds1 is not None
    assert admin_ds1["name"] == "User1.csv"

    admin_meta1 = repo.admin_get_dataset_metadata("ds_maint_01")
    assert admin_meta1 is not None
    assert admin_meta1.dataset_id == "ds_maint_01"

    # 4. Administrative investigation listing and retrieval
    admin_invs = repo.admin_list_all_investigations()
    assert len(admin_invs) == 1
    assert admin_invs[0]["investigation_id"] == "inv_maint_01"

    admin_inv = repo.admin_get_investigation("inv_maint_01")
    assert admin_inv is not None
    assert admin_inv["question"] == "Q1"

    # 5. Administrative deletion
    assert repo.admin_delete_dataset("ds_maint_01", delete_file=False) is True
    assert repo.admin_get_dataset("ds_maint_01") is None
    assert repo.admin_get_investigation("inv_maint_01") is None
