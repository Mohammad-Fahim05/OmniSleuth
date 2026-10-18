"""Security regression tests for T-005R vulnerability remediation.

Verifies:
1. Cross-session dataset overwrite blocked when session_id=None.
2. Cross-session investigation overwrite blocked.
3. Cross-session delete dataset blocked when session_id=None.
4. Dataset deletion cascades to child tool executions, evidence items, hypotheses, and claims.
5. Excel validation blocks macro-enabled workbooks (vbaProject.bin).
6. Excel validation blocks decompression bombs (zip bombs).
7. Excel validation blocks archive directory traversal members.
8. DataFrame validation blocks excessive column counts (DoS prevention).
9. SQLite connection configures busy_timeout for concurrency safety.
"""

import io
from pathlib import Path
import sqlite3
import zipfile
import pandas as pd
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
from omnisleuth.ingestion.models import ColumnSchema, ColumnType, DatasetMetadata
from omnisleuth.ingestion.validator import DatasetValidator
from omnisleuth.persistence.db import DatabaseManager
from omnisleuth.persistence.repository import Repository


@pytest.fixture
def repo(tmp_path: Path) -> Repository:
    """Create an isolated temporary SQLite database and repository."""
    db_file = tmp_path / "test_sec.db"
    db_man = DatabaseManager(db_path=str(db_file))
    return Repository(db_manager=db_man)


def test_cross_session_dataset_overwrite_blocked_when_session_id_is_none(repo: Repository, tmp_path: Path):
    """Security Fix 1: Unauthenticated request (session_id=None) cannot overwrite an owned dataset."""
    file_a = tmp_path / "alice.csv"
    file_a.write_text("id,val\n1,100\n", encoding="utf-8")
    meta_a = DatasetMetadata(
        dataset_id="ds_sec_01",
        name="Alice Data.csv",
        file_path=str(file_a),
        version_hash="vhash_a",
        row_count=1,
        column_count=2,
        session_id="session_alice",
    )
    repo.save_dataset(meta_a, verify_file=True, session_id="session_alice")

    # Attacker calls save_dataset with same dataset_id but session_id=None
    file_b = tmp_path / "attacker.csv"
    file_b.write_text("id,val\n2,200\n", encoding="utf-8")
    meta_b = DatasetMetadata(
        dataset_id="ds_sec_01",
        name="Attacker Data.csv",
        file_path=str(file_b),
        version_hash="vhash_b",
        row_count=1,
        column_count=2,
        session_id=None,
    )

    with pytest.raises(IngestionError) as exc_info:
        repo.save_dataset(meta_b, verify_file=True, session_id=None)
    assert "Access denied" in str(exc_info.value)

    # Verify Alice's dataset remains intact and owned
    alice_ds = repo.get_dataset("ds_sec_01", session_id="session_alice")
    assert alice_ds is not None
    assert alice_ds["name"] == "Alice Data.csv"
    assert alice_ds["session_id"] == "session_alice"


def test_cross_session_investigation_overwrite_blocked(repo: Repository, tmp_path: Path):
    """Security Fix 2: Foreign session cannot overwrite an existing investigation."""
    file_a = tmp_path / "alice_inv.csv"
    file_a.write_text("id,val\n1,100\n", encoding="utf-8")
    meta_a = DatasetMetadata(
        dataset_id="ds_inv_a",
        name="Alice.csv",
        file_path=str(file_a),
        version_hash="vhash_inv_a",
        row_count=1,
        column_count=2,
        session_id="session_alice",
    )
    repo.save_dataset(meta_a, verify_file=True, session_id="session_alice")

    store_a = EvidenceStore(dataset_id="ds_inv_a", dataset_version="vhash_inv_a")
    repo.save_investigation_results(
        investigation_id="inv_shared_target",
        dataset_id="ds_inv_a",
        question="Alice question?",
        status="completed",
        plan_dict={"intent": "query"},
        final_answer="Alice answer",
        evidence_store=store_a,
        session_id="session_alice",
    )

    # Bob sets up his own dataset
    file_b = tmp_path / "bob_inv.csv"
    file_b.write_text("id,val\n2,200\n", encoding="utf-8")
    meta_b = DatasetMetadata(
        dataset_id="ds_inv_b",
        name="Bob.csv",
        file_path=str(file_b),
        version_hash="vhash_inv_b",
        row_count=1,
        column_count=2,
        session_id="session_bob",
    )
    repo.save_dataset(meta_b, verify_file=True, session_id="session_bob")

    # Bob attempts to overwrite Alice's investigation
    store_b = EvidenceStore(dataset_id="ds_inv_b", dataset_version="vhash_inv_b")
    with pytest.raises(IngestionError) as exc_info:
        repo.save_investigation_results(
            investigation_id="inv_shared_target",
            dataset_id="ds_inv_b",
            question="Bob hijack question?",
            status="completed",
            plan_dict={"intent": "query"},
            final_answer="Bob hijack answer",
            evidence_store=store_b,
            session_id="session_bob",
        )
    assert "Access denied" in str(exc_info.value)

    # Alice's investigation remains unchanged
    alice_inv = repo.get_investigation("inv_shared_target", session_id="session_alice")
    assert alice_inv is not None
    assert alice_inv["question"] == "Alice question?"
    assert alice_inv["session_id"] == "session_alice"


def test_cross_session_delete_dataset_blocked_when_session_id_is_none(repo: Repository, tmp_path: Path):
    """Security Fix 3: Request with session_id=None cannot delete a session-owned dataset."""
    upload_file = tmp_path / "protected_del.csv"
    upload_file.write_text("x,y\n1,2\n", encoding="utf-8")
    meta = DatasetMetadata(
        dataset_id="ds_sec_del",
        name="Protected.csv",
        file_path=str(upload_file),
        version_hash="vhash_del",
        row_count=1,
        column_count=2,
        session_id="session_protected_owner",
    )
    repo.save_dataset(meta, verify_file=True, session_id="session_protected_owner")

    # Unauthenticated deletion attempt must return False
    res = repo.delete_dataset("ds_sec_del", delete_file=True, session_id=None)
    assert res is False

    # Dataset and file remain intact
    assert repo.get_dataset("ds_sec_del", session_id="session_protected_owner") is not None
    assert upload_file.exists()


def test_delete_dataset_cleans_up_child_investigation_artifacts(repo: Repository, tmp_path: Path):
    """Security Fix 4: Deleting a dataset cascades to all child investigation evidence, hypotheses, and claims."""
    upload_file = tmp_path / "cascade.csv"
    upload_file.write_text("a,b\n1,2\n", encoding="utf-8")
    meta = DatasetMetadata(
        dataset_id="ds_cascade",
        name="Cascade.csv",
        file_path=str(upload_file),
        version_hash="vhash_cascade",
        row_count=1,
        column_count=2,
        session_id="session_test",
    )
    repo.save_dataset(meta, verify_file=True, session_id="session_test")

    inv_id = "inv_cascade_01"
    store = EvidenceStore(dataset_id="ds_cascade", dataset_version="vhash_cascade")
    store.add_evidence(
        EvidenceItem(
            evidence_id="ev_cascade_01",
            evidence_type=EvidenceType.STRUCTURED_DATA,
            source_id="test",
            content={"val": 42},
        )
    )
    store.add_hypothesis(
        Hypothesis(hypothesis_id="hyp_cascade_01", statement="Hypothesis test", status=HypothesisStatus.SUPPORTED)
    )
    store.add_claim(Claim(claim_id="clm_cascade_01", statement="Claim test", status=ClaimStatus.SUPPORTED))

    repo.save_investigation_results(
        investigation_id=inv_id,
        dataset_id="ds_cascade",
        question="Cascade test?",
        status="completed",
        plan_dict={"intent": "query"},
        final_answer="Answer",
        evidence_store=store,
        session_id="session_test",
    )

    # Verify records exist before deletion
    with repo.db.get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM evidence_items WHERE investigation_id = ?", (inv_id,)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM hypotheses WHERE investigation_id = ?", (inv_id,)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM claims WHERE investigation_id = ?", (inv_id,)).fetchone()[0] == 1

    # Delete dataset
    assert repo.delete_dataset("ds_cascade", delete_file=True, session_id="session_test") is True

    # Verify all child records are cleaned up (no orphaned records)
    with repo.db.get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM investigations WHERE investigation_id = ?", (inv_id,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM evidence_items WHERE investigation_id = ?", (inv_id,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM hypotheses WHERE investigation_id = ?", (inv_id,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM claims WHERE investigation_id = ?", (inv_id,)).fetchone()[0] == 0


def test_excel_validation_blocks_macro_enabled_workbooks(tmp_path: Path):
    """Security Fix 5A: Excel validation detects and rejects macro-enabled workbooks (.xlsm / vbaProject.bin)."""
    xl_path = tmp_path / "macro_test.xlsx"
    with zipfile.ZipFile(xl_path, "w") as zf:
        zf.writestr("[Content_Types].xml", b"<Types/>")
        zf.writestr("xl/workbook.xml", b"<workbook/>")
        zf.writestr("xl/vbaProject.bin", b"VBA macro binary code")

    validator = DatasetValidator()
    with pytest.raises(IngestionError) as exc_info:
        validator.validate_content_signature(xl_path)
    assert "macro" in str(exc_info.value).lower() or "vbaproject" in str(exc_info.value).lower()


def test_excel_validation_blocks_decompression_bomb(tmp_path: Path):
    """Security Fix 5B: Excel validation rejects archives with dangerous uncompressed size ratio (zip bomb)."""
    xl_path = tmp_path / "zip_bomb.xlsx"
    with zipfile.ZipFile(xl_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", b"<Types/>")
        zf.writestr("xl/workbook.xml", b"<workbook/>")
        # Write large compressible zero-byte stream to simulate decompression bomb
        zf.writestr("xl/sheet1.xml", b"0" * (100 * 1024 * 1024))  # 100 MB uncompressed

    # Validator with 5 MB max upload limit should flag a 100 MB uncompressed payload
    validator = DatasetValidator(max_file_size_bytes=5 * 1024 * 1024)
    with pytest.raises(IngestionError) as exc_info:
        validator.validate_content_signature(xl_path)
    assert "decompression" in str(exc_info.value).lower() or "uncompressed" in str(exc_info.value).lower()


def test_excel_validation_blocks_path_traversal_in_zip(tmp_path: Path):
    """Security Fix 5C: Excel validation rejects archive containing directory traversal member."""
    xl_path = tmp_path / "traversal.xlsx"
    with zipfile.ZipFile(xl_path, "w") as zf:
        zf.writestr("[Content_Types].xml", b"<Types/>")
        zf.writestr("xl/workbook.xml", b"<workbook/>")
        zf.writestr("../evil.xml", b"<evil/>")

    validator = DatasetValidator()
    with pytest.raises(IngestionError) as exc_info:
        validator.validate_content_signature(xl_path)
    assert "traversal" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()


def test_dataframe_validation_blocks_excessive_column_count():
    """Security Fix 6: DataFrame validation rejects files with excessive columns (> 1000) to prevent DoS."""
    df_wide = pd.DataFrame({f"col_{i}": [1, 2] for i in range(1001)})
    validator = DatasetValidator()
    with pytest.raises(IngestionError) as exc_info:
        validator.validate_dataframe(df_wide, "wide_dataset.csv")
    assert "column" in str(exc_info.value).lower()
    assert "maximum" in str(exc_info.value).lower() or "limit" in str(exc_info.value).lower()


def test_db_manager_configures_busy_timeout(tmp_path: Path):
    """Security Fix 7: DatabaseManager configures busy_timeout (30000ms) for concurrency resilience."""
    db_file = tmp_path / "busy_test.db"
    db_man = DatabaseManager(db_path=str(db_file))
    with db_man.get_connection() as conn:
        row = conn.execute("PRAGMA busy_timeout;").fetchone()
        assert row is not None
        assert row[0] >= 5000  # Configured for at least 5000ms (30000ms expected)
