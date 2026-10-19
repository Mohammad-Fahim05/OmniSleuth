"""Comprehensive test suite for Phase D: Dataset Row Limits, Rate Limiting, and Upload Retention."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from omnisleuth.core.config import Settings, get_settings, reset_settings
from omnisleuth.core.exceptions import IngestionError, RateLimitExceededError
from omnisleuth.core.rate_limiter import (
    InMemoryRateLimiter,
    get_rate_limiter,
    reset_rate_limiter,
)
from omnisleuth.ingestion import CSVLoader, DatasetValidator, safe_cleanup_file, safe_save_upload
from omnisleuth.ingestion.retention import (
    RetentionCleanupReport,
    cleanup_expired_uploads,
    get_active_dataset_paths,
)
from omnisleuth.nlq.models import QueryResult
from omnisleuth.nlq.pipeline import AnalysisPipeline
from omnisleuth.persistence.db import DatabaseManager
from omnisleuth.persistence.repository import Repository


# ==============================================================================
# 1. Dataset Row Limits Tests
# ==============================================================================


def test_dataset_at_row_limit_accepted(tmp_path: Path):
    """Datasets with row count exactly equal to max_rows are accepted."""
    csv_file = tmp_path / "exact_limit.csv"
    # Create 5 rows with header
    lines = ["id,value"] + [f"{i},val_{i}" for i in range(5)]
    csv_file.write_text("\n".join(lines), encoding="utf-8")

    loader = CSVLoader(max_rows=5)
    df, meta = loader.load_csv(csv_file)
    assert len(df) == 5
    assert meta.row_count == 5


def test_dataset_below_row_limit_accepted(tmp_path: Path):
    """Datasets with row count below max_rows are accepted normally."""
    csv_file = tmp_path / "below_limit.csv"
    lines = ["id,value"] + [f"{i},val_{i}" for i in range(3)]
    csv_file.write_text("\n".join(lines), encoding="utf-8")

    loader = CSVLoader(max_rows=10)
    df, meta = loader.load_csv(csv_file)
    assert len(df) == 3
    assert meta.row_count == 3


def test_dataset_above_row_limit_rejected_with_clear_error(tmp_path: Path):
    """Datasets with row count exceeding max_rows are rejected with clear error message."""
    csv_file = tmp_path / "above_limit.csv"
    lines = ["id,value"] + [f"{i},val_{i}" for i in range(15)]
    csv_file.write_text("\n".join(lines), encoding="utf-8")

    loader = CSVLoader(max_rows=10)
    with pytest.raises(IngestionError) as exc_info:
        loader.load_csv(csv_file)

    msg = str(exc_info.value)
    assert "exceeds maximum allowed row count" in msg
    assert "10" in msg


def test_bounded_streaming_csv_aborts_early_without_loading_all_chunks(tmp_path: Path):
    """CSVLoader streams chunks and aborts immediately when row limit is reached."""
    csv_file = tmp_path / "stream_limit.csv"
    # 50 rows, limit is 10
    lines = ["id,val"] + [f"{i},data" for i in range(50)]
    csv_file.write_text("\n".join(lines), encoding="utf-8")

    loader = CSVLoader(max_rows=10)
    with pytest.raises(IngestionError) as exc_info:
        loader.load_csv(csv_file)
    assert "exceeds maximum allowed row count" in str(exc_info.value)


def test_parquet_row_limit_checked_via_metadata(tmp_path: Path):
    """Parquet files exceeding row limit are rejected before full dataframe loading."""
    parquet_file = tmp_path / "oversized.parquet"
    df = pd.DataFrame({"id": list(range(20)), "name": [f"item_{i}" for i in range(20)]})
    df.to_parquet(parquet_file)

    loader = CSVLoader(max_rows=10)
    with pytest.raises(IngestionError) as exc_info:
        loader.load_csv(parquet_file)
    assert "exceeds maximum allowed row count" in str(exc_info.value)


def test_excel_row_limit_rejected(tmp_path: Path):
    """Excel files exceeding row limit are rejected."""
    xlsx_file = tmp_path / "oversized.xlsx"
    df = pd.DataFrame({"colA": list(range(15)), "colB": [f"val_{i}" for i in range(15)]})
    df.to_excel(xlsx_file, index=False)

    loader = CSVLoader(max_rows=10)
    with pytest.raises(IngestionError) as exc_info:
        loader.load_csv(xlsx_file)
    assert "exceeds maximum allowed row count" in str(exc_info.value)


def test_empty_and_malformed_inputs_handled_cleanly(tmp_path: Path):
    """Empty files, headers-only, and malformed files are cleanly rejected."""
    loader = CSVLoader(max_rows=100)

    # 1. 0-byte file
    empty_file = tmp_path / "empty.csv"
    empty_file.write_bytes(b"")
    with pytest.raises(IngestionError) as exc_info:
        loader.load_csv(empty_file)
    assert "empty" in str(exc_info.value).lower()

    # 2. Headers only (0 data rows)
    headers_only = tmp_path / "headers_only.csv"
    headers_only.write_text("colA,colB,colC\n", encoding="utf-8")
    with pytest.raises(IngestionError) as exc_info:
        loader.load_csv(headers_only)
    assert "no data rows" in str(exc_info.value).lower()


def test_rejected_upload_leaves_no_orphaned_file(tmp_path: Path):
    """If an upload exceeds row limits, the uploaded file is cleaned up and no DB record exists."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    csv_content = b"col1,col2\n" + b"\n".join([f"{i},val".encode() for i in range(25)])

    class MockUploadFile:
        def __init__(self, content: bytes, name: str):
            self.content = content
            self.name = name
            self.size = len(content)
            self._pos = 0

        def read(self, n=-1):
            if n == -1 or n is None:
                data = self.content[self._pos:]
                self._pos = len(self.content)
                return data
            data = self.content[self._pos:self._pos + n]
            self._pos += len(data)
            return data

        def seek(self, pos):
            self._pos = pos

    upload_file = MockUploadFile(csv_content, "big_file.csv")

    # Step 1: safe_save_upload writes to disk
    dest_path, display_name = safe_save_upload(
        file_obj=upload_file,
        raw_filename=upload_file.name,
        upload_dir=upload_dir,
    )
    assert dest_path.exists()

    # Step 2: Loader with max_rows=10 rejects it
    loader = CSVLoader(max_rows=10)
    with pytest.raises(IngestionError):
        try:
            loader.load_csv(dest_path)
        except IngestionError:
            safe_cleanup_file(dest_path, upload_dir=upload_dir)
            raise

    # File must be cleaned up
    assert not dest_path.exists()


# ==============================================================================
# 2. Thread-Safe In-Memory Rate Limiter Tests
# ==============================================================================


def test_rate_limiter_allows_up_to_capacity_and_blocks_excess():
    """Rate limiter allows requests up to capacity and rejects requests exceeding the ceiling."""
    current_time = 1000.0
    limiter = InMemoryRateLimiter(
        questions_per_minute=3,
        uploads_per_hour=2,
        time_func=lambda: current_time,
    )

    session = "sess_user_alpha"

    # First 3 questions succeed
    limiter.acquire("question", session)
    limiter.acquire("question", session)
    limiter.acquire("question", session)

    # 4th question fails
    with pytest.raises(RateLimitExceededError) as exc_info:
        limiter.acquire("question", session)
    assert "Rate limit exceeded" in str(exc_info.value)
    assert exc_info.value.action == "question"
    assert exc_info.value.retry_after_seconds > 0.0


def test_rate_limiter_sliding_window_expiration_with_injectable_clock():
    """Sliding window expires older requests when clock advances past the window duration."""
    current_time = 1000.0

    def get_time():
        return current_time

    limiter = InMemoryRateLimiter(
        questions_per_minute=2,
        time_func=get_time,
    )
    session = "sess_user_beta"

    limiter.acquire("question", session)
    limiter.acquire("question", session)

    # Reaching limit
    assert not limiter.is_allowed("question", session)

    # Advance clock by 30 seconds: still blocked (window is 60s)
    current_time += 30.0
    assert not limiter.is_allowed("question", session)

    # Advance clock past 60s from first request
    current_time += 31.0  # now 1061.0 (> 1000.0 + 60.0)
    # The first timestamp expired, so 1 slot is now open!
    assert limiter.is_allowed("question", session)
    limiter.acquire("question", session)


def test_rate_limiter_session_isolation():
    """Rate limits for session A do not impact session B."""
    current_time = 1000.0
    limiter = InMemoryRateLimiter(
        questions_per_minute=2,
        time_func=lambda: current_time,
    )

    session_a = "sess_user_a"
    session_b = "sess_user_b"

    limiter.acquire("question", session_a)
    limiter.acquire("question", session_a)
    assert not limiter.is_allowed("question", session_a)

    # Session B is unaffected
    assert limiter.is_allowed("question", session_b)
    limiter.acquire("question", session_b)
    assert limiter.is_allowed("question", session_b)


def test_rate_limiter_thread_safety_concurrency():
    """Concurrent requests cannot bypass rate limits due to race conditions."""
    limiter = InMemoryRateLimiter(
        questions_per_minute=10,
        time_func=time.time,
    )
    session = "sess_concurrent_test"

    success_count = 0
    failure_count = 0

    def attempt_acquire():
        nonlocal success_count, failure_count
        try:
            limiter.acquire("question", session)
            return True
        except RateLimitExceededError:
            return False

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(lambda _: attempt_acquire(), range(50)))

    successes = sum(1 for r in results if r is True)
    failures = sum(1 for r in results if r is False)

    # Exactly 10 succeed, exactly 40 fail
    assert successes == 10
    assert failures == 40


def test_rate_limiter_cli_sessions_exempt():
    """CLI sessions are exempt from interactive session rate limiting."""
    limiter = InMemoryRateLimiter(
        questions_per_minute=1,
        time_func=time.time,
    )

    # cli_local is exempt
    for _ in range(10):
        limiter.acquire("question", "cli_local")
        limiter.acquire("upload", "cli_local")


def test_rate_limiter_prunes_expired_entries_and_cleans_memory():
    """Memory cleanup removes expired timestamps and inactive session entries."""
    current_time = 1000.0

    def get_time():
        return current_time

    limiter = InMemoryRateLimiter(
        questions_per_minute=5,
        time_func=get_time,
    )

    limiter.acquire("question", "sess_prune_1")
    limiter.acquire("question", "sess_prune_2")

    assert len(limiter._records["question"]) == 2

    # Advance time by 61 seconds (past window)
    current_time += 61.0
    active_count = limiter.prune_expired()

    # All expired sessions were pruned
    assert active_count == 0
    assert len(limiter._records["question"]) == 0


def test_pipeline_blocked_by_rate_limit_invokes_zero_planners_or_tools():
    """When a session is rate-limited, AnalysisPipeline returns blocked result without calling planners or tools."""
    mock_planner = MagicMock()
    mock_executor = MagicMock()

    current_time = 1000.0
    custom_limiter = InMemoryRateLimiter(
        questions_per_minute=1,
        time_func=lambda: current_time,
    )

    pipeline = AnalysisPipeline(
        mode="rule",
        rule_planner=mock_planner,
        executor=mock_executor,
        rate_limiter=custom_limiter,
    )

    df = pd.DataFrame({"revenue": [100, 200], "category": ["A", "B"]})
    meta = MagicMock()
    meta.dataset_id = "ds_test_123"
    meta.name = "Sales"
    meta.columns = []
    meta.session_id = "sess_rate_target"

    # First question is allowed (rule_planner runs)
    from omnisleuth.nlq.models import AnalysisPlan, PlanIntent

    valid_plan = AnalysisPlan(
        intent=PlanIntent.QUERY,
    )
    mock_planner.plan.return_value = valid_plan
    mock_executor.execute.return_value = QueryResult(
        status="answered",
        answer="Revenue is 300",
        plan=valid_plan,
        plain_language_summary="Revenue is 300",
    )
    routing1 = pipeline.run("What is revenue?", df, meta, session_id="sess_rate_target")
    assert routing1.result.status == "answered"
    assert mock_planner.plan.call_count == 1

    # Second question exceeds rate limit
    mock_planner.reset_mock()
    mock_executor.reset_mock()

    routing2 = pipeline.run("What is revenue again?", df, meta, session_id="sess_rate_target")
    assert routing2.result.status == "blocked"
    assert "Rate Limit Exceeded" in routing2.result.answer

    # Planners and executors were NEVER called!
    assert mock_planner.plan.call_count == 0
    assert mock_executor.execute.call_count == 0


# ==============================================================================
# 3. Upload Retention Policy Tests
# ==============================================================================


def test_retention_cleanup_dry_run_does_not_delete_files(tmp_path: Path):
    """Dry-run mode previews eligible files without modifying the filesystem."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    old_file = upload_dir / "old_upload.csv"
    old_file.write_text("a,b\n1,2\n")

    # Set mtime to 40 days ago
    now = time.time()
    forty_days_ago = now - (40 * 86400)
    os.utime(old_file, (forty_days_ago, forty_days_ago))

    report = cleanup_expired_uploads(
        upload_dir=upload_dir,
        retention_days=30,
        dry_run=True,
        now_ts=now,
    )

    assert report.scanned_files == 1
    assert "old_upload.csv" in report.expired_files
    assert len(report.deleted_files) == 0
    assert report.dry_run is True
    # File must still exist
    assert old_file.exists()


def test_retention_cleanup_deletes_expired_unreferenced_files(tmp_path: Path):
    """When dry_run=False, expired unreferenced files are deleted and space is freed."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    old_file = upload_dir / "expired_unreferenced.csv"
    old_file.write_text("x,y\n10,20\n")

    now = time.time()
    forty_days_ago = now - (40 * 86400)
    os.utime(old_file, (forty_days_ago, forty_days_ago))

    report = cleanup_expired_uploads(
        upload_dir=upload_dir,
        retention_days=30,
        dry_run=False,
        now_ts=now,
    )

    assert report.scanned_files == 1
    assert "expired_unreferenced.csv" in report.expired_files
    assert "expired_unreferenced.csv" in report.deleted_files
    assert report.freed_bytes > 0
    assert not old_file.exists()


def test_retention_cleanup_preserves_recent_files(tmp_path: Path):
    """Recent files within retention window are preserved."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    recent_file = upload_dir / "recent.csv"
    recent_file.write_text("c1,c2\n1,2\n")

    now = time.time()
    five_days_ago = now - (5 * 86400)
    os.utime(recent_file, (five_days_ago, five_days_ago))

    report = cleanup_expired_uploads(
        upload_dir=upload_dir,
        retention_days=30,
        dry_run=False,
        now_ts=now,
    )

    assert report.scanned_files == 1
    assert "recent.csv" in report.skipped_recent_files
    assert len(report.deleted_files) == 0
    assert recent_file.exists()


def test_retention_cleanup_preserves_active_files_referenced_in_db(tmp_path: Path):
    """Expired files that are actively registered in SQLite datasets table are preserved."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    active_file = upload_dir / "active_dataset.csv"
    active_file.write_text("col1,col2\n100,200\n")

    now = time.time()
    forty_days_ago = now - (40 * 86400)
    os.utime(active_file, (forty_days_ago, forty_days_ago))

    # Set up mock DatabaseManager returning this active file
    mock_db = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [{"file_path": str(active_file.resolve())}]
    mock_conn.execute.return_value = mock_cursor
    mock_db.get_connection.return_value.__enter__.return_value = mock_conn

    report = cleanup_expired_uploads(
        upload_dir=upload_dir,
        retention_days=30,
        db_manager=mock_db,
        dry_run=False,
        now_ts=now,
    )

    assert report.scanned_files == 1
    assert "active_dataset.csv" in report.expired_files
    assert "active_dataset.csv" in report.skipped_active_files
    assert len(report.deleted_files) == 0
    # File must be preserved to protect database integrity!
    assert active_file.exists()


def test_retention_cleanup_idempotent(tmp_path: Path):
    """Running retention cleanup repeatedly is idempotent with zero errors."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    old_file = upload_dir / "old.csv"
    old_file.write_text("a,b\n1,2\n")

    now = time.time()
    fifty_days_ago = now - (50 * 86400)
    os.utime(old_file, (fifty_days_ago, fifty_days_ago))

    # Run 1: deletes the file
    report1 = cleanup_expired_uploads(
        upload_dir=upload_dir,
        retention_days=30,
        dry_run=False,
        now_ts=now,
    )
    assert len(report1.deleted_files) == 1
    assert not old_file.exists()

    # Run 2: immediately afterwards, finding 0 files, 0 errors
    report2 = cleanup_expired_uploads(
        upload_dir=upload_dir,
        retention_days=30,
        dry_run=False,
        now_ts=now,
    )
    assert report2.scanned_files == 0
    assert len(report2.deleted_files) == 0
    assert report2.freed_bytes == 0


def test_retention_cleanup_rejects_subdirectories_and_escapes(tmp_path: Path):
    """Subdirectories, symlinks, and exterior path traversal attempts are never deleted."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    sub_dir = upload_dir / "subfolder"
    sub_dir.mkdir()

    now = time.time()
    report = cleanup_expired_uploads(
        upload_dir=upload_dir,
        retention_days=30,
        dry_run=False,
        now_ts=now,
    )

    # Subdirectory was not deleted
    assert sub_dir.exists()
    assert report.scanned_files == 0
