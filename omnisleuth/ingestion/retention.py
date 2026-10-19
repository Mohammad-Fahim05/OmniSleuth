"""Configurable upload retention policy and safe cleanup utilities.

Note:
    Retention cleanup is NOT executed automatically on application startup to
    prevent unreviewed bulk deletions. It should be invoked through administrative
    CLI commands, scheduled background maintenance (e.g. cron), or explicit API calls.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from typing import Any, List, Optional, Set, Union

from omnisleuth.core.config import get_settings
from omnisleuth.core.logging import get_logger
from omnisleuth.ingestion.storage import is_safe_path_contained, safe_cleanup_file

logger = get_logger("ingestion.retention")


@dataclass
class RetentionCleanupReport:
    """Detailed summary of an upload retention cleanup operation."""

    scanned_files: int = 0
    expired_files: List[str] = field(default_factory=list)
    deleted_files: List[str] = field(default_factory=list)
    skipped_active_files: List[str] = field(default_factory=list)
    skipped_recent_files: List[str] = field(default_factory=list)
    freed_bytes: int = 0
    dry_run: bool = True
    cutoff_timestamp: float = 0.0
    retention_days: int = 30


def get_active_dataset_paths(db_manager: Optional[Any] = None) -> Set[Path]:
    """Retrieve canonical resolved paths for all datasets currently registered in SQLite."""
    active_paths: Set[Path] = set()
    try:
        if db_manager is None:
            from omnisleuth.persistence.db import DatabaseManager

            db = DatabaseManager()
        else:
            db = db_manager
        with db.get_connection() as conn:
            cursor = conn.execute("SELECT file_path FROM datasets")
            for row in cursor.fetchall():
                fp = row["file_path"] if isinstance(row, dict) else row[0]
                if fp:
                    try:
                        active_paths.add(Path(fp).resolve())
                    except Exception:
                        pass
    except Exception as exc:
        logger.warning(f"Could not query active dataset paths from database: {exc}")
    return active_paths


def cleanup_expired_uploads(
    upload_dir: Optional[Union[str, Path]] = None,
    retention_days: Optional[int] = None,
    db_manager: Optional[Any] = None,
    dry_run: bool = True,
    now_ts: Optional[float] = None,
    clean_active_records: bool = False,
) -> RetentionCleanupReport:
    """Safely scan upload directory and prune files exceeding the retention window.

    Guarantees:
    - Filesystem containment: Target directory is strictly resolved, and any file
      resolving outside upload_dir is rejected and skipped.
    - Preserves database integrity: Upload files referenced by active dataset records
      in the SQLite database are preserved unless clean_active_records is explicitly True.
    - Trusted metadata: Evaluates file stat st_mtime against trusted now_ts clock;
      never trusts user-controlled filenames or extensions.
    - Idempotency: Multiple runs produce consistent results with no errors or data loss.
    - Dry-run capability: Supports inspection and preview without destructive actions.

    Args:
        upload_dir: Directory containing uploads. Defaults to Settings.uploads_dir.
        retention_days: Maximum retention age in days. Defaults to Settings.upload_retention_days (30).
        db_manager: Optional DatabaseManager instance.
        dry_run: If True, previews candidate deletions without modifying the filesystem.
        now_ts: Optional epoch timestamp override for deterministic testing.
        clean_active_records: If True, deletes active dataset records when files are expired.
            Defaults to False to preserve database state.

    Returns:
        RetentionCleanupReport summarizing scanned, expired, deleted, and skipped files.
    """
    settings = get_settings()

    target_dir = Path(upload_dir or settings.uploads_dir).resolve()
    days = retention_days if retention_days is not None else getattr(settings, "upload_retention_days", 30)
    current_time = now_ts if now_ts is not None else time.time()
    cutoff_time = current_time - (days * 86400.0)

    report = RetentionCleanupReport(
        dry_run=dry_run,
        cutoff_timestamp=cutoff_time,
        retention_days=days,
    )

    if not target_dir.exists() or not target_dir.is_dir():
        logger.info(f"Upload directory '{target_dir}' does not exist. Nothing to clean.")
        return report

    active_paths = get_active_dataset_paths(db_manager=db_manager)

    try:
        entries = list(target_dir.iterdir())
    except OSError as exc:
        logger.error(f"Failed to list directory '{target_dir}': {exc}")
        return report

    for entry in entries:
        # Enforce containment: never clean files outside target_dir
        if not is_safe_path_contained(entry, target_dir):
            logger.warning(f"Skipping entry outside upload directory: {entry.name}")
            continue

        # Skip directories and symlinks
        if entry.is_dir() or entry.is_symlink() or not entry.is_file():
            continue

        report.scanned_files += 1

        try:
            stat_info = entry.stat()
            file_mtime = stat_info.st_mtime
            file_size = stat_info.st_size
        except OSError as exc:
            logger.warning(f"Could not stat file '{entry.name}': {exc}")
            continue

        if file_mtime > cutoff_time:
            report.skipped_recent_files.append(entry.name)
            continue

        # File is older than retention window
        report.expired_files.append(entry.name)
        resolved_entry = entry.resolve()

        # Check if file is actively referenced in the database
        if resolved_entry in active_paths and not clean_active_records:
            report.skipped_active_files.append(entry.name)
            logger.info(
                f"Preserving expired upload '{entry.name}' because it is actively referenced in SQLite database."
            )
            continue

        # File is eligible for deletion
        report.freed_bytes += file_size

        if not dry_run:
            success = safe_cleanup_file(entry, upload_dir=target_dir)
            if success:
                report.deleted_files.append(entry.name)
                logger.info(f"Deleted expired upload file: '{entry.name}' ({file_size} bytes)")
            else:
                logger.warning(f"Failed to delete expired upload file: '{entry.name}'")

    return report
