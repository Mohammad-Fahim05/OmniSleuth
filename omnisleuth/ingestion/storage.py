"""Secure filesystem storage, filename sanitization, and path containment utilities for dataset uploads."""

import os
from pathlib import Path
import re
import time
from typing import Any, Optional, Set, Tuple, Union
import unicodedata
import uuid

from omnisleuth.core.exceptions import IngestionError
from omnisleuth.core.logging import get_logger
from omnisleuth.ingestion.validator import DEFAULT_MAX_UPLOAD_SIZE_BYTES, SUPPORTED_EXTENSIONS

logger = get_logger("ingestion.storage")

# Windows reserved device names (case-insensitive)
WINDOWS_RESERVED_NAMES: Set[str] = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


def sanitize_filename(raw_filename: str) -> str:
    """Validate and sanitize a user-supplied upload filename for safe display metadata.

    Rejects directory traversal sequences, absolute paths, drive letters, UNC paths,
    control characters, Windows reserved device names, and empty or dot-only filenames.
    Preserves valid unicode characters and validated extensions (.csv, .parquet, .xlsx).

    Args:
        raw_filename: Raw filename string provided by the user or client.

    Returns:
        Clean, sanitized display filename.

    Raises:
        IngestionError: If the filename contains forbidden traversal sequences,
            reserved device names, control characters, or unsupported extensions.
    """
    if raw_filename is None or not isinstance(raw_filename, str):
        raise IngestionError("Invalid filename: Filename must be a non-empty string.")

    stripped = raw_filename.strip()
    if not stripped:
        raise IngestionError("Invalid filename: Filename cannot be empty.")

    # 1. Null byte check
    if "\x00" in raw_filename:
        raise IngestionError("Invalid filename: Null bytes are not permitted.")

    # 2. Control characters check (ASCII 0-31 and 127)
    for char in raw_filename:
        code = ord(char)
        if code < 32 or code == 127:
            raise IngestionError(
                f"Invalid filename: Control character (code {code}) is not permitted."
            )

    # 3. Unicode normalization
    normalized = unicodedata.normalize("NFKC", stripped)

    # 4. Dot-only filenames check (e.g. '.', '..', '...')
    if all(c == "." for c in normalized):
        raise IngestionError(
            f"Invalid filename '{raw_filename}': Dot-only filenames are not permitted."
        )

    # 5. Extension-only / hidden files without stem (e.g. '.csv', '.parquet', '.xlsx')
    if normalized.startswith(".") and ("." + normalized.lstrip(".").split(".")[-1].lower() in SUPPORTED_EXTENSIONS) and len(normalized.lstrip(".").split(".")) == 1:
        raise IngestionError(
            f"Invalid filename '{raw_filename}': Filename must have a valid non-empty stem."
        )

    # 6. Traversal sequences check
    if ".." in normalized or ".." in raw_filename:
        raise IngestionError(
            f"Invalid filename '{raw_filename}': Directory traversal sequences ('..') are not permitted."
        )

    # 7. Directory separators check (POSIX '/' and Windows '\')
    if "/" in normalized or "\\" in normalized:
        raise IngestionError(
            f"Invalid filename '{raw_filename}': Directory separators are not permitted in filenames."
        )

    # 8. Windows drive letters check (e.g. C:, D:)
    if re.search(r"^[a-zA-Z]:", normalized):
        raise IngestionError(
            f"Invalid filename '{raw_filename}': Drive letter prefixes are not permitted."
        )

    # 9. UNC network path check
    if normalized.startswith(r"\\") or normalized.startswith("//"):
        raise IngestionError(
            f"Invalid filename '{raw_filename}': UNC paths are not permitted."
        )

    # 9. Extension check
    suffix = Path(normalized).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        sorted_exts = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise IngestionError(
            f"Unsupported file format for '{normalized}': '{suffix or 'none'}'. Supported formats are: {sorted_exts}"
        )

    # 10. Stem verification and Windows reserved device names
    stem = Path(normalized).stem.strip()
    if not stem or all(c == "." for c in stem):
        raise IngestionError(
            f"Invalid filename '{raw_filename}': Filename must have a valid non-empty stem."
        )

    base_device = stem.split(".")[0].upper()
    if base_device in WINDOWS_RESERVED_NAMES:
        raise IngestionError(
            f"Invalid filename '{raw_filename}': Windows reserved device name '{base_device}' is not permitted."
        )

    return normalized


def is_safe_path_contained(
    candidate: Union[str, Path],
    base_dir: Union[str, Path],
) -> bool:
    """Verify that candidate path strictly resolves within base_dir without escaping.

    Args:
        candidate: Candidate file or directory path.
        base_dir: Intended parent directory.

    Returns:
        True if candidate is strictly inside base_dir, False otherwise.
    """
    try:
        resolved_candidate = Path(candidate).resolve()
        resolved_base = Path(base_dir).resolve()
        return (
            resolved_candidate.is_relative_to(resolved_base)
            and resolved_candidate != resolved_base
        )
    except (ValueError, RuntimeError):
        return False


def generate_safe_storage_path(
    display_name: str,
    upload_dir: Optional[Union[str, Path]] = None,
    extension: Optional[str] = None,
) -> Path:
    """Generate a collision-resistant, isolated storage path strictly within upload_dir.

    The storage filename is strictly generated using uuid4 and high-precision timestamps,
    never using user-controlled characters for the storage filename stem.

    Args:
        display_name: Sanitized display filename (used only to extract extension if not provided).
        upload_dir: Designated destination directory (defaults to Settings.uploads_dir).
        extension: Optional explicit extension override.

    Returns:
        Resolved Path guaranteed to be inside upload_dir and non-colliding.

    Raises:
        IngestionError: If upload_dir cannot be created, extension is unsupported,
            or storage path escapes upload_dir.
    """
    if upload_dir is None:
        try:
            from omnisleuth.core.config import get_settings

            upload_dir = get_settings().uploads_dir
        except Exception:
            upload_dir = "data/uploads"

    base_dir = Path(upload_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    ext = extension or Path(display_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        sorted_exts = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise IngestionError(
            f"Unsupported file format: '{ext or 'none'}'. Supported formats are: {sorted_exts}"
        )

    # Generate unique collision-resistant storage filename
    for _ in range(10):
        unique_token = uuid.uuid4().hex
        timestamp_ns = time.time_ns()
        filename = f"up_{unique_token[:16]}_{timestamp_ns}{ext}"
        candidate = (base_dir / filename).resolve()

        if not is_safe_path_contained(candidate, base_dir):
            raise IngestionError(
                "Security violation: Storage path escaped target upload directory."
            )

        if not candidate.exists():
            return candidate

    raise IngestionError("Failed to generate unique non-colliding storage path.")


def safe_save_upload(
    file_obj: Any,
    raw_filename: str,
    upload_dir: Optional[Union[str, Path]] = None,
    max_size_bytes: Optional[int] = None,
) -> Tuple[Path, str]:
    """Safely validate filename, stream content to an isolated storage path, and return (path, display_name).

    Streams in 64 KB chunks, checks size ceilings, ensures containment, uses exclusive creation,
    and cleans up only the newly created file if any validation or write error occurs.

    Args:
        file_obj: File-like object supporting .read() and optionally .seek() and .size.
        raw_filename: User-provided filename string.
        upload_dir: Target directory for stored files.
        max_size_bytes: Maximum allowed size in bytes.

    Returns:
        Tuple of (stored_file_path, sanitized_display_name).

    Raises:
        IngestionError: If filename is invalid, file is empty or oversized,
            or storage writing fails.
    """
    display_name = sanitize_filename(raw_filename)

    if max_size_bytes is None:
        try:
            from omnisleuth.core.config import get_settings

            max_size_bytes = get_settings().max_upload_size_bytes
        except Exception:
            max_size_bytes = DEFAULT_MAX_UPLOAD_SIZE_BYTES

    # Pre-check size if file_obj provides size attribute
    file_size = getattr(file_obj, "size", None)
    if file_size is not None:
        if file_size == 0:
            raise IngestionError(f"Uploaded file '{display_name}' is empty (0 bytes).")
        if file_size > max_size_bytes:
            raise IngestionError(
                f"File size ({file_size / (1024 * 1024):.1f} MB) exceeds maximum allowed size "
                f"({max_size_bytes / (1024 * 1024):.1f} MB)."
            )

    dest_path = generate_safe_storage_path(
        display_name=display_name, upload_dir=upload_dir
    )

    bytes_written = 0
    created = False
    try:
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)

        # Exclusive creation mode 'xb' ensures no file is accidentally overwritten
        with open(dest_path, "xb") as out_f:
            created = True
            while True:
                chunk = file_obj.read(65536)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_size_bytes:
                    raise IngestionError(
                        f"File size exceeds maximum allowed upload size ({max_size_bytes / (1024 * 1024):.1f} MB)."
                    )
                out_f.write(chunk)

        if bytes_written == 0:
            raise IngestionError(f"Uploaded file '{display_name}' is empty (0 bytes).")

        return dest_path, display_name

    except Exception:
        if created and dest_path.exists():
            safe_cleanup_file(dest_path, upload_dir=upload_dir)
        raise


def safe_cleanup_file(
    file_path: Union[str, Path, None],
    upload_dir: Optional[Union[str, Path]] = None,
) -> bool:
    """Safely delete an upload file only if it is a regular file contained within upload_dir.

    Never deletes directories, symlinks, or files outside the designated upload directory.
    If deletion fails, logs a safe diagnostic without exposing absolute paths.

    Args:
        file_path: Path to the file to delete.
        upload_dir: Permitted upload directory. Defaults to Settings.uploads_dir.

    Returns:
        True if file was deleted or did not exist; False if deletion failed or was rejected.
    """
    if file_path is None:
        return False

    path = Path(file_path)
    if not path.exists():
        return True

    # Never delete directories or symlinks
    if path.is_dir() or path.is_symlink():
        logger.warning("Refusing to clean up directory or symlink target.")
        return False

    # Enforce containment: target must be inside upload_dir
    if upload_dir is None:
        try:
            from omnisleuth.core.config import get_settings

            upload_dir = get_settings().uploads_dir
        except Exception:
            upload_dir = "data/uploads"

    base_dir = Path(upload_dir).resolve()
    if not is_safe_path_contained(path, base_dir):
        logger.warning("Refusing to clean up file outside upload directory.")
        return False

    try:
        path.unlink()
        logger.info(f"Safely removed upload file '{path.name}'.")
        return True
    except OSError as exc:
        logger.warning(
            f"Failed to remove file '{path.name}': {exc.strerror or 'Permission denied'}"
        )
        return False
