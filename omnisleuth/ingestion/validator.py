"""Dataset upload validation and format verification utilities."""

import os
from pathlib import Path
from typing import Optional, Set, Tuple, Union
import zipfile
import pandas as pd

from omnisleuth.core.exceptions import IngestionError
from omnisleuth.core.logging import get_logger

logger = get_logger("ingestion.validator")

# Supported file extensions per product requirements
SUPPORTED_EXTENSIONS: Set[str] = {".csv", ".parquet", ".xlsx"}

# Default upload size limit (50 MB)
DEFAULT_MAX_UPLOAD_SIZE_BYTES: int = 50 * 1024 * 1024

# Default maximum allowed column count (DoS prevention)
DEFAULT_MAX_COLUMNS: int = 1000

# Default maximum allowed row count (DoS prevention)
DEFAULT_MAX_ROWS: int = 1_000_000

# Known binary magic prefixes that must not be parsed as plain text CSV
BINARY_MAGIC_PREFIXES: list[Tuple[bytes, str]] = [
    (b"MZ", "Windows executable (PE)"),
    (b"\x7fELF", "Linux executable (ELF)"),
    (b"PK\x03\x04", "ZIP or OpenXML archive"),
    (b"PAR1", "Parquet file"),
    (b"%PDF", "PDF document"),
    (b"\x89PNG\r\n\x1a\n", "PNG image"),
    (b"GIF87a", "GIF image"),
    (b"GIF89a", "GIF image"),
    (b"\xff\xd8\xff", "JPEG image"),
    (b"\x1f\x8b", "GZIP compressed archive"),
    (b"\x42\x5a\x68", "BZIP2 compressed archive"),
    (b"\xfd7zXZ\x00", "XZ compressed archive"),
]


class DatasetValidator:
    """Validates dataset file existence, size, extensions, content signatures, and tabular viability."""

    def __init__(
        self,
        max_file_size_bytes: Optional[int] = None,
        allowed_extensions: Optional[Set[str]] = None,
        max_columns: Optional[int] = None,
        max_rows: Optional[int] = None,
    ):
        if max_file_size_bytes is not None:
            self.max_file_size_bytes = max_file_size_bytes
        else:
            try:
                from omnisleuth.core.config import get_settings

                self.max_file_size_bytes = get_settings().max_upload_size_bytes
            except Exception:
                self.max_file_size_bytes = DEFAULT_MAX_UPLOAD_SIZE_BYTES

        self.allowed_extensions = allowed_extensions or set(SUPPORTED_EXTENSIONS)
        self.max_columns = max_columns or DEFAULT_MAX_COLUMNS

        if max_rows is not None:
            self.max_rows = max_rows
        else:
            try:
                from omnisleuth.core.config import get_settings

                self.max_rows = get_settings().max_upload_rows
            except Exception:
                self.max_rows = DEFAULT_MAX_ROWS

    def validate_file_metadata(self, file_path: Union[str, Path]) -> Path:
        """Validate file existence, file type, allowed extensions, and size limits."""
        path = Path(file_path)
        if not path.exists():
            raise IngestionError(f"File does not exist: {path.name}")

        if not path.is_file():
            raise IngestionError(f"Provided path is not a file: {path.name}")

        suffix = path.suffix.lower()
        if suffix not in self.allowed_extensions:
            sorted_exts = ", ".join(sorted(self.allowed_extensions))
            raise IngestionError(
                f"Unsupported file format for '{path.name}': '{suffix or 'none'}'. Supported formats are: {sorted_exts}"
            )

        file_size = path.stat().st_size
        if file_size == 0:
            raise IngestionError(f"Uploaded file '{path.name}' is empty (0 bytes).")

        if file_size > self.max_file_size_bytes:
            raise IngestionError(
                f"File size ({file_size / (1024 * 1024):.1f} MB) exceeds maximum allowed size "
                f"({self.max_file_size_bytes / (1024 * 1024):.1f} MB)"
            )

        return path

    def validate_content_signature(self, path: Path) -> None:
        """Inspect file content signatures/magic bytes to catch misleading extensions or corrupt headers."""
        suffix = path.suffix.lower()
        file_size = path.stat().st_size

        if suffix == ".parquet":
            if file_size < 8:
                raise IngestionError(
                    f"Invalid Parquet file '{path.name}': File size ({file_size} bytes) is too small to be a valid Parquet file."
                )
            with open(path, "rb") as f:
                head = f.read(4)
                f.seek(-4, os.SEEK_END)
                tail = f.read(4)
            if head != b"PAR1":
                raise IngestionError(
                    f"Invalid Parquet file '{path.name}': Missing PAR1 header signature."
                )
            if tail != b"PAR1":
                raise IngestionError(
                    f"Invalid Parquet file '{path.name}': Missing PAR1 footer signature."
                )

        elif suffix == ".xlsx":
            with open(path, "rb") as f:
                header = f.read(4)
            if not header.startswith(b"PK\x03\x04"):
                raise IngestionError(
                    f"Invalid Excel file '{path.name}': Not a valid OpenXML (.xlsx) archive."
                )
            try:
                with zipfile.ZipFile(path, "r") as zf:
                    namelist = zf.namelist()

                    # Check for path traversal or dangerous archive member names
                    for entry_name in namelist:
                        if entry_name.startswith("/") or entry_name.startswith("\\") or ".." in entry_name:
                            raise IngestionError(
                                f"Invalid Excel file '{path.name}': Malicious path traversal entry in archive."
                            )
                        # Reject macro-enabled workbooks disguised as .xlsx
                        if "vbaproject.bin" in entry_name.lower():
                            raise IngestionError(
                                f"Invalid Excel file '{path.name}': Macro-enabled workbooks (.xlsm / VBA) are not permitted."
                            )

                    # Zip bomb protection: check total uncompressed size and compression ratio
                    total_uncompressed = sum(info.file_size for info in zf.infolist())
                    max_uncompressed = self.max_file_size_bytes * 10
                    if total_uncompressed > max_uncompressed:
                        raise IngestionError(
                            f"Invalid Excel file '{path.name}': Uncompressed archive size exceeds safe limit."
                        )

                    compressed_size = sum(info.compress_size for info in zf.infolist())
                    if compressed_size > 0 and (total_uncompressed / compressed_size) > 100.0:
                        raise IngestionError(
                            f"Invalid Excel file '{path.name}': Suspicious compression ratio (zip bomb protection)."
                        )

                    if not any(
                        entry in namelist
                        for entry in [
                            "[Content_Types].xml",
                            "xl/workbook.xml",
                            "xl/_rels/workbook.xml.rels",
                        ]
                    ):
                        raise IngestionError(
                            f"Invalid Excel file '{path.name}': Archive does not contain valid OpenXML workbook structure."
                        )
                    bad_member = zf.testzip()
                    if bad_member:
                        raise IngestionError(
                            f"Invalid Excel file '{path.name}': Corrupted archive member '{bad_member}'."
                        )
            except zipfile.BadZipFile:
                raise IngestionError(
                    f"Invalid Excel file '{path.name}': Corrupted or malformed ZIP archive."
                )

        elif suffix == ".csv":
            with open(path, "rb") as f:
                sample = f.read(1024)
            if not sample:
                raise IngestionError(f"Uploaded file '{path.name}' is empty (0 bytes).")
            for magic, desc in BINARY_MAGIC_PREFIXES:
                if sample.startswith(magic):
                    raise IngestionError(
                        f"Invalid CSV file '{path.name}': File appears to be a {desc}, not a plain text CSV."
                    )
            if b"\x00" in sample:
                raise IngestionError(
                    f"Invalid CSV file '{path.name}': Binary content (null bytes) detected."
                )

    def validate_path(self, file_path: Union[str, Path]) -> Path:
        """Validate both file metadata (existence, extension, size) and content signature."""
        path = self.validate_file_metadata(file_path)
        self.validate_content_signature(path)
        return path

    def validate_dataframe(self, df: Optional[pd.DataFrame], dataset_name: str) -> None:
        """Validate that a parsed DataFrame contains usable tabular data (non-empty rows and columns)."""
        if df is None or (df.empty and len(df.columns) == 0):
            raise IngestionError(f"Dataset file '{dataset_name}' has no columns or data.")
        if len(df.columns) == 0:
            raise IngestionError(f"Dataset file '{dataset_name}' contains no columns.")
        if len(df.columns) > self.max_columns:
            raise IngestionError(
                f"Dataset file '{dataset_name}' exceeds maximum allowed column count ({len(df.columns)} > {self.max_columns})."
            )
        if self.max_rows is not None and len(df) > self.max_rows:
            raise IngestionError(
                f"Dataset file '{dataset_name}' exceeds maximum allowed row count "
                f"({len(df):,} > {self.max_rows:,} rows)."
            )
        if len(df) == 0:
            raise IngestionError(f"Dataset file '{dataset_name}' contains headers but no data rows.")


def validate_dataset_file(
    file_path: Union[str, Path],
    max_file_size_bytes: Optional[int] = None,
) -> Path:
    """Convenience helper to validate a dataset file using default configuration."""
    validator = DatasetValidator(max_file_size_bytes=max_file_size_bytes)
    return validator.validate_path(file_path)
