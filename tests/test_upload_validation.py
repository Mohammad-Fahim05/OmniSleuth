"""Unit and regression tests for T-005.1: Secure file upload validation and format verification.

Covers:
1. Valid CSV, Parquet, and XLSX uploads.
2. Unsupported file extensions (.txt, .xls, .json, .exe, etc.).
3. Misleading/renamed extensions (text -> parquet, csv -> xlsx, zip -> xlsx, exe -> csv).
4. Oversized uploads rejected before parsing.
5. Empty files (0 bytes, whitespace-only, headers-only).
6. Malformed CSV, Parquet, and XLSX inputs.
7. Missing optional Parquet engine dependency handling.
8. Error message sanitization (no internal path leakage).
9. Safe processing (failed uploads do not leave partial state).
"""

import io
import sys
from pathlib import Path
import unicodedata
from unittest.mock import patch
import zipfile
import pandas as pd
import pytest

from omnisleuth.core.config import Settings
from omnisleuth.core.exceptions import IngestionError
from omnisleuth.ingestion import (
    CSVLoader,
    DatasetValidator,
    generate_safe_storage_path,
    is_safe_path_contained,
    safe_save_upload,
    sanitize_filename,
    validate_dataset_file,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Standard tabular dataframe with multiple data types."""
    return pd.DataFrame({
        "order_id": ["ORD-101", "ORD-102", "ORD-103", "ORD-104"],
        "customer": ["Alice", "Bob", "Charlie", "David"],
        "amount": [150.50, 220.00, 89.99, 450.00],
        "quantity": [2, 1, 3, 5],
        "is_returned": [False, False, True, False],
    })


@pytest.fixture
def valid_csv_file(tmp_path: Path, sample_df: pd.DataFrame) -> Path:
    csv_path = tmp_path / "valid_data.csv"
    sample_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def valid_parquet_file(tmp_path: Path, sample_df: pd.DataFrame) -> Path:
    pq_path = tmp_path / "valid_data.parquet"
    sample_df.to_parquet(pq_path, index=False)
    return pq_path


@pytest.fixture
def valid_xlsx_file(tmp_path: Path, sample_df: pd.DataFrame) -> Path:
    xlsx_path = tmp_path / "valid_data.xlsx"
    sample_df.to_excel(xlsx_path, index=False)
    return xlsx_path


# ---------------------------------------------------------------------------
# 1. Valid Format Tests
# ---------------------------------------------------------------------------

def test_valid_csv_upload(valid_csv_file: Path):
    """Test standard CSV file loading and metadata generation."""
    loader = CSVLoader()
    df, meta = loader.load_csv(valid_csv_file)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert meta.row_count == 4
    assert meta.column_count == 5
    assert "order_id" in meta.identifier_columns
    assert "amount" in meta.numeric_columns
    assert meta.dataset_id.startswith("ds_")


def test_valid_parquet_upload(valid_parquet_file: Path):
    """Test standard Parquet file loading, schema detection, and version hashing."""
    loader = CSVLoader()
    df, meta = loader.load_dataset(valid_parquet_file)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert meta.row_count == 4
    assert meta.column_count == 5
    assert "customer" in meta.categorical_columns or "customer" in meta.identifier_columns
    assert "amount" in meta.numeric_columns
    assert len(meta.version_hash) == 64


def test_valid_xlsx_upload(valid_xlsx_file: Path):
    """Test standard XLSX file loading with OpenXML archive validation."""
    loader = CSVLoader()
    df, meta = loader.load_csv(valid_xlsx_file)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert meta.row_count == 4
    assert meta.column_count == 5
    assert "quantity" in meta.numeric_columns


# ---------------------------------------------------------------------------
# 2. Unsupported Extensions & Misleading Files
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ext", [".txt", ".xls", ".json", ".tsv", ".exe", ".sh", ""])
def test_unsupported_extensions_rejected(tmp_path: Path, ext: str):
    """Reject formats outside allowed set (.csv, .parquet, .xlsx), including legacy .txt and .xls."""
    filename = f"dataset{ext}" if ext else "dataset_no_ext"
    target = tmp_path / filename
    target.write_bytes(b"col1,col2\n1,2\n")

    loader = CSVLoader()
    with pytest.raises(IngestionError, match="Unsupported file format"):
        loader.load_csv(target)


def test_renamed_text_as_parquet_rejected(tmp_path: Path):
    """A plain text or CSV file renamed to .parquet must be rejected by magic bytes."""
    fake_parquet = tmp_path / "fake.parquet"
    fake_parquet.write_text("order_id,amount\n1,100\n2,200\n", encoding="utf-8")

    loader = CSVLoader()
    with pytest.raises(IngestionError, match="Missing PAR1 header"):
        loader.load_csv(fake_parquet)


def test_renamed_csv_as_xlsx_rejected(tmp_path: Path):
    """A plain text CSV file renamed to .xlsx must be rejected by missing PK zip signature."""
    fake_xlsx = tmp_path / "fake.xlsx"
    fake_xlsx.write_text("order_id,amount\n1,100\n2,200\n", encoding="utf-8")

    loader = CSVLoader()
    with pytest.raises(IngestionError, match="Not a valid OpenXML"):
        loader.load_csv(fake_xlsx)


def test_generic_zip_renamed_as_xlsx_rejected(tmp_path: Path):
    """A non-OpenXML zip file renamed to .xlsx must be rejected for missing workbook structure."""
    zip_path = tmp_path / "non_excel.xlsx"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("readme.txt", "This is just a text file inside a zip.")

    loader = CSVLoader()
    with pytest.raises(IngestionError, match="Archive does not contain valid OpenXML"):
        loader.load_csv(zip_path)


def test_executable_renamed_as_csv_rejected(tmp_path: Path):
    """A Windows PE executable (starting with MZ) renamed to .csv must be rejected."""
    fake_csv = tmp_path / "malware.csv"
    fake_csv.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00")

    loader = CSVLoader()
    with pytest.raises(IngestionError, match="Windows executable"):
        loader.load_csv(fake_csv)


def test_linux_elf_binary_renamed_as_csv_rejected(tmp_path: Path):
    """A Linux ELF binary (starting with \x7fELF) renamed to .csv must be rejected."""
    fake_csv = tmp_path / "program.csv"
    fake_csv.write_bytes(b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00")

    loader = CSVLoader()
    with pytest.raises(IngestionError, match="Linux executable"):
        loader.load_csv(fake_csv)


def test_binary_null_bytes_in_csv_rejected(tmp_path: Path):
    """A file containing binary null bytes renamed to .csv must be rejected."""
    fake_csv = tmp_path / "binary_data.csv"
    fake_csv.write_bytes(b"col_a,col_b\nval_1\x00\x00\x01,val_2\n")

    loader = CSVLoader()
    with pytest.raises(IngestionError, match="Binary content.*null bytes"):
        loader.load_csv(fake_csv)


# ---------------------------------------------------------------------------
# 3. File Size Limit Tests
# ---------------------------------------------------------------------------

def test_oversized_file_rejected_before_parsing(tmp_path: Path):
    """Files exceeding max_file_size_bytes must be rejected immediately."""
    oversized_file = tmp_path / "big_dataset.csv"
    # Write 1000 bytes
    oversized_file.write_bytes(b"a,b\n" + b"1,2\n" * 200)

    # Set tiny threshold of 50 bytes
    loader = CSVLoader(max_file_size_bytes=50)
    with pytest.raises(IngestionError, match="exceeds maximum allowed size"):
        loader.load_csv(oversized_file)


def test_default_upload_size_from_settings():
    """CSVLoader defaults to max_upload_size_bytes configured in Settings (50 MB)."""
    loader = CSVLoader()
    assert loader.max_file_size_bytes == 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# 4. Empty and Unusable Datasets
# ---------------------------------------------------------------------------

def test_zero_byte_empty_file_rejected(tmp_path: Path):
    """0-byte file must be rejected with clear empty file error."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_bytes(b"")

    loader = CSVLoader()
    with pytest.raises(IngestionError, match="empty \\(0 bytes\\)"):
        loader.load_csv(empty_file)


def test_whitespace_only_file_rejected(tmp_path: Path):
    """A file containing only spaces or newlines must be rejected."""
    blank_file = tmp_path / "blank.csv"
    blank_file.write_text("   \n\n  \t  \n", encoding="utf-8")

    loader = CSVLoader()
    with pytest.raises(IngestionError, match="empty"):
        loader.load_csv(blank_file)


def test_headers_with_no_rows_rejected(tmp_path: Path):
    """A CSV containing headers but 0 data rows must be rejected as unusable."""
    headers_only = tmp_path / "no_data.csv"
    headers_only.write_text("order_id,customer,amount\n", encoding="utf-8")

    loader = CSVLoader()
    with pytest.raises(IngestionError, match="contains headers but no data rows"):
        loader.load_csv(headers_only)


# ---------------------------------------------------------------------------
# 5. Malformed Inputs
# ---------------------------------------------------------------------------

def test_malformed_parquet_corrupted_data_rejected(tmp_path: Path):
    """Parquet file with valid PAR1 header but truncated or corrupted body must fail gracefully."""
    corrupt_pq = tmp_path / "corrupt.parquet"
    corrupt_pq.write_bytes(b"PAR1\x00\x01\x02\x03\x04\x05\x06\x07PAR1")

    loader = CSVLoader()
    with pytest.raises(IngestionError, match="Invalid or corrupt Parquet data"):
        loader.load_csv(corrupt_pq)


def test_malformed_xlsx_corrupted_archive_rejected(tmp_path: Path):
    """XLSX file with valid PK header but corrupted ZIP central directory must fail gracefully."""
    corrupt_xlsx = tmp_path / "corrupt.xlsx"
    corrupt_xlsx.write_bytes(b"PK\x03\x04\x14\x00\x00\x00\x08\x00corrupted_zip_payload")

    loader = CSVLoader()
    with pytest.raises(IngestionError, match="Corrupted or malformed ZIP archive"):
        loader.load_csv(corrupt_xlsx)


# ---------------------------------------------------------------------------
# 6. Missing Parquet Engine Handling
# ---------------------------------------------------------------------------

def test_missing_parquet_engine_handling(valid_parquet_file: Path, monkeypatch):
    """When both pyarrow and fastparquet are unavailable, a clear IngestionError is raised."""
    # Hide pyarrow and fastparquet
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "fastparquet", None)

    loader = CSVLoader()
    with pytest.raises(IngestionError, match="Parquet engine \\(pyarrow or fastparquet\\) is required"):
        loader.load_csv(valid_parquet_file)


# ---------------------------------------------------------------------------
# 7. Internal Path Leakage Prevention
# ---------------------------------------------------------------------------

def test_error_message_does_not_leak_server_directory_path(tmp_path: Path):
    """Error messages should mention only the filename, not full server paths."""
    secret_dir = tmp_path / "super_secret_server_path_directory"
    secret_dir.mkdir()
    bad_file = secret_dir / "bad_format.json"
    bad_file.write_text("{}", encoding="utf-8")

    validator = DatasetValidator()
    with pytest.raises(IngestionError) as exc_info:
        validator.validate_path(bad_file)

    msg = str(exc_info.value)
    assert "bad_format.json" in msg
    assert "super_secret_server_path_directory" not in msg


# ---------------------------------------------------------------------------
# 8. Safe State Handling
# ---------------------------------------------------------------------------

def test_failed_upload_does_not_leave_partial_state(tmp_path: Path):
    """Verify that an invalid upload attempt does not mutate application state."""
    # Simulated existing state
    app_state = {
        "df": pd.DataFrame({"existing_col": [1, 2]}),
        "metadata": {"name": "initial_dataset"},
    }

    invalid_file = tmp_path / "corrupt_data.parquet"
    invalid_file.write_text("not_parquet_content", encoding="utf-8")

    loader = CSVLoader()
    try:
        new_df, new_meta = loader.load_csv(invalid_file)
        app_state["df"] = new_df
        app_state["metadata"] = new_meta
    except IngestionError:
        pass  # Caught cleanly as in app.py

    # State must be unchanged
    assert len(app_state["df"]) == 2
    assert "existing_col" in app_state["df"].columns
    assert app_state["metadata"]["name"] == "initial_dataset"


# ---------------------------------------------------------------------------
# 9. T-005.2 Safe Filename & Path Traversal Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("traversal_name", [
    "../secret.csv",
    "..\\secret.csv",
    "../../etc/passwd.csv",
    "..\\..\\windows\\system32\\calc.csv",
    "data/../evil.csv",
    "data\\..\\evil.csv",
    "....//evil.csv",
    "foo/bar/sales.csv",
    "foo\\bar\\sales.csv",
])
def test_path_traversal_filenames_rejected(traversal_name: str):
    """Path traversal sequences (..), mixed slashes, and directory separators must be rejected."""
    with pytest.raises(IngestionError, match="(Directory traversal|Directory separators)"):
        sanitize_filename(traversal_name)


@pytest.mark.parametrize("abs_name", [
    "/etc/shadow.csv",
    "/var/log/data.parquet",
    "\\windows\\system32\\cmd.xlsx",
    "C:\\Users\\admin\\secrets.csv",
    "D:/OmniSleuth/data.parquet",
    "e:\\temp\\file.xlsx",
    "\\\\server\\share\\data.csv",
    "//nas/share/dataset.parquet",
])
def test_absolute_and_drive_and_unc_paths_rejected(abs_name: str):
    """Absolute POSIX paths, Windows drive letter prefixes, and UNC network paths must be rejected."""
    with pytest.raises(IngestionError, match="(Absolute paths|Drive letter|UNC paths|Directory separators)"):
        sanitize_filename(abs_name)


@pytest.mark.parametrize("reserved_name", [
    "CON.csv",
    "con.xlsx",
    "PRN.parquet",
    "prn.csv",
    "AUX.csv",
    "aux.parquet",
    "NUL.csv",
    "nul.xlsx",
    "COM1.csv",
    "com5.parquet",
    "com9.xlsx",
    "LPT1.csv",
    "lpt2.xlsx",
    "lpt8.parquet",
    "CON.tar.csv",
])
def test_windows_reserved_device_names_rejected(reserved_name: str):
    """Windows reserved device names (CON, PRN, AUX, NUL, COM1-9, LPT1-9) must be rejected."""
    with pytest.raises(IngestionError, match="Windows reserved device name"):
        sanitize_filename(reserved_name)


@pytest.mark.parametrize("malformed_name", [
    "",
    "   ",
    ".",
    "..",
    "...",
    ".csv",
    "  .parquet  ",
    "file\x00name.csv",
    "file\x07name.csv",
    "file\x1fname.csv",
])
def test_empty_dot_only_and_control_char_filenames_rejected(malformed_name: str):
    """Empty, whitespace-only, dot-only, extension-only, and control/null-byte filenames must be rejected."""
    with pytest.raises(IngestionError, match="(cannot be empty|Dot-only|valid non-empty stem|Null bytes|Control character)"):
        sanitize_filename(malformed_name)


@pytest.mark.parametrize("valid_name", [
    "sales_2024.csv",
    "annual-report (Q1).parquet",
    "quarterly [final] v2.xlsx",
    "données_ventes_2024.csv",
    "销售数据.xlsx",
    "данные_клиентов.parquet",
    "customer.metrics.v1.csv",
])
def test_valid_and_unicode_filenames_accepted(valid_name: str):
    """Valid filenames with standard characters, spaces, parentheses, and Unicode must be accepted."""
    cleaned = sanitize_filename(valid_name)
    assert cleaned == unicodedata.normalize("NFKC", valid_name.strip())


# ---------------------------------------------------------------------------
# 10. T-005.2 Filesystem Containment & Collision Protection
# ---------------------------------------------------------------------------

def test_generate_safe_storage_path_isolation_and_containment(tmp_path: Path):
    """Storage paths must strictly resolve inside upload_dir and use uuid/timestamp stems."""
    upload_dir = tmp_path / "custom_uploads"
    upload_dir.mkdir()

    path1 = generate_safe_storage_path("sales_data.csv", upload_dir=upload_dir)
    assert path1.parent == upload_dir.resolve()
    assert is_safe_path_contained(path1, upload_dir)
    assert path1.suffix.lower() == ".csv"
    assert path1.name.startswith("up_")
    assert "sales_data" not in path1.name  # User filename is never used as filesystem path


def test_duplicate_uploads_receive_distinct_storage_paths(tmp_path: Path):
    """Two uploads with identical original filenames receive distinct non-colliding storage paths."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    path1 = generate_safe_storage_path("my_report.xlsx", upload_dir=upload_dir)
    path2 = generate_safe_storage_path("my_report.xlsx", upload_dir=upload_dir)

    assert path1 != path2
    assert path1.parent == upload_dir.resolve()
    assert path2.parent == upload_dir.resolve()


def test_is_safe_path_contained_blocks_escape(tmp_path: Path):
    """Path containment helper accurately detects path escape attempts."""
    upload_dir = tmp_path / "safe_box"
    upload_dir.mkdir()

    safe_file = upload_dir / "file.csv"
    assert is_safe_path_contained(safe_file, upload_dir) is True

    # Escapes to parent
    escape_file = upload_dir / ".." / "secret.csv"
    assert is_safe_path_contained(escape_file, upload_dir) is False

    # Target directory itself is not a contained file
    assert is_safe_path_contained(upload_dir, upload_dir) is False


def test_safe_save_upload_exclusive_creation_and_streaming(tmp_path: Path):
    """safe_save_upload streams chunks to a contained path and returns sanitized display name."""
    upload_dir = tmp_path / "uploads"
    csv_bytes = b"col1,col2\n10,20\n30,40\n"
    file_obj = io.BytesIO(csv_bytes)
    file_obj.size = len(csv_bytes)

    dest_path, display_name = safe_save_upload(
        file_obj=file_obj,
        raw_filename="  my_data (2024).csv  ",
        upload_dir=upload_dir,
    )

    assert dest_path.exists()
    assert dest_path.read_bytes() == csv_bytes
    assert display_name == "my_data (2024).csv"
    assert is_safe_path_contained(dest_path, upload_dir)


def test_safe_save_upload_cleans_up_on_failure_without_affecting_unrelated_files(tmp_path: Path):
    """Failed upload streams clean up only their own target file, preserving existing files."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    # Pre-existing file
    existing_file = upload_dir / "unrelated_existing.csv"
    existing_file.write_text("keep,me\n1,1\n", encoding="utf-8")

    # Upload that exceeds tiny size limit
    oversized_bytes = b"order_id,amount\n" + b"1,100\n" * 100
    file_obj = io.BytesIO(oversized_bytes)

    with pytest.raises(IngestionError, match="exceeds maximum allowed upload size"):
        safe_save_upload(
            file_obj=file_obj,
            raw_filename="large.csv",
            upload_dir=upload_dir,
            max_size_bytes=20,  # Tiny threshold
        )

    # Pre-existing file must remain completely untouched
    assert existing_file.exists()
    assert existing_file.read_text(encoding="utf-8") == "keep,me\n1,1\n"

    # No stray files created
    files = list(upload_dir.iterdir())
    assert files == [existing_file]
