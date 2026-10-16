"""CSV ingestion, validation, version hashing, and metadata generation."""

import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple, Union
import pandas as pd
from omnisleuth.core.exceptions import IngestionError
from omnisleuth.core.logging import get_logger
from omnisleuth.ingestion.models import ColumnType, DatasetMetadata
from omnisleuth.ingestion.schema_detector import SchemaDetector

from omnisleuth.ingestion.validator import DatasetValidator

logger = get_logger("ingestion")


class CSVLoader:
    """Handles secure loading, validation, hashing, and metadata extraction for tabular datasets (.csv, .parquet, .xlsx)."""

    def __init__(
        self,
        schema_detector: Optional[SchemaDetector] = None,
        max_file_size_bytes: Optional[int] = None,
    ):
        self.schema_detector = schema_detector or SchemaDetector()
        if max_file_size_bytes is not None:
            self.max_file_size_bytes = max_file_size_bytes
        else:
            try:
                from omnisleuth.core.config import get_settings

                self.max_file_size_bytes = get_settings().max_upload_size_bytes
            except Exception:
                self.max_file_size_bytes = 50 * 1024 * 1024

        self.validator = DatasetValidator(max_file_size_bytes=self.max_file_size_bytes)

    def calculate_file_hash(self, file_path: Union[str, Path]) -> str:
        """Calculate SHA-256 hash of the raw file content for version tracking."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def generate_dataset_id(self, version_hash: str, session_id: Optional[str] = None) -> str:
        """Generate a deterministic dataset ID based on the file version hash and optional session."""
        if session_id:
            clean_sess = session_id.replace("sess_", "")[:8]
            return f"ds_{version_hash[:8]}_{clean_sess}"
        return f"ds_{version_hash[:12]}"

    def validate_file(self, file_path: Union[str, Path]) -> Path:
        """Validate file existence, extension, size, and content signature."""
        return self.validator.validate_path(file_path)

    def load_csv(
        self,
        file_path: Union[str, Path],
        dataset_name: Optional[str] = None,
        parse_dates: bool = True,
        session_id: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, DatasetMetadata]:
        """Load a CSV, Parquet, or Excel file, validate schema, compute checksum, and return DataFrame and metadata.

        Args:
            file_path: Path to the dataset file (.csv, .parquet, .xlsx).
            dataset_name: Optional custom name. Defaults to filename.
            parse_dates: Whether to automatically parse detected date columns into pd.Timestamp.
            session_id: Optional owner session ID to scope dataset identifier and metadata.

        Returns:
            Tuple of (pd.DataFrame, DatasetMetadata)
        """
        path = self.validate_file(file_path)
        version_hash = self.calculate_file_hash(path)
        dataset_id = self.generate_dataset_id(version_hash, session_id=session_id)
        name = dataset_name or path.name

        logger.info(f"Loading dataset '{name}' (ID: {dataset_id}) from {path}")

        df: Optional[pd.DataFrame] = None
        suffix = path.suffix.lower()

        if suffix == ".parquet":
            try:
                import pyarrow  # type: ignore # noqa: F401
            except ImportError:
                try:
                    import fastparquet  # type: ignore # noqa: F401
                except ImportError:
                    raise IngestionError(
                        "Parquet engine (pyarrow or fastparquet) is required to read Parquet files."
                    )
            try:
                df = pd.read_parquet(path)
            except Exception as exc:
                raise IngestionError(f"Failed to parse Parquet file '{name}': Invalid or corrupt Parquet data.") from None

        elif suffix == ".xlsx":
            try:
                import openpyxl  # type: ignore # noqa: F401
            except ImportError:
                raise IngestionError("Excel engine (openpyxl) is required to read .xlsx files.")
            try:
                df = pd.read_excel(path, engine="openpyxl")
            except Exception as exc:
                raise IngestionError(f"Failed to parse Excel file '{name}': Invalid or corrupt XLSX workbook.") from None

        else:
            # Try reading with standard encodings for CSV
            encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
            last_exc: Optional[Exception] = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(path, encoding=encoding)
                    break
                except (UnicodeDecodeError, pd.errors.ParserError) as exc:
                    last_exc = exc
                    continue
                except pd.errors.EmptyDataError:
                    raise IngestionError(f"Dataset file '{name}' is empty (0 bytes).")

            if df is None:
                raise IngestionError(f"Failed to parse CSV file '{name}': Invalid formatting or encoding.")

        self.validator.validate_dataframe(df, name)

        # Detect initial schema
        columns = self.schema_detector.detect_schema(df)

        # If parse_dates is True, convert detected datetime columns in place
        if parse_dates:
            for col_schema in columns:
                if col_schema.data_type == ColumnType.DATETIME and not pd.api.types.is_datetime64_any_dtype(df[col_schema.name]):
                    try:
                        df[col_schema.name] = pd.to_datetime(df[col_schema.name], errors="coerce")
                        col_schema.dtype = str(df[col_schema.name].dtype)
                    except Exception as e:
                        logger.warning(f"Could not parse column {col_schema.name} as datetime: {e}")

        # Count duplicate rows
        duplicate_count = int(df.duplicated().sum())

        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            name=name,
            file_path=str(path.resolve()),
            version_hash=version_hash,
            row_count=len(df),
            column_count=len(df.columns),
            duplicate_rows=duplicate_count,
            columns=columns,
            session_id=session_id,
        )

        logger.info(
            f"Successfully ingested '{name}': {metadata.row_count} rows, "
            f"{metadata.column_count} columns, {metadata.duplicate_rows} duplicates."
        )

        return df, metadata

    def load_dataset(
        self,
        file_path: Union[str, Path],
        dataset_name: Optional[str] = None,
        parse_dates: bool = True,
        session_id: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, DatasetMetadata]:
        """Alias for load_csv that semantically reflects multi-format tabular support."""
        return self.load_csv(
            file_path,
            dataset_name=dataset_name,
            parse_dates=parse_dates,
            session_id=session_id,
        )
