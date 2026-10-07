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

logger = get_logger("ingestion")


class CSVLoader:
    """Handles secure loading, validation, hashing, and metadata extraction for CSV datasets."""

    def __init__(
        self,
        schema_detector: Optional[SchemaDetector] = None,
        max_file_size_bytes: int = 100 * 1024 * 1024,  # 100 MB default safeguard
    ):
        self.schema_detector = schema_detector or SchemaDetector()
        self.max_file_size_bytes = max_file_size_bytes

    def calculate_file_hash(self, file_path: Union[str, Path]) -> str:
        """Calculate SHA-256 hash of the raw file content for version tracking."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def generate_dataset_id(self, version_hash: str) -> str:
        """Generate a deterministic dataset ID based on the file version hash."""
        return f"ds_{version_hash[:12]}"

    def validate_file(self, file_path: Union[str, Path]) -> Path:
        """Validate file existence, extension, and non-empty status."""
        path = Path(file_path)
        if not path.exists():
            raise IngestionError(f"File does not exist: {path}")

        if not path.is_file():
            raise IngestionError(f"Provided path is not a file: {path}")

        valid_suffixes = [".csv", ".txt", ".xlsx", ".xls"]
        if path.suffix.lower() not in valid_suffixes:
            raise IngestionError(f"Unsupported file format: {path.suffix}. Expected one of {valid_suffixes}")

        file_size = path.stat().st_size
        if file_size == 0:
            raise IngestionError(f"Uploaded file is empty (0 bytes): {path.name}")

        if file_size > self.max_file_size_bytes:
            raise IngestionError(
                f"File size ({file_size / (1024 * 1024):.1f} MB) exceeds maximum allowed size ({self.max_file_size_bytes / (1024 * 1024):.1f} MB)"
            )

        return path

    def load_csv(
        self,
        file_path: Union[str, Path],
        dataset_name: Optional[str] = None,
        parse_dates: bool = True,
    ) -> Tuple[pd.DataFrame, DatasetMetadata]:
        """Load a CSV or Excel file, validate schema, compute checksum, and return DataFrame and metadata.

        Args:
            file_path: Path to the CSV or Excel file (.csv, .xlsx, .xls).
            dataset_name: Optional custom name. Defaults to filename.
            parse_dates: Whether to automatically parse detected date columns into pd.Timestamp.

        Returns:
            Tuple of (pd.DataFrame, DatasetMetadata)
        """
        path = self.validate_file(file_path)
        version_hash = self.calculate_file_hash(path)
        dataset_id = self.generate_dataset_id(version_hash)
        name = dataset_name or path.name

        logger.info(f"Loading dataset '{name}' (ID: {dataset_id}) from {path}")

        df: Optional[pd.DataFrame] = None
        suffix = path.suffix.lower()

        if suffix in [".xlsx", ".xls"]:
            try:
                df = pd.read_excel(path)
            except Exception as exc:
                raise IngestionError(f"Failed to parse Excel file '{name}': {exc}")
        else:
            # Try reading with standard encodings for CSV/text
            encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
            last_exc: Optional[Exception] = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(path, encoding=encoding)
                    break
                except (UnicodeDecodeError, pd.errors.ParserError) as exc:
                    last_exc = exc
                    continue

            if df is None:
                raise IngestionError(f"Failed to parse CSV with standard encodings: {last_exc}")

        if df.empty and len(df.columns) == 0:
            raise IngestionError(f"Dataset file '{name}' has no columns or data.")

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
        )

        logger.info(
            f"Successfully ingested '{name}': {metadata.row_count} rows, "
            f"{metadata.column_count} columns, {metadata.duplicate_rows} duplicates."
        )

        return df, metadata
