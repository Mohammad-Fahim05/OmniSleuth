"""Dataset ingestion, validation, and schema detection module."""

from omnisleuth.ingestion.csv_loader import CSVLoader
from omnisleuth.ingestion.models import ColumnSchema, ColumnType, DatasetMetadata
from omnisleuth.ingestion.schema_detector import SchemaDetector
from omnisleuth.ingestion.storage import (
    generate_safe_storage_path,
    is_safe_path_contained,
    safe_cleanup_file,
    safe_save_upload,
    sanitize_filename,
)
from omnisleuth.ingestion.retention import (
    RetentionCleanupReport,
    cleanup_expired_uploads,
)
from omnisleuth.ingestion.validator import DatasetValidator, validate_dataset_file

DatasetLoader = CSVLoader

__all__ = [
    "ColumnType",
    "ColumnSchema",
    "DatasetMetadata",
    "SchemaDetector",
    "CSVLoader",
    "DatasetLoader",
    "DatasetValidator",
    "validate_dataset_file",
    "sanitize_filename",
    "generate_safe_storage_path",
    "is_safe_path_contained",
    "safe_save_upload",
    "safe_cleanup_file",
    "RetentionCleanupReport",
    "cleanup_expired_uploads",
]
