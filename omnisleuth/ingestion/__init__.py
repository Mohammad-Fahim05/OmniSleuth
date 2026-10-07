"""Dataset ingestion, validation, and schema detection module."""

from omnisleuth.ingestion.csv_loader import CSVLoader
from omnisleuth.ingestion.models import ColumnSchema, ColumnType, DatasetMetadata
from omnisleuth.ingestion.schema_detector import SchemaDetector

DatasetLoader = CSVLoader

__all__ = [
    "ColumnType",
    "ColumnSchema",
    "DatasetMetadata",
    "SchemaDetector",
    "CSVLoader",
    "DatasetLoader",
]
