"""Data models for dataset ingestion and schema representation."""

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ColumnType(str, Enum):
    """Semantic data type classification of a column."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    IDENTIFIER = "identifier"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


class ColumnSchema(BaseModel):
    """Schema and statistical metadata for a single column."""

    name: str = Field(description="Column name")
    data_type: ColumnType = Field(description="Detected semantic type")
    dtype: str = Field(description="Underlying pandas/numpy dtype string")
    missing_count: int = Field(default=0, ge=0, description="Total missing/null values")
    missing_percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="Percentage of missing values")
    unique_count: int = Field(default=0, ge=0, description="Number of distinct values")
    sample_values: List[Any] = Field(default_factory=list, description="Small preview of non-null sample values")


class DatasetMetadata(BaseModel):
    """Complete metadata descriptor for an ingested dataset."""

    dataset_id: str = Field(description="Unique identifier for the dataset (e.g., ds_a1b2c3d4)")
    name: str = Field(description="User-friendly dataset name or original filename")
    file_path: str = Field(description="Local file path where the dataset is stored")
    version_hash: str = Field(description="SHA-256 checksum of the raw file content")
    row_count: int = Field(ge=0, description="Total number of rows")
    column_count: int = Field(ge=0, description="Total number of columns")
    duplicate_rows: int = Field(default=0, ge=0, description="Number of duplicate rows")
    columns: List[ColumnSchema] = Field(default_factory=list, description="Column schemas")
    session_id: Optional[str] = Field(default=None, description="Owner session identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of ingestion")

    @property
    def numeric_columns(self) -> List[str]:
        """List of column names classified as numeric."""
        return [col.name for col in self.columns if col.data_type == ColumnType.NUMERIC]

    @property
    def categorical_columns(self) -> List[str]:
        """List of column names classified as categorical."""
        return [col.name for col in self.columns if col.data_type == ColumnType.CATEGORICAL]

    @property
    def datetime_columns(self) -> List[str]:
        """List of column names classified as datetime."""
        return [col.name for col in self.columns if col.data_type == ColumnType.DATETIME]

    @property
    def identifier_columns(self) -> List[str]:
        """List of column names classified as identifiers."""
        return [col.name for col in self.columns if col.data_type == ColumnType.IDENTIFIER]

    @property
    def boolean_columns(self) -> List[str]:
        """List of column names classified as boolean."""
        return [col.name for col in self.columns if col.data_type == ColumnType.BOOLEAN]

    def get_column(self, name: str) -> Optional[ColumnSchema]:
        """Retrieve schema for a specific column by name."""
        for col in self.columns:
            if col.name == name:
                return col
        return None
