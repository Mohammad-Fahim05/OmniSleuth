"""Deterministic schema detector for tabular dataframes."""

import re
from typing import Any, List
import numpy as np
import pandas as pd
from omnisleuth.ingestion.models import ColumnSchema, ColumnType

# Patterns indicating an identifier column by name
ID_NAME_PATTERNS = re.compile(r"(_id|id$|^id_|uuid|guid|code$|key$|ref$|sku$|number$)", re.IGNORECASE)


class SchemaDetector:
    """Infers semantic column types and extracts column metadata from pandas DataFrames."""

    def __init__(self, id_cardinality_threshold: float = 0.95, sample_size: int = 5):
        """Initialize detector.

        Args:
            id_cardinality_threshold: Ratio of unique non-null values to total rows above which
                a text or integer column might be treated as an identifier if naming matches.
            sample_size: Number of non-null sample values to record per column.
        """
        self.id_cardinality_threshold = id_cardinality_threshold
        self.sample_size = sample_size

    def detect_column_type(self, series: pd.Series, col_name: str) -> ColumnType:
        """Infer the semantic type of a pandas Series."""
        # 1. Check boolean
        if pd.api.types.is_bool_dtype(series):
            return ColumnType.BOOLEAN

        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return ColumnType.UNKNOWN

        # 2. Check datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return ColumnType.DATETIME

        # Attempt datetime parse for object/string columns if likely date
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            if self._is_datetime_series(non_null_series):
                return ColumnType.DATETIME

        # 3. Check identifier based on name pattern and high uniqueness
        unique_count = non_null_series.nunique()
        total_count = len(non_null_series)
        uniqueness_ratio = unique_count / total_count if total_count > 0 else 0.0

        if ID_NAME_PATTERNS.search(col_name):
            if uniqueness_ratio > 0.5 or pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                return ColumnType.IDENTIFIER

        # 4. Check numeric
        if pd.api.types.is_numeric_dtype(series):
            # Check if an integer column is actually an ID (e.g., 100% unique sequence)
            if ID_NAME_PATTERNS.search(col_name) and uniqueness_ratio >= self.id_cardinality_threshold:
                return ColumnType.IDENTIFIER
            return ColumnType.NUMERIC

        # 5. Check categorical (strings, categories, or low-cardinality values)
        if isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            if uniqueness_ratio >= self.id_cardinality_threshold and ID_NAME_PATTERNS.search(col_name):
                return ColumnType.IDENTIFIER
            return ColumnType.CATEGORICAL

        return ColumnType.UNKNOWN

    def _is_datetime_series(self, sample_series: pd.Series) -> bool:
        """Check if a string series contains recognizable datetime values."""
        sample = sample_series.head(20)
        try:
            # Avoid purely numeric strings being parsed as unix epochs unless structured like dates
            first_val = str(sample.iloc[0]).strip()
            if first_val.isdigit() and len(first_val) < 8:
                return False

            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            # If at least 80% of sample successfully parsed and isn't all NaT
            valid_ratio = parsed.notna().mean()
            return valid_ratio >= 0.8
        except Exception:
            return False

    def profile_column(self, series: pd.Series, col_name: str) -> ColumnSchema:
        """Extract statistical and type metadata for a single column."""
        total_rows = len(series)
        missing_count = int(series.isna().sum())
        missing_pct = round((missing_count / total_rows * 100.0), 2) if total_rows > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))

        col_type = self.detect_column_type(series, col_name)

        # Get non-null sample values formatted cleanly
        non_null = series.dropna()
        samples: List[Any] = []
        if len(non_null) > 0:
            raw_samples = non_null.head(self.sample_size).tolist()
            # Convert timestamp or numpy types to standard Python primitives
            for val in raw_samples:
                if isinstance(val, (np.integer, int)):
                    samples.append(int(val))
                elif isinstance(val, (np.floating, float)):
                    samples.append(float(val) if not np.isnan(val) else None)
                elif isinstance(val, (pd.Timestamp, np.datetime64)):
                    samples.append(str(val))
                else:
                    samples.append(str(val))

        return ColumnSchema(
            name=col_name,
            data_type=col_type,
            dtype=str(series.dtype),
            missing_count=missing_count,
            missing_percentage=missing_pct,
            unique_count=unique_count,
            sample_values=samples,
        )

    def detect_schema(self, df: pd.DataFrame) -> List[ColumnSchema]:
        """Detect schemas for all columns in the DataFrame."""
        return [self.profile_column(df[col], col) for col in df.columns]
