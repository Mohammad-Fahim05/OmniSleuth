"""Schema context representation for datasets inspected during natural language planning."""

from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, Field
from omnisleuth.ingestion.models import DatasetMetadata


class ColumnContext(BaseModel, extra="forbid"):
    """Metadata and distribution context for a single column."""

    name: str
    data_type: str  # "numeric", "categorical", "datetime", "boolean", "identifier", "text"
    dtype: str
    missing_count: int = 0
    unique_count: int = 0
    sample_values: List[Any] = Field(default_factory=list)
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None


class SchemaContext(BaseModel, extra="forbid"):
    """Compact context summary describing a dataset for deterministic and LLM planners."""

    dataset_id: str
    dataset_name: str
    row_count: int
    column_count: int
    columns: Dict[str, ColumnContext] = Field(default_factory=dict)
    date_columns: List[str] = Field(default_factory=list)
    numeric_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    min_date: Optional[str] = None
    max_date: Optional[str] = None


def build_schema_context(df: pd.DataFrame, metadata: DatasetMetadata) -> SchemaContext:
    """Build a SchemaContext model from active DataFrame and DatasetMetadata."""
    col_contexts: Dict[str, ColumnContext] = {}
    date_cols: List[str] = []
    numeric_cols: List[str] = []
    cat_cols: List[str] = []

    min_date_val: Optional[str] = None
    max_date_val: Optional[str] = None

    for col in metadata.columns:
        c_name = col.name
        c_type = col.data_type.value
        dtype_str = col.dtype

        # Compute sample preview and bounds if column exists in DataFrame
        sample_vals: List[Any] = []
        c_min: Optional[Any] = None
        c_max: Optional[Any] = None

        if c_name in df.columns:
            series = df[c_name].dropna()
            if not series.empty:
                if c_type == "numeric":
                    numeric_cols.append(c_name)
                    try:
                        c_min = float(series.min())
                        c_max = float(series.max())
                    except Exception:
                        pass
                elif c_type == "datetime":
                    date_cols.append(c_name)
                    try:
                        dt_series = pd.to_datetime(series, errors="coerce").dropna()
                        if not dt_series.empty:
                            c_min = str(dt_series.min().date())
                            c_max = str(dt_series.max().date())
                            if min_date_val is None or c_min < min_date_val:
                                min_date_val = c_min
                            if max_date_val is None or c_max > max_date_val:
                                max_date_val = c_max
                    except Exception:
                        pass
                else:
                    cat_cols.append(c_name)
                    # For low-cardinality columns, sample top distinct values
                    if series.nunique() <= 20:
                        sample_vals = [str(v) for v in series.unique()[:10]]
                    else:
                        sample_vals = [str(v) for v in series.head(5)]

        col_contexts[c_name] = ColumnContext(
            name=c_name,
            data_type=c_type,
            dtype=dtype_str,
            missing_count=col.missing_count,
            unique_count=col.unique_count,
            sample_values=sample_vals,
            min_value=c_min,
            max_value=c_max,
        )

    return SchemaContext(
        dataset_id=metadata.dataset_id,
        dataset_name=metadata.name,
        row_count=metadata.row_count,
        column_count=metadata.column_count,
        columns=col_contexts,
        date_columns=date_cols,
        numeric_columns=numeric_cols,
        categorical_columns=cat_cols,
        min_date=min_date_val,
        max_date=max_date_val,
    )
