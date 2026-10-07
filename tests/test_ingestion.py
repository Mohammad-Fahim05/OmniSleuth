"""Unit tests for Phase 1 - Dataset Ingestion & Schema Detection."""

from pathlib import Path
import pandas as pd
import pytest
from omnisleuth.core.exceptions import IngestionError
from omnisleuth.ingestion import (
    CSVLoader,
    ColumnType,
    DatasetMetadata,
    SchemaDetector,
)


@pytest.fixture
def sample_sales_csv(tmp_path: Path) -> Path:
    """Create a realistic sample sales CSV."""
    csv_file = tmp_path / "sales_data.csv"
    csv_content = """order_id,order_date,customer_id,region,category,revenue,units_sold,is_returned
ORD-1001,2024-01-15,CUST-001,North,Electronics,1250.50,2,False
ORD-1002,2024-01-16,CUST-002,South,Furniture,450.00,1,False
ORD-1003,2024-02-01,CUST-003,East,Electronics,890.00,1,True
ORD-1004,2024-02-14,CUST-001,North,Clothing,120.00,3,False
ORD-1005,2024-03-05,CUST-004,West,Electronics,2100.00,4,False
ORD-1005,2024-03-05,CUST-004,West,Electronics,2100.00,4,False
"""
    csv_file.write_text(csv_content, encoding="utf-8")
    return csv_file


@pytest.fixture
def dirty_csv(tmp_path: Path) -> Path:
    """Create a CSV with missing values."""
    csv_file = tmp_path / "dirty_data.csv"
    csv_content = """sku_id,product_name,price,quantity
SKU_A,Laptop,999.99,5
SKU_B,Mouse,,10
SKU_C,,25.50,
SKU_D,Keyboard,45.00,2
"""
    csv_file.write_text(csv_content, encoding="utf-8")
    return csv_file


def test_load_valid_sales_csv(sample_sales_csv: Path):
    """Test standard sales CSV loading and schema detection."""
    loader = CSVLoader()
    df, metadata = loader.load_csv(sample_sales_csv)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 6
    assert metadata.row_count == 6
    assert metadata.column_count == 8
    assert metadata.duplicate_rows == 1  # ORD-1005 is duplicated

    # Check identifiers
    assert "order_id" in metadata.identifier_columns
    assert "customer_id" in metadata.identifier_columns

    # Check datetime
    assert "order_date" in metadata.datetime_columns
    assert pd.api.types.is_datetime64_any_dtype(df["order_date"])

    # Check categoricals
    assert "region" in metadata.categorical_columns
    assert "category" in metadata.categorical_columns

    # Check numerics
    assert "revenue" in metadata.numeric_columns
    assert "units_sold" in metadata.numeric_columns

    # Check dataset ID and version hash
    assert metadata.dataset_id.startswith("ds_")
    assert len(metadata.version_hash) == 64  # SHA-256


def test_version_hash_determinism(sample_sales_csv: Path):
    """Test that same file content produces identical version hash and dataset ID."""
    loader = CSVLoader()
    hash1 = loader.calculate_file_hash(sample_sales_csv)
    hash2 = loader.calculate_file_hash(sample_sales_csv)
    assert hash1 == hash2
    assert loader.generate_dataset_id(hash1) == loader.generate_dataset_id(hash2)


def test_missing_values_detection(dirty_csv: Path):
    """Test detection of nulls and percentages per column."""
    loader = CSVLoader()
    df, metadata = loader.load_csv(dirty_csv)

    price_col = metadata.get_column("price")
    assert price_col is not None
    assert price_col.missing_count == 1
    assert price_col.missing_percentage == 25.0

    product_col = metadata.get_column("product_name")
    assert product_col is not None
    assert product_col.missing_count == 1

    sku_col = metadata.get_column("sku_id")
    assert sku_col is not None
    assert sku_col.data_type == ColumnType.IDENTIFIER
    assert sku_col.missing_count == 0


def test_empty_file_error(tmp_path: Path):
    """Test that empty file raises IngestionError."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")

    loader = CSVLoader()
    with pytest.raises(IngestionError, match="empty"):
        loader.load_csv(empty_file)


def test_nonexistent_file_error():
    """Test that nonexistent file raises IngestionError."""
    loader = CSVLoader()
    with pytest.raises(IngestionError, match="does not exist"):
        loader.load_csv(Path("non_existent_file.csv"))


def test_invalid_extension_error(tmp_path: Path):
    """Test that non-csv file raises IngestionError."""
    invalid_file = tmp_path / "data.json"
    invalid_file.write_text("{}", encoding="utf-8")

    loader = CSVLoader()
    with pytest.raises(IngestionError, match="Unsupported file format"):
        loader.load_csv(invalid_file)


def test_latin1_encoding_support(tmp_path: Path):
    """Test that CSV with latin-1 encoded characters loads successfully."""
    latin1_file = tmp_path / "latin1.csv"
    # Write characters outside standard 7-bit ASCII
    latin1_file.write_bytes("item_id,café_name,rating\n1,Café de Paris,4.8\n2,Köln Café,4.2\n".encode("latin-1"))

    loader = CSVLoader()
    df, metadata = loader.load_csv(latin1_file)
    assert len(df) == 2
    assert "café_name" in df.columns


def test_excel_file_loading(tmp_path: Path):
    """Test loading of Excel (.xlsx) file."""
    excel_file = tmp_path / "sales.xlsx"
    df_src = pd.DataFrame({
        "order_id": ["ORD-1", "ORD-2", "ORD-3"],
        "region": ["North", "South", "East"],
        "revenue": [100.0, 200.0, 300.0],
        "is_active": [True, False, True],
    })
    df_src.to_excel(excel_file, index=False)

    loader = CSVLoader()
    df, metadata = loader.load_csv(excel_file)
    assert len(df) == 3
    assert metadata.row_count == 3
    assert "order_id" in metadata.identifier_columns
    assert "revenue" in metadata.numeric_columns
    assert "is_active" in metadata.boolean_columns or metadata.get_column("is_active").data_type in [ColumnType.BOOLEAN, ColumnType.CATEGORICAL]


def test_oversized_file_error(tmp_path: Path):
    """Test that file exceeding max_file_size_bytes raises IngestionError."""
    large_file = tmp_path / "large.csv"
    large_file.write_text("id,val\n1,100\n2,200\n", encoding="utf-8")

    # Set threshold very low (e.g. 5 bytes)
    loader = CSVLoader(max_file_size_bytes=5)
    with pytest.raises(IngestionError, match="exceeds maximum allowed size"):
        loader.load_csv(large_file)

