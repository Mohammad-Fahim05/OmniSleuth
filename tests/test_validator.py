"""Unit tests for Phase 7 - Programmatic Claim Validator."""

from pathlib import Path
import pytest
from omnisleuth.evidence import (
    Claim,
    ClaimStatus,
    ClaimValidator,
    EvidenceStore,
)
from omnisleuth.ingestion import CSVLoader
from omnisleuth.tools import get_default_tool_registry


@pytest.fixture
def populated_evidence_store(tmp_path: Path) -> EvidenceStore:
    """Create an evidence store populated with actual trend tool output."""
    csv_file = tmp_path / "sales.csv"
    csv_file.write_text(
        """order_id,order_date,category,revenue
ORD-1,2024-01-10,Electronics,1000.0
ORD-2,2024-02-10,Electronics,1200.0
ORD-3,2024-03-10,Electronics,400.0
""",
        encoding="utf-8",
    )
    loader = CSVLoader()
    df, metadata = loader.load_csv(csv_file)
    store = EvidenceStore(dataset_id=metadata.dataset_id, dataset_version=metadata.version_hash)
    registry = get_default_tool_registry()

    trend_tool = registry.get("calculate_trend")
    output = trend_tool.run({"date_column": "order_date", "metric_column": "revenue", "frequency": "M"}, df, metadata)
    store.record_tool_execution("calculate_trend", {}, output)

    return store


def test_supported_numerical_claim(populated_evidence_store: EvidenceStore):
    """Test verification of accurate numerical statement grounded in evidence."""
    validator = ClaimValidator()
    # In fixture, delta is -600.0 (from 1000 to 400 total, overall change -60%)
    claim = Claim(statement="Overall revenue delta across the period was -600.0")
    result = validator.validate_claim(claim, populated_evidence_store)

    assert result.status == ClaimStatus.SUPPORTED
    assert len(result.verified_numbers) == 1
    assert -600.0 in result.verified_numbers
    assert len(result.matched_evidence_ids) > 0


def test_unsupported_hallucinated_claim(populated_evidence_store: EvidenceStore):
    """Test rejection of fabricated / hallucinated numbers."""
    validator = ClaimValidator()
    claim = Claim(statement="Revenue grew by 850.50 and total sales reached 9999.0")
    result = validator.validate_claim(claim, populated_evidence_store)

    assert result.status == ClaimStatus.UNSUPPORTED
    assert len(result.unverified_numbers) == 2
    assert 850.5 in result.unverified_numbers
    assert 9999.0 in result.unverified_numbers


def test_unsupported_causality_detection_and_calibration():
    """Test detection of definitive causal words and suggested calibrated language."""
    validator = ClaimValidator()
    text = "The marketing outage caused the decline and was the sole reason revenue dropped."
    has_issue, words, calibrated = validator.check_unsupported_causality(text)

    assert has_issue is True
    assert any("caused" in w.lower() for w in words)
    assert any("sole reason" in w.lower() for w in words)
    assert "contributed to" in calibrated
    assert "a primary observed factor" in calibrated


def test_inconclusive_when_store_empty():
    """Test that claims against an empty store return INCONCLUSIVE."""
    empty_store = EvidenceStore(dataset_id="ds_test", dataset_version="abc")
    validator = ClaimValidator()
    claim = Claim(statement="Sales dropped by 45.5%")
    result = validator.validate_claim(claim, empty_store)

    assert result.status == ClaimStatus.INCONCLUSIVE
