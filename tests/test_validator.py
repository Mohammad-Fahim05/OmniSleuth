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


def test_grouped_regional_claim_supported():
    """Test that a valid grouped revenue statement with comma formatting and ranks is supported."""
    from omnisleuth.evidence.models import EvidenceItem, EvidenceType
    store = EvidenceStore(dataset_id="ds_regional", dataset_version="v1")
    ev = EvidenceItem(
        source_id="flexible_query_group_by",
        evidence_type=EvidenceType.DETERMINISTIC_CALC,
        location="group_by(region)",
        content={
            "group_by": ["region"],
            "rows_before": 1015,
            "rows_after": 1015,
            "group_count": 4,
            "records": [
                {"region": "North", "revenue": 398325.40},
                {"region": "West", "revenue": 254256.15},
                {"region": "East", "revenue": 251091.56},
                {"region": "South", "revenue": 237703.96},
            ],
        },
        confidence=1.0,
    )
    store.add_evidence(ev)

    validator = ClaimValidator(numerical_tolerance_pct=1.0)
    statement = (
        "Breakdown of sum revenue by region (4 groups): "
        "1. North ($398,325.40), 2. West ($254,256.15), 3. East ($251,091.56), 4. South ($237,703.96)."
    )
    claim = Claim(statement=statement)
    result = validator.validate_claim(claim, store)

    assert result.status == ClaimStatus.SUPPORTED
    assert len(result.verified_numbers) == 4
    assert len(result.unverified_numbers) == 0
    assert 398325.40 in result.verified_numbers
    assert 254256.15 in result.verified_numbers
    assert 251091.56 in result.verified_numbers
    assert 237703.96 in result.verified_numbers
    assert ev.evidence_id in result.matched_evidence_ids
    assert claim.status == ClaimStatus.SUPPORTED


def test_grouped_regional_claim_unsupported_when_hallucinated():
    """Test that fabricated numbers in a grouped claim statement are rejected."""
    from omnisleuth.evidence.models import EvidenceItem, EvidenceType
    store = EvidenceStore(dataset_id="ds_regional", dataset_version="v1")
    ev = EvidenceItem(
        source_id="flexible_query_group_by",
        evidence_type=EvidenceType.DETERMINISTIC_CALC,
        location="group_by(region)",
        content={
            "records": [
                {"region": "North", "revenue": 398325.40},
                {"region": "West", "revenue": 254256.15},
            ],
        },
        confidence=1.0,
    )
    store.add_evidence(ev)

    validator = ClaimValidator()
    statement = (
        "Breakdown of sum revenue by region (4 groups): "
        "1. North ($999,999.00), 2. West ($888,888.00)."
    )
    claim = Claim(statement=statement)
    result = validator.validate_claim(claim, store)

    assert result.status == ClaimStatus.UNSUPPORTED
    assert 999999.0 in result.unverified_numbers
    assert 888888.0 in result.unverified_numbers
    assert claim.status == ClaimStatus.UNSUPPORTED


def test_grouped_regional_claim_partial_mismatch_rejected():
    """Test that a claim with some true numbers and one fabricated number is rejected."""
    from omnisleuth.evidence.models import EvidenceItem, EvidenceType
    store = EvidenceStore(dataset_id="ds_regional", dataset_version="v1")
    ev = EvidenceItem(
        source_id="flexible_query_group_by",
        evidence_type=EvidenceType.DETERMINISTIC_CALC,
        location="group_by(region)",
        content={
            "records": [
                {"region": "North", "revenue": 398325.40},
                {"region": "West", "revenue": 254256.15},
            ],
        },
        confidence=1.0,
    )
    store.add_evidence(ev)

    validator = ClaimValidator()
    # North is correct, West is fabricated
    statement = (
        "Breakdown of sum revenue by region (2 groups): "
        "1. North ($398,325.40), 2. West ($999,999.00)."
    )
    claim = Claim(statement=statement)
    result = validator.validate_claim(claim, store)

    assert result.status == ClaimStatus.UNSUPPORTED
    assert 398325.40 in result.verified_numbers
    assert 999999.0 in result.unverified_numbers
    assert claim.status == ClaimStatus.UNSUPPORTED


def test_comma_formatted_scalar_claim_supported():
    """Test verification of comma-separated numbers in scalar aggregation statements."""
    from omnisleuth.evidence.models import EvidenceItem, EvidenceType
    store = EvidenceStore(dataset_id="ds_scalar", dataset_version="v1")
    ev = EvidenceItem(
        source_id="flexible_query_scalar",
        evidence_type=EvidenceType.DETERMINISTIC_CALC,
        location="dataset_aggregate",
        content={
            "scalars": {"sum_revenue": 1141377.07},
            "rows_after": 1015,
        },
        confidence=1.0,
    )
    store.add_evidence(ev)

    validator = ClaimValidator()
    statement = "Total sum of 'revenue' across 1,015 records is $1,141,377.07."
    claim = Claim(statement=statement)
    result = validator.validate_claim(claim, store)

    assert result.status == ClaimStatus.SUPPORTED
    assert 1015.0 in result.verified_numbers
    assert 1141377.07 in result.verified_numbers
    assert len(result.unverified_numbers) == 0
    assert claim.status == ClaimStatus.SUPPORTED


def test_tolerance_safeguard_preserved():
    """Test that numbers within tolerance pass and numbers outside tolerance fail."""
    from omnisleuth.evidence.models import EvidenceItem, EvidenceType
    store = EvidenceStore(dataset_id="ds_tol", dataset_version="v1")
    ev = EvidenceItem(
        source_id="calc",
        evidence_type=EvidenceType.DETERMINISTIC_CALC,
        content={"revenue": 10000.0},
    )
    store.add_evidence(ev)

    validator = ClaimValidator(numerical_tolerance_pct=1.0)

    # 10050.0 is +0.5% off (<= 1.0% tolerance) -> SUPPORTED
    claim_within = Claim(statement="Revenue was 10050.0")
    res_within = validator.validate_claim(claim_within, store)
    assert res_within.status == ClaimStatus.SUPPORTED

    # 10500.0 is +5.0% off (> 1.0% tolerance) -> UNSUPPORTED
    claim_outside = Claim(statement="Revenue was 10500.0")
    res_outside = validator.validate_claim(claim_outside, store)
    assert res_outside.status == ClaimStatus.UNSUPPORTED
