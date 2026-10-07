"""Unit tests for Phase 13 - Discovery and Investigation Tree."""

from pathlib import Path
import pytest
from omnisleuth.discovery import (
    DatasetDiscoveryScanner,
    DiscoveredFinding,
    FindingCategory,
)
from omnisleuth.ingestion import CSVLoader


@pytest.fixture
def discovery_sales_csv(tmp_path: Path) -> Path:
    """Create sample dataset with known trend drop and category concentration."""
    csv_file = tmp_path / "sales_discovery.csv"
    csv_file.write_text(
        """order_id,order_date,category,region,revenue
ORD-1,2024-01-10,Electronics,North,3000.0
ORD-2,2024-01-20,Electronics,North,2500.0
ORD-3,2024-02-10,Electronics,North,3500.0
ORD-4,2024-02-20,Electronics,North,2800.0
ORD-5,2024-03-10,Furniture,South,500.0
ORD-6,2024-03-20,Furniture,South,400.0
ORD-7,2024-03-25,Furniture,South,10000.0
""",
        encoding="utf-8",
    )
    return csv_file


def test_discovery_scanner_finds_trends_concentration_anomalies(discovery_sales_csv: Path):
    """Test automated discovery surfaces trend drops, category concentration, and outliers."""
    loader = CSVLoader()
    df, metadata = loader.load_csv(discovery_sales_csv)

    scanner = DatasetDiscoveryScanner()
    findings = scanner.scan(df, metadata)

    assert len(findings) > 0

    categories = {f.category for f in findings}
    assert FindingCategory.TREND_INFLECTION in categories
    assert FindingCategory.SEGMENT_CONCENTRATION in categories
    assert FindingCategory.ANOMALY_SPIKE in categories

    # Verify ranking
    for i in range(len(findings) - 1):
        assert findings[i].composite_rank_score >= findings[i + 1].composite_rank_score

    top_finding = findings[0]
    assert top_finding.composite_rank_score > 0.5
    assert len(top_finding.recommended_question) > 5


def test_build_investigation_tree(discovery_sales_csv: Path):
    """Test building full 5-tier investigation tree from discovered finding."""
    loader = CSVLoader()
    df, metadata = loader.load_csv(discovery_sales_csv)

    scanner = DatasetDiscoveryScanner()
    findings = scanner.scan(df, metadata)
    assert len(findings) > 0

    tree = scanner.build_investigation_tree(findings[0])
    assert tree.node_type == "finding"
    assert len(tree.children) == 1

    hyp_node = tree.children[0]
    assert hyp_node.node_type == "hypothesis"
    assert len(hyp_node.children) == 1

    ev_node = hyp_node.children[0]
    assert ev_node.node_type == "evidence"
    assert len(ev_node.children) == 1

    test_node = ev_node.children[0]
    assert test_node.node_type == "test"
    assert len(test_node.children) == 1

    res_node = test_node.children[0]
    assert res_node.node_type == "result"
