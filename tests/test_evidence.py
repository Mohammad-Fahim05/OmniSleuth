"""Unit tests for Phase 6 - Evidence System."""

from pathlib import Path
import pytest
from omnisleuth.evidence import (
    Claim,
    ClaimStatus,
    EvidenceItem,
    EvidenceStore,
    EvidenceType,
    Hypothesis,
    HypothesisStatus,
    InvestigationNode,
)
from omnisleuth.ingestion import CSVLoader
from omnisleuth.tools import get_default_tool_registry


@pytest.fixture
def store_fixture(tmp_path: Path):
    """Create sample sales dataset and initialized EvidenceStore."""
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
    return df, metadata, store, registry


def test_tool_execution_and_auto_evidence_minting(store_fixture):
    """Test tool execution recording and automatic evidence generation."""
    df, metadata, store, registry = store_fixture

    # Execute trend tool
    trend_tool = registry.get("calculate_trend")
    params = {"date_column": "order_date", "metric_column": "revenue", "frequency": "M"}
    output = trend_tool.run(params, df, metadata)

    exec_record = store.record_tool_execution("calculate_trend", params, output)

    assert exec_record.success is True
    assert exec_record.execution_id in store.executions

    # Verify automatically minted evidence
    ev_items = store.list_evidence(evidence_type=EvidenceType.DETERMINISTIC_CALC)
    assert len(ev_items) == 1
    ev = ev_items[0]
    assert ev.related_tool_execution_id == exec_record.execution_id
    assert ev.content["overall_delta"] == -600.0


def test_claim_and_hypothesis_management(store_fixture):
    """Test claims and hypotheses lifecycle in EvidenceStore."""
    _, _, store, _ = store_fixture

    # Add hypothesis
    hyp = Hypothesis(
        statement="Revenue declined primarily in March due to drop in Electronics sales",
        rationale="Preliminary inspection showed steep volume drop",
    )
    store.add_hypothesis(hyp)
    assert store.get_hypothesis(hyp.hypothesis_id) is not None

    # Add claim
    claim = Claim(
        statement="Revenue dropped by 66.67% between February and March",
        extracted_numbers=[66.67],
        status=ClaimStatus.PENDING,
    )
    store.add_claim(claim)
    assert claim.claim_id in store.claims


def test_investigation_tree_nodes(store_fixture):
    """Test parent-child relationship in InvestigationNode tree."""
    _, _, store, _ = store_fixture

    root = InvestigationNode(
        node_type="finding",
        title="Revenue dropped significantly in March",
    )
    store.add_node(root)

    child = InvestigationNode(
        node_type="hypothesis",
        title="Electronics segment drove majority of drop",
        parent_id=root.node_id,
    )
    store.add_node(child)

    assert child.node_id in store.nodes[root.node_id].children_ids


def test_list_hypotheses_and_claims_query_api(store_fixture):
    """Regression test ensuring list_hypotheses and list_claims work seamlessly in UI and persistence."""
    _, _, store, _ = store_fixture

    # Initially empty
    assert store.list_hypotheses() == []
    assert store.list_claims() == []
    assert store.list_nodes() == []

    # Add multiple hypotheses with distinct statuses
    h1 = Hypothesis(
        statement="Hypothesis 1: Trend drop in Electronics",
        status=HypothesisStatus.SUPPORTED,
    )
    h2 = Hypothesis(
        statement="Hypothesis 2: Region shift in South",
        status=HypothesisStatus.REJECTED,
    )
    h3 = Hypothesis(
        statement="Hypothesis 3: Price increase anomaly",
        status=HypothesisStatus.INCONCLUSIVE,
    )
    store.add_hypothesis(h1)
    store.add_hypothesis(h2)
    store.add_hypothesis(h3)

    # Test listing all and filtering
    all_hyps = store.list_hypotheses()
    assert len(all_hyps) == 3
    assert store.list_hypotheses(status=HypothesisStatus.SUPPORTED) == [h1]
    assert store.list_hypotheses(status=HypothesisStatus.REJECTED) == [h2]
    assert store.list_hypotheses(status=HypothesisStatus.INCONCLUSIVE) == [h3]

    # Add claims and test listing
    c1 = Claim(statement="Claim 1", status=ClaimStatus.SUPPORTED)
    c2 = Claim(statement="Claim 2", status=ClaimStatus.UNSUPPORTED)
    store.add_claim(c1)
    store.add_claim(c2)

    assert len(store.list_claims()) == 2
    assert store.list_claims(status=ClaimStatus.SUPPORTED) == [c1]
    assert store.get_claim(c1.claim_id) == c1

    # Add node and test listing
    n1 = InvestigationNode(node_type="finding", title="Root node")
    store.add_node(n1)
    assert len(store.list_nodes()) == 1
    assert store.get_node(n1.node_id) == n1
