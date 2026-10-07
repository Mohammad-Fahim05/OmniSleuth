"""Unit and integration tests for Phases 10, 11, 12 - Multi-Agent System."""

from pathlib import Path
import pytest
from omnisleuth.agents import (
    AgentRole,
    MultiAgentOrchestrator,
    SupervisorAgent,
)
from omnisleuth.evidence.models import ClaimStatus, HypothesisStatus
from omnisleuth.ingestion import CSVLoader


@pytest.fixture
def sales_bundle(tmp_path: Path):
    """Create sample sales dataset for multi-agent testing."""
    csv_file = tmp_path / "sales_agents.csv"
    csv_file.write_text(
        """order_id,order_date,customer_id,region,category,revenue,units
ORD-101,2024-01-15,C-1,North,Electronics,1500.0,2
ORD-102,2024-01-20,C-2,South,Furniture,500.0,1
ORD-103,2024-02-05,C-3,North,Electronics,1800.0,2
ORD-104,2024-02-25,C-4,South,Furniture,600.0,1
ORD-105,2024-03-01,C-5,North,Electronics,400.0,1
ORD-106,2024-03-15,C-6,South,Furniture,300.0,1
""",
        encoding="utf-8",
    )
    loader = CSVLoader()
    df, metadata = loader.load_csv(csv_file)
    return df, metadata


def test_supervisor_planning(sales_bundle):
    """Test supervisor creates structured, bounded plan."""
    _, metadata = sales_bundle
    supervisor = SupervisorAgent()

    plan = supervisor.create_investigation_plan(
        question="Why did revenue drop in March?",
        metadata=metadata,
    )

    assert plan.target_metric == "revenue"
    assert plan.requires_rag is True
    assert len(plan.tasks) <= 10
    assert plan.max_depth == 3

    # Check task assignments
    assigned_roles = {t.assigned_to for t in plan.tasks}
    assert AgentRole.DATA_ANALYST in assigned_roles
    assert AgentRole.RETRIEVAL in assigned_roles
    assert AgentRole.INVESTIGATOR in assigned_roles


def test_full_multi_agent_investigation_lifecycle(sales_bundle):
    """Test full multi-agent orchestration on sales decline question."""
    df, metadata = sales_bundle
    orchestrator = MultiAgentOrchestrator()

    question = "Why did revenue decline in March for Electronics in North region?"
    result = orchestrator.run_investigation(question=question, df=df, metadata=metadata)

    assert result["status"] == "completed"
    assert len(result["executed_tasks"]) >= 4

    # Check evidence store
    store = result["evidence_store"]
    assert len(store.list_evidence()) >= 4

    # Check tested hypotheses
    hypotheses = result["hypotheses"]
    assert len(hypotheses) >= 2
    for h in hypotheses:
        assert h.status in [HypothesisStatus.SUPPORTED, HypothesisStatus.INCONCLUSIVE, HypothesisStatus.REJECTED]

    # Check claims and validation
    claims = result["claims"]
    assert len(claims) > 0
    for vr in result["validation_results"]:
        assert vr.status == ClaimStatus.SUPPORTED

    # Check synthesized final answer report
    report = result["final_answer"]
    assert "# Multi-Agent Investigation Report" in report
    assert "Verified Analytical Claims" in report
    assert "Contextual RAG Business Evidence" in report
    assert "Evidence Provenance & Audit Trail" in report
