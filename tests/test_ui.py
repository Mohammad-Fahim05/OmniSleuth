"""Unit tests for Phase 15 - Streamlit UI module."""

import importlib
import pytest


def test_ui_module_imports():
    """Verify that the UI app module can be imported and defined without syntax errors."""
    mod = importlib.import_module("omnisleuth.ui.app")
    assert mod is not None


def test_ui_investigation_result_kpis_and_hypotheses_path(tmp_path):
    """Verify that the investigation result structure consumed by UI at line 419 executes cleanly."""
    from omnisleuth.agents import MultiAgentOrchestrator
    from omnisleuth.evidence.models import HypothesisStatus
    from omnisleuth.ingestion import CSVLoader

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

    orchestrator = MultiAgentOrchestrator()
    inv_res = orchestrator.run_investigation(
        question="What is the monthly revenue trend across 2024?",
        df=df,
        metadata=metadata,
    )

    # Mimic UI extraction at app.py lines 407-425
    store = inv_res["evidence_store"]
    assert hasattr(store, "list_hypotheses")
    hypotheses = store.list_hypotheses()
    assert isinstance(hypotheses, list)

    evidence_items = store.list_evidence()
    assert isinstance(evidence_items, list)

    hyp_supported = sum(1 for h in hypotheses if h.status == HypothesisStatus.SUPPORTED)
    hyp_rejected = sum(1 for h in hypotheses if h.status == HypothesisStatus.REJECTED)
    assert hyp_supported >= 0
    assert hyp_rejected >= 0
