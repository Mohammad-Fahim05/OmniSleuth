"""End-to-end tests for Streamlit UI flow and interface regressions without LLM keys.

Covers:
1. UI smoke test (AppTest) with cleared LLM keys across pages/tabs.
2. UI flow test: load sample -> investigate -> verify results -> reopen from history.
3. UI file upload simulation test.
4. Regression test for EvidenceStore and SQLiteRepository interfaces against a real temp SQLite DB.
5. Deterministic numeric accuracy comparison with independent pandas calculation.
"""

import os
from pathlib import Path
import pytest
import pandas as pd
from streamlit.testing.v1 import AppTest

from omnisleuth.agents.orchestrator import MultiAgentOrchestrator
from omnisleuth.evidence.models import (
    Claim,
    ClaimStatus,
    EvidenceItem,
    EvidenceType,
    Hypothesis,
    HypothesisStatus,
    InvestigationNode,
)
from omnisleuth.evidence.store import EvidenceStore
from omnisleuth.ingestion.csv_loader import CSVLoader
from omnisleuth.persistence import DatabaseManager, Repository
from omnisleuth.evaluation.dataset_generator import generate_benchmark_sales_dataset


APP_FILE = str(Path(__file__).resolve().parent.parent / "omnisleuth" / "ui" / "app.py")


@pytest.fixture(autouse=True)
def clean_llm_env(monkeypatch, tmp_path):
    """Ensure all LLM provider env vars are unset and a temporary database is used."""
    for key in list(os.environ.keys()):
        if any(prefix in key for prefix in ("OPENAI", "AZURE", "LLM", "ANTHROPIC", "COHERE")):
            monkeypatch.delenv(key, raising=False)
    temp_db = str(tmp_path / "omnisleuth_test.db")
    monkeypatch.setenv("SQLITE_DB_PATH", temp_db)
    monkeypatch.setenv("APP_ENV", "testing")
    return temp_db


def test_ui_smoke_no_llm_key():
    """Smoke test: Launch real Streamlit app with no LLM key and assert 0 exceptions."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    assert len(at.exception) == 0, f"Startup exceptions: {at.exception}"

    # Verify notice when LLM is not configured
    info_boxes = [inf.value for inf in at.info]
    assert any("LLM not configured; showing deterministic results only" in val for val in info_boxes)

    # Click Load Sample Dataset
    load_btns = [b for b in at.button if "Sample" in b.label]
    assert len(load_btns) > 0
    load_btns[0].click().run()
    assert len(at.exception) == 0, f"Exceptions after loading sample: {at.exception}"

    # Verify dataset loaded in session_state
    assert at.session_state.df is not None
    assert len(at.session_state.df) > 0


def test_ui_flow_question_render_and_history_reopen():
    """Flow test: Sample dataset -> ask question -> result rendered -> history lists it -> reopen."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # 1. Load sample dataset
    load_btn = [b for b in at.button if "Sample" in b.label][0]
    load_btn.click().run()
    assert len(at.exception) == 0
    assert at.session_state.df is not None

    # 2. Ask question
    text_input = at.text_input[0]
    text_input.input("What is the monthly revenue trend across 2024?")
    inv_btn = [b for b in at.button if b.label == "Investigate"][0]
    inv_btn.click().run()
    assert len(at.exception) == 0, f"Exceptions after investigation: {at.exception}"

    # 3. Assert result rendered
    last_inv = at.session_state.last_inv
    assert last_inv is not None
    store = last_inv["evidence_store"]
    assert len(store.list_evidence()) > 0
    assert len(store.list_hypotheses()) > 0
    assert len(store.list_claims()) > 0

    # 4. Assert history reopen flow
    reopen_btns = [b for b in at.button if b.label == "Reopen Investigation"]
    assert len(reopen_btns) > 0, "Expected Reopen Investigation button in history expander"
    reopen_btns[0].click().run()
    assert len(at.exception) == 0, f"Exceptions after reopening from history: {at.exception}"

    # Verify reopened investigation retains loaded evidence and hypotheses
    reopened_inv = at.session_state.last_inv
    reopened_store = reopened_inv["evidence_store"]
    assert len(reopened_store.list_evidence()) == len(store.list_evidence())
    assert len(reopened_store.list_hypotheses()) == len(store.list_hypotheses())


def test_ui_upload_widget_custom_csv():
    """Exercise file upload simulation via Streamlit AppTest."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Select custom upload radio
    upload_radio = [r for r in at.radio if r.label == "Dataset source"][0]
    upload_radio.set_value("Upload Custom File").run()
    assert len(at.file_uploader) > 0

    # Upload custom CSV
    custom_csv = (
        b"order_id,order_date,category,revenue\n"
        b"ORD-101,2024-01-15,Electronics,1200.50\n"
        b"ORD-102,2024-01-16,Furniture,850.25\n"
        b"ORD-103,2024-01-17,Electronics,990.00\n"
    )
    at.file_uploader[0].upload("custom_orders.csv", custom_csv).run()
    assert len(at.exception) == 0
    assert at.session_state.df is not None
    assert len(at.session_state.df) == 3
    assert at.session_state.metadata.name.startswith("custom_orders")


def test_regression_store_and_repository_interfaces(tmp_path):
    """Regression test for list_hypotheses, list_claims, and list_investigations on real SQLite DB."""
    db_file = str(tmp_path / "test_reg.db")
    db_mgr = DatabaseManager(db_path=db_file)
    repo = Repository(db_manager=db_mgr)

    # 1. Test EvidenceStore methods
    store = EvidenceStore(dataset_id="ds_test_001", dataset_version="v1.0")
    hyp1 = Hypothesis(
        hypothesis_id="hyp_001",
        statement="Revenue declined due to seasonality.",
        status=HypothesisStatus.SUPPORTED,
        confidence=0.85,
        findings_summary="Statistical verification supported seasonal dip.",
    )
    hyp2 = Hypothesis(
        hypothesis_id="hyp_002",
        statement="Pricing surge drove volume down.",
        status=HypothesisStatus.REJECTED,
        confidence=0.15,
        findings_summary="No price increases observed.",
    )
    store.add_hypothesis(hyp1)
    store.add_hypothesis(hyp2)

    # Verify list_hypotheses exists and filters
    all_hyp = store.list_hypotheses()
    assert len(all_hyp) == 2
    supported_hyp = store.list_hypotheses(status=HypothesisStatus.SUPPORTED)
    assert len(supported_hyp) == 1
    assert supported_hyp[0].hypothesis_id == "hyp_001"

    # Verify list_claims exists
    clm = Claim(
        claim_id="clm_001",
        statement="Revenue changed by -23.19%.",
        status=ClaimStatus.SUPPORTED,
        extracted_numbers=[-23.19],
    )
    store.add_claim(clm)
    # Verify list_nodes and add_node exists
    node = InvestigationNode(
        node_id="node_001",
        node_type="finding",
        title="Observed dip in March",
    )
    store.add_node(node)
    nodes = store.list_nodes()
    assert len(nodes) == 1
    assert nodes[0].node_id == "node_001"

    # 2. Test SQLiteRepository persistence and list_investigations
    from omnisleuth.ingestion.models import DatasetMetadata, ColumnSchema, ColumnType
    meta = DatasetMetadata(
        dataset_id="ds_test_001",
        name="Test Dataset",
        file_path="test.csv",
        version_hash="hash_001",
        row_count=10,
        column_count=2,
        duplicate_rows=0,
        columns=[
            ColumnSchema(name="col_a", data_type=ColumnType.CATEGORICAL, dtype="object"),
            ColumnSchema(name="col_b", data_type=ColumnType.NUMERIC, dtype="float64"),
        ],
    )
    repo.save_dataset(meta)

    repo.save_investigation_results(
        investigation_id="inv_test_001",
        dataset_id="ds_test_001",
        question="What is the monthly trend?",
        status="completed",
        plan_dict={"objective": "Test", "tasks": []},
        final_answer="Revenue dropped in March.",
        evidence_store=store,
    )

    # Verify list_investigations exists and returns list of dicts
    inv_list = repo.list_investigations(dataset_id="ds_test_001")
    assert isinstance(inv_list, list)
    assert len(inv_list) == 1
    assert inv_list[0]["investigation_id"] == "inv_test_001"
    assert inv_list[0]["question"] == "What is the monthly trend?"

    # Verify get_investigation retrieves decoded items
    reopened = repo.get_investigation("inv_test_001")
    assert reopened is not None
    assert len(reopened["hypotheses"]) == 2
    assert len(reopened["claims"]) == 1


def test_no_key_deterministic_numeric_accuracy_and_llm_reporting(tmp_path):
    """Verify numerical answers match independent pandas calculations and LLM reports not configured."""
    # 1. Generate sample data
    csv_file = tmp_path / "sales.csv"
    generate_benchmark_sales_dataset(n_rows=1000, random_seed=42, output_path=csv_file)
    loader = CSVLoader()
    df, metadata = loader.load_csv(csv_file)

    # 2. Independent pandas calculation of monthly trend
    df_calc = df.copy()
    df_calc["order_date"] = pd.to_datetime(df_calc["order_date"])
    df_calc["period"] = df_calc["order_date"].dt.to_period("M").astype(str)
    monthly_series = df_calc.groupby("period")["revenue"].sum().sort_index()
    expected_start = monthly_series.iloc[0]
    expected_end = monthly_series.iloc[-1]
    expected_delta = expected_end - expected_start
    expected_change_pct = (expected_delta / expected_start) * 100
    expected_total = df_calc["revenue"].sum()

    # 3. Run investigation through orchestrator with no LLM key
    orchestrator = MultiAgentOrchestrator()
    assert not orchestrator.llm_gateway.is_configured, "LLM gateway must report not configured"

    res = orchestrator.run_investigation(
        question="What is the monthly revenue trend across 2024?",
        df=df,
        metadata=metadata,
    )

    # 4. Assert numeric accuracy against pandas ground truth
    store = res["evidence_store"]
    trend_ev = [
        ev for ev in store.list_evidence()
        if ev.content.get("overall_change_pct") is not None
    ]
    assert len(trend_ev) > 0, "Expected trend calculation evidence item"
    computed_trend = trend_ev[0].content

    assert computed_trend["overall_delta"] == pytest.approx(round(expected_delta, 2), abs=0.01)
    assert computed_trend["overall_change_pct"] == pytest.approx(round(expected_change_pct, 2), abs=0.01)

    # Check segment totals match pandas
    segment_ev = [
        ev for ev in store.list_evidence()
        if ev.content.get("total_metric_value") is not None
    ]
    assert len(segment_ev) > 0
    assert segment_ev[0].content["total_metric_value"] == pytest.approx(round(expected_total, 2), abs=0.01)

    # 5. Assert honest report without fabrication
    final_answer = res["final_answer"]
    assert "Verified Analytical Claims" in final_answer
    assert "Tested Hypotheses" in final_answer
    assert "No unverified causal claims are asserted" in final_answer
