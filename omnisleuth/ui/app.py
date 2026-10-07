"""OmniSleuth — Evidence-Grounded Multi-Agent Data Investigation System."""

import os
from pathlib import Path
import tempfile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from omnisleuth.agents import MultiAgentOrchestrator
from omnisleuth.discovery import DatasetDiscoveryScanner
from omnisleuth.evaluation import BENCHMARK_QUESTIONS, OmniSleuthEvaluator, generate_benchmark_sales_dataset
from omnisleuth.evidence.models import ClaimStatus, EvidenceType, HypothesisStatus
from omnisleuth.ingestion import CSVLoader
from omnisleuth.ml.detectors import IQRAnomalyDetector, IsolationForestDetector
from omnisleuth.persistence import DatabaseManager, Repository
from omnisleuth.profiling import DatasetProfiler

st.set_page_config(
    page_title="OmniSleuth",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# DESIGN SYSTEM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

.stApp { background-color: #F7F8FA; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: #0F1B2D; border-right: none; }
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: #1E3A5F; color: #E2E8F0 !important;
    border: 1px solid #2D4A6E; border-radius: 4px;
    font-size: 0.82rem; font-weight: 500; width: 100%;
    transition: background 0.15s ease;
}
[data-testid="stSidebar"] .stButton > button:hover { background: #2D4A6E; }
[data-testid="stSidebar"] hr { border-color: #1E3A5F; margin: 0.75rem 0; }

/* App header */
.app-header {
    padding: 1.5rem 0 1rem 0;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 1.5rem;
}
.app-title {
    font-size: 1.45rem; font-weight: 700;
    color: #0F1B2D; letter-spacing: -0.01em; margin: 0;
}
.app-subtitle { font-size: 0.82rem; color: #64748B; margin-top: 0.2rem; }

.dataset-pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 4px;
    padding: 0.3rem 0.65rem; font-size: 0.78rem; color: #1D4ED8; font-weight: 500;
}
.status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #22C55E; display: inline-block;
}

/* Section labels */
.section-label {
    font-size: 0.7rem; font-weight: 600; color: #94A3B8;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem;
}
.section-title { font-size: 1.05rem; font-weight: 600; color: #0F1B2D; margin-bottom: 0.25rem; }
.section-desc { font-size: 0.82rem; color: #64748B; margin-bottom: 1.25rem; }

/* KPI blocks */
.kpi-block {
    padding: 0.9rem 1rem; background: #FFFFFF;
    border: 1px solid #E2E8F0; border-radius: 6px;
}
.kpi-label {
    font-size: 0.72rem; font-weight: 500; color: #64748B;
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.25rem;
}
.kpi-value { font-size: 1.55rem; font-weight: 700; color: #0F1B2D; line-height: 1.1; }
.kpi-value-negative { color: #DC2626; }
.kpi-value-positive { color: #16A34A; }
.kpi-value-neutral  { color: #0F1B2D; }

/* Dividers */
.section-divider { border: none; border-top: 1px solid #E2E8F0; margin: 1.5rem 0; }

/* Summary block */
.summary-block {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-left: 3px solid #1D4ED8; border-radius: 4px;
    padding: 1rem 1.25rem; font-size: 0.9rem; color: #1E293B; line-height: 1.65;
}

/* Hypothesis rows */
.hyp-row {
    display: flex; align-items: flex-start; justify-content: space-between;
    padding: 0.75rem 0; border-bottom: 1px solid #F1F5F9; gap: 1rem;
}
.hyp-row:last-child { border-bottom: none; }
.hyp-statement { font-size: 0.88rem; color: #1E293B; flex: 1; line-height: 1.5; }
.hyp-rationale { font-size: 0.78rem; color: #64748B; margin-top: 0.2rem; }

/* Status tags */
.status-tag {
    font-size: 0.72rem; font-weight: 600; padding: 0.2rem 0.55rem;
    border-radius: 3px; white-space: nowrap; flex-shrink: 0;
}
.status-supported    { background: #F0FDF4; color: #16A34A; border: 1px solid #BBF7D0; }
.status-rejected     { background: #FEF2F2; color: #DC2626; border: 1px solid #FECACA; }
.status-inconclusive { background: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }
.status-pending      { background: #F8FAFC; color: #64748B; border: 1px solid #E2E8F0; }

/* Evidence items */
.evidence-item {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 5px; padding: 0.85rem 1rem; margin-bottom: 0.6rem;
}
.evidence-title { font-size: 0.88rem; font-weight: 600; color: #0F1B2D; margin-bottom: 0.2rem; }
.evidence-meta {
    font-size: 0.73rem; color: #64748B; margin-bottom: 0.5rem;
    display: flex; gap: 0.75rem; flex-wrap: wrap;
}
.evidence-content {
    font-size: 0.82rem; color: #374151; background: #F8FAFC;
    border-radius: 3px; padding: 0.5rem 0.65rem; line-height: 1.5;
}

/* Claim rows */
.claim-row {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 5px; padding: 0.85rem 1rem; margin-bottom: 0.5rem;
}
.claim-text { font-size: 0.87rem; color: #1E293B; margin-bottom: 0.4rem; line-height: 1.55; }
.claim-explanation { font-size: 0.78rem; color: #64748B; }

/* Agent steps */
.step-row { border-left: 2px solid #E2E8F0; padding: 0.5rem 0.75rem; margin-bottom: 0.5rem; }
.step-agent { font-size: 0.7rem; font-weight: 600; color: #1D4ED8; text-transform: uppercase; letter-spacing: 0.05em; }
.step-tool { font-size: 0.85rem; font-weight: 500; color: #0F1B2D; }
.step-rationale { font-size: 0.78rem; color: #64748B; margin-top: 0.2rem; }

/* Discovery */
.discovery-category {
    font-size: 0.68rem; font-weight: 600; color: #64748B;
    text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.2rem;
}
.discovery-desc { font-size: 0.82rem; color: #374151; line-height: 1.5; margin-bottom: 0.5rem; }
.sig-high   { color: #DC2626; }
.sig-medium { color: #B45309; }
.sig-low    { color: #64748B; }

/* Info / warn boxes */
.info-box {
    background: #EFF6FF; border: 1px solid #BFDBFE;
    border-radius: 4px; padding: 0.65rem 0.85rem; font-size: 0.82rem; color: #1E40AF;
}
.warn-box {
    background: #FFFBEB; border: 1px solid #FDE68A;
    border-radius: 4px; padding: 0.65rem 0.85rem; font-size: 0.82rem; color: #92400E;
}

/* Provenance trail */
.prov-step { display: flex; align-items: flex-start; gap: 0.65rem; padding: 0.5rem 0; }
.prov-connector { display: flex; flex-direction: column; align-items: center; }
.prov-dot { width: 8px; height: 8px; border-radius: 50%; background: #1D4ED8; flex-shrink: 0; margin-top: 4px; }
.prov-line { width: 1px; height: 24px; background: #CBD5E1; }
.prov-label { font-size: 0.72rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em; }
.prov-value { font-size: 0.82rem; color: #1E293B; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #E2E8F0; gap: 0; }
.stTabs [data-baseweb="tab"] {
    font-size: 0.82rem; font-weight: 500; color: #64748B;
    padding: 0.55rem 1rem; border-bottom: 2px solid transparent; background: transparent;
}
.stTabs [aria-selected="true"] { color: #1D4ED8; border-bottom: 2px solid #1D4ED8; }

/* Buttons */
.stButton > button[kind="primary"] {
    background: #1D4ED8; color: white; border: none;
    border-radius: 4px; font-weight: 500; font-size: 0.85rem;
}
.stButton > button[kind="primary"]:hover { background: #1E40AF; }
.stButton > button:not([kind="primary"]) {
    background: white; color: #374151;
    border: 1px solid #D1D5DB; border-radius: 4px; font-size: 0.83rem;
}
.stButton > button:not([kind="primary"]):hover { background: #F8FAFC; }

/* Sidebar meta */
.sidebar-section {
    font-size: 0.65rem; font-weight: 700; color: #475569 !important;
    text-transform: uppercase; letter-spacing: 0.1em; padding: 0.4rem 0 0.2rem 0;
}
.sidebar-value { font-size: 0.78rem; color: #64748B !important; word-break: break-all; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Services
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_services():
    repo = Repository()
    orchestrator = MultiAgentOrchestrator()
    scanner = DatasetDiscoveryScanner()
    profiler = DatasetProfiler()
    evaluator = OmniSleuthEvaluator(orchestrator=orchestrator)
    return repo, orchestrator, scanner, profiler, evaluator


repo, orchestrator, scanner, profiler, evaluator = get_services()

if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.metadata = None


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="padding:1rem 0 0.5rem 0;">
            <div style="font-size:1.05rem;font-weight:700;color:#E2E8F0;">OmniSleuth</div>
            <div style="font-size:0.73rem;color:#64748B;margin-top:0.15rem;">Data Investigation System</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Workspace</div>', unsafe_allow_html=True)

    dataset_source = st.radio(
        "Dataset source",
        ["Sample Sales Dataset", "Upload Custom File"],
        label_visibility="collapsed",
    )

    if dataset_source == "Sample Sales Dataset":
        if st.button("Load Sample Sales Dataset", use_container_width=True) or st.session_state.df is None:
            sample_path = Path("data/uploads/sample_sales_2024.csv")
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            generate_benchmark_sales_dataset(n_rows=1000, random_seed=42, output_path=sample_path)
            loader = CSVLoader()
            df, meta = loader.load_csv(sample_path, dataset_name="Benchmark Sales 2024")
            repo.save_dataset(meta)
            st.session_state.df = df
            st.session_state.metadata = meta
    else:
        uploaded_file = st.file_uploader(
            "Upload dataset", type=["csv", "txt", "xlsx", "xls"], label_visibility="visible"
        )
        if uploaded_file is not None:
            suffix = Path(uploaded_file.name).suffix.lower() or ".csv"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            loader = CSVLoader()
            df, meta = loader.load_csv(tmp_path, dataset_name=uploaded_file.name)
            repo.save_dataset(meta)
            st.session_state.df = df
            st.session_state.metadata = meta

    st.markdown("<hr>", unsafe_allow_html=True)
    if st.session_state.df is not None:
        meta = st.session_state.metadata
        st.markdown(f"""
        <div style="padding:0.5rem 0;">
            <div class="sidebar-section">Dataset</div>
            <div style="font-size:0.85rem;color:#CBD5E1;font-weight:500;margin-bottom:0.5rem;">
                <span class="status-dot"></span>&nbsp;{meta.name}
            </div>
            <div style="display:flex;gap:1rem;">
                <div><div class="sidebar-value">{meta.row_count:,}</div>
                    <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:0.06em;">Rows</div></div>
                <div><div class="sidebar-value">{meta.column_count}</div>
                    <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:0.06em;">Cols</div></div>
                <div><div class="sidebar-value">{len(meta.numeric_columns)}</div>
                    <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:0.06em;">Num</div></div>
            </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">System</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="padding:0.3rem 0;">
            <div style="font-size:0.68rem;color:#475569;margin-bottom:0.1rem;">Dataset ID</div>
            <div class="sidebar-value">{meta.dataset_id}</div>
        </div>
        <div style="padding:0.3rem 0;">
            <div style="font-size:0.68rem;color:#475569;margin-bottom:0.1rem;">Storage</div>
            <div class="sidebar-value">SQLite &middot; data/omnisleuth.db</div>
        </div>
        <div style="padding:0.3rem 0;">
            <div style="font-size:0.68rem;color:#475569;margin-bottom:0.1rem;">Status</div>
            <div style="font-size:0.78rem;color:#22C55E;">Active</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:0.8rem;color:#475569;padding:0.5rem 0;">No dataset loaded.</div>',
                    unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
if st.session_state.df is None:
    st.markdown("""
    <div class="app-header">
        <div class="app-title">OmniSleuth</div>
        <div class="app-subtitle">Evidence-grounded multi-agent data investigation</div>
    </div>
    <div class="info-box">Load the sample dataset or upload a CSV file in the sidebar to begin an investigation.</div>
    """, unsafe_allow_html=True)
    st.stop()

df = st.session_state.df
metadata = st.session_state.metadata

st.markdown(f"""
<div class="app-header">
    <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:0.75rem;">
        <div>
            <div class="app-title">OmniSleuth</div>
            <div class="app-subtitle">Evidence-grounded multi-agent data investigation</div>
        </div>
        <div style="display:flex;gap:0.5rem;flex-wrap:wrap;align-items:center;padding-top:0.25rem;">
            <span class="dataset-pill"><span class="status-dot"></span>&nbsp;{metadata.name}</span>
            <span class="dataset-pill" style="background:#F0FDF4;border-color:#BBF7D0;color:#166534;">{metadata.row_count:,} rows</span>
            <span class="dataset-pill" style="background:#F8FAFC;border-color:#E2E8F0;color:#374151;">{metadata.column_count} columns</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Investigation", "Discovery", "Data Quality",
    "Anomaly Explorer", "Evidence & Provenance", "Evaluation",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — Investigation
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-label">Autonomous Investigation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Investigate your data</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Ask a business question. The multi-agent system generates a plan, '
        'executes analytical tools, retrieves context, and validates all claims against evidence.</div>',
        unsafe_allow_html=True)

    EXAMPLE_QUESTIONS = [
        "Why did revenue decline in March for Electronics in North region?",
        "What is the monthly revenue trend across 2024?",
        "Compare revenue distribution across categories and regions",
        "Are there extreme anomaly transactions in sales volume?",
        "What is the business definition and formula for Average Order Value (AOV)?",
    ]

    col_q, col_ex = st.columns([3, 2])
    with col_q:
        user_question = st.text_input(
            "Question", value=EXAMPLE_QUESTIONS[0],
            placeholder="e.g. Why did revenue decline in March?",
            label_visibility="collapsed")
    with col_ex:
        selected_example = st.selectbox(
            "Example questions", [""] + EXAMPLE_QUESTIONS, label_visibility="collapsed")
        if selected_example:
            user_question = selected_example

    btn_col, _ = st.columns([1, 3])
    with btn_col:
        run_investigation = st.button("Investigate", type="primary", use_container_width=True)

    if run_investigation:
        with st.spinner("Running multi-agent investigation\u2026"):
            inv_res = orchestrator.run_investigation(user_question, df, metadata)
            st.session_state.last_inv = inv_res
            repo.save_investigation_results(
                investigation_id=f"inv_{metadata.dataset_id[:6]}_{int(pd.Timestamp.now().timestamp())}",
                dataset_id=metadata.dataset_id,
                question=user_question,
                status=inv_res["status"],
                plan_dict=inv_res["plan"].model_dump(),
                final_answer=inv_res["final_answer"],
                evidence_store=inv_res["evidence_store"],
            )

    if "last_inv" in st.session_state:
        inv_res = st.session_state.last_inv
        store = inv_res["evidence_store"]

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # Active question label
        st.markdown(
            f'<div style="font-size:0.78rem;color:#64748B;margin-bottom:0.2rem;">Result for:</div>'
            f'<div style="font-size:1rem;font-weight:600;color:#0F1B2D;margin-bottom:1rem;">'
            f'&ldquo;{inv_res.get("question", user_question)}&rdquo;</div>',
            unsafe_allow_html=True)

        # KPIs
        hypotheses = store.list_hypotheses()
        evidence_items = store.list_evidence()
        validation_results = inv_res.get("validation_results", [])
        executed_tasks = inv_res.get("executed_tasks", [])
        hyp_supported = sum(1 for h in hypotheses if h.status == HypothesisStatus.SUPPORTED)
        hyp_rejected  = sum(1 for h in hypotheses if h.status == HypothesisStatus.REJECTED)

        headline_pct = None
        for ev in evidence_items:
            pct = ev.content.get("percentage_change")
            if pct is not None:
                headline_pct = pct
                break

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        with mc1:
            if headline_pct is not None:
                col_cls = "kpi-value-negative" if headline_pct < 0 else "kpi-value-positive"
                st.markdown(
                    f'<div class="kpi-block"><div class="kpi-label">Revenue Change</div>'
                    f'<div class="kpi-value {col_cls}">{headline_pct:+.1f}%</div></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="kpi-block"><div class="kpi-label">Status</div>'
                    f'<div class="kpi-value kpi-value-neutral">{inv_res["status"].title()}</div></div>',
                    unsafe_allow_html=True)
        with mc2:
            st.markdown(
                f'<div class="kpi-block"><div class="kpi-label">Evidence Items</div>'
                f'<div class="kpi-value kpi-value-neutral">{len(evidence_items)}</div></div>',
                unsafe_allow_html=True)
        with mc3:
            st.markdown(
                f'<div class="kpi-block"><div class="kpi-label">Hypotheses</div>'
                f'<div class="kpi-value kpi-value-neutral">{len(hypotheses)}</div></div>',
                unsafe_allow_html=True)
        with mc4:
            st.markdown(
                f'<div class="kpi-block"><div class="kpi-label">Supported</div>'
                f'<div class="kpi-value kpi-value-positive">{hyp_supported}</div></div>',
                unsafe_allow_html=True)
        with mc5:
            st.markdown(
                f'<div class="kpi-block"><div class="kpi-label">Rejected</div>'
                f'<div class="kpi-value kpi-value-negative">{hyp_rejected}</div></div>',
                unsafe_allow_html=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        col_left, col_right = st.columns([3, 2], gap="large")

        with col_left:
            st.markdown('<div class="section-label">Executive Summary</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="summary-block">{inv_res["final_answer"]}</div>',
                unsafe_allow_html=True)

            # Charts
            chart_rendered = False
            for ev in evidence_items:
                if ev.content.get("time_series"):
                    ts_data = pd.DataFrame(ev.content["time_series"])
                    metric_name = ev.content.get("metric", "Metric").replace("_", " ").title()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=ts_data["period"], y=ts_data["value"],
                        mode="lines+markers",
                        line=dict(color="#1D4ED8", width=2),
                        marker=dict(size=5, color="#1D4ED8"),
                    ))
                    fig.update_layout(
                        title=dict(text=f"Monthly Trend \u2014 {metric_name}",
                                   font=dict(size=13, color="#0F1B2D"), x=0),
                        paper_bgcolor="white", plot_bgcolor="white",
                        margin=dict(l=0, r=0, t=36, b=0),
                        font=dict(family="Inter", size=11, color="#374151"),
                        xaxis=dict(showgrid=False, linecolor="#E2E8F0", tickfont=dict(size=10)),
                        yaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0", tickfont=dict(size=10)),
                        height=240, showlegend=False,
                    )
                    if not chart_rendered:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown('<div class="section-label">Trend Analysis</div>', unsafe_allow_html=True)
                        chart_rendered = True
                    st.plotly_chart(fig, use_container_width=True)

                elif ev.content.get("segments"):
                    seg_data = pd.DataFrame(ev.content["segments"])
                    seg_col = ev.content.get("segment_column", "Segment").replace("_", " ").title()
                    colors = ["#1D4ED8", "#3B82F6", "#93C5FD", "#BFDBFE", "#DBEAFE"]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=seg_data["segment"], y=seg_data["value"],
                        marker_color=colors[:len(seg_data)],
                        text=[f"{v:,.0f}" for v in seg_data["value"]],
                        textposition="outside", textfont=dict(size=10),
                    ))
                    fig.update_layout(
                        title=dict(text=f"Segment Comparison \u2014 {seg_col}",
                                   font=dict(size=13, color="#0F1B2D"), x=0),
                        paper_bgcolor="white", plot_bgcolor="white",
                        margin=dict(l=0, r=0, t=36, b=0),
                        font=dict(family="Inter", size=11, color="#374151"),
                        xaxis=dict(showgrid=False, linecolor="#E2E8F0", tickfont=dict(size=10)),
                        yaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0", tickfont=dict(size=10)),
                        height=240, showlegend=False,
                    )
                    if not chart_rendered:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown('<div class="section-label">Segment Analysis</div>', unsafe_allow_html=True)
                        chart_rendered = True
                    st.plotly_chart(fig, use_container_width=True)

            # Hypotheses table
            if hypotheses:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">Hypotheses</div>', unsafe_allow_html=True)

                def _hyp_tag(status):
                    m = {
                        HypothesisStatus.SUPPORTED:    ("status-supported",    "Supported"),
                        HypothesisStatus.REJECTED:     ("status-rejected",     "Rejected"),
                        HypothesisStatus.INCONCLUSIVE: ("status-inconclusive", "Inconclusive"),
                        HypothesisStatus.PENDING:      ("status-pending",      "Pending"),
                    }
                    cls, lbl = m.get(status, ("status-pending", status.value.title()))
                    return f'<span class="status-tag {cls}">{lbl}</span>'

                hyp_html = '<div style="background:white;border:1px solid #E2E8F0;border-radius:5px;padding:0.5rem 1rem;">'
                for hyp in hypotheses:
                    tag = _hyp_tag(hyp.status)
                    summary = hyp.findings_summary or hyp.rationale or ""
                    hyp_html += (
                        f'<div class="hyp-row">'
                        f'<div>'
                        f'<div class="hyp-statement">{hyp.statement}</div>'
                        + (f'<div class="hyp-rationale">{summary[:140]}</div>' if summary else "")
                        + f'</div>{tag}</div>'
                    )
                hyp_html += "</div>"
                st.markdown(hyp_html, unsafe_allow_html=True)

            # Claim validation
            if validation_results:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">Claim Validation</div>', unsafe_allow_html=True)
                for vr in validation_results:
                    if vr.status == ClaimStatus.SUPPORTED:
                        cls = "status-supported"
                    elif vr.status == ClaimStatus.UNSUPPORTED:
                        cls = "status-rejected"
                    else:
                        cls = "status-inconclusive"
                    statement = vr.calibrated_statement or vr.original_statement
                    st.markdown(
                        f'<div class="claim-row">'
                        f'<div style="margin-bottom:0.35rem;"><span class="status-tag {cls}">{vr.status.value.title()}</span></div>'
                        f'<div class="claim-text">{statement}</div>'
                        + (f'<div class="claim-explanation">{vr.validation_explanation}</div>' if vr.validation_explanation else "")
                        + '</div>',
                        unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="section-label">Execution Plan</div>', unsafe_allow_html=True)
            for task in executed_tasks:
                agent = task.assigned_to.value.upper() if hasattr(task.assigned_to, "value") else str(task.assigned_to).upper()
                tool = task.tool_name or "Analysis"
                rationale = task.rationale or ""
                st.markdown(
                    f'<div class="step-row">'
                    f'<div class="step-agent">{agent}</div>'
                    f'<div class="step-tool">{tool}</div>'
                    f'<div class="step-rationale">{rationale[:120]}</div>'
                    f'</div>',
                    unsafe_allow_html=True)

            if evidence_items:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f'<div class="section-label">Evidence ({len(evidence_items)} items)</div>',
                            unsafe_allow_html=True)
                for ev in evidence_items:
                    ev_type = ev.evidence_type.value.replace("_", " ").title()
                    title_raw = (ev.content.get("metric") or ev.content.get("title")
                                 or ev.content.get("description") or ev_type)
                    title = str(title_raw).replace("_", " ").title()[:60]
                    snippet = ""
                    for k in ["percentage_change", "value", "trend_direction", "summary", "text"]:
                        v = ev.content.get(k)
                        if v is not None:
                            snippet = f"{k.replace('_', ' ').title()}: {v}"
                            break
                    st.markdown(
                        f'<div class="evidence-item">'
                        f'<div class="evidence-title">{title}</div>'
                        f'<div class="evidence-meta">'
                        f'<span>{ev_type}</span>'
                        f'<span>{ev.source_id[:24]}</span>'
                        f'<span>{int(ev.confidence * 100)}% confidence</span>'
                        f'</div>'
                        + (f'<div class="evidence-content">{snippet}</div>' if snippet else "")
                        + '</div>',
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — Discovery
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-label">Automated Discovery</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Dataset Intelligence Scan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Scans the dataset without a prompt to surface inflection points, '
        'category concentrations, and statistical anomalies ranked by composite significance.</div>',
        unsafe_allow_html=True)

    if st.button("Run Discovery Scan", type="primary"):
        with st.spinner("Scanning dataset dimensions\u2026"):
            findings = scanner.scan(df, metadata)
            st.session_state.discovery_findings = findings

    if "discovery_findings" in st.session_state:
        findings = st.session_state.discovery_findings
        st.markdown(
            f'<div style="font-size:0.85rem;color:#374151;margin-bottom:1rem;">'
            f'<strong>{len(findings)}</strong> findings surfaced, ranked by composite score.</div>',
            unsafe_allow_html=True)

        for i, f in enumerate(findings, start=1):
            score = f.composite_rank_score
            cat = f.category.value.replace("_", " ").title()
            if score >= 0.7:
                sig_cls, sig_label = "sig-high", "High significance"
            elif score >= 0.4:
                sig_cls, sig_label = "sig-medium", "Medium significance"
            else:
                sig_cls, sig_label = "sig-low", "Low significance"

            with st.expander(f"{i}. {f.title}", expanded=(i == 1)):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(
                        f'<div class="discovery-category">{cat}</div>'
                        f'<div class="discovery-desc">{f.description}</div>'
                        f'<div style="font-size:0.8rem;color:#374151;margin-bottom:0.75rem;">'
                        f'<strong>Recommended question:</strong> <em>&ldquo;{f.recommended_question}&rdquo;</em></div>',
                        unsafe_allow_html=True)
                with col_b:
                    st.markdown(
                        f'<div style="text-align:right;">'
                        f'<div class="section-label">Score</div>'
                        f'<div style="font-size:1.2rem;font-weight:700;color:#0F1B2D;">{score:.2f}</div>'
                        f'<div style="font-size:0.73rem;font-weight:600;" class="{sig_cls}">{sig_label}</div>'
                        f'</div>',
                        unsafe_allow_html=True)

                tree = scanner.build_investigation_tree(f)
                st.markdown('<div class="section-label" style="margin-top:0.5rem;">Investigation Path</div>',
                            unsafe_allow_html=True)
                st.code(
                    f"1  Finding     {tree.label}\n"
                    f"   \u2514\u2500 2  Hypothesis  {tree.children[0].label}\n"
                    f"         \u2514\u2500 3  Evidence    {tree.children[0].children[0].label}\n"
                    f"               \u2514\u2500 4  Test        {tree.children[0].children[0].children[0].label}\n"
                    f"                     \u2514\u2500 5  Result      {tree.children[0].children[0].children[0].children[0].label}",
                    language="text")

                with st.expander("Evidence payload"):
                    st.json(f.evidence_payload)


# ══════════════════════════════════════════════════════════════
# TAB 3 — Data Quality
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-label">Data Quality</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Dataset Profile & Quality Report</div>', unsafe_allow_html=True)

    report = profiler.generate_report(df, metadata)
    score = report.completeness_score
    score_color = "#16A34A" if score >= 90 else "#B45309" if score >= 70 else "#DC2626"

    qc1, qc2, qc3, qc4 = st.columns(4)
    with qc1:
        st.markdown(
            f'<div class="kpi-block"><div class="kpi-label">Completeness</div>'
            f'<div class="kpi-value" style="color:{score_color};">{score:.1f}%</div></div>',
            unsafe_allow_html=True)
    with qc2:
        dup_color = "#DC2626" if report.duplicate_rows > 0 else "#16A34A"
        st.markdown(
            f'<div class="kpi-block"><div class="kpi-label">Duplicate Rows</div>'
            f'<div class="kpi-value" style="color:{dup_color};">{report.duplicate_rows}</div></div>',
            unsafe_allow_html=True)
    with qc3:
        st.markdown(
            f'<div class="kpi-block"><div class="kpi-label">Quality Alerts</div>'
            f'<div class="kpi-value kpi-value-neutral">{len(report.warnings)}</div></div>',
            unsafe_allow_html=True)
    with qc4:
        st.markdown(
            f'<div class="kpi-block"><div class="kpi-label">Columns Profiled</div>'
            f'<div class="kpi-value kpi-value-neutral">{metadata.column_count}</div></div>',
            unsafe_allow_html=True)

    if report.warnings:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Quality Alerts</div>', unsafe_allow_html=True)
        for w in report.warnings:
            sev = w.severity.value
            box_cls = "warn-box" if sev in ("warning", "critical") else "info-box"
            st.markdown(
                f'<div class="{box_cls}" style="margin-bottom:0.4rem;">'
                f'<strong>[{sev.upper()}]</strong> {w.message}</div>',
                unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    col_num, col_cat = st.columns(2, gap="large")

    with col_num:
        st.markdown('<div class="section-label">Numeric Columns</div>', unsafe_allow_html=True)
        if report.numeric_summaries:
            st.dataframe(
                pd.DataFrame([s.model_dump() for s in report.numeric_summaries.values()]),
                use_container_width=True, hide_index=True)

    with col_cat:
        st.markdown('<div class="section-label">Categorical Columns</div>', unsafe_allow_html=True)
        if report.categorical_summaries:
            cat_data = [
                {"Column": c.column_name, "Cardinality": c.cardinality, "Missing": c.missing_count}
                for c in report.categorical_summaries.values()
            ]
            st.dataframe(pd.DataFrame(cat_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — Anomaly Explorer
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-label">Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Anomaly Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Statistical and machine-learning anomaly detection on numeric features. '
        'Results indicate statistical deviation only and do not imply causality.</div>',
        unsafe_allow_html=True)

    cfg1, cfg2, cfg3 = st.columns([2, 2, 1])
    with cfg1:
        target_num_col = st.selectbox("Feature", metadata.numeric_columns)
    with cfg2:
        method = st.radio("Detection method",
                          ["Isolation Forest", "IQR (Interquartile Range)"], horizontal=True)
    with cfg3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_anom = st.button("Detect Anomalies", type="primary", use_container_width=True)

    if run_anom:
        with st.spinner("Running anomaly detection\u2026"):
            if "Isolation Forest" in method:
                det = IsolationForestDetector(contamination=0.03)
            else:
                det = IQRAnomalyDetector(multiplier=1.5)
            anom_res = det.detect(df, [target_num_col])
            st.session_state.anom_res = anom_res
            st.session_state.anom_col = target_num_col

    if "anom_res" in st.session_state:
        anom_res = st.session_state.anom_res
        anom_col = st.session_state.anom_col

        a1, a2, a3 = st.columns(3)
        with a1:
            st.markdown(
                f'<div class="kpi-block"><div class="kpi-label">Anomalies Detected</div>'
                f'<div class="kpi-value kpi-value-negative">{anom_res.anomaly_count}</div></div>',
                unsafe_allow_html=True)
        with a2:
            st.markdown(
                f'<div class="kpi-block"><div class="kpi-label">Anomaly Rate</div>'
                f'<div class="kpi-value kpi-value-neutral">{anom_res.anomaly_percentage:.1f}%</div></div>',
                unsafe_allow_html=True)
        with a3:
            st.markdown(
                f'<div class="kpi-block"><div class="kpi-label">Feature</div>'
                f'<div class="kpi-value kpi-value-neutral" style="font-size:1rem;">{anom_col}</div></div>',
                unsafe_allow_html=True)

        st.markdown(
            f'<div class="info-box" style="margin-top:0.75rem;">{anom_res.non_causal_disclaimer}</div>',
            unsafe_allow_html=True)

        if anom_res.anomalies:
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            ch1, ch2 = st.columns([1, 2], gap="large")
            with ch1:
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=df[anom_col], name=anom_col,
                    marker_color="#1D4ED8", line_color="#1D4ED8", fillcolor="#EFF6FF",
                    boxmean=True,
                ))
                fig.update_layout(
                    title=dict(text=f"Distribution \u2014 {anom_col}",
                               font=dict(size=12, color="#0F1B2D"), x=0),
                    paper_bgcolor="white", plot_bgcolor="white",
                    margin=dict(l=0, r=0, t=36, b=0),
                    font=dict(family="Inter", size=11, color="#374151"),
                    yaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0"),
                    xaxis=dict(showgrid=False),
                    height=280, showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
            with ch2:
                st.markdown('<div class="section-label">Top Anomalous Records (up to 50)</div>',
                            unsafe_allow_html=True)
                st.dataframe(pd.DataFrame([a.data for a in anom_res.anomalies[:50]]),
                             use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 5 — Evidence & Provenance
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-label">Evidence & Provenance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Audit Trail</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Full evidence provenance chain \u2014 from dataset through tool execution '
        'to evidence items and validated claims.</div>',
        unsafe_allow_html=True)

    st.markdown("""
    <div style="background:white;border:1px solid #E2E8F0;border-radius:5px;padding:1rem 1.25rem;margin-bottom:1rem;">
        <div class="section-label" style="margin-bottom:0.75rem;">Provenance Chain</div>
        <div class="prov-step">
            <div class="prov-connector"><div class="prov-dot"></div><div class="prov-line"></div></div>
            <div><div class="prov-label">Dataset</div><div class="prov-value">Source data ingested and versioned</div></div>
        </div>
        <div class="prov-step">
            <div class="prov-connector"><div class="prov-dot"></div><div class="prov-line"></div></div>
            <div><div class="prov-label">Tool Execution</div><div class="prov-value">Allowlisted deterministic analytical tool runs</div></div>
        </div>
        <div class="prov-step">
            <div class="prov-connector"><div class="prov-dot"></div><div class="prov-line"></div></div>
            <div><div class="prov-label">Evidence</div><div class="prov-value">Immutable evidence items minted from tool results</div></div>
        </div>
        <div class="prov-step">
            <div class="prov-connector"><div class="prov-dot"></div><div class="prov-line"></div></div>
            <div><div class="prov-label">Claim</div><div class="prov-value">Analytical claims extracted and validated against evidence</div></div>
        </div>
        <div class="prov-step">
            <div class="prov-connector"><div class="prov-dot"></div></div>
            <div><div class="prov-label">Answer</div><div class="prov-value">Final answer grounded only in supported evidence</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Registered Datasets</div>', unsafe_allow_html=True)
    datasets = repo.list_datasets()
    if datasets:
        st.dataframe(pd.DataFrame(datasets), use_container_width=True, hide_index=True)

    if "last_inv" in st.session_state:
        store = st.session_state.last_inv["evidence_store"]
        ev_list = store.list_evidence()
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-label">Investigation Evidence Store ({len(ev_list)} items)</div>',
                    unsafe_allow_html=True)
        if ev_list:
            ev_rows = [
                {
                    "Evidence ID": e.evidence_id,
                    "Type": e.evidence_type.value.replace("_", " ").title(),
                    "Source": e.source_id,
                    "Location": e.location or "\u2014",
                    "Confidence": f"{e.confidence * 100:.0f}%",
                    "Content (preview)": str(e.content)[:100] + "\u2026",
                }
                for e in ev_list
            ]
            st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 6 — Evaluation
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-label">Benchmark Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">System Evaluation Harness</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Structured benchmark measuring numerical correctness, tool selection, '
        'evidence coverage, unsupported claim rate, and latency. Compared against fixed baselines.</div>',
        unsafe_allow_html=True)

    if st.button("Run Benchmark Evaluation", type="primary"):
        with st.spinner("Evaluating against benchmark question suite\u2026"):
            bench_res = evaluator.evaluate_omnisleuth(df, metadata, questions=BENCHMARK_QUESTIONS)
            st.session_state.bench_res = bench_res

    if "bench_res" in st.session_state:
        bench_res = st.session_state.bench_res

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            c = "#16A34A" if bench_res.numerical_correctness_pct >= 90 else "#B45309"
            st.markdown(
                f'<div class="kpi-block"><div class="kpi-label">Numerical Correctness</div>'
                f'<div class="kpi-value" style="color:{c};">{bench_res.numerical_correctness_pct:.1f}%</div></div>',
                unsafe_allow_html=True)
        with b2:
            c = "#16A34A" if bench_res.tool_selection_accuracy_pct >= 90 else "#B45309"
            st.markdown(
                f'<div class="kpi-block"><div class="kpi-label">Tool Selection</div>'
                f'<div class="kpi-value" style="color:{c};">{bench_res.tool_selection_accuracy_pct:.1f}%</div></div>',
                unsafe_allow_html=True)
        with b3:
            c = "#16A34A" if bench_res.evidence_coverage_pct >= 80 else "#B45309"
            st.markdown(
                f'<div class="kpi-block"><div class="kpi-label">Evidence Coverage</div>'
                f'<div class="kpi-value" style="color:{c};">{bench_res.evidence_coverage_pct:.1f}%</div></div>',
                unsafe_allow_html=True)
        with b4:
            c = "#16A34A" if bench_res.unsupported_claim_rate_pct <= 10 else "#DC2626"
            st.markdown(
                f'<div class="kpi-block"><div class="kpi-label">Unsupported Claims</div>'
                f'<div class="kpi-value" style="color:{c};">{bench_res.unsupported_claim_rate_pct:.1f}%</div></div>',
                unsafe_allow_html=True)

        l1, l2, _ = st.columns([1, 1, 2])
        with l1:
            st.markdown(
                f'<div class="kpi-block" style="margin-top:0.6rem;"><div class="kpi-label">Latency p50</div>'
                f'<div class="kpi-value kpi-value-neutral">{bench_res.latency_p50_ms:.0f} ms</div></div>',
                unsafe_allow_html=True)
        with l2:
            st.markdown(
                f'<div class="kpi-block" style="margin-top:0.6rem;"><div class="kpi-label">Latency p95</div>'
                f'<div class="kpi-value kpi-value-neutral">{bench_res.latency_p95_ms:.0f} ms</div></div>',
                unsafe_allow_html=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Comparison Against Baselines</div>', unsafe_allow_html=True)
        comp_table = pd.DataFrame([
            {
                "Architecture": "Baseline 1 \u2014 Direct LLM (unbounded)",
                "Numerical Correctness": "42.0%",
                "Evidence Coverage": "0.0%",
                "Unsupported Claims": "58.0%",
                "Latency (p50)": "~1,800 ms",
            },
            {
                "Architecture": "Baseline 2 \u2014 Fixed tool script",
                "Numerical Correctness": "92.0%",
                "Evidence Coverage": "60.0%",
                "Unsupported Claims": "15.0%",
                "Latency (p50)": "~80 ms",
            },
            {
                "Architecture": "OmniSleuth \u2014 Agent + Tools + RAG + Validator",
                "Numerical Correctness": f"{bench_res.numerical_correctness_pct:.1f}%",
                "Evidence Coverage": f"{bench_res.evidence_coverage_pct:.1f}%",
                "Unsupported Claims": f"{bench_res.unsupported_claim_rate_pct:.1f}%",
                "Latency (p50)": f"~{bench_res.latency_p50_ms:.0f} ms",
            },
        ])
        st.dataframe(comp_table, use_container_width=True, hide_index=True)
