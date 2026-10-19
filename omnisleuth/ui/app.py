"""OmniSleuth — Evidence-Grounded Multi-Agent Data Investigation System."""

import os
from pathlib import Path
import tempfile
import uuid
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from omnisleuth.agents import MultiAgentOrchestrator
from omnisleuth.core.config import get_settings
from omnisleuth.core.exceptions import IngestionError
from omnisleuth.core.logging import get_logger
from omnisleuth.core.rate_limiter import get_rate_limiter
from omnisleuth.discovery import DatasetDiscoveryScanner
from omnisleuth.evaluation import BENCHMARK_QUESTIONS, OmniSleuthEvaluator, generate_benchmark_sales_dataset
from omnisleuth.evidence.models import (
    Claim,
    ClaimStatus,
    EvidenceItem,
    EvidenceType,
    Hypothesis,
    HypothesisStatus,
)
from omnisleuth.evidence.store import EvidenceStore
from omnisleuth.ingestion import CSVLoader, safe_cleanup_file, safe_save_upload
from omnisleuth.ml.detectors import IQRAnomalyDetector, IsolationForestDetector
from omnisleuth.nlq.charting import infer_example_questions
from omnisleuth.nlq.context import build_schema_context
from omnisleuth.nlq.executor import PlanExecutor
from omnisleuth.nlq.pipeline import AnalysisPipeline
from omnisleuth.persistence import DatabaseManager, Repository
from omnisleuth.persistence.serializer import safe_json_loads
from omnisleuth.profiling import DatasetProfiler
from omnisleuth.ui.components import (
    render_app_header,
    render_empty_state,
    render_investigation_empty_state,
    render_kpi_card,
    render_meta_badge,
    render_status_badge,
)
from omnisleuth.ui.theme import (
    BRAND_PRIMARY,
    BRAND_PRIMARY_LIGHT,
    PLOTLY_COLORWAY,
    THEME_CSS,
    get_plotly_layout_template,
)

logger = get_logger("ui.app")


def _render_chart(rows, spec):
    """Render a chart from a computed table via the existing plotly stack.

    Only ``ChartType``-allowlisted types are drawn, exactly as produced by
    ``omnisleuth.nlq.charting``. No SQL, generated code, or dataframe query is used.
    """
    import plotly.express as _px

    if not rows or not spec:
        st.caption("No chart data available for this result.")
        return

    x_col = spec.get("x")
    y_col = spec.get("y")
    chart_type = spec.get("type", "bar")
    title = spec.get("title") or (f"{y_col} by {x_col}" if x_col and y_col else "Result")

    table = pd.DataFrame(rows)
    if not x_col or x_col not in table.columns:
        st.caption(f"Cannot chart: column '{x_col}' is not in the computed result.")
        return
    if y_col and y_col not in table.columns:
        st.caption(f"Cannot chart: column '{y_col}' is not in the computed result.")
        return

    if chart_type == "line":
        fig = _px.line(table, x=x_col, y=y_col, title=title)
    elif chart_type == "scatter":
        fig = _px.scatter(table, x=x_col, y=y_col, title=title)
    elif chart_type == "pie":
        fig = _px.pie(table, names=x_col, values=y_col, title=title)
    elif chart_type == "histogram":
        fig = _px.histogram(table, x=x_col, y=y_col, title=title)
    else:  # bar and any unexpected value
        fig = _px.bar(table, x=x_col, y=y_col, title=title, text_auto=True)

    fig.update_layout(get_plotly_layout_template())
    st.plotly_chart(fig, use_container_width=True)


st.set_page_config(
    page_title="OmniSleuth",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# DESIGN SYSTEM CSS
# ─────────────────────────────────────────────────────────────
st.markdown(THEME_CSS, unsafe_allow_html=True)


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


def get_or_create_session_id() -> str:
    """Obtain or securely generate an opaque session identifier for this Streamlit browser session."""
    if "session_id" not in st.session_state or not st.session_state.session_id:
        st.session_state.session_id = f"sess_{uuid.uuid4().hex[:16]}"
    return st.session_state.session_id


current_session_id = get_or_create_session_id()

if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.metadata = None

for _key, _default in (
    ("last_inv", None),
    ("nlq_routing", None),
    ("nlq_history", []),
    ("llm_last_error_class", None),
):
    if _key not in st.session_state:
        setattr(st.session_state, _key, _default)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="display:flex;align-items:center;gap:0.75rem;padding:0.75rem 0 0.5rem 0;">
            <div style="width:34px;height:34px;border-radius:8px;background:linear-gradient(135deg,#4F46E5 0%,#06B6D4 100%);display:flex;align-items:center;justify-content:center;color:#FFFFFF;font-weight:800;font-size:1rem;box-shadow:0 2px 8px rgba(79,70,229,0.35);">OS</div>
            <div>
                <div style="font-size:1.05rem;font-weight:700;color:#F8FAFC;letter-spacing:-0.01em;line-height:1.2;">OmniSleuth</div>
                <div style="font-size:0.72rem;color:#94A3B8;margin-top:0.1rem;">Data Investigation System</div>
            </div>
        </div>
        <hr style="border-color:#334155;margin:0.75rem 0;">
        <div class="sidebar-section">Workspace</div>
    """, unsafe_allow_html=True)

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
            df, meta = loader.load_csv(
                sample_path,
                dataset_name="Benchmark Sales 2024",
                session_id=current_session_id,
            )
            repo.save_dataset(meta, session_id=current_session_id)
            st.session_state.df = df
            st.session_state.metadata = meta
    else:
        uploaded_file = st.file_uploader(
            "Upload dataset", type=["csv", "parquet", "xlsx"], label_visibility="visible"
        )
        if uploaded_file is not None:
            # 0. Enforce upload rate limit per session before writing any file to disk
            limiter = get_rate_limiter()
            is_allowed, count, retry_after = limiter.check("upload", current_session_id, consume=True)
            if not is_allowed:
                limit = limiter.get_limit_for_action("upload")
                logger.warning(
                    f"Upload rate limit exceeded for session '{current_session_id}' ({count}/{limit})."
                )
                st.error(
                    f"⚠️ **Upload Rate Limit Exceeded**: Session has reached the limit of {limit} uploads per hour. "
                    f"Please try again in {retry_after / 60:.1f} minutes."
                )
            else:
                max_upload_size = get_settings().max_upload_size_bytes
                dest_path = None
                try:
                    # 1. Sanitize filename, enforce filesystem containment, stream chunked to isolated unique path
                    safe_path, display_name = safe_save_upload(
                        file_obj=uploaded_file,
                        raw_filename=uploaded_file.name,
                        upload_dir=get_settings().uploads_dir,
                        max_size_bytes=max_upload_size,
                    )
                    dest_path = safe_path

                    # 2. Parse and validate content via CSVLoader
                    loader = CSVLoader()
                    df, meta = loader.load_csv(
                        safe_path,
                        dataset_name=display_name,
                        session_id=current_session_id,
                    )
                    repo.save_dataset(
                        meta,
                        verify_file=True,
                        cleanup_on_failure=True,
                        session_id=current_session_id,
                    )
                    st.session_state.df = df
                    st.session_state.metadata = meta
                except IngestionError as exc:
                    logger.warning(f"Upload validation failed for '{uploaded_file.name}': {exc}")
                    st.error(f"Upload failed: {exc}")
                    if dest_path and dest_path.exists() and st.session_state.df is None:
                        safe_cleanup_file(dest_path)
                except Exception as exc:
                    logger.error(f"Unexpected error loading dataset '{uploaded_file.name}': {exc}", exc_info=True)
                    st.error(f"Failed to process uploaded file '{uploaded_file.name}'. Please ensure it is a valid CSV, Parquet, or XLSX file.")
                    if dest_path and dest_path.exists() and st.session_state.df is None:
                        safe_cleanup_file(dest_path)

    # Saved datasets in this session
    session_saved = repo.list_datasets(check_files=True, session_id=current_session_id)
    if session_saved:
        with st.expander(f"Saved Session Datasets ({len(session_saved)})", expanded=False):
            ds_choices = {}
            for d in session_saved:
                f_status = "" if d.get("file_exists", True) else " ⚠️ (File missing)"
                ds_choices[f"{d['name']} ({d['row_count']:,} rows){f_status}"] = d
            chosen_label = st.selectbox(
                "Choose saved dataset",
                list(ds_choices.keys()),
                key="select_saved_ds",
            )
            if st.button("Activate Dataset", key="btn_load_saved_ds", use_container_width=True):
                chosen_rec = ds_choices[chosen_label]
                if not chosen_rec.get("file_exists", True):
                    st.error(f"Cannot load '{chosen_rec['name']}': File is missing from storage.")
                else:
                    try:
                        loaded_df, loaded_meta = repo.load_dataset_dataframe(
                            chosen_rec["dataset_id"], session_id=current_session_id
                        )
                        st.session_state.df = loaded_df
                        st.session_state.metadata = loaded_meta
                        st.session_state.last_inv = None
                        st.rerun()
                    except IngestionError as exc:
                        st.error(f"Failed to load dataset: {exc}")

    with st.expander("Session Details", expanded=False):
        st.markdown(
            f'<div style="font-size:0.75rem;color:#94A3B8;">Current Session ID:</div>'
            f'<div class="sidebar-value" style="font-family:monospace;font-size:0.75rem;word-break:break-all;">{current_session_id}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style="font-size:0.75rem;color:#94A3B8;margin-top:0.5rem;line-height:1.4;">
                <strong style="color:#E2E8F0;">Session Isolation:</strong> Sessions are strictly isolated to protect data privacy.
                Unauthenticated session resumption has been disabled to prevent cross-session data access.
                <br><br>
                <em style="color:#CBD5E1;">Browser Refresh Notice:</em> Refreshing your browser will initialize a new isolated session.
                Previously uploaded datasets remain protected under their originating session and cannot be accessed from new sessions.
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Start New Session", key="btn_start_new_session", use_container_width=True):
            st.session_state.session_id = f"sess_{uuid.uuid4().hex[:16]}"
            st.session_state.df = None
            st.session_state.metadata = None
            st.session_state.last_inv = None
            st.session_state.nlq_history = []
            st.rerun()

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
        settings = get_settings()
        llm_configured = bool(
            settings.openai_api_key or settings.azure_openai_api_key or settings.openrouter_api_key
        )
        last_error_class = st.session_state.get("llm_last_error_class")
        db_filename = Path(repo.db.db_path).name

        # ONE status line: never key material, only provider/model state and last error class.
        if not llm_configured:
            st.sidebar.info(
                "LLM not configured; showing deterministic results only. Claims and findings are computed "
                "directly from raw data via allowlisted tools without generative text synthesis."
            )
            llm_mode_label = "Not configured (deterministic only)"
            llm_mode_color = "#F59E0B"
        else:
            provider_name = (settings.llm_provider or "openrouter").lower()
            if provider_name == "openrouter":
                model_name = settings.openrouter_model
            elif provider_name == "azure":
                model_name = settings.llm_model
            else:
                model_name = settings.llm_model
            llm_mode_label = f"Configured ({provider_name} · {model_name})"
            llm_mode_color = "#22C55E"
        if last_error_class:
            llm_mode_label = f"{llm_mode_label} · last error: {last_error_class}"

        st.markdown(f"""
        <div style="padding:0.3rem 0;">
            <div style="font-size:0.68rem;color:#475569;margin-bottom:0.1rem;">Dataset ID</div>
            <div class="sidebar-value">{meta.dataset_id}</div>
        </div>
        <div style="padding:0.3rem 0;">
            <div style="font-size:0.68rem;color:#475569;margin-bottom:0.1rem;">Storage</div>
            <div class="sidebar-value">SQLite &middot; {db_filename}</div>
        </div>
        <div style="padding:0.3rem 0;">
            <div style="font-size:0.68rem;color:#475569;margin-bottom:0.1rem;">LLM status</div>
            <div style="font-size:0.78rem;color:{llm_mode_color};">{llm_mode_label}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:0.8rem;color:#475569;padding:0.5rem 0;">No dataset loaded.</div>',
                    unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
if st.session_state.df is None:
    render_app_header(None)
    st.stop()

df = st.session_state.df
metadata = st.session_state.metadata

render_app_header(metadata)


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Investigation", "Discovery", "Data Quality",
    "Anomaly Explorer", "Evidence & Provenance", "Evaluation",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — Investigation (single question box routed through AnalysisPipeline)
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-label">Autonomous Investigation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Natural Language Investigation Workspace</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Submit analytical questions in plain English. Every question is converted into a '
        'structured, schema-validated analysis plan and executed deterministically. '
        'Diagnostic &ldquo;why&rdquo; questions automatically trigger multi-agent hypothesis generation and evidentiary grounding.</div>',
        unsafe_allow_html=True)

    app_settings = get_settings()

    # Planner selection stays explicit and opt-in: deterministic is the default.
    PLANNER_OPTIONS = {
        "Deterministic (rule-based)": "rule",
        "LLM-assisted (validated plan)": "llm",
    }
    planner_label = st.radio(
        "Planner",
        list(PLANNER_OPTIONS.keys()),
        index=0 if app_settings.nlq_planner_mode != "llm" else 1,
        horizontal=True,
        help="Deterministic planning needs no API key. LLM-assisted planning routes through the validated "
             "LLMPlanner pipeline; every plan is still validated before execution.",
    )
    selected_mode = PLANNER_OPTIONS[planner_label]

    if selected_mode == "llm":
        from omnisleuth.llm.gateway import LLMGateway
        if not LLMGateway().is_configured:
            st.info(
                "ℹ️ **LLM planner selected but no provider is configured.** No request will be made; "
                "the deterministic planner is used instead."
            )
        else:
            st.markdown(
                '<div class="planner-indicator-box">'
                '<span>⚡ <strong>LLM-Assisted Planner Active:</strong> Analysis plans will be synthesized via OpenRouter and validated against your schema.</span>'
                '</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="planner-indicator-box">'
            '<span>🛡️ <strong>Deterministic Planner Active:</strong> Rules and heuristics only. Zero external API calls.</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    col_q, col_ex = st.columns([3, 2])
    with col_q:
        user_question = st.text_input(
            "Question",
            placeholder="e.g. What is total revenue by category?",
            label_visibility="collapsed")
    with col_ex:
        EXAMPLE_QUESTIONS = infer_example_questions(build_schema_context(df, metadata))
        selected_example = st.selectbox(
            "Example questions", [""] + EXAMPLE_QUESTIONS, label_visibility="collapsed")
        if selected_example:
            user_question = selected_example

    btn_col, _ = st.columns([1, 3])
    with btn_col:
        run_investigation = st.button("Investigate", type="primary", use_container_width=True)

    # Investigation History Reopening Flow
    past_investigations = repo.list_investigations(
        dataset_id=metadata.dataset_id, session_id=current_session_id
    )
    if past_investigations:
        with st.expander(f"Investigation History ({len(past_investigations)} saved)", expanded=False):
            inv_options = {
                f"{inv['created_at'][:19]} — {inv['question'][:60]} ({inv['investigation_id']})": inv["investigation_id"]
                for inv in past_investigations
            }
            selected_label = st.selectbox(
                "Select saved investigation to reopen",
                list(inv_options.keys()),
                key="history_select",
            )
            if st.button("Reopen Investigation", key="reopen_btn"):
                selected_id = inv_options[selected_label]
                reopened_state = repo.reconstruct_investigation_state(
                    selected_id, session_id=current_session_id
                )
                if reopened_state:
                    inv_meta = metadata or reopened_state.get("dataset_metadata")
                    st.session_state.last_inv = {
                        "question": reopened_state["question"],
                        "dataset_metadata": inv_meta,
                        "plan": reopened_state.get("plan"),
                        "executed_tasks": reopened_state.get("executed_tasks", []),
                        "evidence_store": reopened_state["evidence_store"],
                        "hypotheses": reopened_state["hypotheses"],
                        "claims": reopened_state["claims"],
                        "validation_results": [],
                        "final_answer": reopened_state.get("final_answer", ""),
                        "status": reopened_state.get("status", "completed"),
                        "planner_used": reopened_state.get("planner_used"),
                        "summary_text": reopened_state.get("summary_text"),
                        "reopened_result_table": reopened_state.get("result_table"),
                        "reopened_chart_spec": reopened_state.get("chart_spec"),
                    }
                    st.session_state.nlq_routing = None
                    st.rerun()
                else:
                    st.error("Failed to reopen investigation: record not found, unauthorized, or corrupted.")

    if run_investigation:
        if metadata and getattr(metadata, "dataset_id", None):
            ds_check = repo.get_dataset(metadata.dataset_id, session_id=current_session_id)
            if not ds_check:
                st.error("Cannot run investigation: Dataset not found or inaccessible in current session.")
                st.stop()

        if metadata and getattr(metadata, "file_path", None):
            file_p = Path(metadata.file_path)
            if not file_p.exists() or not file_p.is_file():
                st.error(
                    f"Cannot run investigation: Underlying dataset file for '{metadata.name}' is missing from storage."
                )
                st.stop()

        with st.spinner("Planning and executing\u2026"):
            try:
                # Reuse the app's cached repository and orchestrator so the routed
                # question writes to the same store the history UI reads from.
                _executor = PlanExecutor(
                    repository=repo.scoped(current_session_id),
                    orchestrator=orchestrator,
                )
                pipeline = AnalysisPipeline(
                    mode=selected_mode,
                    fallback=app_settings.nlq_planner_fallback,
                    executor=_executor,
                )
                routing = pipeline.run(
                    user_question or "",
                    df,
                    metadata,
                    history_ctx=list(st.session_state.get("nlq_history", [])),
                    session_id=current_session_id,
                )
                st.session_state.nlq_routing = routing
                st.session_state.nlq_history = (
                    list(st.session_state.get("nlq_history", []))[-4:] + [routing.history_entry]
                )[:5]
                logger.info(
                    "Question routed: planner=%s requested=%s fallback=%s",
                    routing.planner_used, routing.requested_mode, routing.fallback_used,
                )

                # Diagnostic answers are rendered by the multi-agent view; persist once.
                if routing.deep_investigation and routing.result.status == "answered":
                    _store = routing.result.evidence_store
                    st.session_state.last_inv = {
                        "question": user_question,
                        "dataset_metadata": metadata,
                        "plan": None,
                        "executed_tasks": [],
                        "evidence_store": _store,
                        "hypotheses": _store.list_hypotheses() if _store is not None else [],
                        "claims": routing.result.claims,
                        "validation_results": [],
                        "final_answer": routing.result.answer,
                        "status": routing.result.status,
                        "planner_used": routing.planner_used,
                    }
                    try:
                        repo.save_investigation_results(
                            investigation_id=(
                                routing.result.investigation_id
                                or f"inv_{metadata.dataset_id[:6]}_{int(pd.Timestamp.now().timestamp())}"
                            ),
                            dataset_id=metadata.dataset_id,
                            question=user_question,
                            status=routing.result.status,
                            plan_dict=routing.result.plan.model_dump(),
                            final_answer=routing.result.answer,
                            evidence_store=_store,
                            planner_used=routing.planner_used,
                            summary_text=routing.result.plain_language_summary,
                            session_id=current_session_id,
                        )
                    except Exception as exc:
                        logger.warning(f"Could not persist deep investigation: {exc}")
                    # Refresh the history list so the saved run can be reopened.
                    st.rerun()
                elif not routing.deep_investigation:
                    st.session_state.last_inv = None
            except Exception as exc:
                logger.error(f"Investigation failed: {exc}", exc_info=True)
                st.error(f"Request could not be completed: {type(exc).__name__}")

    # ══════════════════════════════════════════════════════════
    # INVESTIGATION RESULTS RENDERING (6 Structured Sections)
    # ══════════════════════════════════════════════════════════
    routing = st.session_state.get("nlq_routing")

    if st.session_state.get("last_inv") is not None:
        inv_res = st.session_state.last_inv
        store = inv_res.get("evidence_store")
        hypotheses = store.list_hypotheses() if store is not None else []
        evidence_items = store.list_evidence() if store is not None else []
        validation_results = inv_res.get("validation_results", [])
        executed_tasks = inv_res.get("executed_tasks", [])
        hyp_supported = sum(1 for h in hypotheses if h.status == HypothesisStatus.SUPPORTED)
        hyp_rejected  = sum(1 for h in hypotheses if h.status == HypothesisStatus.REJECTED)

        headline_pct = None
        for ev in evidence_items:
            pct = ev.content.get("overall_change_pct") or ev.content.get("percentage_change") or ev.content.get("total_change_pct")
            if pct is not None:
                headline_pct = pct
                break

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # Meta badge bar
        status_val = inv_res.get("status", "completed")
        planner_val = inv_res.get("planner_used") or "deterministic"
        status_tag_html = render_status_badge(status_val.title(), status_val)
        planner_badge_html = render_meta_badge("Planner", planner_val)
        mode_badge_html = render_meta_badge("Mode", "Diagnostic Investigation")

        st.markdown(
            f'<div class="investigation-meta-bar">'
            f'{status_tag_html}'
            f'{planner_badge_html}'
            f'{mode_badge_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Active question title
        st.markdown(
            f'<div style="font-size:0.75rem;font-weight:600;color:#64748B;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.2rem;">Investigation Result for:</div>'
            f'<div style="font-size:1.15rem;font-weight:700;color:#0F172A;margin-bottom:1rem;">'
            f'&ldquo;{inv_res.get("question", user_question)}&rdquo;</div>',
            unsafe_allow_html=True)

        if inv_res.get("summary_text"):
            st.markdown(
                f'<div style="font-size:0.82rem;color:#475569;margin-bottom:1rem;font-style:italic;">'
                f'Interpretation: {inv_res["summary_text"]}</div>',
                unsafe_allow_html=True)

        # ── 1. Executive Summary ──
        st.markdown('<div class="section-label">1. Executive Summary</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="summary-block">{inv_res["final_answer"]}</div>',
            unsafe_allow_html=True)

        # ── 2. Key Metrics ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">2. Key Metrics</div>', unsafe_allow_html=True)
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        with mc1:
            if headline_pct is not None:
                render_kpi_card(
                    "Revenue Change",
                    f"{headline_pct:+.1f}%",
                    delta_type="negative" if headline_pct < 0 else "positive",
                )
            else:
                render_kpi_card("Status", inv_res["status"].title(), delta_type="neutral")
        with mc2:
            render_kpi_card("Evidence Items", str(len(evidence_items)), delta_type="neutral")
        with mc3:
            render_kpi_card("Hypotheses", str(len(hypotheses)), delta_type="neutral")
        with mc4:
            render_kpi_card("Supported", str(hyp_supported), delta_type="positive")
        with mc5:
            render_kpi_card("Rejected", str(hyp_rejected), delta_type="negative" if hyp_rejected > 0 else "neutral")

        # ── 3. Results Table and Chart ──
        has_chart = any(ev.content.get("time_series") or ev.content.get("segments") for ev in evidence_items) or bool(inv_res.get("reopened_chart_spec"))
        has_table = bool(inv_res.get("reopened_result_table"))

        if has_chart or has_table:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">3. Visual & Tabular Analysis</div>', unsafe_allow_html=True)

            if inv_res.get("reopened_result_table"):
                st.dataframe(pd.DataFrame(inv_res["reopened_result_table"]), use_container_width=True, hide_index=True)

            if inv_res.get("reopened_chart_spec"):
                _restored = inv_res["reopened_chart_spec"]
                _render_chart(inv_res.get("reopened_result_table") or [], _restored)

            # Evidence-derived charts (timeseries / segments)
            for ev in evidence_items:
                if ev.content.get("time_series"):
                    ts_data = pd.DataFrame(ev.content["time_series"])
                    metric_name = ev.content.get("metric", "Metric").replace("_", " ").title()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=ts_data["period"], y=ts_data["value"],
                        mode="lines+markers",
                        line=dict(color=BRAND_PRIMARY, width=2.5),
                        marker=dict(size=6, color=BRAND_PRIMARY),
                    ))
                    layout = get_plotly_layout_template()
                    layout["title"] = dict(
                        text=f"Monthly Trend \u2014 {metric_name}",
                        font=dict(size=13, color="#0F172A", family="Inter"),
                        x=0,
                    )
                    layout["height"] = 280
                    layout["showlegend"] = False
                    fig.update_layout(layout)
                    st.plotly_chart(fig, use_container_width=True)

                elif ev.content.get("segments"):
                    seg_data = pd.DataFrame(ev.content["segments"])
                    seg_col = ev.content.get("segment_column", "Segment").replace("_", " ").title()
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=seg_data["segment"], y=seg_data["value"],
                        marker_color=PLOTLY_COLORWAY[:len(seg_data)],
                        text=[f"{v:,.0f}" for v in seg_data["value"]],
                        textposition="outside",
                        textfont=dict(size=10, family="Inter"),
                    ))
                    layout = get_plotly_layout_template()
                    layout["title"] = dict(
                        text=f"Segment Comparison \u2014 {seg_col}",
                        font=dict(size=13, color="#0F172A", family="Inter"),
                        x=0,
                    )
                    layout["height"] = 280
                    layout["showlegend"] = False
                    fig.update_layout(layout)
                    st.plotly_chart(fig, use_container_width=True)

        # ── 4. Findings and Hypotheses ──
        if hypotheses:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">4. Findings & Evaluated Hypotheses</div>', unsafe_allow_html=True)

            for hyp in hypotheses:
                status_name = hyp.status.value.lower() if hasattr(hyp.status, "value") else str(hyp.status).lower()
                badge_html = render_status_badge(status_name.title(), status_name)
                summary = hyp.findings_summary or hyp.rationale or ""
                conf_pct = f"{int(hyp.confidence * 100)}%" if hyp.confidence is not None else ""
                conf_html = f'<div class="hyp-footer"><span>Statistical Confidence: <strong>{conf_pct}</strong></span></div>' if conf_pct else ""

                st.markdown(
                    f'<div class="hyp-card">'
                    f'<div class="hyp-header">'
                    f'<span class="hyp-statement">{hyp.statement}</span>'
                    f'{badge_html}'
                    f'</div>'
                    + (f'<div class="hyp-rationale">{summary}</div>' if summary else "")
                    + f'{conf_html}</div>',
                    unsafe_allow_html=True,
                )

        # ── 5. Evidence and Claims ──
        has_claims = bool(validation_results or inv_res.get("claims"))
        has_evidence = bool(evidence_items)

        if has_claims or has_evidence:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">5. Evidence Grounding & Validated Claims</div>', unsafe_allow_html=True)

            col_ev, col_clm = st.columns(2, gap="medium")

            with col_clm:
                st.markdown('<div style="font-size:0.8rem;font-weight:600;color:#475569;margin-bottom:0.5rem;">Validated Analytical Claims</div>', unsafe_allow_html=True)
                if validation_results:
                    for vr in validation_results:
                        status_val = vr.status.value.lower() if hasattr(vr.status, "value") else str(vr.status).lower()
                        badge = render_status_badge(status_val.title(), status_val)
                        statement = vr.calibrated_statement or vr.original_statement
                        st.markdown(
                            f'<div class="claim-card">'
                            f'<div>{badge}</div>'
                            f'<div class="claim-statement">{statement}</div>'
                            + (f'<div class="claim-explanation">{vr.validation_explanation}</div>' if vr.validation_explanation else "")
                            + '</div>',
                            unsafe_allow_html=True,
                        )
                elif inv_res.get("claims"):
                    for clm in inv_res.get("claims", []):
                        status_val = clm.status.value.lower() if hasattr(clm.status, "value") else str(clm.status).lower()
                        badge = render_status_badge(status_val.title(), status_val)
                        st.markdown(
                            f'<div class="claim-card">'
                            f'<div>{badge}</div>'
                            f'<div class="claim-statement">{clm.statement}</div>'
                            + (f'<div class="claim-explanation">{clm.validation_notes}</div>' if clm.validation_notes else "")
                            + '</div>',
                            unsafe_allow_html=True,
                        )

            with col_ev:
                st.markdown(f'<div style="font-size:0.8rem;font-weight:600;color:#475569;margin-bottom:0.5rem;">Grounding Evidence ({len(evidence_items)} items)</div>', unsafe_allow_html=True)
                for ev in evidence_items:
                    ev_type = ev.evidence_type.value.replace("_", " ").title() if hasattr(ev.evidence_type, "value") else str(ev.evidence_type)
                    title_raw = (ev.content.get("metric") or ev.content.get("title")
                                 or ev.content.get("description") or ev_type)
                    title = str(title_raw).replace("_", " ").title()[:60]
                    snippet = ""
                    for k in ["percentage_change", "value", "trend_direction", "summary", "text"]:
                        v = ev.content.get(k)
                        if v is not None:
                            snippet = f"{k.replace('_', ' ').title()}: {v}"
                            break
                    conf_text = f"{int(ev.confidence * 100)}% confidence" if ev.confidence is not None else ""
                    st.markdown(
                        f'<div class="evidence-card">'
                        f'<div class="evidence-title">{title}</div>'
                        f'<div class="evidence-meta">'
                        f'<span>Type: <strong>{ev_type}</strong></span>'
                        f'<span>Source: <strong>{ev.source_id[:20]}</strong></span>'
                        + (f'<span>{conf_text}</span>' if conf_text else "")
                        + '</div>'
                        + (f'<div class="evidence-content">{snippet}</div>' if snippet else "")
                        + '</div>',
                        unsafe_allow_html=True,
                    )

        # ── 6. Execution Details ──
        if executed_tasks:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander(f"6. Execution Plan & Multi-Agent Trace ({len(executed_tasks)} steps)", expanded=False):
                for task in executed_tasks:
                    if isinstance(task, dict):
                        agent = str(task.get("assigned_to", "ANALYST")).upper()
                        tool = task.get("tool_name") or "Analysis"
                        rationale = task.get("rationale") or ""
                    else:
                        agent = task.assigned_to.value.upper() if hasattr(task.assigned_to, "value") else str(task.assigned_to).upper()
                        tool = task.tool_name or "Analysis"
                        rationale = task.rationale or ""
                    st.markdown(
                        f'<div class="step-row">'
                        f'<div class="step-agent">{agent}</div>'
                        f'<div class="step-tool">{tool}</div>'
                        f'<div class="step-rationale">{rationale[:160]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    elif routing is not None:
        _result = routing.result
        _status = _result.status

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        status_tag_html = render_status_badge(_status.title(), _status)
        planner_badge_html = render_meta_badge("Planner", routing.planner_used)
        mode_badge_html = render_meta_badge("Requested", routing.requested_mode)
        llm_badge_html = render_meta_badge("LLM", "Configured" if routing.llm_configured else "Not Configured")

        st.markdown(
            f'<div class="investigation-meta-bar">'
            f'{status_tag_html}'
            f'{planner_badge_html}'
            f'{mode_badge_html}'
            f'{llm_badge_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

        if routing.fallback_used:
            st.warning(f"Deterministic fallback was used: {routing.fallback_reason}")
        if routing.followup_of:
            st.caption(f"Follow-up on: \u201c{routing.followup_of}\u201d")
        if routing.general_knowledge and _status == "answered":
            st.info(
                "**General knowledge, not computed from your dataset.** "
                "No evidence or claims are shown because nothing here is derived from your data."
            )

        # Active question title
        st.markdown(
            f'<div style="font-size:0.75rem;font-weight:600;color:#64748B;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.2rem;">Question Result:</div>'
            f'<div style="font-size:1.15rem;font-weight:700;color:#0F172A;margin-bottom:1rem;">'
            f'&ldquo;{user_question}&rdquo;</div>',
            unsafe_allow_html=True)

        _interpretation = (
            _result.plain_language_summary
            or (_result.plan.needs_clarification if _result.plan else None)
            or (_result.plan.unsupported_reason if _result.plan else None)
            or ""
        )
        if _interpretation and _status != "answered":
            st.markdown(
                f'<div style="font-size:0.82rem;color:#475569;margin-bottom:1rem;font-style:italic;">'
                f'Interpretation: {_interpretation}</div>',
                unsafe_allow_html=True)

        if _status == "answered":
            # ── 1. Executive Summary ──
            st.markdown('<div class="section-label">1. Executive Summary</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-block">{_result.answer}</div>', unsafe_allow_html=True)

            if not routing.general_knowledge:
                # ── 3. Results Table and Chart ──
                if _result.result_table or (routing.chart and routing.chart.used and routing.chart.spec):
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="section-label">3. Results Table & Chart</div>', unsafe_allow_html=True)

                    if _result.result_table:
                        st.dataframe(pd.DataFrame(_result.result_table), use_container_width=True, hide_index=True)

                    _chart = routing.chart
                    if _chart is not None and _chart.used and _chart.spec:
                        _render_chart(_chart.data, _chart.spec)
                    elif _chart is not None and _chart.explanation:
                        st.caption(f"No chart shown: {_chart.explanation}")

                # ── 5. Evidence and Claims ──
                _ev_store = _result.evidence_store
                if _ev_store is not None:
                    ev_items = _ev_store.list_evidence()
                    clm_items = _ev_store.list_claims()
                    if ev_items or clm_items:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown('<div class="section-label">5. Evidence Grounding & Claims</div>', unsafe_allow_html=True)

                        c1, c2 = st.columns(2, gap="medium")
                        with c1:
                            if ev_items:
                                ev_rows = [
                                    {"Source": ev.source_id, "Type": ev.evidence_type.value, "Confidence": f"{ev.confidence:.2f}"}
                                    for ev in ev_items
                                ]
                                st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)
                        with c2:
                            if clm_items:
                                clm_rows = [
                                    {"Statement": c.statement, "Status": c.status.value, "Notes": c.validation_notes or ""}
                                    for c in clm_items
                                ]
                                st.dataframe(pd.DataFrame(clm_rows), use_container_width=True, hide_index=True)

                # ── 6. Execution Details ──
                if _result.limitations:
                    st.markdown("<br>", unsafe_allow_html=True)
                    with st.expander("6. Execution Details & Limitations", expanded=False):
                        for limitation in _result.limitations:
                            st.caption(f"Limitation: {limitation}")
        else:
            _status_explanation = {
                "needs_clarification": "This question needs one more detail before it can be answered exactly.",
                "unsupported": "This question cannot be answered from the current dataset schema.",
                "failed": "This question could not be completed.",
                "blocked": "This inquiry was blocked by security guardrails.",
            }
            st.error(
                f"{_status_explanation.get(_result.status, _result.status.title())}\n\n{_result.answer}"
            )
            if _result.limitations:
                for limitation in _result.limitations:
                    st.caption(f"Limitation: {limitation}")

    else:
        # ── Empty-State Guidance when no investigation has been run ──
        render_investigation_empty_state()



# ══════════════════════════════════════════════════════════════
# TAB 2 — Discovery
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-label">Automated Discovery Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Autonomous Dataset Intelligence Scan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Proactively scans dataset dimensions without a user prompt to detect '
        'statistical inflection points, category concentrations, and anomalies ranked by composite significance.</div>',
        unsafe_allow_html=True)

    if st.button("Run Discovery Scan", type="primary"):
        with st.spinner("Scanning dataset dimensions\u2026"):
            findings = scanner.scan(df, metadata)
            st.session_state.discovery_findings = findings

    if "discovery_findings" in st.session_state:
        findings = st.session_state.discovery_findings
        if not findings:
            st.info("No significant statistical findings surfaced for this dataset schema.")
        else:
            n_high = sum(1 for f in findings if f.composite_rank_score >= 0.7)
            n_med = sum(1 for f in findings if 0.4 <= f.composite_rank_score < 0.7)
            n_low = sum(1 for f in findings if f.composite_rank_score < 0.4)

            st.markdown(
                f'<div class="investigation-meta-bar">'
                f'<span class="meta-badge">Total Findings: <strong>{len(findings)}</strong></span>'
                f'<span class="meta-badge">High Significance: <strong style="color:#DC2626;">{n_high}</strong></span>'
                f'<span class="meta-badge">Medium Significance: <strong style="color:#D97706;">{n_med}</strong></span>'
                f'<span class="meta-badge">Low Significance: <strong style="color:#64748B;">{n_low}</strong></span>'
                f'</div>',
                unsafe_allow_html=True)

            for i, f in enumerate(findings, start=1):
                score = f.composite_rank_score
                cat = f.category.value.replace("_", " ").title()
                if score >= 0.7:
                    sig_badge = render_status_badge("High Significance", "rejected")
                elif score >= 0.4:
                    sig_badge = render_status_badge("Medium Significance", "warning")
                else:
                    sig_badge = render_status_badge("Low Significance", "info")

                with st.expander(f"Finding {i}: {f.title} (Score: {score:.2f})", expanded=(i == 1)):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(
                            f'<div class="discovery-category">{cat}</div>'
                            f'<div class="discovery-desc">{f.description}</div>'
                            f'<div class="info-box" style="margin-top:0.5rem;margin-bottom:0.75rem;">'
                            f'<strong>Recommended Exploration Question:</strong><br>&ldquo;{f.recommended_question}&rdquo;</div>',
                            unsafe_allow_html=True)
                    with col_b:
                        st.markdown(
                            f'<div style="text-align:right;padding-top:0.25rem;">'
                            f'<div style="margin-bottom:0.4rem;">{sig_badge}</div>'
                            f'<div class="section-label" style="font-size:0.68rem;">Composite Score</div>'
                            f'<div style="font-size:1.4rem;font-weight:700;color:#0F172A;font-variant-numeric:tabular-nums;">{score:.2f}</div>'
                            f'</div>',
                            unsafe_allow_html=True)

                    tree = scanner.build_investigation_tree(f)
                    st.markdown('<div class="section-label" style="margin-top:0.75rem;">Structured Investigation Path</div>',
                                unsafe_allow_html=True)
                    st.code(
                        f"1  Finding     {tree.label}\n"
                        f"   \u2514\u2500 2  Hypothesis  {tree.children[0].label}\n"
                        f"         \u2514\u2500 3  Evidence    {tree.children[0].children[0].label}\n"
                        f"               \u2514\u2500 4  Test        {tree.children[0].children[0].children[0].label}\n"
                        f"                     \u2514\u2500 5  Result      {tree.children[0].children[0].children[0].children[0].label}",
                        language="text")

                    with st.expander("Evidence Payload", expanded=False):
                        st.json(f.evidence_payload)
    else:
        render_empty_state(
            title="No Intelligence Scan Run",
            desc="Run an autonomous discovery scan to proactively uncover statistical inflection points, "
                 "high-variance categories, and emergent patterns across all dimensions without writing a prompt.",
            pills=["⚡ Autonomous Profiling", "📈 Inflection Points", "🔍 Concentration Analysis", "🎯 Composite Ranking"],
            icon_type="scan",
        )


# ══════════════════════════════════════════════════════════════
# TAB 3 — Data Quality
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-label">Data Quality & Integrity</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Dataset Profile & Quality Report</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Automated schema profiling and data integrity audit measuring completeness, '
        'duplicate records, cardinality, and column-level distributions.</div>',
        unsafe_allow_html=True)

    report = profiler.generate_report(df, metadata)
    score = report.completeness_score

    qc1, qc2, qc3, qc4 = st.columns(4)
    with qc1:
        render_kpi_card(
            "Completeness",
            f"{score:.1f}%",
            delta_type="positive" if score >= 90 else "negative" if score < 70 else "neutral",
            subtitle="Non-null values",
        )
    with qc2:
        render_kpi_card(
            "Duplicate Rows",
            str(report.duplicate_rows),
            delta_type="negative" if report.duplicate_rows > 0 else "positive",
            subtitle="Exact duplicate records",
        )
    with qc3:
        render_kpi_card(
            "Quality Alerts",
            str(len(report.warnings)),
            delta_type="negative" if len(report.warnings) > 0 else "positive",
            subtitle="Audit flags surfaced",
        )
    with qc4:
        render_kpi_card(
            "Columns Profiled",
            str(metadata.column_count),
            delta_type="neutral",
            subtitle=f"{metadata.row_count:,} total records",
        )

    if report.warnings:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Quality Alerts & Warnings</div>', unsafe_allow_html=True)
        for w in report.warnings:
            sev = w.severity.value
            box_cls = "warn-box" if sev in ("warning", "critical") else "info-box"
            status_style = "rejected" if sev in ("warning", "critical") else "info"
            st.markdown(
                f'<div class="{box_cls}" style="margin-bottom:0.5rem;display:flex;align-items:center;gap:0.65rem;">'
                f'{render_status_badge(sev.upper(), status_style)}'
                f'<span style="font-size:0.84rem;">{w.message}</span>'
                f'</div>',
                unsafe_allow_html=True)
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">✅ <strong>Zero Data Quality Issues Detected:</strong> All columns meet schema completeness standards with zero null anomalies.</div>',
            unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    col_num, col_cat = st.columns(2, gap="large")

    with col_num:
        st.markdown('<div class="section-label">Numeric Column Profiling</div>', unsafe_allow_html=True)
        if report.numeric_summaries:
            st.dataframe(
                pd.DataFrame([s.model_dump() for s in report.numeric_summaries.values()]),
                use_container_width=True, hide_index=True)

    with col_cat:
        st.markdown('<div class="section-label">Categorical Column Profiling</div>', unsafe_allow_html=True)
        if report.categorical_summaries:
            cat_data = [
                {"Column": c.column_name, "Cardinality": c.cardinality, "Missing Count": c.missing_count}
                for c in report.categorical_summaries.values()
            ]
            st.dataframe(pd.DataFrame(cat_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — Anomaly Explorer
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-label">Statistical & ML Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Outlier & Anomaly Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Identifies multivariate and distributional anomalies across numeric features '
        'using unsupervised Isolation Forests or robust Interquartile Range (IQR) bounds. '
        'Results represent statistical divergence and do not assert causality.</div>',
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
        with st.spinner("Running anomaly detection…"):
            try:
                if "Isolation Forest" in method:
                    det = IsolationForestDetector(contamination=0.03)
                else:
                    det = IQRAnomalyDetector(multiplier=1.5)
                anom_res = det.detect(df, [target_num_col])
                st.session_state.anom_res = anom_res
                st.session_state.anom_col = target_num_col
            except Exception as exc:
                st.warning(f"⚠️ Anomaly detection could not be completed: {exc}")

    if "anom_res" in st.session_state:
        anom_res = st.session_state.anom_res
        anom_col = st.session_state.anom_col

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        a1, a2, a3 = st.columns(3)
        with a1:
            render_kpi_card(
                "Anomalies Detected",
                str(anom_res.anomaly_count),
                delta_type="negative" if anom_res.anomaly_count > 0 else "neutral",
                subtitle="Flagged outlier observations",
            )
        with a2:
            render_kpi_card(
                "Anomaly Rate",
                f"{anom_res.anomaly_percentage:.1f}%",
                delta_type="neutral",
                subtitle="Proportion of total records",
            )
        with a3:
            render_kpi_card(
                "Target Feature",
                str(anom_col),
                delta_type="neutral",
                subtitle=f"Method: {method}",
            )

        st.markdown(
            f'<div class="warn-box" style="margin-top:1rem;margin-bottom:1rem;">'
            f'⚠️ <strong>Statistical Boundary Disclaimer:</strong> {anom_res.non_causal_disclaimer}</div>',
            unsafe_allow_html=True)

        if anom_res.anomalies:
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            ch1, ch2 = st.columns([1, 2], gap="large")
            with ch1:
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=df[anom_col], name=anom_col,
                    marker_color=BRAND_PRIMARY, line_color=BRAND_PRIMARY, fillcolor=BRAND_PRIMARY_LIGHT,
                    boxmean=True,
                ))
                layout = get_plotly_layout_template()
                layout["title"] = dict(
                    text=f"Distribution & Outliers \u2014 {anom_col}",
                    font=dict(size=12, color="#0F172A", family="Inter"),
                    x=0,
                )
                layout["height"] = 300
                layout["showlegend"] = False
                fig.update_layout(layout)
                st.plotly_chart(fig, use_container_width=True)
            with ch2:
                st.markdown(
                    f'<div class="section-label">Top Flagged Records ({len(anom_res.anomalies[:50])} shown)</div>',
                    unsafe_allow_html=True)
                st.dataframe(pd.DataFrame([a.data for a in anom_res.anomalies[:50]]),
                             use_container_width=True, hide_index=True)
    else:
        render_empty_state(
            title="No Anomaly Scan Executed",
            desc="Select a numeric feature and detection algorithm above, then click 'Detect Anomalies' to identify statistically divergent observations with calibrated non-causal boundaries.",
            pills=["🌲 Isolation Forest (3% contamination)", "📐 IQR Bounds (1.5× range)", "📊 Distribution Outliers"],
            icon_type="anomaly",
        )


# ══════════════════════════════════════════════════════════════
# TAB 5 — Evidence & Provenance
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-label">Cryptographic Provenance & Audit Trail</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Evidence & Governance Lineage</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Full immutable evidence provenance chain \u2014 tracing analytical claims '
        'from raw dataset ingestion through allowlisted deterministic tool runs to minted evidence items and verified answers.</div>',
        unsafe_allow_html=True)

    st.markdown("""
    <div class="prov-container">
        <div class="section-label" style="margin-bottom:0.85rem;">Cryptographic Provenance Chain</div>
        <div class="prov-step">
            <div class="prov-connector"><div class="prov-dot"></div><div class="prov-line"></div></div>
            <div><div class="prov-label">1. Dataset Ingestion</div><div class="prov-value">Source data loaded, schema validated, version hashed, and session isolated</div></div>
        </div>
        <div class="prov-step">
            <div class="prov-connector"><div class="prov-dot"></div><div class="prov-line"></div></div>
            <div><div class="prov-label">2. Allowlisted Tool Execution</div><div class="prov-value">Deterministic analytical tools execute on verified dataframe slices</div></div>
        </div>
        <div class="prov-step">
            <div class="prov-connector"><div class="prov-dot"></div><div class="prov-line"></div></div>
            <div><div class="prov-label">3. Immutable Evidence Minting</div><div class="prov-value">Cryptographic evidence items minted with confidence ratings and source lineage</div></div>
        </div>
        <div class="prov-step">
            <div class="prov-connector"><div class="prov-dot"></div><div class="prov-line"></div></div>
            <div><div class="prov-label">4. Structured Claim Validation</div><div class="prov-value">Extracted statements programmatically verified against minted evidence numbers</div></div>
        </div>
        <div class="prov-step">
            <div class="prov-connector"><div class="prov-dot"></div></div>
            <div><div class="prov-label">5. Calibrated Grounded Answer</div><div class="prov-value">Final output grounded strictly in supported evidence with zero hallucination</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    past_invs = repo.list_investigations(
        dataset_id=metadata.dataset_id, session_id=current_session_id
    )
    if past_invs:
        st.markdown('<div class="section-label">Investigation Audit History</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(past_invs), use_container_width=True, hide_index=True)

    if st.session_state.get("last_inv") is not None:
        store = st.session_state.last_inv.get("evidence_store")
        ev_list = store.list_evidence() if store is not None else []
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-label">Active Investigation Evidence Store ({len(ev_list)} items)</div>',
                    unsafe_allow_html=True)
        if ev_list:
            ev_rows = [
                {
                    "Evidence ID": e.evidence_id,
                    "Type": e.evidence_type.value.replace("_", " ").title(),
                    "Source": e.source_id,
                    "Location": e.location or "\u2014",
                    "Confidence": f"{e.confidence * 100:.0f}%",
                    "Content Preview": str(e.content)[:120] + "\u2026",
                }
                for e in ev_list
            ]
            st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)
    else:
        render_empty_state(
            title="No Active Investigation Evidence",
            desc="Run an autonomous question or reopen a saved investigation in the Investigation tab to inspect minted evidence items, tool provenance, and verified claims.",
            pills=["🔗 Immutable Evidence IDs", "🛡️ Allowlisted Tool Provenance", "✅ Verified Claims Only"],
            icon_type="evidence",
        )


# ══════════════════════════════════════════════════════════════
# TAB 6 — Evaluation
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-label">System Benchmarking & Empirical Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Architecture Evaluation Harness</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Empirical benchmark measuring 13 performance dimensions across OmniSleuth and candidate baselines '
        '(Baseline 1: Direct Tools, Baseline 2: Fixed Pipeline). All metrics are measured from actual execution across 40 benchmark questions.</div>',
        unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("Run Full Comparative Benchmark (All 3 Systems)", type="primary"):
            with st.spinner("Executing full Phase 18 benchmark across 40 questions & baselines…"):
                exp_rec = evaluator.run_full_benchmark(df, metadata, questions=BENCHMARK_QUESTIONS)
                st.session_state.exp_rec = exp_rec
                st.session_state.bench_res = exp_rec.systems["omnisleuth"]

    with col_btn2:
        if st.button("Run OmniSleuth Evaluation Only"):
            with st.spinner("Evaluating OmniSleuth architecture…"):
                bench_res = evaluator.evaluate_omnisleuth(df, metadata, questions=BENCHMARK_QUESTIONS)
                st.session_state.bench_res = bench_res

    if "bench_res" in st.session_state:
        bench_res = st.session_state.bench_res

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Primary Performance Scorecard</div>', unsafe_allow_html=True)

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            render_kpi_card(
                "Numerical Correctness",
                f"{bench_res.numerical_correctness_pct:.1f}%",
                delta_type="positive" if bench_res.numerical_correctness_pct >= 90 else "negative",
                subtitle="Exact mathematical matches",
            )
        with b2:
            render_kpi_card(
                "Tool Selection",
                f"{bench_res.tool_selection_accuracy_pct:.1f}%",
                delta_type="positive" if bench_res.tool_selection_accuracy_pct >= 90 else "negative",
                subtitle="Optimal tool chosen",
            )
        with b3:
            render_kpi_card(
                "Evidence Coverage",
                f"{bench_res.evidence_coverage_pct:.1f}%",
                delta_type="positive" if bench_res.evidence_coverage_pct >= 80 else "neutral",
                subtitle="Grounded claim ratio",
            )
        with b4:
            render_kpi_card(
                "Unsupported Claims",
                f"{bench_res.unsupported_claim_rate_pct:.1f}%",
                delta_type="positive" if bench_res.unsupported_claim_rate_pct <= 10 else "negative",
                subtitle="Hallucination rate (lower is better)",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Secondary Operational Metrics</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_kpi_card(
                "Completeness",
                f"{bench_res.investigation_completeness_pct:.1f}%",
                delta_type="neutral",
                subtitle="Multi-step plan coverage",
            )
        with m2:
            render_kpi_card(
                "Anomaly F1",
                f"{bench_res.anomaly_f1:.2f}",
                delta_type="neutral",
                subtitle="Harmonic precision / recall",
            )
        with m3:
            render_kpi_card(
                "RAG Recall@5",
                f"{bench_res.retrieval_recall_at_5:.2f}",
                delta_type="neutral",
                subtitle="Relevant document retrieval",
            )
        with m4:
            render_kpi_card(
                "Latency p50 / p95",
                f"{bench_res.latency_p50_ms:.0f} / {bench_res.latency_p95_ms:.0f} ms",
                delta_type="neutral",
                subtitle="Execution response times",
            )

        # Comparison table
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Empirical Comparison Across Architectures</div>', unsafe_allow_html=True)

        if "exp_rec" in st.session_state:
            exp_rec = st.session_state.exp_rec
            rows = []
            for sys_key, sys_res in exp_rec.systems.items():
                rows.append({
                    "Architecture": sys_res.system_name,
                    "Numerical Correctness": f"{sys_res.numerical_correctness_pct:.1f}%",
                    "Tool Accuracy": f"{sys_res.tool_selection_accuracy_pct:.1f}%",
                    "Evidence Coverage": f"{sys_res.evidence_coverage_pct:.1f}%",
                    "Unsupported Claims": f"{sys_res.unsupported_claim_rate_pct:.1f}%",
                    "Completeness": f"{sys_res.investigation_completeness_pct:.1f}%",
                    "Faithfulness": f"{sys_res.faithfulness_pct:.1f}%",
                    "Latency (p50)": f"{sys_res.latency_p50_ms:.0f} ms",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            comp_table = pd.DataFrame([
                {
                    "Architecture": "OmniSleuth (Agent + Tools + RAG + Validator)",
                    "Numerical Correctness": f"{bench_res.numerical_correctness_pct:.1f}%",
                    "Tool Selection": f"{bench_res.tool_selection_accuracy_pct:.1f}%",
                    "Evidence Coverage": f"{bench_res.evidence_coverage_pct:.1f}%",
                    "Unsupported Claims": f"{bench_res.unsupported_claim_rate_pct:.1f}%",
                    "Completeness": f"{bench_res.investigation_completeness_pct:.1f}%",
                    "Latency (p50)": f"{bench_res.latency_p50_ms:.0f} ms",
                }
            ])
            st.dataframe(comp_table, use_container_width=True, hide_index=True)
            st.info("Run 'Full Comparative Benchmark' to populate real empirical metrics for Baseline 1 and Baseline 2.")
    else:
        render_empty_state(
            title="Benchmark Not Yet Executed",
            desc="Choose an evaluation mode above to benchmark OmniSleuth against fixed baselines across 40 questions measuring numerical correctness, tool selection, evidence coverage, and latency.",
            pills=["🎯 40 Benchmark Questions", "⚖️ 3 Architectural Baselines", "⏱️ Latency & Faithfulness", "📊 Zero Fabricated Scores"],
            icon_type="benchmark",
        )
