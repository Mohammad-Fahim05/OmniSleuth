"""OmniSleuth Design System — Reusable UI components and presentation primitives."""

from typing import Any, Optional
import streamlit as st


def render_app_header(
    metadata: Optional[Any] = None,
    title: str = "OmniSleuth",
    subtitle: str = "Evidence-grounded multi-agent data investigation",
) -> None:
    """Render the application header with brand mark and active dataset metadata pills."""
    if metadata is not None:
        name = getattr(metadata, "name", "Dataset")
        row_count = getattr(metadata, "row_count", 0)
        col_count = getattr(metadata, "column_count", 0)

        st.markdown(
            f"""
            <div class="app-header-container">
                <div class="brand-wrapper">
                    <div class="brand-icon-box">OS</div>
                    <div>
                        <div class="app-brand-title">{title}</div>
                        <div class="app-brand-subtitle">{subtitle}</div>
                    </div>
                </div>
                <div class="header-meta-group">
                    <span class="dataset-pill">
                        <span class="status-dot-active"></span>
                        {name}
                    </span>
                    <span class="meta-pill" style="color:#166534;background:#F0FDF4;border-color:#BBF7D0;">
                        {row_count:,} rows
                    </span>
                    <span class="meta-pill">
                        {col_count} columns
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="app-header-container">
                <div class="brand-wrapper">
                    <div class="brand-icon-box">OS</div>
                    <div>
                        <div class="app-brand-title">{title}</div>
                        <div class="app-brand-subtitle">{subtitle}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_empty_state(
            title="No Dataset Loaded",
            desc="Load the sample dataset or upload a CSV, Parquet, or XLSX file in the sidebar to begin an evidence-grounded investigation.",
            pills=[
                "📊 Autonomous Multi-Agent Analysis",
                "🛡️ Schema-Grounding & Validation",
                "🔍 Automated Anomaly & Discovery Engine",
                "🔒 Zero Synthetic Hallucinations",
            ],
            icon_type="scan",
        )


def render_kpi_card(
    label: str,
    value: str,
    delta_type: str = "neutral",
    subtitle: Optional[str] = None,
) -> None:
    """Render a KPI card with tabular numerals and semantic value coloring."""
    val_class = {
        "positive": "kpi-value-positive",
        "negative": "kpi-value-negative",
        "neutral": "kpi-value-neutral",
    }.get(delta_type, "kpi-value-neutral")

    sub_html = f'<div style="font-size:0.75rem;color:#64748B;margin-top:0.25rem;">{subtitle}</div>' if subtitle else ""

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value {val_class}">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(text: str, status_type: str = "info") -> str:
    """Generate HTML for an accessible status badge."""
    css_class = {
        "supported": "status-supported",
        "verified": "status-supported",
        "answered": "status-supported",
        "completed": "status-supported",
        "rejected": "status-rejected",
        "unsupported": "status-rejected",
        "failed": "status-rejected",
        "warning": "status-inconclusive",
        "inconclusive": "status-inconclusive",
        "needs_clarification": "status-inconclusive",
        "pending": "status-pending",
    }.get(status_type.lower(), "status-pending")

    return f'<span class="status-tag {css_class}">{text}</span>'


def render_meta_badge(label: str, value: str) -> str:
    """Generate HTML for a metadata badge pill."""
    return f'<span class="meta-badge">{label}: <strong style="color:#0F172A;">{value}</strong></span>'


def render_empty_state(
    title: str,
    desc: str,
    pills: Optional[Any] = None,
    icon_type: str = "search",
) -> None:
    """Render a unified, accessible empty-state card matching the design system."""
    icon_map = {
        "search": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>',
        "scan": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7V5a2 2 0 0 1 2-2h2"></path><path d="M17 3h2a2 2 0 0 1 2 2v2"></path><path d="M21 17v2a2 2 0 0 1-2 2h-2"></path><path d="M7 21H5a2 2 0 0 1-2-2v-2"></path><line x1="7" y1="12" x2="17" y2="12"></line></svg>',
        "anomaly": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>',
        "benchmark": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>',
        "evidence": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>',
    }
    icon_svg = icon_map.get(icon_type, icon_map["search"])
    pills_html = ""
    if pills:
        pills_str = "".join([f'<span class="cap-pill">{p}</span>' for p in pills])
        pills_html = f'<div class="capability-pills">{pills_str}</div>'

    st.markdown(
        f"""
        <div class="empty-state-card">
            <div class="empty-state-icon">
                {icon_svg}
            </div>
            <div class="empty-state-title">{title}</div>
            <div class="empty-state-desc">{desc}</div>
            {pills_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_investigation_empty_state() -> None:
    """Render helpful empty-state guidance when no investigation has been run."""
    render_empty_state(
        title="Ready for Investigation",
        desc="Submit an analytical question or choose a suggested example above to begin. "
             "OmniSleuth will generate a schema-grounded execution plan, test diagnostic hypotheses, "
             "and mathematically ground all claims against verified evidence without hallucination.",
        pills=[
            "🛡️ Deterministic Execution",
            "🔬 Multi-Agent Hypothesis Testing",
            "📊 Grounded Visualizations",
            "🔒 Zero LLM Hallucination",
        ],
        icon_type="search",
    )
