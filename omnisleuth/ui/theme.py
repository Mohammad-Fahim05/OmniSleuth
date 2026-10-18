"""OmniSleuth Design System — Theme tokens, CSS definitions, and Plotly templates."""

from typing import Any, Dict


# Core palette tokens
BRAND_PRIMARY = "#4F46E5"        # Indigo 600
BRAND_PRIMARY_HOVER = "#4338CA"  # Indigo 700
BRAND_PRIMARY_LIGHT = "#EEF2FF"  # Indigo 50
BRAND_SECONDARY = "#06B6D4"      # Cyan 500
BRAND_ACCENT = "#8B5CF6"         # Violet 500

BG_MAIN = "#F8FAFC"              # Slate 50
BG_SURFACE = "#FFFFFF"           # Pure White
BG_SIDEBAR = "#0F172A"           # Slate 900
BG_SIDEBAR_SURFACE = "#1E293B"   # Slate 800

TEXT_PRIMARY = "#0F172A"         # Slate 900
TEXT_SECONDARY = "#64748B"       # Slate 500
TEXT_MUTED = "#94A3B8"           # Slate 400
TEXT_SIDEBAR = "#E2E8F0"         # Slate 200
TEXT_SIDEBAR_MUTED = "#94A3B8"   # Slate 400

BORDER_SUBTLE = "#E2E8F0"        # Slate 200
BORDER_FOCUS = "#818CF8"         # Indigo 400
BORDER_SIDEBAR = "#334155"       # Slate 700

COLOR_SUCCESS = "#16A34A"        # Green 600
COLOR_SUCCESS_BG = "#F0FDF4"     # Green 50
COLOR_SUCCESS_BORDER = "#BBF7D0" # Green 200

COLOR_WARNING = "#D97706"        # Amber 600
COLOR_WARNING_BG = "#FFFBEB"     # Amber 50
COLOR_WARNING_BORDER = "#FDE68A" # Amber 200

COLOR_ERROR = "#DC2626"          # Red 600
COLOR_ERROR_BG = "#FEF2F2"       # Red 50
COLOR_ERROR_BORDER = "#FECACA"   # Red 200

COLOR_INFO = "#0284C7"           # Sky 600
COLOR_INFO_BG = "#F0F9FF"        # Sky 50
COLOR_INFO_BORDER = "#BAE6FD"    # Sky 200

PLOTLY_COLORWAY = [
    "#4F46E5",  # Brand Indigo
    "#06B6D4",  # Cyan
    "#8B5CF6",  # Violet
    "#10B981",  # Emerald
    "#F59E0B",  # Amber
    "#EC4899",  # Pink
    "#3B82F6",  # Blue
]


THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {{
    --brand-primary: {BRAND_PRIMARY};
    --brand-primary-hover: {BRAND_PRIMARY_HOVER};
    --brand-primary-light: {BRAND_PRIMARY_LIGHT};
    --brand-secondary: {BRAND_SECONDARY};
    --brand-accent: {BRAND_ACCENT};
    --bg-main: {BG_MAIN};
    --bg-surface: {BG_SURFACE};
    --bg-sidebar: {BG_SIDEBAR};
    --bg-sidebar-surface: {BG_SIDEBAR_SURFACE};
    --text-primary: {TEXT_PRIMARY};
    --text-secondary: {TEXT_SECONDARY};
    --text-muted: {TEXT_MUTED};
    --text-sidebar: {TEXT_SIDEBAR};
    --text-sidebar-muted: {TEXT_SIDEBAR_MUTED};
    --border-subtle: {BORDER_SUBTLE};
    --border-focus: {BORDER_FOCUS};
    --border-sidebar: {BORDER_SIDEBAR};
    --color-success: {COLOR_SUCCESS};
    --color-warning: {COLOR_WARNING};
    --color-error: {COLOR_ERROR};
    --color-info: {COLOR_INFO};
    --shadow-sm: 0 1px 2px 0 rgba(15, 23, 42, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(15, 23, 42, 0.07), 0 2px 4px -2px rgba(15, 23, 42, 0.05);
    --radius-sm: 6px;
    --radius-md: 8px;
    --radius-lg: 12px;
}}

/* Typography defaults */
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: var(--text-primary);
}}

/* Clean app chrome */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ visibility: hidden; }}

.stApp {{
    background-color: var(--bg-main);
}}

/* ── SIDEBAR STYLING ── */
[data-testid="stSidebar"] {{
    background-color: var(--bg-sidebar);
    border-right: 1px solid var(--border-sidebar);
}}

[data-testid="stSidebar"] * {{
    color: var(--text-sidebar);
}}

[data-testid="stSidebar"] hr {{
    border-color: var(--border-sidebar);
    margin: 1rem 0;
}}

/* Sidebar section titles */
.sidebar-section {{
    font-size: 0.68rem;
    font-weight: 700;
    color: var(--text-sidebar-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.5rem 0 0.35rem 0;
}}

.sidebar-value {{
    font-size: 0.82rem;
    color: var(--text-sidebar) !important;
    font-weight: 500;
}}

/* Sidebar button contrast fix */
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {{
    background-color: #1E293B !important;
    color: #F8FAFC !important;
    border: 1px solid #475569 !important;
    border-radius: var(--radius-sm);
    font-size: 0.83rem;
    font-weight: 600;
    width: 100%;
    padding: 0.45rem 0.75rem;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    transition: all 0.15s ease-in-out;
}}

[data-testid="stSidebar"] button p,
[data-testid="stSidebar"] button span,
[data-testid="stSidebar"] button div,
[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stButton > button span,
[data-testid="stSidebar"] .stButton > button div {{
    color: #F8FAFC !important;
}}

[data-testid="stSidebar"] button:hover,
[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {{
    background-color: var(--brand-primary) !important;
    border-color: var(--brand-primary) !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.4);
}}

[data-testid="stSidebar"] button:hover p,
[data-testid="stSidebar"] button:hover span,
[data-testid="stSidebar"] button:hover div,
[data-testid="stSidebar"] .stButton > button:hover p,
[data-testid="stSidebar"] .stButton > button:hover span,
[data-testid="stSidebar"] .stButton > button:hover div {{
    color: #FFFFFF !important;
}}

[data-testid="stSidebar"] .stButton > button:active {{
    transform: translateY(1px);
}}

/* Primary button inside sidebar */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
    background-color: var(--brand-primary) !important;
    color: #FFFFFF !important;
    border: none !important;
}}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
    background-color: var(--brand-primary-hover) !important;
}}

/* Sidebar Expanders */
[data-testid="stSidebar"] [data-testid="stExpander"] {{
    background-color: var(--bg-sidebar-surface);
    border: 1px solid var(--border-sidebar) !important;
    border-radius: var(--radius-md);
    margin-bottom: 0.65rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
}}

[data-testid="stSidebar"] [data-testid="stExpander"] details summary {{
    color: #F1F5F9 !important;
    font-weight: 600;
    font-size: 0.82rem;
    padding: 0.5rem 0.75rem;
}}

[data-testid="stSidebar"] [data-testid="stExpander"] details summary svg {{
    fill: var(--text-sidebar-muted) !important;
}}

[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
    background-color: var(--bg-sidebar-surface);
    border-top: 1px solid var(--border-sidebar);
    padding: 0.75rem;
}}

[data-testid="stSidebar"] [data-testid="stExpander"] .stSelectbox label,
[data-testid="stSidebar"] [data-testid="stExpander"] .stTextInput label {{
    color: var(--text-sidebar-muted) !important;
    font-size: 0.75rem;
}}

/* Sidebar radio group accent */
[data-testid="stSidebar"] div[role="radiogroup"] > label {{
    padding: 0.35rem 0.5rem;
    border-radius: var(--radius-sm);
    transition: background 0.12s ease;
}}

[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
    background: rgba(255, 255, 255, 0.04);
}}

[data-testid="stSidebar"] div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p {{
    color: var(--text-sidebar) !important;
    font-size: 0.85rem;
    font-weight: 500;
}}

[data-testid="stSidebar"] div[role="radiogroup"] [aria-checked="true"] {{
    border-color: var(--brand-primary) !important;
}}

/* ── APP HEADER & BRAND ── */
.app-header-container {{
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: var(--shadow-sm);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
}}

.brand-wrapper {{
    display: flex;
    align-items: center;
    gap: 0.85rem;
}}

.brand-icon-box {{
    width: 42px;
    height: 42px;
    border-radius: 10px;
    background: linear-gradient(135deg, var(--brand-primary) 0%, var(--brand-secondary) 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 10px rgba(79, 70, 229, 0.25);
    color: #FFFFFF;
    font-weight: 800;
    font-size: 1.25rem;
    letter-spacing: -0.03em;
    flex-shrink: 0;
}}

.app-brand-title {{
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    line-height: 1.2;
    margin: 0;
}}

.app-brand-subtitle {{
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin-top: 0.15rem;
    font-weight: 400;
}}

.header-meta-group {{
    display: flex;
    gap: 0.5rem;
    align-items: center;
    flex-wrap: wrap;
}}

.dataset-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: var(--brand-primary-light);
    border: 1px solid #C7D2FE;
    border-radius: var(--radius-sm);
    padding: 0.35rem 0.75rem;
    font-size: 0.8rem;
    color: var(--brand-primary);
    font-weight: 600;
}}

.meta-pill {{
    display: inline-flex;
    align-items: center;
    background: #F1F5F9;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 0.35rem 0.65rem;
    font-size: 0.78rem;
    color: var(--text-secondary);
    font-weight: 500;
    font-feature-settings: 'tnum';
    font-variant-numeric: tabular-nums;
}}

.status-dot-active {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10B981;
    display: inline-block;
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
}}

/* ── TABS NAVIGATION ── */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent;
    border-bottom: 1px solid var(--border-subtle);
    gap: 0.2rem;
    padding: 0.2rem 0;
    overflow-x: auto;
    flex-wrap: nowrap !important;
    scrollbar-width: thin;
    scrollbar-color: #CBD5E1 transparent;
    -webkit-overflow-scrolling: touch;
}}

.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {{
    height: 3px;
}}

.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {{
    background: #CBD5E1;
    border-radius: 4px;
}}

.stTabs [data-baseweb="tab"] {{
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-secondary);
    padding: 0.5rem 0.8rem;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0;
    border-bottom: 2px solid transparent;
    white-space: nowrap !important;
    flex-shrink: 0;
    transition: all 0.15s ease;
}}

.stTabs [data-baseweb="tab"]:hover {{
    color: var(--text-primary);
    background: rgba(255, 255, 255, 0.6);
}}

.stTabs [aria-selected="true"] {{
    color: var(--brand-primary) !important;
    border-bottom: 2px solid var(--brand-primary) !important;
    background: #FFFFFF !important;
}}

/* ── KPI CARDS (Tabular Numerals) ── */
.kpi-card {{
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1rem 1.15rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.15s ease, border-color 0.15s ease;
}}

.kpi-card:hover {{
    box-shadow: var(--shadow-md);
    border-color: #CBD5E1;
}}

.kpi-label {{
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.35rem;
}}

.kpi-value {{
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.15;
    font-feature-settings: 'tnum', 'cv02', 'cv03', 'cv04';
    font-variant-numeric: tabular-nums;
}}

.kpi-value-positive {{
    color: var(--color-success);
}}

.kpi-value-negative {{
    color: var(--color-error);
}}

.kpi-value-neutral {{
    color: var(--text-primary);
}}

/* ── SECTION LABELS & HEADINGS ── */
.section-label {{
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--brand-primary);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.35rem;
}}

.section-title {{
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    margin-bottom: 0.3rem;
}}

.section-desc {{
    font-size: 0.85rem;
    color: var(--text-secondary);
    line-height: 1.55;
    margin-bottom: 1.25rem;
}}

.section-divider {{
    border: none;
    border-top: 1px solid var(--border-subtle);
    margin: 1.5rem 0;
}}

/* ── SUMMARY HERO BOX ── */
.summary-block {{
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-left: 4px solid var(--brand-primary);
    border-radius: var(--radius-sm);
    padding: 1rem 1.35rem;
    font-size: 0.92rem;
    color: var(--text-primary);
    line-height: 1.65;
    box-shadow: var(--shadow-sm);
}}

/* ── STATUS TAGS ── */
.status-tag {{
    font-size: 0.73rem;
    font-weight: 600;
    padding: 0.22rem 0.65rem;
    border-radius: 999px;
    white-space: nowrap;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}}

.status-supported {{
    background: var(--color-success-bg);
    color: var(--color-success);
    border: 1px solid var(--color-success-border);
}}

.status-rejected {{
    background: var(--color-error-bg);
    color: var(--color-error);
    border: 1px solid var(--color-error-border);
}}

.status-inconclusive {{
    background: var(--color-warning-bg);
    color: var(--color-warning);
    border: 1px solid var(--color-warning-border);
}}

.status-pending {{
    background: #F8FAFC;
    color: var(--text-secondary);
    border: 1px solid var(--border-subtle);
}}

/* ── CALLOUT BOXES ── */
.info-box {{
    background: var(--color-info-bg);
    border: 1px solid var(--color-info-border);
    border-radius: var(--radius-sm);
    padding: 0.75rem 1rem;
    font-size: 0.84rem;
    color: #0369A1;
    line-height: 1.5;
}}

.warn-box {{
    background: var(--color-warning-bg);
    border: 1px solid var(--color-warning-border);
    border-radius: var(--radius-sm);
    padding: 0.75rem 1rem;
    font-size: 0.84rem;
    color: #92400E;
    line-height: 1.5;
}}

/* ── BUTTONS ── */
.stButton > button[kind="primary"] {{
    background: var(--brand-primary) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: var(--radius-sm);
    font-weight: 600;
    font-size: 0.86rem;
    padding: 0.5rem 1.1rem;
    box-shadow: 0 1px 2px rgba(79, 70, 229, 0.2);
    transition: all 0.15s ease;
}}

.stButton > button[kind="primary"]:hover {{
    background: var(--brand-primary-hover) !important;
    box-shadow: 0 4px 10px rgba(79, 70, 229, 0.35);
}}

/* Main content buttons */
.main .stButton > button:not([kind="primary"]),
[data-testid="stMainBlockContainer"] .stButton > button:not([kind="primary"]) {{
    background: #FFFFFF !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm);
    font-size: 0.84rem;
    font-weight: 500;
    padding: 0.45rem 0.9rem;
    transition: all 0.15s ease;
}}

.main .stButton > button:not([kind="primary"]):hover,
[data-testid="stMainBlockContainer"] .stButton > button:not([kind="primary"]):hover {{
    background: #F8FAFC !important;
    border-color: #CBD5E1 !important;
}}

/* ── FORM INPUTS FOCUS ── */
.stTextInput > div > div > input:focus,
.stSelectbox > div > div:focus-within {{
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 2px rgba(129, 140, 248, 0.2) !important;
}}

/* ── INVESTIGATION WORKSPACE & RESULTS ── */
.investigation-meta-bar {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    align-items: center;
    margin: 0.75rem 0 1rem 0;
}}

.meta-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: #F1F5F9;
    color: #334155;
    border: 1px solid var(--border-subtle);
    border-radius: 999px;
    padding: 0.2rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 600;
}}

.planner-indicator-box {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 0.85rem;
    background: #FFFFFF;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-bottom: 0.85rem;
}}

/* Empty state guidance */
.empty-state-card {{
    background: var(--bg-surface);
    border: 1px dashed #CBD5E1;
    border-radius: var(--radius-lg);
    padding: 2.75rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
}}

.empty-state-icon {{
    width: 48px;
    height: 48px;
    border-radius: 12px;
    background: var(--brand-primary-light);
    color: var(--brand-primary);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1.35rem;
    margin-bottom: 0.85rem;
}}

.empty-state-title {{
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.35rem;
}}

.empty-state-desc {{
    font-size: 0.85rem;
    color: var(--text-secondary);
    max-width: 560px;
    margin: 0 auto 1.35rem auto;
    line-height: 1.55;
}}

.capability-pills {{
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0.5rem;
}}

.cap-pill {{
    background: #F8FAFC;
    border: 1px solid var(--border-subtle);
    border-radius: 999px;
    padding: 0.3rem 0.75rem;
    font-size: 0.75rem;
    color: #475569;
    font-weight: 500;
}}

/* Findings & Hypotheses cards */
.hyp-card {{
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.65rem;
    transition: box-shadow 0.15s ease, border-color 0.15s ease;
}}

.hyp-card:hover {{
    box-shadow: var(--shadow-sm);
    border-color: #CBD5E1;
}}

.hyp-header {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.75rem;
}}

.hyp-statement {{
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1.45;
    flex: 1;
}}

.hyp-rationale {{
    font-size: 0.8rem;
    color: var(--text-secondary);
    line-height: 1.5;
    margin-top: 0.35rem;
}}

.hyp-footer {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-size: 0.74rem;
    color: var(--text-muted);
    margin-top: 0.45rem;
}}

.hyp-row {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    padding: 0.75rem 0;
    border-bottom: 1px solid #F1F5F9;
    gap: 1rem;
}}
.hyp-row:last-child {{ border-bottom: none; }}

/* Evidence cards */
.evidence-card, .evidence-item {{
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 0.85rem 1rem;
    margin-bottom: 0.65rem;
}}

.evidence-title {{
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.25rem;
}}

.evidence-meta {{
    display: flex;
    gap: 0.65rem;
    flex-wrap: wrap;
    font-size: 0.74rem;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}}

.evidence-content {{
    font-size: 0.82rem;
    color: #334155;
    background: #F8FAFC;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 0.5rem 0.75rem;
    line-height: 1.5;
    font-feature-settings: 'tnum';
    font-variant-numeric: tabular-nums;
}}

/* Claim cards */
.claim-card, .claim-row {{
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
}}

.claim-text, .claim-statement {{
    font-size: 0.88rem;
    font-weight: 500;
    color: var(--text-primary);
    line-height: 1.5;
    margin-bottom: 0.35rem;
}}

.claim-explanation {{
    font-size: 0.78rem;
    color: var(--text-secondary);
    line-height: 1.45;
}}

/* Agent steps & timeline */
.step-row {{
    border-left: 3px solid var(--brand-primary);
    padding: 0.5rem 0.85rem;
    margin-bottom: 0.55rem;
    background: var(--bg-surface);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    border-top: 1px solid var(--border-subtle);
    border-right: 1px solid var(--border-subtle);
    border-bottom: 1px solid var(--border-subtle);
}}

.step-agent {{
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--brand-primary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

.step-tool {{
    font-size: 0.84rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-top: 0.1rem;
}}

.step-rationale {{
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin-top: 0.2rem;
    line-height: 1.4;
}}

/* ── PROVENANCE CHAIN ── */
.prov-container {{
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: var(--shadow-sm);
}}

.prov-step {{
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
    padding: 0.5rem 0;
}}

.prov-connector {{
    display: flex;
    flex-direction: column;
    align-items: center;
}}

.prov-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--brand-primary);
    flex-shrink: 0;
    margin-top: 4px;
    box-shadow: 0 0 0 3px var(--brand-primary-light);
}}

.prov-line {{
    width: 2px;
    height: 32px;
    background: #E2E8F0;
    margin-top: 2px;
}}

.prov-label {{
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--brand-primary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

.prov-value {{
    font-size: 0.84rem;
    color: var(--text-primary);
    margin-top: 0.15rem;
}}

/* ── DISCOVERY CARDS & SIGNIFICANCE ── */
.discovery-card {{
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    box-shadow: var(--shadow-sm);
}}

.discovery-category {{
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--brand-primary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.25rem;
}}

.discovery-desc {{
    font-size: 0.86rem;
    color: var(--text-primary);
    line-height: 1.55;
    margin-bottom: 0.65rem;
}}

.sig-high {{ color: var(--color-error); font-weight: 600; }}
.sig-medium {{ color: var(--color-warning); font-weight: 600; }}
.sig-low {{ color: var(--text-secondary); font-weight: 600; }}

/* ── RESPONSIVE & MOBILE ADAPTATIONS ── */
@media (max-width: 1280px) {{
    .stTabs [data-baseweb="tab"] {{
        font-size: 0.79rem;
        padding: 0.45rem 0.65rem;
    }}
    .app-brand-title {{
        font-size: 1.3rem;
    }}
    .kpi-value {{
        font-size: 1.45rem;
    }}
}}

@media (max-width: 1024px) {{
    .stTabs [data-baseweb="tab"] {{
        font-size: 0.75rem;
        padding: 0.4rem 0.5rem;
    }}
    .app-header-container {{
        padding: 1rem 1.15rem;
    }}
    .app-brand-title {{
        font-size: 1.2rem;
    }}
    .app-brand-subtitle {{
        font-size: 0.78rem;
    }}
    .kpi-card {{
        padding: 0.85rem 1rem;
    }}
    .kpi-value {{
        font-size: 1.35rem;
    }}
    .prov-container {{
        gap: 0.5rem;
        padding: 0.75rem 0.5rem;
        overflow-x: auto;
    }}
    .prov-label {{
        font-size: 0.62rem;
    }}
}}

@media (max-width: 768px) {{
    .stTabs [data-baseweb="tab"] {{
        font-size: 0.72rem;
        padding: 0.35rem 0.45rem;
    }}
    .app-header-container {{
        flex-direction: column;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.85rem 1rem;
    }}
    .header-meta-group {{
        width: 100%;
    }}
    .kpi-card {{
        padding: 0.75rem;
    }}
    .kpi-label {{
        font-size: 0.68rem;
    }}
    .kpi-value {{
        font-size: 1.25rem;
    }}
    .prov-container {{
        flex-wrap: wrap;
        gap: 0.75rem;
    }}
    .prov-connector {{
        display: none;
    }}
    .empty-state-card {{
        padding: 2rem 1.25rem;
    }}
    .empty-state-title {{
        font-size: 1.15rem;
    }}
    .empty-state-desc {{
        font-size: 0.82rem;
    }}
}}
</style>
"""


def get_plotly_layout_template() -> Dict[str, Any]:
    """Return a unified, accessible Plotly layout dictionary matching the design system."""
    return dict(
        autosize=True,
        paper_bgcolor=BG_SURFACE,
        plot_bgcolor=BG_SURFACE,
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            size=11,
            color=TEXT_PRIMARY,
        ),
        colorway=PLOTLY_COLORWAY,
        margin=dict(l=24, r=24, t=36, b=24),
        xaxis=dict(
            showgrid=False,
            linecolor=BORDER_SUBTLE,
            tickfont=dict(size=10, color=TEXT_SECONDARY),
            title_font=dict(size=11, color=TEXT_SECONDARY),
        ),
        yaxis=dict(
            gridcolor="#F1F5F9",
            linecolor=BORDER_SUBTLE,
            tickfont=dict(size=10, color=TEXT_SECONDARY),
            title_font=dict(size=11, color=TEXT_SECONDARY),
        ),
        legend=dict(
            font=dict(size=10, color=TEXT_SECONDARY),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=BORDER_SUBTLE,
            borderwidth=1,
        ),
    )
