"""
CHEM 120 Catalyst Insight Studio
=================================

Streamlit UI entry point. All data loading, cleaning, validation, descriptors,
plotting, and ML logic lives in pipeline.py. This file only handles page layout,
widgets, and calls into pipeline.

Layout
------
The screen is organized as TABS (not a long scroll). `main()` loads the data and
then hands each tab to a small `render_*()` function:

    Check    -> render_check_tab()        data-entry issues, outliers, clean table
    Explore  -> render_explore_tab()      response counts, per-element + mixing tables
    Heatmap  -> render_heatmap_tab()      interactive correlation map + glossary
    Structure-> render_pca_tab()          PCA map with optional KMeans clusters
    Predict  -> render_ml_tab()           bubble + purity models and predictions
    Add      -> render_add_compound_tab() stage a new compound by hand
    Export   -> render_export_tab()       download cleaned data and reports
    Semester -> render_new_semester_tab() merge a new semester's raw export

Design notes (June 2026 revamp)
-------------------------------
* Dark "lab glass" theme: .streamlit/config.toml sets base colors, inject_css()
  adds the glassmorphism cards, gradients, and micro-animations.
* All charts are Plotly (interactive hover/zoom) — see pipeline.py section 9.
* Privacy: emails / student names / member lists are never DISPLAYED anywhere
  in the app (instructor request). They stay in the data and in Export downloads.

Run locally:
    py -m pip install -r requirements.txt
    py -m streamlit run app.py
"""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from pipeline import (
    APP_TITLE,
    APP_SUBTITLE,
    INSTRUCTOR_PASSCODE,
    add_chemical_descriptors,
    apply_element_fixes,
    bubble_relationships,
    build_feature_matrix,
    build_html_report,
    clean_and_encode_data,
    compute_pca,
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    detect_numeric_outliers,
    display_label,
    element_outcome_summary,
    find_default_database,
    find_merge_duplicates,
    friendly_label,
    glossary_table,
    label_count_table,
    load_atomic_table_from_bytes,
    load_en_table_from_bytes,
    load_radii_table_from_bytes,
    make_demo_dataset,
    make_single_prediction_row,
    mixing_summary,
    nearest_neighbors,
    normalize_bubble_label,
    normalize_phase_label,
    normalize_to_long_format,
    numeric_correlation_table,
    ordered_columns,
    plot_bar_chart,
    plot_confusion,
    plot_heatmap,
    plot_pca_scatter,
    plot_periodic_heat,
    plot_what_if,
    propose_element_fixes,
    public_view,
    read_local_table,
    read_table_from_bytes,
    remove_invalid_element_rows,
    summarize_by_element,
    valid_element_symbols,
    train_ml_model,
    train_purity_model,
    validate_compound_rows,
)

# Shared Plotly toolbar config: keep zoom/pan/hover, hide the noisy extras.
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "zoomIn2d", "zoomOut2d"],
}


# =============================================================================
# PAGE STYLE — dark "lab glass" theme with micro-animations
# =============================================================================

def inject_css() -> None:
    """Inject the app-wide CSS. Base colors come from .streamlit/config.toml."""

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

        :root {
            --ink: #e2e8f0;
            --muted: #94a3b8;
            --accent: #38bdf8;
            --accent-2: #8b5cf6;
            --card-bg: rgba(17, 26, 46, 0.62);
            --card-border: rgba(148, 163, 184, 0.16);
            --ok: #34d399;
            --warn: #fbbf24;
            --bad: #f87171;
        }

        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
        }

        /* Ambient background glows behind everything. */
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(1100px 700px at 88% -12%, rgba(56, 189, 248, 0.13), transparent 60%),
                radial-gradient(900px 650px at -8% 108%, rgba(139, 92, 246, 0.11), transparent 55%),
                radial-gradient(700px 500px at 55% 120%, rgba(34, 211, 238, 0.06), transparent 60%);
        }

        .main .block-container {
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }

        h1, h2, h3, h4 { letter-spacing: -0.02em; color: var(--ink); }

        /* ---------- animations ---------- */
        @keyframes riseIn {
            from { opacity: 0; transform: translateY(14px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes gradientFlow {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes floatOrb {
            from { transform: translate(0, 0) scale(1); }
            to   { transform: translate(26px, -18px) scale(1.12); }
        }
        @keyframes barGrow { from { width: 0; } }

        .stTabs [data-baseweb="tab-panel"] { animation: riseIn 0.35s ease both; }

        /* ---------- hero ---------- */
        .hero {
            position: relative;
            overflow: hidden;
            padding: 1.9rem 2.2rem 1.7rem;
            border: 1px solid var(--card-border);
            border-radius: 22px;
            background: linear-gradient(160deg, rgba(19, 30, 55, 0.92), rgba(11, 17, 32, 0.88));
            box-shadow: 0 18px 60px rgba(2, 6, 18, 0.55), inset 0 1px 0 rgba(148, 163, 184, 0.12);
            margin-bottom: 1.1rem;
            animation: riseIn 0.5s ease both;
        }
        .hero::before, .hero::after {
            content: "";
            position: absolute;
            border-radius: 50%;
            filter: blur(46px);
            pointer-events: none;
        }
        .hero::before {
            width: 320px; height: 320px; right: -80px; top: -140px;
            background: radial-gradient(circle, rgba(56, 189, 248, 0.28), transparent 70%);
            animation: floatOrb 9s ease-in-out infinite alternate;
        }
        .hero::after {
            width: 260px; height: 260px; left: 22%; bottom: -170px;
            background: radial-gradient(circle, rgba(139, 92, 246, 0.22), transparent 70%);
            animation: floatOrb 11s ease-in-out infinite alternate-reverse;
        }
        .hero-kicker {
            font-size: 0.74rem; font-weight: 700; letter-spacing: 0.16em;
            text-transform: uppercase; color: var(--accent);
            margin-bottom: 0.45rem;
        }
        .hero-title {
            font-size: 2.35rem; line-height: 1.06; font-weight: 800; margin: 0;
            background: linear-gradient(90deg, #e8eefb 0%, #7dd3fc 45%, #c4b5fd 75%, #e8eefb 100%);
            background-size: 200% auto;
            -webkit-background-clip: text; background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientFlow 9s ease infinite;
        }
        .hero-subtitle { color: var(--muted); font-size: 1.02rem; margin-top: 0.55rem; max-width: 860px; }
        .hero-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1rem; position: relative; z-index: 1; }
        .chip {
            font-size: 0.8rem; font-weight: 600; color: #bae6fd;
            padding: 0.28rem 0.75rem; border-radius: 999px;
            border: 1px solid rgba(56, 189, 248, 0.3);
            background: rgba(56, 189, 248, 0.09);
        }

        /* ---------- cards ---------- */
        .glass-card, .soft-card {
            padding: 1.1rem 1.25rem;
            border: 1px solid var(--card-border);
            border-radius: 18px;
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            box-shadow: 0 10px 34px rgba(2, 6, 18, 0.4);
            margin-bottom: 0.9rem;
            color: var(--ink);
            transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
            animation: riseIn 0.45s ease both;
        }
        .glass-card:hover { transform: translateY(-2px); border-color: rgba(56, 189, 248, 0.35); }

        .kpi-card { display: flex; align-items: center; gap: 0.85rem; padding: 1rem 1.15rem; margin-bottom: 0.55rem; }
        .kpi-icon {
            width: 42px; height: 42px; flex: 0 0 42px; border-radius: 12px;
            display: flex; align-items: center; justify-content: center; font-size: 1.25rem;
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.16), rgba(139, 92, 246, 0.16));
            border: 1px solid rgba(56, 189, 248, 0.25);
        }
        .kpi-value {
            font-size: 1.5rem; font-weight: 800; line-height: 1.05;
            background: linear-gradient(90deg, #e8eefb, #7dd3fc);
            -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
        }
        .kpi-label {
            font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
            text-transform: uppercase; color: var(--muted); margin-top: 0.15rem;
        }
        .kpi-ok   { border-left: 3px solid var(--ok); }
        .kpi-warn { border-left: 3px solid var(--warn); }
        .kpi-bad  { border-left: 3px solid var(--bad); }

        .helper-text { color: var(--muted); font-size: 0.92rem; line-height: 1.5; }
        .big-number { color: var(--ink); font-size: 1.4rem; font-weight: 800; }

        .status-ok   { border-left: 3px solid var(--ok); }
        .status-warn { border-left: 3px solid var(--warn); }
        .status-bad  { border-left: 3px solid var(--bad); }

        /* ---------- landing feature grid ---------- */
        .feat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.9rem; margin-top: 1rem; }
        @media (max-width: 900px) { .feat-grid { grid-template-columns: 1fr; } }
        .feat-card {
            padding: 1.2rem 1.25rem;
            border: 1px solid var(--card-border); border-radius: 18px;
            background: var(--card-bg); backdrop-filter: blur(12px);
            transition: transform 0.25s ease, border-color 0.25s ease;
            animation: riseIn 0.5s ease both;
        }
        .feat-card:nth-child(2) { animation-delay: 0.08s; }
        .feat-card:nth-child(3) { animation-delay: 0.16s; }
        .feat-card:hover { transform: translateY(-3px); border-color: rgba(56, 189, 248, 0.4); }
        .feat-icon { font-size: 1.5rem; }
        .feat-title { font-weight: 700; color: var(--ink); margin: 0.5rem 0 0.3rem; }

        /* ---------- probability cards ---------- */
        .prob-card { padding: 1.15rem 1.3rem; }
        .prob-label { font-size: 0.86rem; font-weight: 600; color: var(--muted); }
        .prob-value { font-size: 2.15rem; font-weight: 800; margin: 0.15rem 0 0.45rem; }
        .prob-track {
            height: 10px; border-radius: 999px; background: rgba(148, 163, 184, 0.14);
            overflow: hidden;
        }
        .prob-fill {
            height: 100%; border-radius: 999px;
            animation: barGrow 0.9s cubic-bezier(0.22, 1, 0.36, 1) both;
            box-shadow: 0 0 14px rgba(56, 189, 248, 0.4);
        }
        .prob-verdict { margin-top: 0.55rem; font-size: 0.88rem; font-weight: 600; }

        .formula {
            font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #7dd3fc;
        }
        .formula sub { font-size: 0.68em; }

        /* ---------- tabs ---------- */
        .stTabs [data-baseweb="tab-list"] { gap: 7px; flex-wrap: wrap; margin-top: 0.55rem; padding-top: 3px; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.5rem 1.05rem;
            background: rgba(18, 27, 48, 0.85);
            border: 1px solid var(--card-border);
            color: var(--muted) !important;
            transition: all 0.22s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            border-color: rgba(56, 189, 248, 0.4); color: var(--ink) !important;
            transform: translateY(-1px);
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(139, 92, 246, 0.2)) !important;
            border-color: rgba(56, 189, 248, 0.55) !important;
            color: #7dd3fc !important;
            box-shadow: 0 0 18px rgba(56, 189, 248, 0.18);
        }
        .stTabs [data-baseweb="tab"] p { color: inherit !important; font-weight: 600; }
        .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

        /* ---------- widgets ---------- */
        .stButton > button, [data-testid="stDownloadButton"] > button, [data-testid="stFormSubmitButton"] > button {
            border-radius: 12px;
            border: 1px solid var(--card-border);
            background: rgba(22, 35, 60, 0.9);
            color: var(--ink);
            font-weight: 600;
            transition: all 0.18s ease;
        }
        .stButton > button:hover, [data-testid="stDownloadButton"] > button:hover,
        [data-testid="stFormSubmitButton"] > button:hover {
            border-color: rgba(56, 189, 248, 0.55);
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(2, 6, 18, 0.45);
        }
        .stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"],
        [data-testid="stDownloadButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, #0ea5e9, #6366f1);
            border: none; color: white;
            box-shadow: 0 6px 22px rgba(14, 165, 233, 0.35);
        }
        .stButton > button[kind="primary"]:hover { filter: brightness(1.1); }

        /* Pills (filters) — rounded with a glow when active. */
        button[data-testid="stBaseButton-pills"], button[kind="pills"] {
            border-radius: 999px !important;
            border: 1px solid var(--card-border) !important;
            background: rgba(18, 27, 48, 0.85) !important;
            color: var(--muted) !important;
            transition: all 0.18s ease !important;
        }
        button[data-testid="stBaseButton-pillsActive"], button[kind="pillsActive"] {
            border-radius: 999px !important;
            border: 1px solid rgba(56, 189, 248, 0.55) !important;
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.22), rgba(139, 92, 246, 0.22)) !important;
            color: #7dd3fc !important;
            box-shadow: 0 0 14px rgba(56, 189, 248, 0.16) !important;
        }

        [data-testid="stMetric"] {
            padding: 0.85rem 1rem;
            border: 1px solid var(--card-border);
            border-radius: 14px;
            background: var(--card-bg);
            transition: border-color 0.25s ease;
        }
        [data-testid="stMetric"]:hover { border-color: rgba(56, 189, 248, 0.35); }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        div[data-testid="stMetricValue"] { font-weight: 800; letter-spacing: -0.03em; }

        [data-testid="stExpander"] details {
            border: 1px solid var(--card-border);
            border-radius: 14px;
            background: rgba(17, 26, 46, 0.5);
            transition: border-color 0.25s ease;
        }
        [data-testid="stExpander"] details:hover { border-color: rgba(56, 189, 248, 0.3); }
        [data-testid="stExpander"] summary { font-weight: 600; }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--card-border);
            border-radius: 12px;
            overflow: hidden;
        }

        [data-testid="stAlert"] { border-radius: 12px; }

        [data-testid="stSidebarContent"] {
            background: linear-gradient(180deg, #0e1626 0%, #0a0f1c 100%);
        }
        .sb-section {
            font-size: 0.7rem; font-weight: 700; letter-spacing: 0.14em;
            text-transform: uppercase; color: var(--accent);
            margin: 0.9rem 0 0.2rem;
        }

        ::-webkit-scrollbar { width: 9px; height: 9px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #24324f; border-radius: 999px; }
        ::-webkit-scrollbar-thumb:hover { background: #33456b; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SMALL UI HELPERS
# =============================================================================

def formula_html(formula: str) -> str:
    """Render a formula string like La2NiO4 with real subscripts."""

    safe = str(formula or "").replace("<", "").replace(">", "")
    return '<span class="formula">' + re.sub(r"(\d+(?:\.\d+)?)", r"<sub>\1</sub>", safe) + "</span>"


def instructor_mode_on() -> bool:
    """True when the instructor passcode has been entered this session."""

    return bool(st.session_state.get("instructor_mode", False))


def visible(df: pd.DataFrame) -> pd.DataFrame:
    """Contact/group columns stay hidden for students, shown in instructor mode."""

    return df if instructor_mode_on() else public_view(df)


def kpi_card(icon: str, value: str, label: str, tone: str = "", delay: float = 0.0) -> str:
    """Return HTML for one KPI card in the header row."""

    return f"""
    <div class="glass-card kpi-card {tone}" style="animation-delay:{delay}s;">
        <div class="kpi-icon">{icon}</div>
        <div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>
    </div>"""


def prob_card(formula_str: str, prob: float, kind: str) -> str:
    """Return HTML for an animated probability card (bubble or purity)."""

    pct = max(0.0, min(1.0, float(prob)))
    if pct >= 0.6:
        verdict, color = ("Likely to bubble 🎉" if kind == "bubble" else "Likely to come out pure ✨"), "var(--ok)"
        fill = "linear-gradient(90deg, #34d399, #38bdf8)"
    elif pct >= 0.4:
        verdict, color = "Could go either way — a good experiment!", "var(--warn)"
        fill = "linear-gradient(90deg, #fbbf24, #38bdf8)"
    else:
        verdict, color = ("Unlikely to bubble" if kind == "bubble" else "Likely to be impure"), "var(--bad)"
        fill = "linear-gradient(90deg, #f87171, #fbbf24)"

    title = "Will it bubble?" if kind == "bubble" else "Will it be pure?"
    return f"""
    <div class="glass-card prob-card">
        <div class="prob-label">{title} &nbsp;·&nbsp; {formula_html(formula_str)}</div>
        <div class="prob-value">{pct:.0%}</div>
        <div class="prob-track"><div class="prob-fill" style="width:{pct * 100:.0f}%; background:{fill};"></div></div>
        <div class="prob-verdict" style="color:{color};">{verdict}</div>
    </div>"""


def render_hero(chips: list[str] | None = None) -> None:
    """Draw the animated hero header, optionally with dataset chips."""

    chips_html = ""
    if chips:
        chips_html = '<div class="hero-chips">' + "".join(f'<span class="chip">{c}</span>' for c in chips) + "</div>"

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">CSUF · CURE Research · Perovskite Catalysts</div>
            <div class="hero-title">{APP_TITLE}</div>
            <div class="hero-subtitle">{APP_SUBTITLE}</div>
            {chips_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def multi_pills_with_all_none(label: str, options: list[str], key: str) -> list[str]:
    """
    A pills multi-select with All / None quick buttons (instructor request:
    "say none and select only the one you want" instead of un-clicking each).

    Returns the current selection. An EMPTY selection means "no filter" —
    the caller shows everything and we surface a small hint.
    """

    if not options:
        return []

    # Sanitize stale session values (e.g. after a new file upload).
    if key in st.session_state:
        st.session_state[key] = [v for v in st.session_state[key] if v in options]

    head = st.columns([4, 0.7, 0.9])
    head[0].markdown(f"**{label}**")
    if head[1].button("All", key=f"{key}_all", help=f"Select every {label.lower()}"):
        st.session_state[key] = list(options)
    if head[2].button("None", key=f"{key}_none", help="Clear the selection, then pick just the ones you want"):
        st.session_state[key] = []

    kwargs = {} if key in st.session_state else {"default": list(options)}
    selected = st.pills(label, options, selection_mode="multi", key=key,
                        label_visibility="collapsed", **kwargs)
    if not selected:
        st.caption(f"No {label.lower()} selected — showing all. Click a pill to filter.")
    return list(selected or [])


# =============================================================================
# TAB RENDERERS
# =============================================================================

def render_did_you_mean(fix_proposals: pd.DataFrame, all_symbols: list) -> None:
    """Pre-cleaning panel: a table of invalid elements, each with a fix dropdown."""

    n = len(fix_proposals)
    with st.expander(f"🛠️ Fix invalid elements ({n} to review)", expanded=True):
        st.caption("Each row is an entry that isn't a real element symbol. The **Fix to** dropdown "
                   "is pre-set to the best guess — change it or pick **Remove row**, then click Apply.")

        editor_df = pd.DataFrame({
            "Row": fix_proposals["SourceRow"],
            "Slot": fix_proposals["Slot"],
            "Field": fix_proposals["Field"],
            "Invalid value": fix_proposals["Original"],
            "Did you mean": fix_proposals["Suggestions"].apply(
                lambda s: ", ".join(s) if s else "(no close match)"),
            "Fix to": fix_proposals["Suggestions"].apply(lambda s: s[0] if s else "Remove row"),
        })

        edited = st.data_editor(
            editor_df,
            hide_index=True,
            use_container_width=True,
            key="fix_editor",
            column_config={
                "Row": st.column_config.Column("Row", disabled=True),
                "Slot": st.column_config.Column("Slot", disabled=True),
                "Field": st.column_config.Column("Field", disabled=True),
                "Invalid value": st.column_config.Column("⚠️ Invalid value", disabled=True),
                "Did you mean": st.column_config.Column("Did you mean", disabled=True),
                "Fix to": st.column_config.SelectboxColumn(
                    "Fix to", options=["Remove row"] + all_symbols, required=True),
            },
        )

        col_a, col_b = st.columns([1, 4])
        with col_a:
            if st.button("Apply", type="primary", key="apply_element_fixes"):
                fixes = dict(st.session_state.element_fixes)
                for _, r in edited.iterrows():
                    key = f"{r['Row']}|{r['Slot']}|{r['Field']}"
                    if r["Fix to"] == "Remove row":
                        fixes.pop(key, None)          # no correction -> row is removed
                    else:
                        fixes[key] = r["Fix to"]      # replace with chosen symbol
                st.session_state.element_fixes = fixes
                st.rerun()
        with col_b:
            if st.session_state.element_fixes and st.button("Reset all corrections", key="reset_element_fixes"):
                st.session_state.element_fixes = {}
                st.rerun()


def render_check_tab(described_df: pd.DataFrame, issues_df: pd.DataFrame,
                     outlier_df: pd.DataFrame, cleaning_log: pd.DataFrame,
                     fix_proposals: pd.DataFrame, corrections_log: pd.DataFrame,
                     all_symbols: list) -> None:
    """Tab 1 — surface data-entry problems before any analysis."""

    st.caption("Find and fix data-entry mistakes before using the data for graphs or ML.")

    if not fix_proposals.empty:
        render_did_you_mean(fix_proposals, all_symbols)

    # Privacy: email-match checks run in the background and are included in the
    # downloadable validation report, but are only shown in instructor mode.
    if not instructor_mode_on() and not issues_df.empty and "Field" in issues_df.columns:
        shown_issues = issues_df[issues_df["Field"] != "Email"].reset_index(drop=True)
        n_hidden = len(issues_df) - len(shown_issues)
    else:
        shown_issues, n_hidden = issues_df, 0

    check_cols = st.columns(2)
    with check_cols[0]:
        st.markdown("#### Validation issues")
        if shown_issues.empty:
            st.caption("No validation issues found. ✅")
        else:
            st.caption(f"{len(shown_issues):,} possible issues. Review before final analysis.")
        st.dataframe(shown_issues, use_container_width=True, height=300)
        if n_hidden:
            st.caption(f"🔒 {n_hidden} email-match check(s) run in the background — see the "
                       "validation report in **Export** (instructor download).")
    with check_cols[1]:
        st.markdown("#### Possible numeric outliers")
        if outlier_df.empty:
            st.caption("No outliers at the current sensitivity. ✅")
        else:
            st.caption("Outliers aren't automatically wrong — just worth a second look.")
        st.dataframe(outlier_df, use_container_width=True, height=300)

    st.markdown("#### Cleaned compound table")
    if described_df.empty:
        st.warning("No compound rows found. Check that the file uses columns like 1A, 1AN, 1B, 1BN, 1O, 1ON, 1P, 1Bub.")
    else:
        table = visible(described_df)
        st.dataframe(table[ordered_columns(table)], use_container_width=True, height=380)
        if instructor_mode_on():
            st.caption("🔓 Instructor mode — contact and group-member columns are visible.")
        else:
            st.caption("🔒 Contact and group-member info is collected but never displayed in the app.")

    n_removed = len(cleaning_log)
    n_fixed = len(corrections_log)
    with st.expander(f"What cleaning changed ({n_fixed} corrected, {n_removed} removed)"):
        if not corrections_log.empty:
            st.markdown("**Corrections applied (Did you mean?)**")
            st.dataframe(corrections_log, use_container_width=True, hide_index=True)
        if not cleaning_log.empty:
            st.markdown("**Rows removed (invalid element)**")
            st.caption("Removed because they contained an element that isn't a real periodic-table symbol.")
            st.dataframe(cleaning_log, use_container_width=True, hide_index=True)
        if corrections_log.empty and cleaning_log.empty:
            st.caption("No rows removed or corrected. Symbols, formatting, and placeholder text were standardized in place.")

    with st.expander("Data-entry rules"):
        st.markdown(
            "- Element **symbols**, not names: `La`, not `Lanthanum`.\n"
            "- **Numbers** for ratios: `2`, not `two`.\n"
            "- Oxygen is **O**. Phase is **pure** or **impure**. Bubble is **yes / no / maybe**."
        )


def render_explore_tab(described_df: pd.DataFrame) -> None:
    """Tab 2 — response counts, per-element summaries, and cation-mixing views."""

    st.caption("Descriptive tables only — patterns to discuss, not proof of cause.")

    semesters = sorted([x for x in described_df.get("Semester", pd.Series(dtype=str)).dropna().astype(str).unique() if x.strip()])
    instructors = sorted([x for x in described_df.get("Instructor", pd.Series(dtype=str)).dropna().astype(str).unique() if x.strip()])

    filter_cols = st.columns([2.4, 2.4, 1.6])
    with filter_cols[0]:
        selected_semesters = multi_pills_with_all_none("Semester", semesters, "flt_semesters")
    with filter_cols[1]:
        selected_instructors = multi_pills_with_all_none("Instructor", instructors, "flt_instructors")
    with filter_cols[2]:
        st.markdown("**Options**")
        hide_rare = st.checkbox("Hide elements with < 3 compounds", value=True)
        min_rows = 3 if hide_rare else 1

    filtered = described_df.copy()
    if selected_semesters:
        filtered = filtered[filtered["Semester"].astype(str).isin(selected_semesters)]
    if selected_instructors:
        filtered = filtered[filtered["Instructor"].astype(str).isin(selected_instructors)]

    if filtered.empty:
        st.warning("No rows match the selected filters.")
        return

    st.caption(f"Showing **{len(filtered):,}** of {len(described_df):,} compounds.")

    plot_df = filtered.copy()
    phase_src = "P_raw" if "P_raw" in plot_df.columns else "P"
    bub_src = "Bub_raw" if "Bub_raw" in plot_df.columns else "Bub"
    plot_df["PhasePlotLabel"] = plot_df[phase_src].apply(lambda v: display_label(normalize_phase_label(v)))
    plot_df["BubblePlotLabel"] = plot_df[bub_src].apply(lambda v: display_label(normalize_bubble_label(v)))

    st.markdown("#### Response counts")
    count_cols = st.columns(2)
    with count_cols[0]:
        st.markdown("**Bubble response**")
        st.dataframe(
            label_count_table(plot_df, "BubblePlotLabel", kind="bubble",
                              preferred_order=["Yes", "No", "Maybe", "Other / needs review", "Missing"]),
            use_container_width=True, hide_index=True,
        )
    with count_cols[1]:
        st.markdown("**Phase result**")
        st.dataframe(
            label_count_table(plot_df, "PhasePlotLabel", kind="phase",
                              preferred_order=["Pure", "Impure", "Not made", "Other / needs review", "Missing"]),
            use_container_width=True, hide_index=True,
        )

    # --- Periodic-table outcome view ---
    st.markdown("#### Periodic view — where the class has explored")
    scope_choice = st.radio("Show elements used on", ["Any site", "A-site (A/A′)", "B-site (B/B′/B″)"],
                            horizontal=True, key="periodic_scope")
    scope = {"Any site": "any", "A-site (A/A′)": "A", "B-site (B/B′/B″)": "B"}[scope_choice]
    element_stats = element_outcome_summary(filtered, scope=scope)
    if element_stats.empty:
        st.info("Not enough element data for this filter.")
    else:
        st.plotly_chart(plot_periodic_heat(element_stats), use_container_width=True, config=PLOTLY_CONFIG)
        st.caption("Color = bubble-yes rate for compounds using that element (red 0% → green 100%). "
                   "Gray elements haven't been tried yet — hover a tile for its numbers. "
                   "Untried neighbors of green tiles make great candidates.")

    st.markdown("#### Elements vs. bubbling and purity")
    st.caption("Bubble yes % and Pure % for each element. Percentages of that element's compounds.")

    a_summary = summarize_by_element(filtered, "A", min_rows=int(min_rows))
    b_summary = summarize_by_element(filtered, "B", min_rows=int(min_rows))
    view_cols = ["compounds", "bubble_yes_rate", "pure_rate", "avg_formula_mass"]
    rename = {"compounds": "Compounds", "bubble_yes_rate": "Bubble yes %",
              "pure_rate": "Pure %", "avg_formula_mass": "Avg formula mass"}

    elem_cols = st.columns(2)
    with elem_cols[0]:
        st.markdown("**A-site elements**")
        if a_summary.empty:
            st.info("Not enough A-site data for this filter.")
        else:
            st.dataframe(a_summary[["A"] + view_cols].rename(columns=rename).round(1),
                         use_container_width=True, hide_index=True, height=300)
    with elem_cols[1]:
        st.markdown("**B-site elements**")
        if b_summary.empty:
            st.info("Not enough B-site data for this filter.")
        else:
            st.dataframe(b_summary[["B"] + view_cols].rename(columns=rename).round(1),
                         use_container_width=True, hide_index=True, height=300)

    # --- Cation mixing (instructor question #1: "is A/A' mixing reflected anywhere?") ---
    st.markdown("#### Cation mixing — A/A′ and B/B′")
    st.caption("Compounds where a site holds TWO OR MORE elements in differing amounts. "
               "The share columns show how much of the site the secondary cation takes "
               "(50% = a perfect 50/50 mix). The same information feeds the heatmap and the "
               "models as **A mixing frac** and **B mixing frac**.")
    mix_table = mixing_summary(filtered)
    if mix_table.empty:
        st.info("Not enough data to summarize cation mixing yet.")
    else:
        st.dataframe(mix_table, use_container_width=True, hide_index=True)

    with st.expander("Descriptor preview"):
        descriptor_cols = [c for c in [
            "Formula", "A_avg_Z", "B_avg_Z", "A_mix_fraction", "B_mix_fraction",
            "FormulaMass", "O_to_cation_ratio", "B_to_A_ratio", "BubbleLabel", "PhaseLabel",
        ] if c in filtered.columns]
        st.dataframe(filtered[descriptor_cols].round(4), use_container_width=True, height=300)


def render_heatmap_tab(described_df: pd.DataFrame) -> None:
    """Tab 3 — interactive correlation heatmap with matching links table + glossary."""

    st.caption("Each cell compares two numeric features — hover any cell for the exact value. "
               "Read the **Bubbles (yes)** and **Phase (pure=2)** rows for links to outcomes.")

    corr = numeric_correlation_table(described_df)
    if corr.empty:
        st.warning("Not enough numeric columns to build a relationship map.")
        return

    all_features = list(corr.columns)
    default_features = [c for c in [
        "BubbleYes", "PhaseN", "A_avg_Z", "B_avg_Z", "FormulaMass",
        "O_to_cation_ratio", "B_to_A_ratio", "Avg_cation_Z",
        "A_mix_fraction", "B_mix_fraction", "Tolerance_factor",
    ] if c in all_features] or all_features

    selected = st.multiselect(
        "Features to compare",
        options=all_features,
        default=default_features,
        format_func=friendly_label,
        help="Keep Bubbles (yes) and Phase (pure=2) selected to see what links to bubbling and purity.",
    )
    if len(selected) < 2:
        st.info("Select at least two features. Showing defaults for now.")
        selected = default_features
    corr_view = corr.loc[selected, selected]

    map_cols = st.columns([1.25, 1])
    with map_cols[0]:
        st.plotly_chart(plot_heatmap(corr_view), use_container_width=True, config=PLOTLY_CONFIG)
    with map_cols[1]:
        st.markdown("#### Reading it")
        st.markdown("**+1** rise together · **0** no link · **−1** move in opposite directions. "
                    "Correlation is not causation.")

        st.markdown("#### Strongest links to bubbling")
        search_all = st.toggle(
            "Rank across all features", value=False,
            help="Off: this table ranks only the features shown on the map, so the two always match. "
                 "On: it searches every feature — items may then not appear on your map.",
        )
        rel = bubble_relationships(corr, restrict_to=None if search_all else selected)
        if search_all:
            st.caption("Ranking **all** features — add one to the map selection to see it in context.")
        else:
            st.caption("Ranking only the features shown on the map, so the table and map always match.")
        if rel.empty:
            st.info("Not available yet.")
        else:
            st.dataframe(rel.round(3), use_container_width=True, hide_index=True, height=280)

    # Glossary (instructor request: define the heat-map terms).
    with st.expander("📖 Glossary — what every term on this map means", expanded=False):
        glossary = glossary_table(selected)
        st.dataframe(
            glossary, use_container_width=True, hide_index=True,
            height=min(430, 38 * len(glossary) + 40),
            column_config={
                "Term": st.column_config.Column("Term", width="small"),
                "Meaning": st.column_config.Column("Meaning", width="large"),
            },
        )


def render_pca_tab(described_df: pd.DataFrame) -> None:
    """Tab 4 — PCA structure map with optional KMeans clusters."""

    st.caption("Squeezes all numeric descriptors into a 2-D map. Nearby compounds have similar "
               "composition — hover any point to see its formula. Exploration only.")

    k = st.slider("KMeans clusters (0 = off)", min_value=0, max_value=8, value=0,
                  help="Group compounds into clusters by composition. Raise above 1 to see clusters.")

    result = compute_pca(described_df, n_clusters=k)
    if not result.get("ok"):
        st.info(result.get("message", "PCA not available."))
        return

    if result["has_clusters"]:
        color_by = st.radio("Color points by", ["Cluster", "Bubble", "Phase"], horizontal=True)
    else:
        color_by = st.radio("Color points by", ["Bubble", "Phase"], horizontal=True)

    ev = result["explained"]
    map_cols = st.columns([1.45, 1])
    with map_cols[0]:
        st.plotly_chart(plot_pca_scatter(result["coords"], color_by=color_by),
                        use_container_width=True, config=PLOTLY_CONFIG)
    with map_cols[1]:
        st.metric("PC1 variance", f"{ev[0]:.0%}")
        st.metric("PC2 variance", f"{ev[1]:.0%}")
        st.metric("Compounds plotted", f"{result['n_used']:,}")
        st.caption(f"PC1 + PC2 together explain {sum(ev):.0%} of the variation across "
                   f"{len(result['features'])} descriptors.")

    with st.expander("What drives each axis (loadings)", expanded=False):
        sort_pc = st.radio("Sort by biggest weight on", ["PC1", "PC2"], horizontal=True,
                           key="pca_loading_sort")
        st.caption(f"Sorted by influence on {sort_pc} (biggest absolute weight first). "
                   "Bigger absolute values = the descriptor pushes a compound further along that axis.")
        loadings = result["loadings"].copy()
        loadings = loadings.reindex(loadings[sort_pc].abs().sort_values(ascending=False).index)
        loadings.index = [friendly_label(i) for i in loadings.index]
        loadings.index.name = "Descriptor"
        st.dataframe(loadings.round(2), use_container_width=True)


def render_ml_tab(described_df: pd.DataFrame, atomic: pd.DataFrame, en_table: pd.DataFrame,
                  radii: pd.DataFrame, use_phase_in_ml: bool, use_mass_in_ml: bool) -> None:
    """Tab 5 — train the bubble + purity models and predict a proposed compound."""

    st.caption("A hypothesis tool, not proof. Two models: one for bubbling, one for purity.")

    with st.spinner("Training models…"):
        ml_result = train_ml_model(described_df, use_phase=use_phase_in_ml, use_mass=use_mass_in_ml)
    if not ml_result.get("ok"):
        st.warning(ml_result.get("message", "Bubble model could not be trained."))
        return

    if ml_result.get("overfit_warning"):
        st.warning(ml_result["overfit_warning"])

    baseline = ml_result["baseline_accuracy"]
    rf_acc = ml_result["rf_accuracy"]

    st.markdown("#### Bubble model")
    if use_mass_in_ml:
        st.caption("⚖️ Mass descriptors are **on** — they tend to dominate importance because mass "
                   "and atomic number rise together. Turn them off in the sidebar to see the other signal.")
    else:
        st.caption("⚖️ Mass descriptors are **off** (default) so mass can't crowd out the rest of the "
                   "chemistry. Flip **Include mass descriptors** in the sidebar to compare.")

    bal_cols = st.columns(4)
    with bal_cols[0]:
        st.metric("Bubble = yes", f"{ml_result['yes_count']:,}")
    with bal_cols[1]:
        st.metric("No / maybe", f"{ml_result['no_count']:,}")
    with bal_cols[2]:
        st.metric("Accuracy", f"{rf_acc:.0%}", delta=f"{rf_acc - baseline:+.0%} vs baseline")
    with bal_cols[3]:
        st.metric("Precision / recall", f"{ml_result['rf_precision']:.0%} / {ml_result['rf_recall']:.0%}")

    beat = rf_acc > baseline
    note = (f"Always-guess baseline is {baseline:.0%}; the model scores {rf_acc:.0%} — "
            + ("found real signal." if beat else "no useful signal yet."))
    (st.success if beat else st.warning)(note)

    if ml_result.get("cv_mean") is not None:
        st.caption(f"🎯 {ml_result['cv_folds']}-fold cross-validated accuracy: "
                   f"**{ml_result['cv_mean']:.0%} ± {ml_result['cv_std']:.02f}** — a more honest "
                   "estimate than a single split on data this size.")

    with st.expander("What the numbers mean + confusion matrix"):
        st.markdown(
            f"- **Accuracy** can mislead when yes is rare — compare to the {baseline:.0%} baseline.\n"
            "- **Precision**: when it says yes, how often it's right.\n"
            "- **Recall**: of all real bubblers, how many it caught.\n"
            "- **Cross-validation** retrains the model on several different splits and averages "
            "the scores, so one lucky split can't flatter it.\n"
            "- **LR accuracy** "
            f"({ml_result['lr_accuracy']:.0%}) is a simpler sanity-check model."
        )
        if ml_result.get("confusion"):
            cm_cols = st.columns([1, 1])
            with cm_cols[0]:
                st.plotly_chart(plot_confusion(ml_result["confusion"]),
                                use_container_width=True, config=PLOTLY_CONFIG)
            with cm_cols[1]:
                cm = ml_result["confusion"]
                st.markdown(
                    f"On the held-out test set:\n\n"
                    f"- ✅ **{cm['tp']}** real bubblers caught (true positives)\n"
                    f"- ✅ **{cm['tn']}** non-bubblers correctly ruled out\n"
                    f"- ⚠️ **{cm['fp']}** false alarms (said yes, was no)\n"
                    f"- ⚠️ **{cm['fn']}** bubblers missed (said no, was yes)"
                )

    importance = ml_result["importance"]
    if not importance.empty:
        st.markdown("**Top features used** — hover a bar for the exact weight")
        st.plotly_chart(
            plot_bar_chart(importance, "Feature", "Importance", "", "Importance"),
            use_container_width=True, config=PLOTLY_CONFIG,
        )

    st.markdown("#### Purity model")
    purity_result = train_purity_model(described_df, use_mass=use_mass_in_ml)
    if not purity_result.get("ok"):
        st.info(purity_result.get("message", "Purity model not available yet."))
    else:
        if purity_result.get("overfit_warning"):
            st.warning(purity_result["overfit_warning"])
        p = st.columns(4)
        with p[0]:
            st.metric("Pure examples", f"{purity_result['pure_count']:,}")
        with p[1]:
            st.metric("Not-pure examples", f"{purity_result['impure_count']:,}")
        with p[2]:
            st.metric("Accuracy", f"{purity_result['accuracy']:.0%}",
                      delta=f"{purity_result['accuracy'] - purity_result['baseline_accuracy']:+.0%} vs baseline")
        with p[3]:
            st.metric("Precision (pure)", f"{purity_result['precision']:.0%}")
        st.caption("Predicts purity from composition only — the phase label is never an input.")

    # --- Predict a proposed compound (A, A', B, B', B'' all supported) ---
    st.markdown("#### Test a proposed compound")
    st.caption("Pick elements from the dropdowns — every site supports mixing: A + A′ and B + B′ "
               "(plus a third B″ under advanced). Leave A′/B′ as **— none —** for a simple compound.")

    symbols = valid_element_symbols(atomic)
    optional_symbols = [""] + symbols

    def element_select(label: str, key: str, default: str = "", optional: bool = True):
        options = optional_symbols if optional else symbols
        index = options.index(default) if default in options else 0
        return st.selectbox(label, options, index=index, key=key,
                            format_func=lambda s: s if s else "— none —")

    with st.container(border=True):
        st.markdown("**A-site**")
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            pred_a = element_select("A element", "pred_a", default="La", optional=False)
        with a2:
            pred_an = st.number_input("A ratio", value=2.0, min_value=0.0, step=0.5, key="pred_an")
        with a3:
            pred_ap = element_select("A′ element", "pred_ap")
        with a4:
            pred_apn = st.number_input("A′ ratio", value=0.0, min_value=0.0, step=0.5, key="pred_apn")

        st.markdown("**B-site**")
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            pred_b = element_select("B element", "pred_b", default="Ni", optional=False)
        with b2:
            pred_bn = st.number_input("B ratio", value=1.0, min_value=0.0, step=0.5, key="pred_bn")
        with b3:
            pred_bp = element_select("B′ element", "pred_bp")
        with b4:
            pred_bpn = st.number_input("B′ ratio", value=0.0, min_value=0.0, step=0.5, key="pred_bpn")

        o1, o2, o3 = st.columns(3)
        with o1:
            pred_on = st.number_input("Oxygen ratio", value=4.0, min_value=0.0, step=0.5, key="pred_on")
        with o2:
            pred_phase = st.selectbox("Phase", ["pure", "impure"], key="pred_phase")

        with st.expander("Third B-site element (B″, advanced)"):
            adv1, adv2 = st.columns(2)
            with adv1:
                pred_bdp = element_select("B″ element", "pred_bdp")
            with adv2:
                pred_bdpn = st.number_input("B″ ratio", value=0.0, min_value=0.0, step=0.5, key="pred_bdpn")

        predict_clicked = st.button("🔮 Predict bubble & purity", type="primary")

    if predict_clicked:
        pred_row = make_single_prediction_row(
            atomic, en_table, pred_phase, pred_a, pred_an, pred_ap, pred_apn,
            pred_b, pred_bn, pred_bp, pred_bpn, pred_bdp, pred_bdpn, pred_on, radii,
        )
        formula = pred_row.iloc[0]["Formula"]

        bubble_features = build_feature_matrix(
            pred_row, use_phase=use_phase_in_ml, expected_columns=ml_result["feature_columns"])
        bubble_prob = ml_result["model"].predict_proba(bubble_features)[0][1]

        out_cols = st.columns(2)
        with out_cols[0]:
            st.markdown(prob_card(formula, bubble_prob, "bubble"), unsafe_allow_html=True)
        with out_cols[1]:
            if purity_result.get("ok"):
                purity_features = build_feature_matrix(
                    pred_row, use_phase=False, expected_columns=purity_result["feature_columns"])
                pure_prob = purity_result["model"].predict_proba(purity_features)[0][1]
                st.markdown(prob_card(formula, pure_prob, "purity"), unsafe_allow_html=True)
            else:
                st.info("Purity model not available yet.")

        # Goldschmidt tolerance factor readout for the proposed compound.
        t_value = pred_row.iloc[0].get("Tolerance_factor")
        if t_value is not None and pd.notna(t_value):
            if 0.9 <= t_value <= 1.0:
                t_note = "in the ideal cubic perovskite window (0.9–1.0) ✅"
            elif t_value < 0.9:
                t_note = "below 0.9 — a distorted (orthorhombic/rhombohedral) structure is likely"
            else:
                t_note = "above 1.0 — hexagonal tendencies"
            st.info(f"⚖️ Goldschmidt tolerance factor **t = {t_value:.3f}** — {t_note}. "
                    "(Shannon radii with typical oxidation states — an approximation.)")

        st.toast(f"Prediction ready for {formula}", icon="🧪")
        st.caption("Hypotheses from past class data — not guarantees.")

        # Closest past experiments: ground the prediction in real class results.
        st.markdown("**Closest past experiments** — what happened when the class made similar compounds")
        neighbors = nearest_neighbors(described_df, pred_row, k=5)
        if neighbors.empty:
            st.caption("Not enough data to find similar compounds.")
        else:
            st.dataframe(neighbors, use_container_width=True, hide_index=True)
            st.caption("Similarity is measured across composition descriptors (lower distance = more alike).")

        pred_show = visible(pred_row)
        st.dataframe(pred_show[[c for c in ordered_columns(pred_show) if c in pred_show.columns]],
                     use_container_width=True)

    # --- What-if explorer: sweep one input, watch the probabilities move ---
    with st.expander("🔬 What-if explorer — sweep one input and watch the prediction change"):
        st.caption("Uses the compound entered above, varying ONE value across a range. "
                   "Great for questions like “would more oxygen help?”")
        sweep_options = {
            "Oxygen ratio": ("on", 0.5, 8.0),
            "A ratio": ("an", 0.5, 4.0),
            "B ratio": ("bn", 0.5, 4.0),
            "A′ ratio": ("apn", 0.0, 2.0),
            "B′ ratio": ("bpn", 0.0, 2.0),
        }
        sweep_cols = st.columns([2, 1])
        with sweep_cols[0]:
            sweep_label = st.selectbox("Input to sweep", list(sweep_options.keys()), key="whatif_input")
        with sweep_cols[1]:
            run_sweep = st.button("Run sweep", type="primary", key="whatif_run")

        if run_sweep:
            which, lo, hi = sweep_options[sweep_label]
            base = {"a": pred_a, "an": pred_an, "ap": pred_ap, "apn": pred_apn,
                    "b": pred_b, "bn": pred_bn, "bp": pred_bp, "bpn": pred_bpn,
                    "bdp": pred_bdp, "bdpn": pred_bdpn, "on": pred_on}
            points = []
            with st.spinner("Sweeping…"):
                for value in [lo + (hi - lo) * i / 24 for i in range(25)]:
                    inputs = dict(base)
                    inputs[which] = value
                    row = make_single_prediction_row(
                        atomic, en_table, pred_phase,
                        inputs["a"], inputs["an"], inputs["ap"], inputs["apn"],
                        inputs["b"], inputs["bn"], inputs["bp"], inputs["bpn"],
                        inputs["bdp"], inputs["bdpn"], inputs["on"], radii,
                    )
                    feats = build_feature_matrix(
                        row, use_phase=use_phase_in_ml, expected_columns=ml_result["feature_columns"])
                    point = {"value": value,
                             "bubble_prob": float(ml_result["model"].predict_proba(feats)[0][1])}
                    if purity_result.get("ok"):
                        p_feats = build_feature_matrix(
                            row, use_phase=False, expected_columns=purity_result["feature_columns"])
                        point["purity_prob"] = float(purity_result["model"].predict_proba(p_feats)[0][1])
                    points.append(point)
            st.session_state["whatif_result"] = {
                "sweep": pd.DataFrame(points), "label": sweep_label, "current": float(base[which]),
            }

        if st.session_state.get("whatif_result"):
            res = st.session_state["whatif_result"]
            st.plotly_chart(plot_what_if(res["sweep"], res["label"], res["current"]),
                            use_container_width=True, config=PLOTLY_CONFIG)
            st.caption("Model hypotheses, not measurements — flat lines mean the model doesn't "
                       "think that input matters; sharp jumps show decision boundaries it learned.")


def render_add_compound_tab(atomic: pd.DataFrame, en_table: pd.DataFrame, radii: pd.DataFrame) -> None:
    """Tab 6 — build and stage a new compound by hand for this session."""

    st.caption("Enter one compound part by part. The app rebuilds the formula and checks it.")

    symbols = valid_element_symbols(atomic)
    optional_symbols = [""] + symbols

    def element_select(label: str, key: str, default: str = "", optional: bool = True):
        options = optional_symbols if optional else symbols
        index = options.index(default) if default in options else 0
        return st.selectbox(label, options, index=index, key=key,
                            format_func=lambda s: s if s else "— none —")

    with st.form("add_compound_form"):
        m1, m2, m3 = st.columns(3)
        with m1:
            new_group = st.text_input("Group Number / ID", value="New Group", key="add_group")
            new_semester = st.text_input("Semester / Year", value="Spring 2026", key="add_semester")
        with m2:
            new_instructor = st.text_input("Instructor / Section", value="", key="add_instructor")
            new_phase = st.selectbox("Phase", ["pure", "impure"], key="add_phase")
        with m3:
            new_bub = st.selectbox("Bubble response", ["yes", "no", "maybe"], key="add_bub")

        st.markdown("#### Formula parts")
        a_col, b_col, o_col = st.columns(3)
        with a_col:
            new_a = element_select("A-site element", "add_a", default="La", optional=False)
            new_an = st.number_input("A-site ratio", value=2.0, min_value=0.0, step=0.5, key="add_an")
            new_ap = element_select("A′-site element", "add_ap")
            new_apn = st.number_input("A′ ratio", value=0.0, min_value=0.0, step=0.5, key="add_apn")
        with b_col:
            new_b = element_select("B-site element", "add_b", default="Ni", optional=False)
            new_bn = st.number_input("B-site ratio", value=1.0, min_value=0.0, step=0.5, key="add_bn")
            new_bp = element_select("B′-site element", "add_bp")
            new_bpn = st.number_input("B′ ratio", value=0.0, min_value=0.0, step=0.5, key="add_bpn")
        with o_col:
            new_bdp = element_select("B″-site element", "add_bdp")
            new_bdpn = st.number_input("B″ ratio", value=0.0, min_value=0.0, step=0.5, key="add_bdpn")
            new_on = st.number_input("Oxygen ratio", value=4.0, min_value=0.0, step=0.5, key="add_on")

        submitted = st.form_submit_button("Build and check compound", type="primary")

    if submitted:
        new_raw = pd.DataFrame([{
            "GroupNumber": new_group, "Email": "", "Name": "", "Members": "",
            "Instructor": new_instructor, "Semester": new_semester, "SourceRow": "manual",
            "Slot": 1, "FormulaRef": "",
            "A": new_a, "AN": new_an, "AP": new_ap, "APN": new_apn,
            "B": new_b, "BN": new_bn, "BP": new_bp, "BPN": new_bpn,
            "BDP": new_bdp, "BDPN": new_bdpn, "O": "O", "ON": new_on,
            "P": new_phase, "Bub": new_bub,
        }])
        new_clean = clean_and_encode_data(new_raw, atomic)
        new_desc = add_chemical_descriptors(new_clean, atomic, en_table, radii)
        st.session_state.pending_manual_entry = new_desc
        st.session_state.pending_manual_issues = validate_compound_rows(new_desc, atomic)

    if not st.session_state.pending_manual_entry.empty:
        pending = st.session_state.pending_manual_entry
        pending_issues = st.session_state.pending_manual_issues
        formula = pending.iloc[0]["Formula"]

        st.markdown(f"#### Preview: {formula_html(formula)}", unsafe_allow_html=True)
        if pending_issues.empty:
            st.success("This entry passed validation.")
        else:
            st.warning("This entry has issues to review before adding.")
            st.dataframe(pending_issues, use_container_width=True)
        pending_show = visible(pending)
        st.dataframe(pending_show[ordered_columns(pending_show)], use_container_width=True)

        add_cols = st.columns([1, 1, 2])
        with add_cols[0]:
            if st.button("Add to session dataset", type="primary", key="confirm_add_manual_entry"):
                st.session_state.manual_entries = pd.concat(
                    [st.session_state.manual_entries, pending], ignore_index=True)
                st.session_state.pending_manual_entry = pd.DataFrame()
                st.session_state.pending_manual_issues = pd.DataFrame()
                st.toast("Compound added to this session.", icon="➕")
                st.rerun()
        with add_cols[1]:
            if st.button("Discard preview", key="discard_manual_preview"):
                st.session_state.pending_manual_entry = pd.DataFrame()
                st.session_state.pending_manual_issues = pd.DataFrame()
                st.rerun()
        with add_cols[2]:
            st.download_button("Download this entry as CSV", dataframe_to_csv_bytes(pending),
                               file_name=f"{formula or 'new_compound'}_entry.csv",
                               mime="text/csv", key="download_single_pending")

    if not st.session_state.manual_entries.empty:
        st.markdown("#### Added this session")
        manual_show = visible(st.session_state.manual_entries)
        st.dataframe(manual_show[ordered_columns(manual_show)], use_container_width=True)
        if st.button("Clear session-added entries", key="clear_session_entries"):
            st.session_state.manual_entries = pd.DataFrame()
            st.rerun()


def render_export_tab(described_df: pd.DataFrame, issues_df: pd.DataFrame, outlier_df: pd.DataFrame,
                      use_phase_in_ml: bool, use_mass_in_ml: bool) -> None:
    """Tab 7 — download cleaned data, reports, and the one-click class report."""

    st.caption("Download analysis-ready data and reports. Your original file is never changed.")

    # --- One-click class report (aggregates only — safe to share) ---
    st.markdown("#### 📄 Class report")
    st.caption("A single print-ready HTML file: dataset stats, response counts, strongest "
               "correlations, cation mixing, element summaries, and both models with "
               "cross-validation. Open it in a browser and print to PDF if needed.")

    plot_df = described_df.copy()
    phase_src = "P_raw" if "P_raw" in plot_df.columns else "P"
    bub_src = "Bub_raw" if "Bub_raw" in plot_df.columns else "Bub"
    plot_df["BubblePlotLabel"] = plot_df[bub_src].apply(lambda v: display_label(normalize_bubble_label(v)))
    plot_df["PhasePlotLabel"] = plot_df[phase_src].apply(lambda v: display_label(normalize_phase_label(v)))

    corr = numeric_correlation_table(described_df)
    ml = train_ml_model(described_df, use_phase=use_phase_in_ml, use_mass=use_mass_in_ml)
    purity = train_purity_model(described_df, use_mass=use_mass_in_ml)

    n_semesters = described_df.get("Semester", pd.Series(dtype=str)).dropna().astype(str).str.strip().replace("", pd.NA).dropna().nunique()
    stats = {
        "generated": pd.Timestamp.now().strftime("%B %d, %Y at %H:%M"),
        "n_compounds": len(described_df),
        "n_semesters": int(n_semesters),
        "bubble_rate": f"{described_df['BubbleYes'].mean():.0%}" if "BubbleYes" in described_df.columns and len(described_df) else "—",
        "pure_rate": f"{(described_df['PhaseN'] == 2).mean():.0%}" if "PhaseN" in described_df.columns and len(described_df) else "—",
        "n_issues": int(len(issues_df)),
        "mass_note": "included" if use_mass_in_ml else "excluded (default)",
    }
    report_bytes = build_html_report(
        stats,
        label_count_table(plot_df, "BubblePlotLabel", kind="bubble",
                          preferred_order=["Yes", "No", "Maybe", "Other / needs review", "Missing"]),
        label_count_table(plot_df, "PhasePlotLabel", kind="phase",
                          preferred_order=["Pure", "Impure", "Not made", "Other / needs review", "Missing"]),
        bubble_relationships(corr) if not corr.empty else None,
        mixing_summary(described_df),
        summarize_by_element(described_df, "A", min_rows=3).round(1),
        summarize_by_element(described_df, "B", min_rows=3).round(1),
        ml if ml.get("ok") else None,
        purity if purity.get("ok") else None,
    )
    st.download_button("📄 Download class report (HTML)", report_bytes,
                       file_name=f"chem120_class_report_{pd.Timestamp.now():%Y-%m-%d}.html",
                       mime="text/html", type="primary")

    st.divider()

    export_cols = st.columns(2)
    with export_cols[0]:
        st.markdown("#### Cleaned data")
        st.download_button("Cleaned data (CSV)",
                           dataframe_to_csv_bytes(described_df[ordered_columns(described_df)]),
                           file_name="chem120_cleaned_compound_data.csv", mime="text/csv")
        st.download_button("Cleaned data (Excel)",
                           dataframe_to_excel_bytes(described_df[ordered_columns(described_df)], "Cleaned_Data"),
                           file_name="chem120_cleaned_compound_data.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        no_contact = public_view(described_df)
        st.download_button("Cleaned data — no contact info (CSV)",
                           dataframe_to_csv_bytes(no_contact[ordered_columns(no_contact)]),
                           file_name="chem120_cleaned_public.csv", mime="text/csv",
                           help="Same data with emails, names, and member lists removed — safe to share.")
    with export_cols[1]:
        st.markdown("#### Reports")
        st.download_button("Validation report (CSV)", dataframe_to_csv_bytes(issues_df),
                           file_name="chem120_validation_report.csv", mime="text/csv")
        st.download_button("Outlier report (CSV)", dataframe_to_csv_bytes(outlier_df),
                           file_name="chem120_outlier_report.csv", mime="text/csv")

    st.caption("🔒 The full downloads keep contact/group info for instructor records — the app itself "
               "never displays it. Use the no-contact-info file for anything student-facing.")


def render_new_semester_tab(long_df: pd.DataFrame) -> None:
    """Tab 8 — merge a new semester's raw export into the current dataset."""

    st.caption("Upload a new semester's raw Excel; the app expands it to one row per compound and merges it.")

    new_sem_file = st.file_uploader(
        "New Semester Excel file", type=["xlsx", "xlsm", "xls"],
        help="Raw export where each row is one group and compounds are in columns like 1A, 1AN, 2A, 2AN…",
        key="new_semester_upload",
    )
    if new_sem_file is None:
        st.info("Upload the new semester file to get started.")
        return

    try:
        new_raw = read_table_from_bytes(new_sem_file.getvalue(), new_sem_file.name)
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        return

    new_long, new_warnings = normalize_to_long_format(new_raw)
    if new_warnings:
        with st.expander("⚠️ Column mapping problems in new file", expanded=True):
            st.warning("Some required columns could not be matched. Check column names (1A, 1AN, 1B, 1BN…). "
                       "See DEVELOPER_NOTES.md.")
            for w in new_warnings:
                st.markdown(f"- {w}")

    if new_long.empty:
        st.error("No compound rows could be extracted. Check the column names.")
        return

    stat_cols = st.columns(3)
    with stat_cols[0]:
        st.metric("Groups in new file", f"{len(new_raw):,}")
    with stat_cols[1]:
        st.metric("Compound rows extracted", f"{len(new_long):,}")
    with stat_cols[2]:
        st.metric("Existing compound rows", f"{len(long_df):,}")

    st.markdown("#### Preview — first 10 new rows")
    preview_cols = [c for c in ["GroupNumber", "Semester", "Slot", "A", "AN", "B", "BN", "O", "ON", "P", "Bub"] if c in new_long.columns]
    st.dataframe(new_long[preview_cols].head(10), use_container_width=True)

    existing_semesters = set(long_df["Semester"].dropna().astype(str).unique())
    new_semesters = set(new_long["Semester"].dropna().astype(str).unique())
    overlap = existing_semesters & new_semesters
    if overlap:
        st.warning(f"New file contains semester(s) already present: **{', '.join(sorted(overlap))}**. "
                   "Check for duplicate data.")
    else:
        st.success(f"No semester overlap. New semesters: **{', '.join(sorted(new_semesters)) or 'unknown'}**")

    # Row-level duplicate detection (same group + semester + composition + outcome).
    duplicates = find_merge_duplicates(long_df, new_long)
    exclude_dups = False
    if not duplicates.empty:
        st.warning(f"⚠️ **{len(duplicates)} row(s)** in the new file exactly match rows already "
                   "in the dataset (same group, semester, composition, and outcomes).")
        with st.expander("Show duplicate rows"):
            dup_cols = [c for c in ["GroupNumber", "Semester", "Slot", "A", "AN", "B", "BN", "ON", "P", "Bub"]
                        if c in duplicates.columns]
            st.dataframe(duplicates[dup_cols], use_container_width=True, hide_index=True)
        exclude_dups = st.checkbox("Exclude these duplicates from the merged download", value=True)
    else:
        st.caption("✅ No exact duplicate rows detected between the new file and the existing data.")

    new_to_merge = new_long.drop(index=duplicates.index) if (exclude_dups and not duplicates.empty) else new_long
    merged = pd.concat([long_df, new_to_merge], ignore_index=True)
    st.markdown(f"#### Merged dataset — {len(merged):,} total rows")
    st.download_button("⬇️ Download merged Combined_Data.xlsx",
                       dataframe_to_excel_bytes(merged, "Combined_Data"),
                       file_name="Combined_Data_merged.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       type="primary")
    st.caption("Replace data/Combined_Data.xlsx with this file to load the merged dataset next startup.")


def render_landing() -> None:
    """Shown before any data is loaded: hero + how-to cards."""

    render_hero()
    st.markdown(
        """
        <div class="glass-card" style="text-align:center; padding:2rem 1.6rem;">
            <div class="big-number">Load your class data to begin</div>
            <div class="helper-text" style="margin-top:0.5rem;">
                Use <b>Data</b> in the sidebar — upload a CHEM 120 spreadsheet (CSV or Excel),
                or tick <b>"Load the bundled class dataset"</b> to explore right away.
            </div>
        </div>
        <div class="feat-grid">
            <div class="feat-card">
                <div class="feat-icon">🧹</div>
                <div class="feat-title">Clean &amp; check</div>
                <div class="helper-text">Fix invalid elements with one click, catch missing ratios,
                and rebuild every formula automatically.</div>
            </div>
            <div class="feat-card">
                <div class="feat-icon">🌡️</div>
                <div class="feat-title">Explore &amp; correlate</div>
                <div class="helper-text">Interactive heatmaps, element summaries, cation-mixing views,
                and a PCA structure map — hover anything for details.</div>
            </div>
            <div class="feat-card">
                <div class="feat-icon">🔮</div>
                <div class="feat-title">Predict</div>
                <div class="helper-text">Train bubble and purity models on class data, then test your
                own proposed compound — mixed A′/B′ sites included.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# STREAMLIT APP
# =============================================================================

def main() -> None:
    """Load data, then render every part as a tab."""

    st.set_page_config(page_title=APP_TITLE, page_icon="🧪", layout="wide",
                       initial_sidebar_state="expanded")
    inject_css()

    # ----- Sidebar: data source + controls -----
    st.sidebar.markdown("## 🧪 CHEM 120")
    st.sidebar.caption("Upload class data, check it, then explore patterns.")

    st.sidebar.markdown('<div class="sb-section">Data</div>', unsafe_allow_html=True)
    uploaded_data = st.sidebar.file_uploader(
        "Class spreadsheet", type=["csv", "xlsx", "xlsm", "xls"],
        help="Upload Combined_Data.xlsx or another CHEM 120 survey/database file.",
    )
    uploaded_atomic = None
    uploaded_en = None

    default_path = find_default_database()
    use_sample_data = st.sidebar.checkbox(
        "Load the bundled class dataset", value=False,
        help="Loads the dataset shipped with the app (data/Combined_Data.xlsx) so you can explore without uploading.",
    )
    if default_path is not None:
        st.sidebar.caption(f"Bundled file: `{default_path.name}`")

    st.sidebar.markdown('<div class="sb-section">Model settings</div>', unsafe_allow_html=True)
    use_phase_in_ml = st.sidebar.toggle(
        "Use phase in bubble model", value=True,
        help="Turn off to make the bubble model ignore pure/impure labels.",
    )
    use_mass_in_ml = st.sidebar.toggle(
        "Include mass descriptors", value=False,
        help="OFF by default: atomic mass and atomic number are highly correlated, so mass features "
             "dominated the models (instructor feedback). Turn on to compare.",
    )

    st.sidebar.markdown('<div class="sb-section">Quality checks</div>', unsafe_allow_html=True)
    z_threshold = st.sidebar.slider(
        "Outlier sensitivity", min_value=2.0, max_value=5.0, value=3.0, step=0.25,
        help="Lower flags more possible outliers; higher flags only extreme values.",
    )

    # ----- Instructor mode: unlocks the hidden contact/group columns -----
    with st.sidebar.expander("🔑 Instructor mode", expanded=False):
        if instructor_mode_on():
            st.success("Unlocked — contact columns are visible.")
            if st.button("Lock again", key="instructor_lock"):
                st.session_state.instructor_mode = False
                st.rerun()
        else:
            passcode = st.text_input("Passcode", type="password", key="instructor_pass")
            if st.button("Unlock", key="instructor_unlock"):
                if passcode == INSTRUCTOR_PASSCODE:
                    st.session_state.instructor_mode = True
                    st.rerun()
                else:
                    st.error("Wrong passcode.")
            st.caption("Reveals emails/names in-app for the teaching team. "
                       "The passcode is set in pipeline.py (INSTRUCTOR_PASSCODE).")

    # ----- Reference tables (atomic mass, electronegativity, ionic radii) -----
    try:
        atomic = load_atomic_table_from_bytes(
            uploaded_atomic.getvalue() if uploaded_atomic else None,
            uploaded_atomic.name if uploaded_atomic else "")
        en_table = load_en_table_from_bytes(
            uploaded_en.getvalue() if uploaded_en else None,
            uploaded_en.name if uploaded_en else "")
        radii = load_radii_table_from_bytes(None)
    except Exception as exc:
        st.error(f"Reference table error: {exc}")
        st.stop()

    # ----- Blank boot: nothing loads until upload or the bundled set is chosen -----
    if uploaded_data is None and not use_sample_data:
        render_landing()
        st.stop()

    # ----- Load the class data (uploaded file wins over the bundled set) -----
    try:
        if uploaded_data is not None:
            raw_df = read_table_from_bytes(uploaded_data.getvalue(), uploaded_data.name)
        else:
            raw_df = read_local_table(str(default_path)) if default_path is not None else make_demo_dataset()
    except Exception as exc:
        st.error(f"Could not load class data: {exc}")
        st.stop()

    # ----- Run the data pipeline (each step is cached for speed) -----
    long_df, column_warnings = normalize_to_long_format(raw_df)

    # Apply any "Did you mean?" corrections the user chose in the Check tab, then
    # work out which invalid elements still need a decision.
    st.session_state.setdefault("element_fixes", {})
    long_df, corrections_log = apply_element_fixes(long_df, st.session_state.element_fixes)
    fix_proposals = propose_element_fixes(long_df, atomic)

    clean_df = clean_and_encode_data(long_df, atomic)
    described_df = add_chemical_descriptors(clean_df, atomic, en_table, radii)

    # Session-staged manual rows (from the Add Compound tab).
    st.session_state.setdefault("manual_entries", pd.DataFrame())
    st.session_state.setdefault("pending_manual_entry", pd.DataFrame())
    st.session_state.setdefault("pending_manual_issues", pd.DataFrame())
    if not st.session_state.manual_entries.empty:
        described_df = pd.concat([described_df, st.session_state.manual_entries], ignore_index=True)

    # Remove rows whose elements aren't real periodic-table symbols (e.g. "BATU").
    described_df, cleaning_log = remove_invalid_element_rows(described_df, atomic)

    issues_df = validate_compound_rows(described_df, atomic)
    outlier_df = detect_numeric_outliers(described_df, z_threshold=z_threshold)

    # ----- Hero with live dataset chips -----
    n_semesters = described_df.get("Semester", pd.Series(dtype=str)).dropna().astype(str).str.strip().replace("", pd.NA).dropna().nunique()
    n_elements = pd.unique(
        pd.concat([described_df.get(c, pd.Series(dtype=str)) for c in ["A", "AP", "B", "BP", "BDP"]])
        .dropna().astype(str).str.strip()
    )
    n_elements = len([e for e in n_elements if e])
    render_hero(chips=[
        f"⚗️ {len(described_df):,} compounds",
        f"📅 {n_semesters} semester{'s' if n_semesters != 1 else ''}",
        f"🧬 {n_elements} elements in play",
        "🔓 Instructor mode" if instructor_mode_on() else "🔒 Contact info hidden app-wide",
    ])

    if column_warnings:
        with st.expander("⚠️ Column mapping problems — expand for details", expanded=True):
            st.warning("One or more required columns could not be matched — usually the form wording changed. "
                       "See DEVELOPER_NOTES.md → When the Google Form changes.")
            for w in column_warnings:
                st.markdown(f"- {w}")

    # ----- KPI row -----
    bubble_rate = described_df["BubbleYes"].mean() if "BubbleYes" in described_df.columns and len(described_df) else 0.0
    pure_rate = (described_df["PhaseN"] == 2).mean() if "PhaseN" in described_df.columns and len(described_df) else 0.0
    student_issues = issues_df[issues_df["Field"] != "Email"] if ("Field" in issues_df.columns and not issues_df.empty) else issues_df

    kpis = st.columns(4)
    kpis[0].markdown(kpi_card("⚗️", f"{len(described_df):,}", "compounds loaded",
                              "kpi-ok" if len(described_df) else "kpi-bad", 0.0), unsafe_allow_html=True)
    kpis[1].markdown(kpi_card("🫧", f"{bubble_rate:.0%}", "bubbled (yes)", "", 0.06), unsafe_allow_html=True)
    kpis[2].markdown(kpi_card("💎", f"{pure_rate:.0%}", "came out pure", "", 0.12), unsafe_allow_html=True)
    kpis[3].markdown(kpi_card("🩺", f"{len(student_issues):,}", "data issues to review",
                              "kpi-ok" if len(student_issues) == 0 else "kpi-warn", 0.18), unsafe_allow_html=True)

    # ----- Everything lives in tabs (no long scroll) -----
    (tab_check, tab_explore, tab_heatmap, tab_structure, tab_ml,
     tab_add, tab_export, tab_semester) = st.tabs(
        ["✅ Check", "📊 Explore", "🌡️ Heatmap", "🧭 Structure",
         "🔮 Predict", "➕ Add", "⬇️ Export", "📥 New Semester"]
    )
    with tab_check:
        render_check_tab(described_df, issues_df, outlier_df, cleaning_log,
                         fix_proposals, corrections_log, valid_element_symbols(atomic))
    with tab_explore:
        render_explore_tab(described_df)
    with tab_heatmap:
        render_heatmap_tab(described_df)
    with tab_structure:
        render_pca_tab(described_df)
    with tab_ml:
        render_ml_tab(described_df, atomic, en_table, radii, use_phase_in_ml, use_mass_in_ml)
    with tab_add:
        render_add_compound_tab(atomic, en_table, radii)
    with tab_export:
        render_export_tab(described_df, issues_df, outlier_df, use_phase_in_ml, use_mass_in_ml)
    with tab_semester:
        render_new_semester_tab(long_df)


if __name__ == "__main__":
    main()
# build: 2026-07-01 revamp
