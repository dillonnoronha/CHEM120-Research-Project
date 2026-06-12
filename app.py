"""
CHEM 120 Catalyst Insight Studio
=================================

Streamlit UI entry point. All data loading, cleaning, validation, descriptors,
plotting, and ML logic lives in pipeline.py. This file only handles the page
layout, widgets, and calls into pipeline.

Run locally:
    py -m pip install -r requirements.txt
    py -m streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pipeline import (
    APP_TITLE,
    APP_SUBTITLE,
    add_chemical_descriptors,
    bubble_relationships,
    build_feature_matrix,
    chi_squared_tests,
    clean_and_encode_data,
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    detect_numeric_outliers,
    display_label,
    find_default_database,
    load_atomic_table_from_bytes,
    load_en_table_from_bytes,
    make_demo_dataset,
    make_single_prediction_row,
    normalize_bubble_label,
    normalize_phase_label,
    normalize_to_long_format,
    numeric_correlation_table,
    ordered_columns,
    plot_bar_chart,
    plot_heatmap,
    read_local_table,
    read_table_from_bytes,
    split_quarantine,
    summarize_by_element,
    train_classification_model,
    validate_compound_rows,
)


# =============================================================================
# PAGE STYLE
# =============================================================================

def inject_css() -> None:
    """Inject CSS for a cleaner, modern, student-friendly interface."""

    st.markdown(
        """
        <style>
        :root {
            --chem-card-bg: rgba(255, 255, 255, 0.82);
            --chem-border: rgba(0, 0, 0, 0.08);
            --chem-muted: #6e6e73;
            --chem-ink: #1d1d1f;
            --chem-soft: #f5f5f7;
            --chem-blue: #0071e3;
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(0, 113, 227, 0.06), transparent 40rem),
                linear-gradient(180deg, #fafafc 0%, #ffffff 60%);
        }

        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1220px;
        }

        h1, h2, h3 {
            letter-spacing: -0.03em;
        }

        h3 {
            margin-top: 0.6rem;
        }

        div[data-testid="stSidebarContent"] {
            background: linear-gradient(180deg, #fbfbfd 0%, #f5f5f7 100%);
        }

        div[data-testid="stSidebarContent"] p,
        div[data-testid="stSidebarContent"] span,
        div[data-testid="stSidebarContent"] label,
        div[data-testid="stSidebarContent"] div,
        div[data-testid="stSidebarContent"] h1,
        div[data-testid="stSidebarContent"] h2,
        div[data-testid="stSidebarContent"] h3,
        div[data-testid="stSidebarContent"] .stMarkdown,
        div[data-testid="stSidebarContent"] .stCaption,
        div[data-testid="stSidebarContent"] [data-testid="stWidgetLabel"] {
            color: #1d1d1f !important;
        }

        /* File uploader dropzone — force light background to match sidebar */
        div[data-testid="stSidebarContent"] [data-testid="stFileUploaderDropzone"],
        div[data-testid="stSidebarContent"] section[data-testid="stFileUploaderDropzone"] {
            background-color: #ffffff !important;
            border: 1.5px dashed #c7c7cc !important;
            border-radius: 12px !important;
        }

        div[data-testid="stSidebarContent"] [data-testid="stFileUploaderDropzone"] *,
        div[data-testid="stSidebarContent"] [data-testid="stFileUploaderDropzone"] span,
        div[data-testid="stSidebarContent"] [data-testid="stFileUploaderDropzone"] p,
        div[data-testid="stSidebarContent"] [data-testid="stFileUploaderDropzone"] small {
            color: #6e6e73 !important;
        }

        div[data-testid="stSidebarContent"] [data-testid="stFileUploaderDropzone"] button {
            background-color: #f5f5f7 !important;
            color: #1d1d1f !important;
            border: 1px solid #c7c7cc !important;
        }

        .hero {
            padding: 2rem 2.2rem;
            border: 1px solid var(--chem-border);
            border-radius: 28px;
            background:
                radial-gradient(circle at top left, rgba(0, 113, 227, 0.12), transparent 28%),
                linear-gradient(135deg, #ffffff 0%, #f5f5f7 100%);
            box-shadow: 0 18px 55px rgba(0,0,0,0.08);
            margin-bottom: 1.5rem;
        }

        .hero-title {
            font-size: 2.45rem;
            line-height: 1.05;
            font-weight: 760;
            color: var(--chem-ink);
            margin: 0;
        }

        .hero-subtitle {
            color: var(--chem-muted);
            font-size: 1.08rem;
            margin-top: 0.85rem;
            max-width: 860px;
        }

        .soft-card {
            padding: 1.1rem 1.2rem;
            border: 1px solid var(--chem-border);
            border-radius: 22px;
            background: var(--chem-card-bg);
            box-shadow: 0 8px 28px rgba(0,0,0,0.045);
            margin-bottom: 0.9rem;
            color: var(--chem-ink);
        }

        .small-label {
            color: var(--chem-muted);
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.25rem;
        }

        .big-number {
            color: var(--chem-ink);
            font-size: 1.85rem;
            font-weight: 760;
            line-height: 1;
        }

        .helper-text {
            color: var(--chem-muted);
            font-size: 0.93rem;
            line-height: 1.45;
        }

        .step-pill {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: #eef5ff;
            color: #0057b8;
            font-size: 0.8rem;
            font-weight: 700;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
        }

        .status-ok {
            border-left: 4px solid #30d158;
        }

        .status-warn {
            border-left: 4px solid #ff9f0a;
        }

        .status-bad {
            border-left: 4px solid #ff453a;
        }

        div[data-testid="stMetricValue"] {
            font-weight: 760;
            letter-spacing: -0.04em;
        }

        /* Tabs — pill style. Top-level tabs are larger; nested tabs inherit a
           slightly smaller version automatically via the same rules. */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            padding: 0.3rem 0;
            border-bottom: 1px solid var(--chem-border);
            margin-bottom: 0.6rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.7rem 1.25rem;
            background-color: #f5f5f7;
            color: #1d1d1f !important;
            font-weight: 600;
            transition: background-color 0.15s ease;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: #ececf0;
        }

        .stTabs [aria-selected="true"] {
            background-color: #0071e3 !important;
            color: #ffffff !important;
            box-shadow: 0 6px 18px rgba(0, 113, 227, 0.28);
        }

        /* Nested tabs (inside ML Lab & Tools) — quieter selected state */
        .stTabs .stTabs [aria-selected="true"] {
            background-color: #e8f1ff !important;
            color: #0057b8 !important;
            box-shadow: none;
        }

        .stTabs [data-baseweb="tab"] p {
            color: inherit !important;
        }

        .stTabs [data-baseweb="tab-highlight"],
        .stTabs [data-baseweb="tab-border"] {
            display: none;
        }

        /* Buttons — primary = solid blue pill, secondary = quiet pill */
        div.stButton > button[kind="primary"],
        div.stDownloadButton > button[kind="primary"] {
            border-radius: 999px !important;
            background: var(--chem-blue) !important;
            color: #ffffff !important;
            border: none !important;
            padding: 0.6rem 1.3rem !important;
            font-weight: 650 !important;
            box-shadow: 0 8px 22px rgba(0, 113, 227, 0.25);
        }

        div.stButton > button[kind="secondary"],
        div.stDownloadButton > button[kind="secondary"] {
            border-radius: 999px !important;
            background: #f5f5f7 !important;
            color: #1d1d1f !important;
            border: 1px solid #d2d2d7 !important;
            padding: 0.6rem 1.3rem !important;
            font-weight: 600 !important;
        }

        /* Dataframes and expanders — soft rounded borders to match the cards */
        [data-testid="stDataFrame"] {
            border: 1px solid var(--chem-border);
            border-radius: 14px;
            overflow: hidden;
        }

        [data-testid="stExpander"] {
            border: 1px solid var(--chem-border) !important;
            border-radius: 16px !important;
            background: rgba(255, 255, 255, 0.7);
        }

        hr {
            border: none;
            border-top: 1px solid var(--chem-border);
            margin: 1.6rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# UI COMPONENTS
# =============================================================================

def metric_card(label: str, value: object, help_text: str = "") -> None:
    """Display a small Apple-style metric card."""

    st.markdown(
        f"""
        <div class="soft-card">
            <div class="small-label">{label}</div>
            <div class="big-number">{value}</div>
            <div class="helper-text">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# STREAMLIT APP
# =============================================================================

def main() -> None:
    """Run the Streamlit app."""

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    # -------------------------------------------------------------------------
    # Sidebar: upload files and select basic settings.
    # -------------------------------------------------------------------------
    st.sidebar.markdown("## 🧪 CHEM 120")
    st.sidebar.caption("Upload class data, check it, then explore patterns.")

    st.sidebar.markdown("### 1. Upload")
    uploaded_data = st.sidebar.file_uploader(
        "Class spreadsheet",
        type=["csv", "xlsx", "xlsm", "xls"],
        help="Upload Combined_Data.xlsx or another CHEM 120 survey/database file.",
    )
    uploaded_atomic = None
    uploaded_en = None

    st.sidebar.markdown("### 2. Controls")
    z_threshold = st.sidebar.slider(
        "Outlier sensitivity",
        min_value=2.0,
        max_value=5.0,
        value=3.0,
        step=0.25,
        help="Lower values flag more possible outliers. Higher values only flag very extreme values.",
    )

    st.sidebar.markdown("### 3. Quick help")
    st.sidebar.info(
        "Best workflow: upload data → check errors → explore charts → use ML as a hypothesis tool → export cleaned results."
    )

    # -------------------------------------------------------------------------
    # Hero header.
    # -------------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">{APP_TITLE}</div>
            <div class="hero-subtitle">{APP_SUBTITLE}</div>
            <div style="margin-top:1rem;">
                <span class="step-pill">1 Upload</span>
                <span class="step-pill">2 Check</span>
                <span class="step-pill">3 Explore</span>
                <span class="step-pill">4 Hypothesize</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Load reference tables.
    # -------------------------------------------------------------------------
    try:
        atomic_bytes = uploaded_atomic.getvalue() if uploaded_atomic is not None else None
        atomic_name = uploaded_atomic.name if uploaded_atomic is not None else ""
        atomic = load_atomic_table_from_bytes(atomic_bytes, atomic_name)

        en_bytes = uploaded_en.getvalue() if uploaded_en is not None else None
        en_name = uploaded_en.name if uploaded_en is not None else ""
        en_table = load_en_table_from_bytes(en_bytes, en_name)
    except Exception as exc:
        st.error(f"Reference table error: {exc}")
        st.stop()

    # -------------------------------------------------------------------------
    # Load class data. Uploaded file has priority, then local default, then demo.
    # -------------------------------------------------------------------------
    try:
        if uploaded_data is not None:
            raw_df = read_table_from_bytes(uploaded_data.getvalue(), uploaded_data.name)
        else:
            default_path = find_default_database()
            if default_path is not None:
                raw_df = read_local_table(str(default_path))
            else:
                raw_df = make_demo_dataset()
    except Exception as exc:
        st.error(f"Could not load class data: {exc}")
        st.stop()

    # -------------------------------------------------------------------------
    # Run the main data pipeline. Each step is cached for speed.
    # -------------------------------------------------------------------------
    long_df, column_warnings = normalize_to_long_format(raw_df)
    if column_warnings:
        with st.expander("⚠️ Column mapping problems — expand to see details", expanded=True):
            st.warning(
                "The app could not match one or more required columns in your file. "
                "This usually means the Google Form question wording changed. "
                "See **DEVELOPER_NOTES.md → When the Google Form changes** for fix instructions."
            )
            for w in column_warnings:
                st.markdown(f"- {w}")
    clean_df = clean_and_encode_data(long_df, atomic)
    described_df = add_chemical_descriptors(clean_df, atomic, en_table)

    # Add manually staged rows from the Add Compound tab.
    if "manual_entries" not in st.session_state:
        st.session_state.manual_entries = pd.DataFrame()
    if "pending_manual_entry" not in st.session_state:
        st.session_state.pending_manual_entry = pd.DataFrame()
    if "pending_manual_issues" not in st.session_state:
        st.session_state.pending_manual_issues = pd.DataFrame()

    if not st.session_state.manual_entries.empty:
        described_df = pd.concat([described_df, st.session_state.manual_entries], ignore_index=True)

    # Validate against all rows first so the report reflects the full upload.
    issues_df = validate_compound_rows(described_df, atomic)

    # Exclude structurally broken rows (missing/unknown element, missing or
    # invalid required ratio) from analysis. These are invalid data, not a
    # judgment call, so they never enter the charts or ML.
    described_df, _excluded_df = split_quarantine(described_df, atomic)

    outlier_df = detect_numeric_outliers(described_df, z_threshold=z_threshold)

    # -------------------------------------------------------------------------
    # Data health cards.
    # -------------------------------------------------------------------------
    health_cols = st.columns(2)
    with health_cols[0]:
        style = "status-ok" if len(described_df) > 0 else "status-bad"
        st.markdown(
            f"""<div class="soft-card {style}">
            <b>Loaded compounds</b><br>
            <span class="helper-text">{len(described_df):,} compound rows are ready for analysis.</span>
            </div>""",
            unsafe_allow_html=True,
        )
    with health_cols[1]:
        style = "status-ok" if len(issues_df) == 0 else "status-warn"
        st.markdown(
            f"""<div class="soft-card {style}">
            <b>Validation report</b><br>
            <span class="helper-text">{len(issues_df):,} possible data-entry issues found. See Check Data below.</span>
            </div>""",
            unsafe_allow_html=True,
        )

    with st.expander("What do A-site, B-site, phase, and bubble mean?"):
        st.markdown(
            """
            - **A-site element**: the first main element position in the perovskite-style formula.
            - **B-site element**: the main element position before oxygen, often a transition metal.
            - **Phase**: whether the sample was recorded as `pure` or `impure`.
            - **Bubble response**: whether visible bubbles were observed: `yes`, `no`, or `maybe`.
            - **Descriptors**: calculated numbers such as atomic mass, atomic number, and formula mass.
            """
        )

    st.divider()

    main_tabs = st.tabs(["✅ Check Data", "📊 Explore & Relationships", "🤖 ML Lab & Tools"])

    with main_tabs[0]:
        # =========================================================================
        # SECTION 1: CHECK DATA
        # =========================================================================
        st.subheader("✅ Check Data")
        st.write(
            "Review data-entry mistakes before using the dataset for graphs or machine learning."
        )

        check_cols = st.columns([1, 1])
        with check_cols[0]:
            st.markdown("#### Validation issues")
            if issues_df.empty:
                st.success("No validation issues found.")
            else:
                n_critical = int((issues_df["Severity"] == "Critical").sum()) if "Severity" in issues_df.columns else 0
                n_minor = len(issues_df) - n_critical
                st.warning(
                    f"{len(issues_df):,} issues found — **{n_critical:,} critical**, {n_minor:,} minor. "
                    "Critical issues mean a row's data is broken; minor issues are cosmetic."
                )
                st.dataframe(issues_df, use_container_width=True, height=300)
        with check_cols[1]:
            st.markdown("#### Possible numeric outliers")
            if outlier_df.empty:
                st.success("No numeric outliers found at the current sensitivity.")
            else:
                st.info("Outliers are not automatically wrong. They are values worth double-checking.")
                st.dataframe(outlier_df, use_container_width=True, height=300)

        st.markdown("#### Cleaned compound table")
        st.caption(
            "Note: rows with **critical** errors above (missing/unknown element or a missing/invalid "
            "required ratio) are automatically excluded from the charts and ML models. They still appear "
            "in the validation report so you can fix them at the source. Rows with only minor issues are kept."
        )
        if described_df.empty:
            st.warning("No compound rows found. Check whether the uploaded file uses recognizable column names like 1A, 1AN, 1B, 1BN, 1O, 1ON, 1P, and 1Bub.")
        else:
            st.dataframe(described_df[ordered_columns(described_df)], use_container_width=True, height=380)

        with st.expander("How to fix common issues"):
            st.markdown(
                """
                - Use **element symbols**, not full names: `La`, not `Lanthanum`.
                - Use **numbers only** for ratios: `2`, not `two`.
                - Oxygen should be entered as **O**.
                - Phase should be exactly **pure** or **impure**.
                - Bubble response should be exactly **yes**, **no**, or **maybe**.
                """
            )


    with main_tabs[1]:
        # =========================================================================
        # SECTION 2: EXPLORE RESULTS
        # =========================================================================
        st.subheader("📊 Explore Results")
        st.write(
            "Compare compounds and look for patterns. These graphs are descriptive — not proof of causation."
        )

        semesters = sorted([x for x in described_df.get("Semester", pd.Series(dtype=str)).dropna().astype(str).unique() if x.strip()])
        instructors = sorted([x for x in described_df.get("Instructor", pd.Series(dtype=str)).dropna().astype(str).unique() if x.strip()])

        filter_cols = st.columns([2, 3, 2])
        with filter_cols[0]:
            selected_semesters = st.pills("Semester", semesters, default=semesters, selection_mode="multi", label_visibility="collapsed")
            st.caption("Semester")
        with filter_cols[1]:
            selected_instructors = st.pills("Instructor", instructors, default=instructors, selection_mode="multi", label_visibility="collapsed")
            st.caption("Instructor")
        with filter_cols[2]:
            hide_rare = st.checkbox("Hide elements with fewer than 3 compounds", value=True)
            min_rows = 3 if hide_rare else 1

        filtered = described_df.copy()
        if selected_semesters:
            filtered = filtered[filtered["Semester"].astype(str).isin(selected_semesters)]
        if selected_instructors:
            filtered = filtered[filtered["Instructor"].astype(str).isin(selected_instructors)]

        if filtered.empty:
            st.warning("No rows match the selected filters.")
        else:
            plot_df = filtered.copy()
            if "P_raw" in plot_df.columns:
                plot_df["PhasePlotLabel"] = plot_df["P_raw"].apply(
                    lambda value: display_label(normalize_phase_label(value))
                )
            else:
                plot_df["PhasePlotLabel"] = plot_df["P"].apply(
                    lambda value: display_label(normalize_phase_label(value))
                )

            if "Bub_raw" in plot_df.columns:
                plot_df["BubblePlotLabel"] = plot_df["Bub_raw"].apply(
                    lambda value: display_label(normalize_bubble_label(value))
                )
            else:
                plot_df["BubblePlotLabel"] = plot_df["Bub"].apply(
                    lambda value: display_label(normalize_bubble_label(value))
                )

            def label_count_table(series: pd.Series, preferred_order: list) -> pd.DataFrame:
                """Build a Count / Share table for a label column, in a stable order."""
                counts = series.replace("", "Missing").fillna("Missing").astype(str).value_counts()
                ordered = [l for l in preferred_order if l in counts.index]
                ordered += sorted([l for l in counts.index if l not in ordered])
                counts = counts.reindex(ordered)
                total = int(counts.sum())
                return pd.DataFrame({
                    "Response": counts.index,
                    "Count": counts.values.astype(int),
                    "Share": [f"{v / total:.0%}" for v in counts.values],
                })

            dist_cols = st.columns(2)
            with dist_cols[0]:
                st.markdown("#### Bubble response counts")
                st.dataframe(
                    label_count_table(plot_df["BubblePlotLabel"], ["Yes", "No", "Maybe", "Other / needs review", "Missing"]),
                    use_container_width=True, hide_index=True,
                )
                st.caption("How many compounds were recorded as yes, no, maybe, or missing.")
            with dist_cols[1]:
                st.markdown("#### Phase counts")
                st.dataframe(
                    label_count_table(plot_df["PhasePlotLabel"], ["Pure", "Impure", "Not made", "Other / needs review", "Missing"]),
                    use_container_width=True, hide_index=True,
                )
                st.caption(
                    "Long phase answers are grouped into readable categories: Pure, Impure, Not made, or Needs review."
                )

            st.markdown("### Which elements appear more often with each outcome?")
            trend_outcome = st.radio(
                "Element trend outcome", ["Bubbling", "Purity"],
                horizontal=True, key="trend_outcome", label_visibility="collapsed",
            )
            if trend_outcome == "Bubbling":
                trend_metric, trend_name, trend_axis = "bubble_yes_rate", "bubble yes rate", "Bubble yes rate (%)"
                st.caption("Bubble yes rate = percentage of compounds in that group where Bubble Response was `yes`.")
            else:
                trend_metric, trend_name, trend_axis = "pure_rate", "pure rate", "Pure rate (%)"
                st.caption("Pure rate = percentage of compounds in that group recorded as `pure`.")

            a_summary = summarize_by_element(filtered, "A", min_rows=int(min_rows))
            b_summary = summarize_by_element(filtered, "B", min_rows=int(min_rows))

            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.markdown("#### A-site trend")
                if a_summary.empty:
                    st.info("Not enough A-site data for the current filter.")
                else:
                    a_sorted = a_summary.sort_values(trend_metric, ascending=False)
                    st.pyplot(
                        plot_bar_chart(a_sorted, "A", trend_metric, f"A-site {trend_name}", trend_axis),
                        use_container_width=True,
                    )
                    st.dataframe(a_sorted.round(3), use_container_width=True, height=240)
            with chart_cols[1]:
                st.markdown("#### B-site trend")
                if b_summary.empty:
                    st.info("Not enough B-site data for the current filter.")
                else:
                    b_sorted = b_summary.sort_values(trend_metric, ascending=False)
                    st.pyplot(
                        plot_bar_chart(b_sorted, "B", trend_metric, f"B-site {trend_name}", trend_axis),
                        use_container_width=True,
                    )
                    st.dataframe(b_sorted.round(3), use_container_width=True, height=240)

            with st.expander("Descriptor preview"):
                descriptor_cols = [
                    "Formula", "A_avg_Z", "B_avg_Z", "A_avg_mass", "B_avg_mass",
                    "FormulaMass", "O_to_cation_ratio", "B_to_A_ratio", "BubbleLabel", "PhaseLabel",
                ]
                descriptor_cols = [c for c in descriptor_cols if c in filtered.columns]
                st.dataframe(filtered[descriptor_cols].round(4), use_container_width=True, height=300)

        st.divider()

        # =========================================================================
        # SECTION 3: RELATIONSHIP MAP
        # =========================================================================
        st.subheader("🧭 Relationship Map")
        st.write(
            "Each square in the heatmap compares two numeric features. Look at the BubbleYes row to see what may relate to bubbling."
        )
        st.caption(
            "The **BubbleYes** row uses the same yes-vs-no definition as the ML models "
            "(“maybe” responses are excluded), so the map and the models stay consistent."
        )

        corr = numeric_correlation_table(described_df)

        if corr.empty:
            st.warning("Not enough numeric columns available to build a relationship map.")
        else:
            heatmap_cols = st.columns([1.2, 1])
            with heatmap_cols[0]:
                st.pyplot(plot_heatmap(corr), use_container_width=True)
            with heatmap_cols[1]:
                st.markdown("#### How to read this")
                st.markdown(
                    """
                    - **+1.00** — two features tend to increase together.
                    - **0.00** — little or no linear relationship.
                    - **-1.00** — one increases while the other decreases.

                    This does **not** prove cause and effect. It suggests patterns worth investigating.
                    """
                )
                rel = bubble_relationships(corr)
                st.markdown("#### Strongest links to bubble result")
                if rel.empty:
                    st.info("Bubble relationship summary not available yet.")
                else:
                    st.dataframe(rel.round(3), use_container_width=True, height=280)

            with st.expander("Example interpretation"):
                st.markdown(
                    """
                    If `B_avg_Z` has a positive correlation with `BubbleYes`, compounds with a higher
                    average B-site atomic number tended to have more `bubble = yes` results in this dataset.
                    That does **not** mean atomic number causes bubbling — it is a pattern worth discussing
                    and testing with more experiments.
                    """
                )


    with main_tabs[2]:
        # =========================================================================
        # TABS: ML LAB | ADD COMPOUND | EXPORT
        # =========================================================================
        action_tabs = st.tabs(["🤖 ML Lab", "➕ Add Compound", "⬇️ Export", "🔄 New Semester"])

        # -------------------------------------------------------------------------
        # TAB: ML LAB
        # -------------------------------------------------------------------------
        with action_tabs[0]:
            st.subheader("ML Lab")
            st.write(
                "Train models to find patterns in two outcomes: whether a compound **bubbles** "
                "and whether it comes out **pure**. Use this as a hypothesis tool, not proof."
            )
            st.info("Predictions describe patterns in past class data. They are not a guarantee of how a new compound will behave.")

            target_label = st.radio(
                "What do you want to predict?",
                ["Bubbling (yes vs no)", "Purity (pure vs impure)"],
                horizontal=True,
                key="ml_target",
            )
            target = "bubble" if target_label.startswith("Bubbling") else "purity"

            ml_result = train_classification_model(described_df, target=target, use_phase=True)

            if not ml_result.get("ok"):
                st.warning(ml_result.get("message", "Model could not be trained."))
            else:
                pos_name = ml_result["pos_name"]
                neg_name = ml_result["neg_name"]
                pos_count = ml_result["pos_count"]
                neg_count = ml_result["neg_count"]
                pos_rate = ml_result["pos_rate"]
                baseline = ml_result["baseline_accuracy"]
                best = ml_result["best"]
                best_name = ml_result["model_name"]

                # --- Dataset balance ---
                st.markdown("#### Dataset balance")
                bal_cols = st.columns(3)
                with bal_cols[0]:
                    st.metric(pos_name, f"{pos_count:,}")
                with bal_cols[1]:
                    st.metric(neg_name, f"{neg_count:,}")
                with bal_cols[2]:
                    st.metric(f"{pos_name} rate", f"{pos_rate:.0%}",
                              help=f"Fraction of labeled compounds that are {pos_name}.")

                # --- Baseline comparison (cross-validated) ---
                cv_bal = best["cv_balanced_accuracy_mean"]
                cv_bal_std = best["cv_balanced_accuracy_std"]
                cv_auc = best["cv_roc_auc_mean"]
                majority_name = pos_name if pos_rate >= 0.5 else neg_name
                baseline_note = (
                    f"A naive model that **always guesses '{majority_name}'** scores **{baseline:.0%}** accuracy. "
                    f"Under 5-fold cross-validation, the best model ({best_name}) scores "
                    f"**{cv_bal:.0%} ± {cv_bal_std:.0%}** balanced accuracy and **{cv_auc:.2f}** ROC-AUC."
                )
                # Balanced accuracy is the honest score on imbalanced data: 50% = random.
                if cv_bal > 0.58:
                    st.success(baseline_note + "  The model finds a real, usable signal.")
                elif cv_bal > 0.53:
                    st.info(baseline_note + "  The model finds a weak but real signal.")
                else:
                    st.warning(baseline_note + "  The model barely beats guessing — treat predictions with caution.")

                # --- Three-model comparison (cross-validated headline + single-split detail) ---
                st.markdown("#### Model comparison")
                st.caption(
                    "**Balanced acc. (CV)** and **ROC-AUC (CV)** come from 5-fold cross-validation — the honest, "
                    "split-independent scores (± shows fold-to-fold spread; 50% / 0.50 = random guessing). "
                    "Precision, Recall, and F1 come from the held-out test split."
                )

                def fmt_pct_pm(mean, std):
                    return f"{mean*100:.0f}% ± {std*100:.0f}%"

                rows = []
                for label, key in [("Random Forest", "rf"), ("Gradient Boosting", "gb"), ("Logistic Regression", "lr")]:
                    s = ml_result[key]
                    rows.append({
                        "Model": label,
                        "Balanced acc. (CV)": fmt_pct_pm(s["cv_balanced_accuracy_mean"], s["cv_balanced_accuracy_std"]),
                        "ROC-AUC (CV)": f"{s['cv_roc_auc_mean']:.2f} ± {s['cv_roc_auc_std']:.2f}",
                        "Precision": f"{s['precision']*100:.0f}%",
                        "Recall": f"{s['recall']*100:.0f}%",
                        "F1": round(s["f1"], 2),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption(f"Predictions below use the best model by cross-validated balanced accuracy: **{best_name}**.")

                with st.expander("What do these numbers mean?"):
                    st.markdown(
                        f"""
                        - **Cross-validation (CV)** — the data is split into 5 parts; the model is trained on 4 and
                          tested on the 1 left out, rotating through all 5. The **± spread** shows how much the score
                          moves between splits. This is far more trustworthy than a single lucky/unlucky split.
                        - **Balanced accuracy** — average of how well it identifies each class. 50% = random.
                          This is the honest headline number (the naive baseline accuracy here is {baseline:.0%}).
                        - **Precision** — when the model predicts *{pos_name}*, how often it is right.
                        - **Recall** — of all the actual *{pos_name}* compounds, how many it caught.
                        - **ROC-AUC** — overall ranking quality. 0.50 = random, 1.00 = perfect.
                        - **Gradient Boosting** is the modern "best tool" for this kind of table; **Random Forest**
                          and **Logistic Regression** are shown for comparison. If they roughly agree, trust the result more.
                        """
                    )

                # --- Permutation importance ---
                st.markdown("#### Which features mattered most?")
                importance = ml_result["importance"]
                if importance.empty:
                    st.info("No feature stood out above noise — another sign the signal is weak for this target.")
                else:
                    st.pyplot(
                        plot_bar_chart(importance, "Feature", "Importance", f"Top features for predicting {ml_result['outcome_name']}", "Permutation importance", horizontal=True),
                        use_container_width=True,
                    )
                    st.caption(
                        "Measured by **permutation importance**: how much accuracy drops when each feature is shuffled. "
                        "This is fair across element (categorical) and mass/ratio (numeric) features, unlike the default "
                        "tree importance which over-weights numeric features like mass."
                    )

                # --- Chi-squared tests ---
                st.markdown("#### Chi-squared test of independence")
                st.caption(
                    f"Tests whether each element position is statistically associated with the {ml_result['outcome_name']} "
                    "outcome. A small p-value (< 0.05) means the association is unlikely to be chance."
                )
                chi = chi_squared_tests(described_df, target=target)
                if chi.empty:
                    st.info("Not enough categorical data to run the chi-squared test.")
                else:
                    chi_show = chi.copy()
                    chi_show["p-value"] = chi_show["p-value"].apply(lambda p: f"{p:.2e}" if p < 0.001 else f"{p:.3f}")
                    st.dataframe(chi_show, use_container_width=True, hide_index=True)

                # --- Interactive prediction (always predicts BOTH outcomes) ---
                st.markdown("### Test a proposed compound")
                st.write(
                    "Enter a compound. The app estimates the chance of **bubbling** and of a "
                    "**pure** phase — each from its own model, regardless of the toggle above."
                )

                bubble_pred_model = train_classification_model(described_df, target="bubble", use_phase=True)
                purity_pred_model = train_classification_model(described_df, target="purity", use_phase=True)

                p1, p2, p3 = st.columns(3)
                with p1:
                    pred_a = st.text_input("A-site element", value="La", key="pred_a")
                    pred_an = st.number_input("A-site ratio", value=2.0, min_value=0.0, step=0.5, key="pred_an")
                with p2:
                    pred_b = st.text_input("B-site element", value="Ni", key="pred_b")
                    pred_bn = st.number_input("B-site ratio", value=1.0, min_value=0.0, step=0.5, key="pred_bn")
                with p3:
                    pred_on = st.number_input("Oxygen ratio", value=4.0, min_value=0.0, step=0.5, key="pred_on")
                    pred_phase = st.selectbox(
                        "Phase", ["pure", "impure"], key="pred_phase",
                        help="Used only by the bubbling model. Purity is predicted without it (it is the outcome).",
                    )

                with st.expander("Mixed-site elements (advanced)"):
                    adv1, adv2, adv3 = st.columns(3)
                    with adv1:
                        pred_ap = st.text_input("A′-site element", value="", key="pred_ap")
                        pred_apn = st.number_input("A′ ratio", value=0.0, min_value=0.0, step=0.5, key="pred_apn")
                    with adv2:
                        pred_bp = st.text_input("B′-site element", value="", key="pred_bp")
                        pred_bpn = st.number_input("B′ ratio", value=0.0, min_value=0.0, step=0.5, key="pred_bpn")
                    with adv3:
                        pred_bdp = st.text_input("B″-site element", value="", key="pred_bdp")
                        pred_bdpn = st.number_input("B″ ratio", value=0.0, min_value=0.0, step=0.5, key="pred_bdpn")

                def render_outcome_prediction(result: dict, pred_row: pd.DataFrame) -> None:
                    """Show one outcome's prediction with confidence scaled to its CV ROC-AUC."""
                    if not result.get("ok"):
                        st.info(f"{result.get('outcome_name', 'This outcome').capitalize()}: model unavailable — {result.get('message', '')}")
                        return
                    feats = build_feature_matrix(
                        pred_row,
                        use_phase=result["feature_use_phase"],
                        expected_columns=result["feature_columns"],
                    )
                    prob = result["model"].predict_proba(feats)[0][1]
                    # Round to the nearest 5% — these models do not justify 1% precision.
                    rounded = round(prob * 20) / 20
                    if prob >= 0.6:
                        verdict = f"leans toward **{result['pos_name']}**"
                    elif prob <= 0.4:
                        verdict = f"leans toward **{result['neg_name']}**"
                    else:
                        verdict = "is a **toss-up**"
                    auc = result["best"]["cv_roc_auc_mean"]
                    line = (
                        f"**{result['outcome_name'].capitalize()}:** {verdict} — estimated "
                        f"{result['pos_name']} chance **{rounded:.0%}**."
                    )
                    if auc >= 0.65:
                        st.success(line)
                        st.caption(f"{result['model_name']} · CV ROC-AUC {auc:.2f} — a useful hint, not a guarantee.")
                    else:
                        st.info(line)
                        st.caption(
                            f"{result['model_name']} · CV ROC-AUC {auc:.2f} — only weakly predictable from "
                            "composition; treat as a rough lean, not a real probability."
                        )

                if st.button("Predict bubbling and purity", type="primary"):
                    pred_row = make_single_prediction_row(
                        atomic, en_table, pred_phase, pred_a, pred_an, pred_ap, pred_apn,
                        pred_b, pred_bn, pred_bp, pred_bpn, pred_bdp, pred_bdpn, pred_on
                    )
                    formula = pred_row.iloc[0]["Formula"]
                    st.markdown(f"#### Results for `{formula}`")
                    result_cols = st.columns(2)
                    with result_cols[0]:
                        render_outcome_prediction(bubble_pred_model, pred_row)
                    with result_cols[1]:
                        render_outcome_prediction(purity_pred_model, pred_row)
                    st.dataframe(
                        pred_row[[c for c in ordered_columns(pred_row) if c in pred_row.columns]],
                        use_container_width=True,
                    )

        # -------------------------------------------------------------------------
        # TAB: ADD COMPOUND
        # -------------------------------------------------------------------------
        with action_tabs[1]:
            st.subheader("Add Compound")
            st.write(
                "Enter a new compound one part at a time. The app rebuilds the formula and checks the entry."
            )

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
                    new_a = st.text_input("A-site element", value="La", key="add_a")
                    new_an = st.number_input("A-site ratio", value=2.0, min_value=0.0, step=0.5, key="add_an")
                    new_ap = st.text_input("A′-site element", value="", key="add_ap")
                    new_apn = st.number_input("A′ ratio", value=0.0, min_value=0.0, step=0.5, key="add_apn")
                with b_col:
                    new_b = st.text_input("B-site element", value="Ni", key="add_b")
                    new_bn = st.number_input("B-site ratio", value=1.0, min_value=0.0, step=0.5, key="add_bn")
                    new_bp = st.text_input("B′-site element", value="", key="add_bp")
                    new_bpn = st.number_input("B′ ratio", value=0.0, min_value=0.0, step=0.5, key="add_bpn")
                with o_col:
                    new_bdp = st.text_input("B″-site element", value="", key="add_bdp")
                    new_bdpn = st.number_input("B″ ratio", value=0.0, min_value=0.0, step=0.5, key="add_bdpn")
                    new_on = st.number_input("Oxygen ratio", value=4.0, min_value=0.0, step=0.5, key="add_on")

                submitted = st.form_submit_button("Build and check compound", type="primary")

            if submitted:
                new_raw = pd.DataFrame(
                    [
                        {
                            "GroupNumber": new_group,
                            "Email": "",
                            "Name": "",
                            "Members": "",
                            "Instructor": new_instructor,
                            "Semester": new_semester,
                            "SourceRow": "manual",
                            "Slot": 1,
                            "FormulaRef": "",
                            "A": new_a,
                            "AN": new_an,
                            "AP": new_ap,
                            "APN": new_apn,
                            "B": new_b,
                            "BN": new_bn,
                            "BP": new_bp,
                            "BPN": new_bpn,
                            "BDP": new_bdp,
                            "BDPN": new_bdpn,
                            "O": "O",
                            "ON": new_on,
                            "P": new_phase,
                            "Bub": new_bub,
                        }
                    ]
                )
                new_clean = clean_and_encode_data(new_raw, atomic)
                new_desc = add_chemical_descriptors(new_clean, atomic, en_table)
                new_issues = validate_compound_rows(new_desc, atomic)
                st.session_state.pending_manual_entry = new_desc
                st.session_state.pending_manual_issues = new_issues

            if not st.session_state.pending_manual_entry.empty:
                pending = st.session_state.pending_manual_entry
                pending_issues = st.session_state.pending_manual_issues
                formula = pending.iloc[0]["Formula"]

                st.markdown(f"### Preview: `{formula}`")

                if pending_issues.empty:
                    st.success("This entry passed validation.")
                else:
                    st.warning("This entry has issues to review before adding.")
                    st.dataframe(pending_issues, use_container_width=True)

                st.dataframe(pending[ordered_columns(pending)], use_container_width=True)

                add_cols = st.columns([1, 1, 2])
                with add_cols[0]:
                    if st.button("Add to session dataset", type="primary", key="confirm_add_manual_entry"):
                        st.session_state.manual_entries = pd.concat(
                            [st.session_state.manual_entries, pending],
                            ignore_index=True,
                        )
                        st.session_state.pending_manual_entry = pd.DataFrame()
                        st.session_state.pending_manual_issues = pd.DataFrame()
                        st.success("Added to this session. Go to Export to download the updated data.")
                        st.rerun()
                with add_cols[1]:
                    if st.button("Discard preview", key="discard_manual_preview"):
                        st.session_state.pending_manual_entry = pd.DataFrame()
                        st.session_state.pending_manual_issues = pd.DataFrame()
                        st.rerun()
                with add_cols[2]:
                    st.download_button(
                        "Download this single entry as CSV",
                        dataframe_to_csv_bytes(pending),
                        file_name=f"{formula or 'new_compound'}_entry.csv",
                        mime="text/csv",
                        key="download_single_pending",
                    )

            if not st.session_state.manual_entries.empty:
                st.markdown("#### Entries added during this session")
                st.dataframe(st.session_state.manual_entries[ordered_columns(st.session_state.manual_entries)], use_container_width=True)
                if st.button("Clear session-added entries", key="clear_session_entries"):
                    st.session_state.manual_entries = pd.DataFrame()
                    st.rerun()

        # -------------------------------------------------------------------------
        # TAB: EXPORT
        # -------------------------------------------------------------------------
        with action_tabs[2]:
            st.subheader("Export")
            st.write("Download cleaned results, validation reports, and analysis-ready data.")

            export_cols = st.columns(2)
            with export_cols[0]:
                st.markdown("#### Cleaned data")
                st.download_button(
                    "Download cleaned data as CSV",
                    dataframe_to_csv_bytes(described_df[ordered_columns(described_df)]),
                    file_name="chem120_cleaned_compound_data.csv",
                    mime="text/csv",
                )
                st.download_button(
                    "Download cleaned data as Excel",
                    dataframe_to_excel_bytes(described_df[ordered_columns(described_df)], "Cleaned_Data"),
                    file_name="chem120_cleaned_compound_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            with export_cols[1]:
                st.markdown("#### Reports")
                st.download_button(
                    "Download validation report",
                    dataframe_to_csv_bytes(issues_df),
                    file_name="chem120_validation_report.csv",
                    mime="text/csv",
                )
                st.download_button(
                    "Download outlier report",
                    dataframe_to_csv_bytes(outlier_df),
                    file_name="chem120_outlier_report.csv",
                    mime="text/csv",
                )

            st.markdown("### What should I save?")
            st.markdown(
                """
                - **Cleaned data** — analysis-ready version of the class database.
                - **Validation report** — use this to correct student-entry mistakes.
                - **Outlier report** — use this to double-check unusual values.
                - The app never overwrites your original file.
                """
            )

        # -------------------------------------------------------------------------
        # TAB: NEW SEMESTER
        # -------------------------------------------------------------------------
        with action_tabs[3]:
            st.subheader("Prepare New Semester Data")
            st.write(
                "Upload a new semester's raw Excel file. The app will split each group's "
                "row into one row per compound, then merge it with the current dataset so "
                "you can download one combined file."
            )

            new_sem_file = st.file_uploader(
                "New semester Excel file",
                type=["xlsx", "xlsm", "xls"],
                help="The raw export where each row is one group and compounds are in columns like 1A, 1AN, 2A, 2AN…",
                key="new_semester_upload",
            )

            if new_sem_file is None:
                st.info("Upload the new semester file above to get started.")
            else:
                try:
                    new_raw = read_table_from_bytes(new_sem_file.getvalue(), new_sem_file.name)
                except Exception as exc:
                    st.error(f"Could not read file: {exc}")
                    st.stop()

                new_long, new_warnings = normalize_to_long_format(new_raw)

                if new_warnings:
                    with st.expander("⚠️ Column mapping problems in new file", expanded=True):
                        st.warning(
                            "Some required columns could not be matched. "
                            "Check that the column names follow the expected pattern (1A, 1AN, 1B, 1BN…). "
                            "See DEVELOPER_NOTES.md for fix instructions."
                        )
                        for w in new_warnings:
                            st.markdown(f"- {w}")

                if new_long.empty:
                    st.error("No compound rows could be extracted from the uploaded file. Check the column names.")
                else:
                    # Preview stats
                    stat_cols = st.columns(3)
                    with stat_cols[0]:
                        st.metric("Groups in new file", f"{len(new_raw):,}")
                    with stat_cols[1]:
                        st.metric("Compound rows extracted", f"{len(new_long):,}")
                    with stat_cols[2]:
                        st.metric("Existing compound rows", f"{len(long_df):,}")

                    st.markdown("#### Preview — first 10 rows from new semester")
                    preview_cols = [c for c in ["GroupNumber", "Semester", "Slot", "A", "AN", "B", "BN", "O", "ON", "P", "Bub"] if c in new_long.columns]
                    st.dataframe(new_long[preview_cols].head(10), use_container_width=True)

                    # Check for semester overlap
                    existing_semesters = set(long_df["Semester"].dropna().astype(str).unique())
                    new_semesters = set(new_long["Semester"].dropna().astype(str).unique())
                    overlap = existing_semesters & new_semesters
                    if overlap:
                        st.warning(
                            f"The new file contains semester(s) already in the existing dataset: "
                            f"**{', '.join(sorted(overlap))}**. "
                            "Check that you are not adding duplicate data."
                        )
                    else:
                        st.success(f"No semester overlap detected. New semesters: **{', '.join(sorted(new_semesters)) or 'unknown'}**")

                    merged = pd.concat([long_df, new_long], ignore_index=True)

                    st.markdown(f"#### Merged dataset — {len(merged):,} total compound rows")

                    dl_bytes = dataframe_to_excel_bytes(merged, "Combined_Data")
                    st.download_button(
                        "⬇️ Download merged Combined_Data.xlsx",
                        dl_bytes,
                        file_name="Combined_Data_merged.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                    )
                    st.caption(
                        "Replace your existing data/Combined_Data.xlsx with this file "
                        "and the app will load the full merged dataset on next startup."
                    )


if __name__ == "__main__":
    main()
