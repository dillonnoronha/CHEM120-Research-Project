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
    plot_distribution,
    plot_heatmap,
    read_local_table,
    read_table_from_bytes,
    summarize_by_element,
    train_ml_model,
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

        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1220px;
        }

        h1, h2, h3 {
            letter-spacing: -0.03em;
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

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.65rem 1rem;
            background-color: #f5f5f7;
            color: #1d1d1f !important;
        }

        .stTabs [aria-selected="true"] {
            background-color: #e8f1ff;
            color: #0057b8 !important;
        }

        .stTabs [data-baseweb="tab"] p {
            color: inherit !important;
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
    use_phase_in_ml = st.sidebar.toggle(
        "Use phase in ML model",
        value=True,
        help="Turn off if you want the model to ignore pure/impure phase labels.",
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

    issues_df = validate_compound_rows(described_df, atomic)
    outlier_df = detect_numeric_outliers(described_df, z_threshold=z_threshold)
    yes_count = int(described_df["BubbleYes"].sum()) if "BubbleYes" in described_df.columns else 0

    # -------------------------------------------------------------------------
    # Data health cards.
    # -------------------------------------------------------------------------
    health_cols = st.columns(3)
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
    with health_cols[2]:
        style = "status-ok" if yes_count > 0 else "status-warn"
        st.markdown(
            f"""<div class="soft-card {style}">
            <b>Bubble outcomes</b><br>
            <span class="helper-text">{yes_count:,} compounds labeled bubble = yes.</span>
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
            st.warning(f"{len(issues_df):,} possible issues found. Review these before final analysis.")
            st.dataframe(issues_df, use_container_width=True, height=300)
    with check_cols[1]:
        st.markdown("#### Possible numeric outliers")
        if outlier_df.empty:
            st.success("No numeric outliers found at the current sensitivity.")
        else:
            st.info("Outliers are not automatically wrong. They are values worth double-checking.")
            st.dataframe(outlier_df, use_container_width=True, height=300)

    st.markdown("#### Cleaned compound table")
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

    st.divider()

    # =========================================================================
    # SECTION 2: EXPLORE RESULTS
    # =========================================================================
    st.subheader("📊 Explore Results")
    st.write(
        "Compare compounds and look for patterns. These graphs are descriptive — not proof of causation."
    )

    filter_cols = st.columns(3)
    with filter_cols[0]:
        semesters = sorted([x for x in described_df.get("Semester", pd.Series(dtype=str)).dropna().astype(str).unique() if x.strip()])
        selected_semesters = st.multiselect("Semester filter", semesters, default=semesters)
    with filter_cols[1]:
        instructors = sorted([x for x in described_df.get("Instructor", pd.Series(dtype=str)).dropna().astype(str).unique() if x.strip()])
        selected_instructors = st.multiselect("Instructor filter", instructors, default=instructors)
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

        dist_cols = st.columns(2)
        with dist_cols[0]:
            st.pyplot(
                plot_distribution(
                    plot_df,
                    "BubblePlotLabel",
                    "Bubble response counts",
                    preferred_order=["Yes", "No", "Maybe", "Other / needs review", "Missing"],
                ),
                use_container_width=True,
            )
            st.caption("How many compounds were recorded as yes, no, maybe, or missing.")
        with dist_cols[1]:
            st.pyplot(
                plot_distribution(
                    plot_df,
                    "PhasePlotLabel",
                    "Phase counts",
                    preferred_order=["Pure", "Impure", "Not made", "Other / needs review", "Missing"],
                ),
                use_container_width=True,
            )
            st.caption(
                "Long phase answers are grouped into readable categories: Pure, Impure, Not made, or Needs review."
            )

        st.markdown("### Which elements appear more often with bubbling?")
        st.caption("Bubble yes rate = percentage of compounds in that group where Bubble Response was `yes`.")

        a_summary = summarize_by_element(filtered, "A", min_rows=int(min_rows))
        b_summary = summarize_by_element(filtered, "B", min_rows=int(min_rows))

        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.markdown("#### A-site trend")
            if a_summary.empty:
                st.info("Not enough A-site data for the current filter.")
            else:
                st.pyplot(
                    plot_bar_chart(a_summary, "A", "bubble_yes_rate", "A-site bubble yes rate", "Bubble yes rate (%)"),
                    use_container_width=True,
                )
                st.dataframe(a_summary.round(3), use_container_width=True, height=240)
        with chart_cols[1]:
            st.markdown("#### B-site trend")
            if b_summary.empty:
                st.info("Not enough B-site data for the current filter.")
            else:
                st.pyplot(
                    plot_bar_chart(b_summary, "B", "bubble_yes_rate", "B-site bubble yes rate", "Bubble yes rate (%)"),
                    use_container_width=True,
                )
                st.dataframe(b_summary.round(3), use_container_width=True, height=240)

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

    st.divider()

    # =========================================================================
    # TABS: ML LAB | ADD COMPOUND | EXPORT
    # =========================================================================
    action_tabs = st.tabs(["🤖 ML Lab", "➕ Add Compound", "⬇️ Export"])

    # -------------------------------------------------------------------------
    # TAB: ML LAB
    # -------------------------------------------------------------------------
    with action_tabs[0]:
        st.subheader("ML Lab")
        st.write(
            "Trains a model to estimate whether a compound looks similar to past compounds labeled bubble = yes."
        )
        st.info("Use this as a hypothesis tool. It should not be treated as proof that a compound will work.")

        ml_result = train_ml_model(described_df, use_phase=use_phase_in_ml)

        if not ml_result.get("ok"):
            st.warning(ml_result.get("message", "ML model could not be trained."))
        else:
            if ml_result.get("overfit_warning"):
                st.warning(ml_result["overfit_warning"])

            ml_cols = st.columns(4)
            with ml_cols[0]:
                st.metric("Random Forest accuracy", f"{ml_result['rf_accuracy']:.0%}")
            with ml_cols[1]:
                st.metric("Logistic Regression accuracy", f"{ml_result['lr_accuracy']:.0%}")
            with ml_cols[2]:
                st.metric("Training rows", ml_result["training_rows"])
            with ml_cols[3]:
                st.metric("Testing rows", ml_result["testing_rows"])

            with st.expander("Why two models?"):
                st.markdown(
                    """
                    **Random Forest** is more powerful but can memorize small datasets
                    (overfitting), making its accuracy look better than it really is.

                    **Logistic Regression** is simpler and gives a more honest baseline.
                    If both agree, the result is more trustworthy. If Random Forest is
                    much higher, treat its predictions with extra caution.

                    Predictions below use the Random Forest model.
                    """
                )

            st.markdown("#### What features did the model use most?")
            importance = ml_result["importance"]
            if importance.empty:
                st.info("Feature importance is not available.")
            else:
                st.pyplot(
                    plot_bar_chart(importance, "Feature", "Importance", "Top model features", "Importance"),
                    use_container_width=True,
                )
                st.caption(
                    "Higher importance means the model relied on that feature more often when making predictions."
                )

            st.markdown("### Test a proposed compound")
            st.write("Enter a compound below. The model will estimate the chance of `bubble = yes` based on past data.")

            p1, p2, p3 = st.columns(3)
            with p1:
                pred_a = st.text_input("A-site element", value="La", key="pred_a")
                pred_an = st.number_input("A-site ratio", value=2.0, min_value=0.0, step=0.5, key="pred_an")
            with p2:
                pred_b = st.text_input("B-site element", value="Ni", key="pred_b")
                pred_bn = st.number_input("B-site ratio", value=1.0, min_value=0.0, step=0.5, key="pred_bn")
            with p3:
                pred_on = st.number_input("Oxygen ratio", value=4.0, min_value=0.0, step=0.5, key="pred_on")
                pred_phase = st.selectbox("Phase", ["pure", "impure"], key="pred_phase")

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

            if st.button("Predict bubble probability", type="primary"):
                pred_row = make_single_prediction_row(
                    atomic, en_table, pred_phase, pred_a, pred_an, pred_ap, pred_apn,
                    pred_b, pred_bn, pred_bp, pred_bpn, pred_bdp, pred_bdpn, pred_on
                )
                pred_features = build_feature_matrix(
                    pred_row,
                    use_phase=use_phase_in_ml,
                    expected_columns=ml_result["feature_columns"],
                )
                probability = ml_result["model"].predict_proba(pred_features)[0][1]
                formula = pred_row.iloc[0]["Formula"]
                st.success(f"Predicted bubble=yes probability for **{formula}**: **{probability:.1%}**")
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


if __name__ == "__main__":
    main()
