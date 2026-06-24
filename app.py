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

    Check   -> render_check_tab()        data-entry issues, outliers, clean table
    Explore -> render_explore_tab()      response-count tables, per-element tables
    Heatmap -> render_heatmap_tab()      correlation map with a feature filter
    Predict -> render_ml_tab()           bubble + purity models and predictions
    Add     -> render_add_compound_tab() stage a new compound by hand
    Export  -> render_export_tab()       download cleaned data and reports
    Semester-> render_new_semester_tab() merge a new semester's raw export

To add a new tab: write a new `render_*_tab()` function and add it to the
`tab_objects` / `render` pairing inside `main()`.

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
    apply_element_fixes,
    bubble_relationships,
    build_feature_matrix,
    clean_and_encode_data,
    compute_pca,
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    detect_numeric_outliers,
    display_label,
    find_default_database,
    label_count_table,
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
    plot_pca_scatter,
    propose_element_fixes,
    read_local_table,
    read_table_from_bytes,
    remove_invalid_element_rows,
    summarize_by_element,
    valid_element_symbols,
    train_ml_model,
    train_purity_model,
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

        h1, h2, h3 { letter-spacing: -0.03em; }

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
            padding: 1.6rem 2rem;
            border: 1px solid var(--chem-border);
            border-radius: 24px;
            background:
                radial-gradient(circle at top left, rgba(0, 113, 227, 0.12), transparent 28%),
                linear-gradient(135deg, #ffffff 0%, #f5f5f7 100%);
            box-shadow: 0 14px 45px rgba(0,0,0,0.07);
            margin-bottom: 1.2rem;
        }

        .hero-title {
            font-size: 2.1rem;
            line-height: 1.05;
            font-weight: 760;
            color: var(--chem-ink);
            margin: 0;
        }

        .hero-subtitle {
            color: var(--chem-muted);
            font-size: 1rem;
            margin-top: 0.5rem;
            max-width: 820px;
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

        .big-number {
            color: var(--chem-ink);
            font-size: 1.5rem;
            font-weight: 760;
            line-height: 1.1;
        }

        .helper-text {
            color: var(--chem-muted);
            font-size: 0.93rem;
            line-height: 1.45;
        }

        .status-ok   { border-left: 4px solid #30d158; }
        .status-warn { border-left: 4px solid #ff9f0a; }
        .status-bad  { border-left: 4px solid #ff453a; }

        div[data-testid="stMetricValue"] {
            font-weight: 760;
            letter-spacing: -0.04em;
        }

        .stTabs [data-baseweb="tab-list"] { gap: 8px; }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.6rem 1rem;
            background-color: #f5f5f7;
            color: #1d1d1f !important;
        }

        .stTabs [aria-selected="true"] {
            background-color: #e8f1ff;
            color: #0057b8 !important;
        }

        .stTabs [data-baseweb="tab"] p { color: inherit !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# TAB RENDERERS
# Each function renders ONE tab. They take only the data they need so they are
# easy to read, test, and reorder. None of them create the tab container itself
# (main() does that) — they just fill it.
# =============================================================================

def render_did_you_mean(fix_proposals: pd.DataFrame, all_symbols: list) -> None:
    """Pre-cleaning panel: a table of invalid elements, each with a fix dropdown.

    Built with st.data_editor so every problem row has a "Fix to" dropdown,
    pre-set to our best guess. Choosing a symbol corrects the row; "Remove row"
    drops it. Choices are saved to st.session_state.element_fixes and applied
    (in main) on the next rerun.
    """

    n = len(fix_proposals)
    with st.expander(f"🛠️ Fix invalid elements ({n} to review)", expanded=True):
        st.caption("Each row is an entry that isn't a real element symbol. The **Fix to** dropdown "
                   "is pre-set to the best guess — change it or pick **Remove row**, then click Apply. "
                   "(Editable tables can't tint cells red, so invalid values are flagged with ⚠️.)")

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

    # Pre-cleaning "did you mean?" suggestions (only when invalid elements remain).
    if not fix_proposals.empty:
        render_did_you_mean(fix_proposals, all_symbols)

    # Both columns use the same lead-in pattern (header + one-line caption + a
    # fixed-height table) so the two tables line up vertically.
    check_cols = st.columns(2)
    with check_cols[0]:
        st.markdown("#### Validation issues")
        if issues_df.empty:
            st.caption("No validation issues found. ✅")
            st.dataframe(issues_df, use_container_width=True, height=300)
        else:
            st.caption(f"{len(issues_df):,} possible issues. Review before final analysis.")
            st.dataframe(issues_df, use_container_width=True, height=300)
    with check_cols[1]:
        st.markdown("#### Possible numeric outliers")
        if outlier_df.empty:
            st.caption("No outliers at the current sensitivity. ✅")
            st.dataframe(outlier_df, use_container_width=True, height=300)
        else:
            st.caption("Outliers aren't automatically wrong — just worth a second look.")
            st.dataframe(outlier_df, use_container_width=True, height=300)

    st.markdown("#### Cleaned compound table")
    if described_df.empty:
        st.warning("No compound rows found. Check that the file uses columns like 1A, 1AN, 1B, 1BN, 1O, 1ON, 1P, 1Bub.")
    else:
        st.dataframe(described_df[ordered_columns(described_df)], use_container_width=True, height=380)

    # Clickable log of what cleaning changed: corrections applied + rows removed.
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
    """Tab 2 — response-count tables and per-element bubble/pure summaries."""

    st.caption("Descriptive tables only — patterns to discuss, not proof of cause.")

    # Optional filters by semester and instructor.
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

    # Build display labels for the count tables. Prefer the raw wording so the
    # tables reflect exactly what students typed (after normalization).
    plot_df = filtered.copy()
    phase_src = "P_raw" if "P_raw" in plot_df.columns else "P"
    bub_src = "Bub_raw" if "Bub_raw" in plot_df.columns else "Bub"
    plot_df["PhasePlotLabel"] = plot_df[phase_src].apply(lambda v: display_label(normalize_phase_label(v)))
    plot_df["BubblePlotLabel"] = plot_df[bub_src].apply(lambda v: display_label(normalize_bubble_label(v)))

    # --- Response counts as tables (not charts) ---
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

    # --- Per-element bubble AND pure summaries as tables ---
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

    with st.expander("Descriptor preview"):
        descriptor_cols = [c for c in [
            "Formula", "A_avg_Z", "B_avg_Z", "A_mix_fraction", "B_mix_fraction",
            "FormulaMass", "O_to_cation_ratio", "B_to_A_ratio", "BubbleLabel", "PhaseLabel",
        ] if c in filtered.columns]
        st.dataframe(filtered[descriptor_cols].round(4), use_container_width=True, height=300)


def render_heatmap_tab(described_df: pd.DataFrame) -> None:
    """Tab 3 — correlation heatmap with a feature-selection filter."""

    st.caption("Each cell compares two numeric features. Read the BubbleYes / PhaseN rows for links to outcomes.")

    corr = numeric_correlation_table(described_df)
    if corr.empty:
        st.warning("Not enough numeric columns to build a relationship map.")
        return

    # Feature filter: let the user focus on a handful of features.
    all_features = list(corr.columns)
    default_features = [c for c in [
        "BubbleYes", "PhaseN", "B_avg_Z", "A_avg_Z", "FormulaMass",
        "O_to_cation_ratio", "B_to_A_ratio", "Avg_cation_Z", "Mixed_B_site",
    ] if c in all_features] or all_features

    selected = st.multiselect(
        "Features to compare",
        options=all_features,
        default=default_features,
        help="Keep BubbleYes and PhaseN to see what links to bubbling and purity.",
    )
    corr_view = corr.loc[selected, selected] if len(selected) >= 2 else corr.loc[default_features, default_features]
    if len(selected) < 2:
        st.info("Select at least two features. Showing defaults for now.")

    map_cols = st.columns([1.2, 1])
    with map_cols[0]:
        st.pyplot(plot_heatmap(corr_view), use_container_width=True)
    with map_cols[1]:
        st.markdown("#### Reading it")
        st.markdown("**+1** rise together · **0** no link · **-1** opposite. Correlation is not causation.")
        rel = bubble_relationships(corr)
        st.markdown("#### Strongest links to bubbling")
        st.caption("Computed across **all** features (not just the ones shown in the map), "
                   "so an item can appear here without being in your current selection.")
        if rel.empty:
            st.info("Not available yet.")
        else:
            st.dataframe(rel.round(3), use_container_width=True, hide_index=True, height=280)

    with st.expander("📖 What the terms mean"):
        st.markdown(
            "- **Bubbles (yes)** — 1 if the compound bubbled, else 0.\n"
            "- **Phase (pure=2)** — phase code: not made = 0, impure = 1, pure = 2.\n"
            "- **A / B ratio**, **A′/B′/B″ ratio**, **O ratio** — the amounts in the formula.\n"
            "- **A atomic # / B atomic #** — ratio-weighted average atomic number on each site.\n"
            "- **A mass / B mass / Formula mass** — ratio-weighted atomic masses and total formula mass.\n"
            "- **O : cation** — oxygen ratio ÷ total cation ratio.\n"
            "- **B : A** — total B-site ratio ÷ total A-site ratio.\n"
            "- **A-B atomic # / A-B mass** — difference between the A-site and B-site averages.\n"
            "- **Avg cation # / Avg cation mass** — averages across both cation sites together.\n"
            "- **# cations / # B elements** — how many distinct elements are present.\n"
            "- **Mixed A / Mixed B** — 1 if that site has more than one element.\n"
            "- **A mixing frac / B mixing frac** — share of the site taken by the *second* element "
            "(0 = single element, 0.5 = a 50/50 mix)."
        )


def render_pca_tab(described_df: pd.DataFrame) -> None:
    """Tab 4 — PCA structure map with optional KMeans clusters."""

    st.caption("Squeezes all numeric descriptors into a 2-D map. Nearby compounds have similar "
               "composition. Exploration only — outcomes don't cluster strongly here.")

    # Slider drives clustering. When it's >= 2, KMeans runs and the color control
    # gains a "Cluster" option that becomes the default, so the map updates right
    # away instead of needing a separate selection.
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
    map_cols = st.columns([1.4, 1])
    with map_cols[0]:
        st.pyplot(plot_pca_scatter(result["coords"], color_by=color_by), use_container_width=True)
    with map_cols[1]:
        st.metric("PC1 variance", f"{ev[0]:.0%}")
        st.metric("PC2 variance", f"{ev[1]:.0%}")
        st.metric("Compounds plotted", f"{result['n_used']:,}")
        st.caption(f"PC1 + PC2 together explain {sum(ev):.0%} of the variation across "
                   f"{len(result['features'])} descriptors.")

    with st.expander("What drives each axis (loadings)"):
        st.caption("Sorted by influence on PC1. Bigger absolute values = the descriptor pushes a "
                   "compound further along that axis.")
        loadings = result["loadings"].copy()
        loadings = loadings.reindex(loadings["PC1"].abs().sort_values(ascending=False).index)
        st.dataframe(loadings.round(2), use_container_width=True)


def render_ml_tab(described_df: pd.DataFrame, atomic: pd.DataFrame, en_table: pd.DataFrame,
                  use_phase_in_ml: bool, use_mass_in_ml: bool) -> None:
    """Tab 4 — train the bubble + purity models and predict a proposed compound."""

    st.caption("A hypothesis tool, not proof. Two models: one for bubbling, one for purity.")

    ml_result = train_ml_model(described_df, use_phase=use_phase_in_ml, use_mass=use_mass_in_ml)
    if not ml_result.get("ok"):
        st.warning(ml_result.get("message", "Bubble model could not be trained."))
        return

    if ml_result.get("overfit_warning"):
        st.warning(ml_result["overfit_warning"])

    # --- Bubble model: balance + headline metrics ---
    baseline = ml_result["baseline_accuracy"]
    rf_acc = ml_result["rf_accuracy"]

    st.markdown("#### Bubble model")
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

    with st.expander("What the numbers mean"):
        st.markdown(
            f"- **Accuracy** can mislead when yes is rare — compare to the {baseline:.0%} baseline.\n"
            "- **Precision**: when it says yes, how often it's right.\n"
            "- **Recall**: of all real bubblers, how many it caught.\n"
            "- **LR accuracy** "
            f"({ml_result['lr_accuracy']:.0%}) is a simpler sanity-check model."
        )

    importance = ml_result["importance"]
    if not importance.empty:
        st.markdown("**Top features used**")
        st.pyplot(plot_bar_chart(importance, "Feature", "Importance", "Top model features", "Importance"),
                  use_container_width=True)

    # --- Purity model: never sees the phase label as input ---
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

    # --- Predict a proposed compound (bubble + purity together) ---
    st.markdown("#### Test a proposed compound")
    st.caption("Leave A′ / B′ blank for a simple compound, or fill them in for a mixed cation site.")

    st.markdown("**A-site**")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        pred_a = st.text_input("A element", value="La", key="pred_a")
    with a2:
        pred_an = st.number_input("A ratio", value=2.0, min_value=0.0, step=0.5, key="pred_an")
    with a3:
        pred_ap = st.text_input("A′ element", value="", key="pred_ap")
    with a4:
        pred_apn = st.number_input("A′ ratio", value=0.0, min_value=0.0, step=0.5, key="pred_apn")

    st.markdown("**B-site**")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        pred_b = st.text_input("B element", value="Ni", key="pred_b")
    with b2:
        pred_bn = st.number_input("B ratio", value=1.0, min_value=0.0, step=0.5, key="pred_bn")
    with b3:
        pred_bp = st.text_input("B′ element", value="", key="pred_bp")
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
            pred_bdp = st.text_input("B″ element", value="", key="pred_bdp")
        with adv2:
            pred_bdpn = st.number_input("B″ ratio", value=0.0, min_value=0.0, step=0.5, key="pred_bdpn")

    if st.button("Predict bubble & purity", type="primary"):
        pred_row = make_single_prediction_row(
            atomic, en_table, pred_phase, pred_a, pred_an, pred_ap, pred_apn,
            pred_b, pred_bn, pred_bp, pred_bpn, pred_bdp, pred_bdpn, pred_on,
        )
        formula = pred_row.iloc[0]["Formula"]

        bubble_features = build_feature_matrix(
            pred_row, use_phase=use_phase_in_ml, expected_columns=ml_result["feature_columns"])
        bubble_prob = ml_result["model"].predict_proba(bubble_features)[0][1]

        out_cols = st.columns(2)
        with out_cols[0]:
            st.metric(f"Will it bubble?  ({formula})", f"{bubble_prob:.0%}")
        with out_cols[1]:
            if purity_result.get("ok"):
                purity_features = build_feature_matrix(
                    pred_row, use_phase=False, expected_columns=purity_result["feature_columns"])
                pure_prob = purity_result["model"].predict_proba(purity_features)[0][1]
                st.metric("Will it be pure?", f"{pure_prob:.0%}")
            else:
                st.metric("Will it be pure?", "n/a")

        st.caption("Hypotheses from past class data — not guarantees.")
        st.dataframe(pred_row[[c for c in ordered_columns(pred_row) if c in pred_row.columns]],
                     use_container_width=True)


def render_add_compound_tab(atomic: pd.DataFrame, en_table: pd.DataFrame) -> None:
    """Tab 5 — build and stage a new compound by hand for this session."""

    st.caption("Enter one compound part by part. The app rebuilds the formula and checks it.")

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
        new_desc = add_chemical_descriptors(new_clean, atomic, en_table)
        st.session_state.pending_manual_entry = new_desc
        st.session_state.pending_manual_issues = validate_compound_rows(new_desc, atomic)

    # Preview the staged (not yet added) entry.
    if not st.session_state.pending_manual_entry.empty:
        pending = st.session_state.pending_manual_entry
        pending_issues = st.session_state.pending_manual_issues
        formula = pending.iloc[0]["Formula"]

        st.markdown(f"#### Preview: `{formula}`")
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
                    [st.session_state.manual_entries, pending], ignore_index=True)
                st.session_state.pending_manual_entry = pd.DataFrame()
                st.session_state.pending_manual_issues = pd.DataFrame()
                st.success("Added. See Export to download the updated data.")
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

    # Entries already added this session.
    if not st.session_state.manual_entries.empty:
        st.markdown("#### Added this session")
        st.dataframe(st.session_state.manual_entries[ordered_columns(st.session_state.manual_entries)],
                     use_container_width=True)
        if st.button("Clear session-added entries", key="clear_session_entries"):
            st.session_state.manual_entries = pd.DataFrame()
            st.rerun()


def render_export_tab(described_df: pd.DataFrame, issues_df: pd.DataFrame, outlier_df: pd.DataFrame) -> None:
    """Tab 6 — download cleaned data and reports. Originals are never overwritten."""

    st.caption("Download analysis-ready data and reports. Your original file is never changed.")

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
    with export_cols[1]:
        st.markdown("#### Reports")
        st.download_button("Validation report (CSV)", dataframe_to_csv_bytes(issues_df),
                           file_name="chem120_validation_report.csv", mime="text/csv")
        st.download_button("Outlier report (CSV)", dataframe_to_csv_bytes(outlier_df),
                           file_name="chem120_outlier_report.csv", mime="text/csv")


def render_new_semester_tab(long_df: pd.DataFrame) -> None:
    """Tab 7 — merge a new semester's raw export into the current dataset."""

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

    # Warn if a semester already exists in the current data (possible duplicate).
    existing_semesters = set(long_df["Semester"].dropna().astype(str).unique())
    new_semesters = set(new_long["Semester"].dropna().astype(str).unique())
    overlap = existing_semesters & new_semesters
    if overlap:
        st.warning(f"New file contains semester(s) already present: **{', '.join(sorted(overlap))}**. "
                   "Check for duplicate data.")
    else:
        st.success(f"No semester overlap. New semesters: **{', '.join(sorted(new_semesters)) or 'unknown'}**")

    merged = pd.concat([long_df, new_long], ignore_index=True)
    st.markdown(f"#### Merged dataset — {len(merged):,} total rows")
    st.download_button("⬇️ Download merged Combined_Data.xlsx",
                       dataframe_to_excel_bytes(merged, "Combined_Data"),
                       file_name="Combined_Data_merged.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       type="primary")
    st.caption("Replace data/Combined_Data.xlsx with this file to load the merged dataset next startup.")


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

    st.sidebar.markdown("### 1. Data")
    uploaded_data = st.sidebar.file_uploader(
        "Class spreadsheet", type=["csv", "xlsx", "xlsm", "xls"],
        help="Upload Combined_Data.xlsx or another CHEM 120 survey/database file.",
    )
    uploaded_atomic = None
    uploaded_en = None

    # The app boots empty. Tick this to explore the bundled sample without uploading.
    use_sample_data = st.sidebar.checkbox(
        "Use sample class data instead", value=False,
        help="Loads the example dataset shipped with the app so you can try the features without uploading.",
    )

    st.sidebar.markdown("### 2. Controls")
    z_threshold = st.sidebar.slider(
        "Outlier sensitivity", min_value=2.0, max_value=5.0, value=3.0, step=0.25,
        help="Lower flags more possible outliers; higher flags only extreme values.",
    )
    use_phase_in_ml = st.sidebar.toggle(
        "Use phase in bubble model", value=True,
        help="Turn off to make the bubble model ignore pure/impure labels.",
    )
    use_mass_in_ml = st.sidebar.toggle(
        "Include mass descriptors", value=True,
        help="Atomic mass and atomic number are highly correlated. Turn off to stop "
             "mass features from dominating the models and see what other signal remains.",
    )

    # ----- Hero header -----
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">{APP_TITLE}</div>
            <div class="hero-subtitle">{APP_SUBTITLE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----- Reference tables (atomic mass, optional electronegativity) -----
    try:
        atomic = load_atomic_table_from_bytes(
            uploaded_atomic.getvalue() if uploaded_atomic else None,
            uploaded_atomic.name if uploaded_atomic else "")
        en_table = load_en_table_from_bytes(
            uploaded_en.getvalue() if uploaded_en else None,
            uploaded_en.name if uploaded_en else "")
    except Exception as exc:
        st.error(f"Reference table error: {exc}")
        st.stop()

    # ----- Blank boot: nothing loads until upload or sample is chosen -----
    if uploaded_data is None and not use_sample_data:
        st.markdown(
            """
            <div class="soft-card" style="text-align:center; padding:2.4rem 1.6rem;">
                <div class="big-number">Upload your class data to begin</div>
                <div class="helper-text" style="margin-top:0.6rem;">
                    Use <b>1. Data</b> in the sidebar to add a CHEM 120 spreadsheet (CSV or Excel).
                    No file handy? Tick <b>“Use sample class data instead”</b> to explore the example dataset.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    # ----- Load the class data (uploaded file wins over the sample) -----
    try:
        if uploaded_data is not None:
            raw_df = read_table_from_bytes(uploaded_data.getvalue(), uploaded_data.name)
        else:
            default_path = find_default_database()
            raw_df = read_local_table(str(default_path)) if default_path is not None else make_demo_dataset()
    except Exception as exc:
        st.error(f"Could not load class data: {exc}")
        st.stop()

    # ----- Run the data pipeline (each step is cached for speed) -----
    long_df, column_warnings = normalize_to_long_format(raw_df)
    if column_warnings:
        with st.expander("⚠️ Column mapping problems — expand for details", expanded=True):
            st.warning("One or more required columns could not be matched — usually the form wording changed. "
                       "See DEVELOPER_NOTES.md → When the Google Form changes.")
            for w in column_warnings:
                st.markdown(f"- {w}")

    # Apply any "Did you mean?" corrections the user chose in the Check tab, then
    # work out which invalid elements still need a decision.
    st.session_state.setdefault("element_fixes", {})
    long_df, corrections_log = apply_element_fixes(long_df, st.session_state.element_fixes)
    fix_proposals = propose_element_fixes(long_df, atomic)

    clean_df = clean_and_encode_data(long_df, atomic)
    described_df = add_chemical_descriptors(clean_df, atomic, en_table)

    # Session-staged manual rows (from the Add Compound tab).
    st.session_state.setdefault("manual_entries", pd.DataFrame())
    st.session_state.setdefault("pending_manual_entry", pd.DataFrame())
    st.session_state.setdefault("pending_manual_issues", pd.DataFrame())
    if not st.session_state.manual_entries.empty:
        described_df = pd.concat([described_df, st.session_state.manual_entries], ignore_index=True)

    # Remove rows whose elements aren't real periodic-table symbols (e.g. "BATU").
    # cleaning_log records what was dropped so the Check tab can show the changes.
    described_df, cleaning_log = remove_invalid_element_rows(described_df, atomic)

    issues_df = validate_compound_rows(described_df, atomic)
    outlier_df = detect_numeric_outliers(described_df, z_threshold=z_threshold)

    # ----- Two compact health cards above the tabs -----
    health_cols = st.columns(2)
    with health_cols[0]:
        style = "status-ok" if len(described_df) > 0 else "status-bad"
        st.markdown(f"""<div class="soft-card {style}"><b>Loaded compounds</b><br>
            <span class="helper-text">{len(described_df):,} rows ready for analysis.</span></div>""",
            unsafe_allow_html=True)
    with health_cols[1]:
        style = "status-ok" if len(issues_df) == 0 else "status-warn"
        st.markdown(f"""<div class="soft-card {style}"><b>Validation report</b><br>
            <span class="helper-text">{len(issues_df):,} possible data-entry issues. See the Check tab.</span></div>""",
            unsafe_allow_html=True)

    # ----- Everything lives in tabs (no long scroll) -----
    (tab_check, tab_explore, tab_heatmap, tab_structure, tab_ml,
     tab_add, tab_export, tab_semester) = st.tabs(
        ["✅ Check", "📊 Explore", "🔥 Heatmap", "🔬 Structure",
         "🔮 Predict", "➕ Add", "⬇️ Export", "🔄 New Semester"]
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
        render_ml_tab(described_df, atomic, en_table, use_phase_in_ml, use_mass_in_ml)
    with tab_add:
        render_add_compound_tab(atomic, en_table)
    with tab_export:
        render_export_tab(described_df, issues_df, outlier_df)
    with tab_semester:
        render_new_semester_tab(long_df)


if __name__ == "__main__":
    main()
