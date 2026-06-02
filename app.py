from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from chem120_excel_processor import process_workbook_bytes


st.set_page_config(
    page_title="CHEM 120 Data Compiler",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --chem-card-bg: rgba(255, 255, 255, 0.78);
        --chem-border: rgba(0, 0, 0, 0.08);
        --chem-text-soft: #6e6e73;
        --chem-blue: #007aff;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(0, 122, 255, 0.12), transparent 32rem),
            linear-gradient(180deg, #f5f5f7 0%, #ffffff 100%);
        color: #1d1d1f;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.72);
        backdrop-filter: blur(24px);
        border-right: 1px solid var(--chem-border);
    }

    .hero-card {
        padding: 2rem 2.25rem;
        border-radius: 28px;
        background: var(--chem-card-bg);
        border: 1px solid var(--chem-border);
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.07);
        margin-bottom: 1.25rem;
    }

    .hero-title {
        font-size: 3rem;
        line-height: 1.02;
        letter-spacing: -0.055em;
        font-weight: 800;
        margin: 0;
    }

    .hero-subtitle {
        font-size: 1.08rem;
        color: var(--chem-text-soft);
        max-width: 760px;
        margin-top: 0.85rem;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: rgba(0, 122, 255, 0.10);
        color: #0057b8;
        font-size: 0.86rem;
        font-weight: 650;
        margin-bottom: 0.85rem;
    }

    .feature-card {
        padding: 1.2rem;
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.74);
        border: 1px solid var(--chem-border);
        min-height: 145px;
    }

    .feature-title {
        font-weight: 750;
        font-size: 1rem;
        margin-bottom: 0.35rem;
    }

    .feature-copy {
        color: var(--chem-text-soft);
        font-size: 0.92rem;
        line-height: 1.5;
    }

    .small-muted {
        color: var(--chem-text-soft);
        font-size: 0.9rem;
    }

    div.stDownloadButton > button,
    div.stButton > button {
        border-radius: 999px !important;
        border: none !important;
        background: #007aff !important;
        color: white !important;
        padding: 0.72rem 1.1rem !important;
        font-weight: 700 !important;
        box-shadow: 0 10px 28px rgba(0, 122, 255, 0.25);
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.74);
        border: 1px solid var(--chem-border);
        border-radius: 22px;
        padding: 1rem 1.1rem;
    }

    hr {
        border: none;
        border-top: 1px solid var(--chem-border);
        margin: 1.75rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def preview_excel(uploaded_bytes: bytes, max_rows: int = 30) -> pd.DataFrame:
    """Return a lightweight preview of the first sheet."""
    return pd.read_excel(BytesIO(uploaded_bytes), nrows=max_rows)


with st.sidebar:
    st.markdown("## Upload")
    uploaded_file = st.file_uploader(
        "Drop the CHEM 120 Excel file here",
        type=["xlsx"],
        help="Upload the long-format or old wide-format spreadsheet. The app will fill atomic-number columns automatically.",
    )

    st.markdown("---")
    st.markdown("## Processing")
    fill_atomic = st.toggle("Fill atomic numbers", value=True)
    fill_codes = st.toggle("Fill phase/bubble codes", value=True)
    overwrite = st.toggle(
        "Correct existing values",
        value=True,
        help="If on, the app overwrites wrong values in yellow auto-fill columns. If off, it only fills blanks.",
    )

    st.markdown("---")
    st.caption("Output stays in Excel format and keeps the original workbook styling.")


st.markdown(
    """
    <div class="hero-card">
        <div class="pill">CHEM 120 · Data Compiler</div>
        <h1 class="hero-title">Clean chemistry data in one upload.</h1>
        <div class="hero-subtitle">
            Upload the student spreadsheet and the app fills the yellow helper columns automatically:
            atomic numbers from element symbols, phase numbers from phase text, and bubble codes from bubble text.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

feature_cols = st.columns(4)
features = [
    ("Atomic numbers", "Converts A, AP, B, BP, BDP, and O symbols into ZA, ZAP, ZB, ZBP, ZBDP, and ZO."),
    ("Legacy compatible", "Supports both long-format columns and old numbered columns like 1A → 1ZA."),
    ("Cleaner results", "Automatically converts pure/impure and yes/no/maybe into numeric codes."),
    ("Excel preserved", "Keeps workbook formatting while only updating the needed output columns."),
]
for col, (title, copy) in zip(feature_cols, features):
    with col:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-title">{title}</div>
                <div class="feature-copy">{copy}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<hr />", unsafe_allow_html=True)

if uploaded_file is None:
    st.info("Upload the Excel file from the top-left sidebar to begin.")
    st.markdown(
        """
        ### Expected columns

        For the SP26 long-format sheet, the app fills:

        `ZA ← A`, `ZAP ← AP`, `ZB ← B`, `ZBP ← BP`, `ZBDP ← BDP`, `ZO ← O`

        It can also fill:

        `PN ← P` and `BubN ← Bub`
        """
    )
else:
    uploaded_bytes = uploaded_file.getvalue()

    with st.spinner("Processing workbook..."):
        processed_bytes, report = process_workbook_bytes(
            uploaded_bytes,
            fill_atomic_numbers=fill_atomic,
            fill_phase_and_bubble_codes=fill_codes,
            overwrite_existing=overwrite,
        )

    st.success("Workbook processed successfully.")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Sheets processed", report["sheets_processed"])
    metric_cols[1].metric("Rows scanned", report["rows_scanned"])
    metric_cols[2].metric("Atomic cells filled", report["atomic_cells_filled"])
    metric_cols[3].metric("Codes filled", report["code_cells_filled"])

    if report["unknown_symbols"]:
        st.warning("Some element symbols could not be converted. Please check these rows.")
        st.dataframe(pd.DataFrame(report["unknown_symbols"]), use_container_width=True, hide_index=True)
    else:
        st.caption("No invalid element symbols were detected.")

    download_name = f"processed_{Path(uploaded_file.name).stem}.xlsx"
    st.download_button(
        label="Download cleaned Excel file",
        data=processed_bytes,
        file_name=download_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    with st.expander("Preview uploaded data", expanded=False):
        try:
            st.dataframe(preview_excel(uploaded_bytes), use_container_width=True)
        except Exception as exc:
            st.error(f"Preview failed: {exc}")

    with st.expander("What was filled?", expanded=True):
        st.markdown(
            """
            - **Atomic number columns:** `ZA`, `ZAP`, `ZB`, `ZBP`, `ZBDP`, `ZO`
            - **Phase code column:** `PN`
            - **Bubble code column:** `BubN`

            Code rules:
            - `impure → 1`
            - `pure → 2`
            - `maybe → 0`
            - `yes → 1`
            - `no → 2`
            """
        )
