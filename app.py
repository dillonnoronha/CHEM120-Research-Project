
"""
CHEM 120 Catalyst Insight Studio
================================

Student-friendly Streamlit dashboard for the CHEM 120 research project.

Purpose
-------
This app turns CHEM 120 lab entries into a clean, analyzable research dataset.
It is designed for future students, so the code is organized into clear blocks:

1. App configuration and styling
2. Data loading
3. Column mapping and reshaping
4. Cleaning and formula reconstruction
5. Validation / data-quality checks
6. Chemical descriptor calculations
7. Outlier detection
8. Plotting helpers
9. Machine-learning helpers
10. Streamlit user interface

How to update this app later
----------------------------
- Add a new descriptor in `add_chemical_descriptors()`.
- Add a new validation rule in `validate_compound_rows()`.
- Add support for a new survey column name in `infer_slot_column()`.
- Add a new chart in the Explore or Relationship Map tabs.
- Change the page look in `inject_css()`.

Run locally
-----------
    py -m pip install -r requirements.txt
    py -m streamlit run app.py
"""

from __future__ import annotations

import io
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except Exception:
    # The rest of the app works without scikit-learn.
    # Only the ML Lab tab will be disabled if sklearn is missing.
    SKLEARN_AVAILABLE = False


# =============================================================================
# 1. APP CONFIGURATION
# =============================================================================
# All high-level settings live here so future students can change app behavior
# without searching through the whole file.

APP_TITLE = "CHEM 120 Catalyst Insight Studio"
APP_SUBTITLE = "Turn class lab entries into clean formulas, visual trends, and testable hypotheses."

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

# The survey stores up to four compounds per group. Each compound is stored in a
# "slot" such as 1A, 1AN, 2A, 2AN, etc.
COMPOUND_SLOTS = [1, 2, 3, 4]

# CHEM 120 project label conversions.
# Keeping these dictionaries near the top makes the app easy to adapt if the
# class changes its coding scheme later.
PHASE_MAP = {"impure": 1, "pure": 2}
BUBBLE_MAP = {"maybe": 0, "yes": 1, "no": 2}

# Optional element/ratio fields used for mixed A-site and B-site compounds.
A_SITE_FIELDS = [("A", "AN"), ("AP", "APN")]
B_SITE_FIELDS = [("B", "BN"), ("BP", "BPN"), ("BDP", "BDPN")]

# Columns that are useful to show first in student-facing tables.
FRONT_COLUMNS = [
    "GroupNumber", "Instructor", "Semester", "Slot", "Formula",
    "A", "AN", "AP", "APN", "B", "BN", "BP", "BPN", "BDP", "BDPN",
    "O", "ON", "P", "Bub", "PhaseN", "BubN",
]


# =============================================================================
# 2. PAGE STYLE
# =============================================================================
# Streamlit apps can be made much cleaner with small CSS changes. This keeps the
# interface calm and modern without changing the underlying data logic.

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
        }

        .stTabs [aria-selected="true"] {
            background-color: #e8f1ff;
            color: #0057b8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# 3. SMALL UTILITY FUNCTIONS
# =============================================================================
# These helpers are intentionally simple. They clean text, parse numbers, and
# create display-friendly values used throughout the app.

def normalize_key(value: object) -> str:
    """Normalize a column name or label for fuzzy matching."""

    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def is_blank(value: object) -> bool:
    """Return True for None, NaN, empty strings, or whitespace-only values."""

    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return str(value).strip() == ""


def clean_text(value: object) -> str:
    """Convert a value into a trimmed string, using empty string for blanks."""

    if is_blank(value):
        return ""
    return str(value).strip()


def clean_label(value: object) -> str:
    """Clean labels like phase and bubble response into lowercase text."""

    return clean_text(value).lower()


def safe_float(value: object) -> float:
    """
    Convert a spreadsheet value into a float.

    Notes for future developers:
    - The survey asks students to enter decimals, not fractions.
    - This parser still accepts simple fractions like 1/2 so older data does not
      crash the app. Validation can still flag unusual formatting separately.
    """

    if is_blank(value):
        return np.nan

    text = clean_text(value).replace(",", "")

    # Accept simple fractions as a courtesy, even though the official template
    # asks for numbers only.
    if "/" in text and re.fullmatch(r"\d+(\.\d+)?/\d+(\.\d+)?", text):
        top, bottom = text.split("/")
        try:
            return float(top) / float(bottom)
        except ZeroDivisionError:
            return np.nan

    try:
        return float(text)
    except Exception:
        return np.nan


def format_ratio(value: object) -> str:
    """
    Format a numeric ratio for a chemical formula.

    Examples:
    - 1.0 becomes "" because chem formulas usually omit 1.
    - 2.0 becomes "2".
    - 0.5 becomes "0.5".
    """

    number = safe_float(value)
    if np.isnan(number) or number == 0:
        return ""
    if abs(number - 1.0) < 1e-9:
        return ""
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:g}"


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


def ordered_columns(df: pd.DataFrame, preferred: Sequence[str] = FRONT_COLUMNS) -> List[str]:
    """Put important columns first, then append the rest."""

    first = [c for c in preferred if c in df.columns]
    rest = [c for c in df.columns if c not in first]
    return first + rest


# =============================================================================
# 4. REFERENCE TABLES
# =============================================================================
# AtomicMass.csv provides element symbols, atomic numbers, and atomic masses. The
# app also supports an optional Pauling electronegativity table if one is added.

@st.cache_data(show_spinner=False)
def load_atomic_table_from_bytes(file_bytes: Optional[bytes], file_name: str = "") -> pd.DataFrame:
    """
    Load the atomic mass table.

    Parameters
    ----------
    file_bytes:
        Bytes from an uploaded CSV. If None, the function looks in data/.
    file_name:
        Kept so Streamlit cache can detect uploaded file changes.
    """

    if file_bytes:
        atomic = pd.read_csv(io.BytesIO(file_bytes))
    else:
        local_path = DATA_DIR / "AtomicMass.csv"
        if local_path.exists():
            atomic = pd.read_csv(local_path)
        else:
            # Small fallback table so the app still works if AtomicMass.csv is missing.
            atomic = pd.DataFrame(
                [
                    {"Element": "Oxygen", "Symbol": "O", "Z": 8, "AtomicMass": 15.999},
                    {"Element": "Calcium", "Symbol": "Ca", "Z": 20, "AtomicMass": 40.078},
                    {"Element": "Manganese", "Symbol": "Mn", "Z": 25, "AtomicMass": 54.938},
                    {"Element": "Iron", "Symbol": "Fe", "Z": 26, "AtomicMass": 55.845},
                    {"Element": "Cobalt", "Symbol": "Co", "Z": 27, "AtomicMass": 58.933},
                    {"Element": "Nickel", "Symbol": "Ni", "Z": 28, "AtomicMass": 58.693},
                    {"Element": "Zinc", "Symbol": "Zn", "Z": 30, "AtomicMass": 65.38},
                    {"Element": "Strontium", "Symbol": "Sr", "Z": 38, "AtomicMass": 87.62},
                    {"Element": "Lanthanum", "Symbol": "La", "Z": 57, "AtomicMass": 138.905},
                ]
            )

    # Standardize likely column names.
    rename_map = {}
    for col in atomic.columns:
        key = normalize_key(col)
        if key in {"element", "name"}:
            rename_map[col] = "Element"
        elif key in {"symbol", "elementsymbol"}:
            rename_map[col] = "Symbol"
        elif key in {"z", "atomicnumber"}:
            rename_map[col] = "Z"
        elif key in {"atomicmass", "mass", "weight", "atomicweight"}:
            rename_map[col] = "AtomicMass"
    atomic = atomic.rename(columns=rename_map)

    required = {"Symbol", "Z", "AtomicMass"}
    missing = required.difference(atomic.columns)
    if missing:
        raise ValueError(f"Atomic mass table is missing required columns: {sorted(missing)}")

    if "Element" not in atomic.columns:
        atomic["Element"] = atomic["Symbol"]

    atomic = atomic[["Element", "Symbol", "Z", "AtomicMass"]].copy()
    atomic["Symbol"] = atomic["Symbol"].astype(str).str.strip()
    atomic["Element"] = atomic["Element"].astype(str).str.strip()
    atomic["Z"] = pd.to_numeric(atomic["Z"], errors="coerce")
    atomic["AtomicMass"] = pd.to_numeric(atomic["AtomicMass"], errors="coerce")
    atomic = atomic.dropna(subset=["Symbol", "Z", "AtomicMass"]).drop_duplicates("Symbol")

    return atomic


@st.cache_data(show_spinner=False)
def load_en_table_from_bytes(file_bytes: Optional[bytes], file_name: str = "") -> pd.DataFrame:
    """Load an optional electronegativity table. Returns empty DataFrame if missing."""

    if not file_bytes:
        local_path = DATA_DIR / "PaulingEN.csv"
        if not local_path.exists():
            return pd.DataFrame(columns=["Symbol", "Electronegativity"])
        data = local_path.read_bytes()
    else:
        data = file_bytes

    en = pd.read_csv(io.BytesIO(data))
    rename_map = {}
    for col in en.columns:
        key = normalize_key(col)
        if key in {"symbol", "elementsymbol"}:
            rename_map[col] = "Symbol"
        elif key in {"en", "electronegativity", "pauling", "paulingen"}:
            rename_map[col] = "Electronegativity"
    en = en.rename(columns=rename_map)

    if {"Symbol", "Electronegativity"}.issubset(en.columns):
        en = en[["Symbol", "Electronegativity"]].copy()
        en["Symbol"] = en["Symbol"].astype(str).str.strip()
        en["Electronegativity"] = pd.to_numeric(en["Electronegativity"], errors="coerce")
        return en.dropna(subset=["Symbol"]).drop_duplicates("Symbol")

    return pd.DataFrame(columns=["Symbol", "Electronegativity"])


def make_element_maps(atomic: pd.DataFrame) -> Tuple[Dict[str, str], set]:
    """
    Build helper maps for cleaning element symbols.

    Returns
    -------
    name_to_symbol:
        Converts full names like "lanthanum" into "La".
    valid_symbols:
        Set of all recognized element symbols.
    """

    name_to_symbol = {
        str(row["Element"]).strip().lower(): str(row["Symbol"]).strip()
        for _, row in atomic.iterrows()
        if not is_blank(row.get("Element")) and not is_blank(row.get("Symbol"))
    }
    valid_symbols = set(atomic["Symbol"].astype(str).str.strip())
    return name_to_symbol, valid_symbols


def clean_symbol(value: object, name_to_symbol: Dict[str, str]) -> str:
    """
    Clean an element entry.

    Examples:
    - "la" becomes "La"
    - "LANthanum" becomes "La" if the atomic table contains Lanthanum
    - blank stays blank
    """

    text = clean_text(value)
    if not text:
        return ""

    lower = text.lower()
    if lower in name_to_symbol:
        return name_to_symbol[lower]

    # Element symbols are usually one or two letters. This turns "FE" into "Fe".
    if len(text) <= 3 and text.isalpha():
        return text[0].upper() + text[1:].lower()

    # Leave unknown text visible so validation can flag it.
    return text


# =============================================================================
# 5. DATA LOADING
# =============================================================================
# The app accepts CSV and Excel files. To improve load times, file reading is
# cached based on the uploaded file's bytes and filename.

@st.cache_data(show_spinner=False)
def read_table_from_bytes(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """Read an uploaded CSV or Excel file into a DataFrame."""

    suffix = Path(file_name).suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(io.BytesIO(file_bytes))

    if suffix in {".xlsx", ".xlsm", ".xls"}:
        # sheet_name=0 loads the first worksheet, which matches typical class exports.
        return pd.read_excel(io.BytesIO(file_bytes), sheet_name=0)

    raise ValueError("Unsupported file type. Please upload a CSV, XLSX, XLSM, or XLS file.")


def find_default_database() -> Optional[Path]:
    """
    Look for a class database saved near the app.

    This is convenient for instructors who want the app to auto-load a local file
    without manually uploading it every time.
    """

    candidates = [
        DATA_DIR / "Combined_Data.xlsx",
        DATA_DIR / "Everything CURE ML.xlsx",
        DATA_DIR / "CURE results FA25.xlsx",
        ROOT_DIR / "Combined_Data.xlsx",
        ROOT_DIR / "Everything CURE ML.xlsx",
        ROOT_DIR / "CURE results FA25.xlsx",
    ]

    for path in candidates:
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def read_local_table(path_string: str) -> pd.DataFrame:
    """Read a local default CSV or Excel file."""

    path = Path(path_string)
    data = path.read_bytes()
    return read_table_from_bytes(data, path.name)


def make_demo_dataset() -> pd.DataFrame:
    """
    Create a tiny demo dataset.

    The app uses this only if no uploaded/local database exists. This lets
    students explore the interface before connecting real class data.
    """

    return pd.DataFrame(
        [
            {
                "Group Number": "Demo 1", "Instructor": "Demo", "Semester": "Spring 2026",
                "1A": "La", "1AN": 2, "1AP": "", "1APN": "",
                "1B": "Ni", "1BN": 1, "1BP": "", "1BPN": "",
                "1BDP": "", "1BDPN": "", "1O": "O", "1ON": 4,
                "1P": "pure", "1Bub": "yes",
            },
            {
                "Group Number": "Demo 2", "Instructor": "Demo", "Semester": "Spring 2026",
                "1A": "Sr", "1AN": 1, "1AP": "", "1APN": "",
                "1B": "Fe", "1BN": 1, "1BP": "", "1BPN": "",
                "1BDP": "", "1BDPN": "", "1O": "O", "1ON": 3,
                "1P": "impure", "1Bub": "no",
            },
            {
                "Group Number": "Demo 3", "Instructor": "Demo", "Semester": "Spring 2026",
                "1A": "La", "1AN": 1, "1AP": "Sr", "1APN": 1,
                "1B": "Fe", "1BN": 1, "1BP": "", "1BPN": "",
                "1BDP": "", "1BDPN": "", "1O": "O", "1ON": 4,
                "1P": "pure", "1Bub": "maybe",
            },
            {
                "Group Number": "Demo 4", "Instructor": "Demo", "Semester": "Spring 2026",
                "1A": "La", "1AN": 1, "1AP": "", "1APN": "",
                "1B": "Fe", "1BN": 1, "1BP": "Co", "1BPN": 1,
                "1BDP": "Ni", "1BDPN": 1, "1O": "O", "1ON": 6,
                "1P": "pure", "1Bub": "yes",
            },
        ]
    )


# =============================================================================
# 6. COLUMN MAPPING AND RESHAPING
# =============================================================================
# Student spreadsheet exports may contain slightly different column names.
# These functions map the uploaded spreadsheet into one consistent structure.

def find_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    """
    Find the first column whose normalized name matches one of the candidates.

    This helps the app understand both short names like "Group Number" and
    alternate names like "Group ID".
    """

    normalized_columns = {normalize_key(c): c for c in columns}
    for candidate in candidates:
        key = normalize_key(candidate)
        if key in normalized_columns:
            return normalized_columns[key]
    return None


def infer_slot_column(columns: Iterable[str], slot: int, field: str) -> Optional[str]:
    """
    Locate a compound-slot column such as 1A, 1AN, 2Bub, or 3BDPN.

    Future update point:
    Add new candidate patterns here if a future survey changes the column names.
    """

    field_aliases = {
        "FormulaRef": ["formula", "fullformula", "compound", "referenceformula"],
        "A": ["a", "asite", "asiteelement", "aelement"],
        "AN": ["an", "aratio", "asiteratio", "aamount"],
        "AP": ["ap", "aprime", "asiteprime", "aprimeelement"],
        "APN": ["apn", "aprimeratio", "apratio", "asiteprimeratio"],
        "B": ["b", "bsite", "bsiteelement", "belement"],
        "BN": ["bn", "bratio", "bsiteratio", "bamount"],
        "BP": ["bp", "bprime", "bsiteprime", "bprimeelement"],
        "BPN": ["bpn", "bprimeratio", "bpratio", "bsiteprimeratio"],
        "BDP": ["bdp", "bdprime", "bdoubleprime", "bsite2prime", "b2prime"],
        "BDPN": ["bdpn", "bdprimeratio", "bdoubleprimeratio", "b2primeratio"],
        "O": ["o", "oxygen", "oxygenelement"],
        "ON": ["on", "oratio", "oxygenratio", "oxygenamount"],
        "P": ["p", "phase"],
        "PN": ["pn", "phasen", "phasenumber"],
        "Bub": ["bub", "bubble", "bubbleresponse", "h2bubble", "bubbles"],
        "BubN": ["bubn", "bubblen", "bubblenumber", "bubbleresponsen"],
    }

    aliases = field_aliases.get(field, [field])
    normalized_columns = {normalize_key(c): c for c in columns}

    # Exact compact patterns are fastest and match the optimized database.
    direct_candidates = []
    for alias in aliases:
        direct_candidates.extend(
            [
                f"{slot}{alias}",
                f"{slot}_{alias}",
                f"compound{slot}{alias}",
                f"compound{slot}_{alias}",
                f"c{slot}{alias}",
            ]
        )

    for candidate in direct_candidates:
        key = normalize_key(candidate)
        if key in normalized_columns:
            return normalized_columns[key]

    # Fuzzy fallback for longer survey exports. This is slower, but only runs
    # when exact matching fails.
    for original in columns:
        key = normalize_key(original)
        contains_slot = (
            key.startswith(str(slot))
            or f"compound{slot}" in key
            or f"section{slot}" in key
            or f"compoundnumber{slot}" in key
        )
        if not contains_slot:
            continue
        if any(alias in key for alias in aliases):
            return original

    return None


@st.cache_data(show_spinner=False)
def normalize_to_long_format(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the uploaded wide spreadsheet into one row per compound.

    Example input:
        Group | 1A | 1AN | 1B | 1BN | 2A | 2AN | 2B | 2BN

    Example output:
        Group | Slot | A | AN | B | BN
        Group | 1    | ... 
        Group | 2    | ...

    This shape is much easier for charts, validation, and machine learning.
    """

    if raw_df.empty:
        return pd.DataFrame()

    columns = list(raw_df.columns)

    # Map common metadata fields. If a column is not present, the app simply
    # leaves that metadata blank.
    metadata_candidates = {
        "GroupNumber": ["Group Number", "Group Number / ID", "Group ID", "Group", "Team"],
        "Email": ["Email", "Email Address"],
        "Name": ["Name", "Student Name"],
        "Members": ["Members", "Group Members"],
        "Instructor": ["Instructor", "Instructor / Section", "Section"],
        "Semester": ["Semester", "Semester / Year", "Term", "Year"],
    }
    metadata_columns = {
        output_name: find_column(columns, candidates)
        for output_name, candidates in metadata_candidates.items()
    }

    # Map compound-slot columns only once for speed.
    field_names = [
        "FormulaRef", "A", "AN", "AP", "APN", "B", "BN", "BP", "BPN",
        "BDP", "BDPN", "O", "ON", "P", "PN", "Bub", "BubN",
    ]
    slot_map = {
        slot: {field: infer_slot_column(columns, slot, field) for field in field_names}
        for slot in COMPOUND_SLOTS
    }

    rows: List[dict] = []

    for source_index, source_row in raw_df.iterrows():
        metadata = {
            output_name: source_row.get(input_col, "")
            if input_col is not None else ""
            for output_name, input_col in metadata_columns.items()
        }

        for slot in COMPOUND_SLOTS:
            mapped = slot_map[slot]

            # If a slot has no meaningful compound values, skip it. This prevents
            # empty Compound 2/3/4 slots from becoming fake rows.
            key_values = [
                source_row.get(mapped.get(field), "")
                for field in ["A", "AN", "B", "BN", "O", "ON", "P", "Bub"]
                if mapped.get(field) is not None
            ]
            if not key_values or all(is_blank(v) for v in key_values):
                continue

            record = dict(metadata)
            record["SourceRow"] = int(source_index) + 2  # +2 matches Excel row numbering with header row
            record["Slot"] = slot

            for field in field_names:
                col = mapped.get(field)
                record[field] = source_row.get(col, "") if col is not None else ""

            rows.append(record)

    return pd.DataFrame(rows)


# =============================================================================
# 7. CLEANING, LABEL ENCODING, AND FORMULA RECONSTRUCTION
# =============================================================================
# The official project asks students to split formulas into separate fields.
# These functions standardize those fields and rebuild a readable formula.

def ratio_for_formula(row: pd.Series, symbol_col: str, ratio_col: str, required: bool = False) -> float:
    """
    Choose a formula ratio for a symbol.

    Required fields like A/AN and B/BN need their ratio from the spreadsheet.
    Optional fields like AP and BP sometimes do not have ratio columns in older
    templates, so the app assumes 1 if the symbol is present but ratio is blank.
    """

    symbol = clean_text(row.get(symbol_col, ""))
    ratio = safe_float(row.get(ratio_col, np.nan))

    if not symbol:
        return 0.0

    if np.isnan(ratio) or ratio == 0:
        return 1.0 if not required else np.nan

    return ratio


def reconstruct_formula(row: pd.Series) -> str:
    """
    Rebuild a formula from the split fields.

    Formula order:
        A, A′, B, B′, B″, O

    Example:
        A=La, AN=2, B=Ni, BN=1, O=O, ON=4 -> La2NiO4
    """

    parts: List[str] = []

    for symbol_col, ratio_col, required in [
        ("A", "AN", True),
        ("AP", "APN", False),
        ("B", "BN", True),
        ("BP", "BPN", False),
        ("BDP", "BDPN", False),
        ("O", "ON", True),
    ]:
        symbol = clean_text(row.get(symbol_col, ""))
        if not symbol:
            continue

        ratio = ratio_for_formula(row, symbol_col, ratio_col, required=required)
        if np.isnan(ratio):
            parts.append(symbol)
        else:
            parts.append(f"{symbol}{format_ratio(ratio)}")

    return "".join(parts)


@st.cache_data(show_spinner=False)
def clean_and_encode_data(long_df: pd.DataFrame, atomic: pd.DataFrame) -> pd.DataFrame:
    """
    Clean text fields, ratios, formula strings, and project label encodings.

    This function is cached because it can be reused by several tabs.
    """

    if long_df.empty:
        return long_df.copy()

    df = long_df.copy()
    name_to_symbol, _valid_symbols = make_element_maps(atomic)

    # Clean element symbols.
    for col in ["A", "AP", "B", "BP", "BDP", "O"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].apply(lambda value: clean_symbol(value, name_to_symbol))

    # Convert ratio columns into numeric values.
    for col in ["AN", "APN", "BN", "BPN", "BDPN", "ON"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].apply(safe_float)

    # Clean text labels.
    for col in ["P", "Bub"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].apply(clean_label)

    # If O is blank but ON is filled, assume O because the survey expects oxygen.
    if "O" in df.columns:
        df.loc[(df["O"] == "") & df["ON"].notna(), "O"] = "O"

    # Convert phase and bubble response into numeric labels used by charts/ML.
    df["PhaseN"] = df["P"].map(PHASE_MAP)
    df["BubN"] = df["Bub"].map(BUBBLE_MAP)
    df["BubbleYes"] = (df["Bub"] == "yes").astype(int)

    # Reconstruct formula after cleaning.
    df["Formula"] = df.apply(reconstruct_formula, axis=1)

    # Create clean student-friendly labels for display.
    df["PhaseLabel"] = df["P"].replace({"": "missing"})
    df["BubbleLabel"] = df["Bub"].replace({"": "missing"})

    return df


# =============================================================================
# 8. VALIDATION AND OUTLIER DETECTION
# =============================================================================
# Data quality is a major part of this project because students may accidentally
# enter full names, missing ratios, invalid labels, or blank values.

@st.cache_data(show_spinner=False)
def validate_compound_rows(clean_df: pd.DataFrame, atomic: pd.DataFrame) -> pd.DataFrame:
    """
    Check cleaned rows against the CHEM 120 data-entry rules.

    Future update point:
    Add new validation checks inside this function if the lab template changes.
    """

    if clean_df.empty:
        return pd.DataFrame(columns=["Row", "GroupNumber", "Slot", "Formula", "Field", "Issue", "Suggestion"])

    _name_to_symbol, valid_symbols = make_element_maps(atomic)
    issues: List[dict] = []

    def add_issue(row: pd.Series, field: str, issue: str, suggestion: str) -> None:
        issues.append(
            {
                "Row": row.get("SourceRow", ""),
                "GroupNumber": row.get("GroupNumber", ""),
                "Slot": row.get("Slot", ""),
                "Formula": row.get("Formula", ""),
                "Field": field,
                "Issue": issue,
                "Suggestion": suggestion,
            }
        )

    for _, row in clean_df.iterrows():
        # Required element fields.
        if is_blank(row.get("A")):
            add_issue(row, "A", "Missing A-site element.", "Enter an element symbol like La, Sr, or Ca.")
        if is_blank(row.get("B")):
            add_issue(row, "B", "Missing B-site element.", "Enter an element symbol like Ni, Fe, Co, Mn, or Zn.")

        # Oxygen must be O.
        if clean_text(row.get("O")) != "O":
            add_issue(row, "O", "Oxygen field is not O.", "Enter the symbol O, not the word oxygen.")

        # Check element symbols against the atomic table.
        for field in ["A", "AP", "B", "BP", "BDP", "O"]:
            symbol = clean_text(row.get(field, ""))
            if symbol and symbol not in valid_symbols:
                add_issue(row, field, f"Unknown element symbol '{symbol}'.", "Use a valid periodic-table symbol.")

        # Required numeric ratios.
        for field in ["AN", "BN", "ON"]:
            value = safe_float(row.get(field))
            if np.isnan(value):
                add_issue(row, field, "Missing or non-numeric ratio.", "Use numbers only, such as 1, 2, 3, or 0.5.")
            elif value <= 0:
                add_issue(row, field, "Ratio must be greater than zero.", "Use a positive number.")

        # Optional ratio without optional element.
        for symbol_field, ratio_field in [("AP", "APN"), ("BP", "BPN"), ("BDP", "BDPN")]:
            symbol = clean_text(row.get(symbol_field, ""))
            ratio = safe_float(row.get(ratio_field))
            if symbol and not np.isnan(ratio) and ratio <= 0:
                add_issue(row, ratio_field, "Optional element ratio is zero or negative.", "Use a positive ratio or leave both optional fields blank.")
            if not symbol and not np.isnan(ratio) and ratio > 0:
                add_issue(row, symbol_field, f"{ratio_field} has a number but {symbol_field} is blank.", "Enter the optional element symbol or clear the ratio.")

        # Allowed labels.
        if clean_label(row.get("P")) not in PHASE_MAP:
            add_issue(row, "P", "Invalid phase label.", "Use exactly: pure or impure.")
        if clean_label(row.get("Bub")) not in BUBBLE_MAP:
            add_issue(row, "Bub", "Invalid bubble response.", "Use exactly: yes, no, or maybe.")

    return pd.DataFrame(issues)


@st.cache_data(show_spinner=False)
def detect_numeric_outliers(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """
    Flag unusual numeric values using z-scores.

    This is a screening tool, not a final judgment. It helps students notice
    entries that may deserve a second look.
    """

    numeric_cols = [
        "AN", "APN", "BN", "BPN", "BDPN", "ON",
        "FormulaMass", "O_to_cation_ratio", "B_to_A_ratio",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    rows: List[dict] = []
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 4:
            continue
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            continue
        mean = series.mean()
        for idx, value in pd.to_numeric(df[col], errors="coerce").items():
            if np.isnan(value):
                continue
            z = (value - mean) / std
            if abs(z) >= z_threshold:
                rows.append(
                    {
                        "Row": df.loc[idx, "SourceRow"] if "SourceRow" in df.columns else idx,
                        "GroupNumber": df.loc[idx, "GroupNumber"] if "GroupNumber" in df.columns else "",
                        "Slot": df.loc[idx, "Slot"] if "Slot" in df.columns else "",
                        "Formula": df.loc[idx, "Formula"] if "Formula" in df.columns else "",
                        "Column": col,
                        "Value": value,
                        "Z_score": round(float(z), 2),
                        "Meaning": "Unusually high/low compared with the rest of the uploaded data.",
                    }
                )
    return pd.DataFrame(rows)


# =============================================================================
# 9. CHEMICAL DESCRIPTORS
# =============================================================================
# Descriptors are numeric chemistry features created from the split formula.
# They make the data usable for graphs and machine learning.

def lookup_atomic_value(atomic_index: pd.DataFrame, symbol: object, column: str) -> float:
    """Look up an atomic property by element symbol."""

    symbol = clean_text(symbol)
    if not symbol or symbol not in atomic_index.index:
        return np.nan
    return float(atomic_index.loc[symbol, column])


def weighted_average(values: Sequence[float], weights: Sequence[float]) -> float:
    """Calculate a weighted average while ignoring missing values."""

    pairs = [(v, w) for v, w in zip(values, weights) if not np.isnan(v) and not np.isnan(w) and w > 0]
    if not pairs:
        return np.nan
    total_weight = sum(w for _v, w in pairs)
    if total_weight == 0:
        return np.nan
    return sum(v * w for v, w in pairs) / total_weight


@st.cache_data(show_spinner=False)
def add_chemical_descriptors(clean_df: pd.DataFrame, atomic: pd.DataFrame, en_table: pd.DataFrame) -> pd.DataFrame:
    """
    Add calculated chemistry descriptors.

    Future update point:
    Add new descriptors here, such as ionic radius, oxidation state estimates,
    tolerance factor, electronegativity difference, or group/period features.
    """

    if clean_df.empty:
        return clean_df.copy()

    df = clean_df.copy()

    atomic_index = atomic.set_index("Symbol")
    en_index = en_table.set_index("Symbol") if not en_table.empty and "Symbol" in en_table.columns else pd.DataFrame()

    # Add per-element atomic number and atomic mass columns.
    for symbol_col in ["A", "AP", "B", "BP", "BDP", "O"]:
        df[f"{symbol_col}_Z"] = df[symbol_col].apply(lambda s: lookup_atomic_value(atomic_index, s, "Z"))
        df[f"{symbol_col}_Mass"] = df[symbol_col].apply(lambda s: lookup_atomic_value(atomic_index, s, "AtomicMass"))

        if not en_index.empty and "Electronegativity" in en_index.columns:
            df[f"{symbol_col}_EN"] = df[symbol_col].apply(
                lambda s: float(en_index.loc[s, "Electronegativity"]) if clean_text(s) in en_index.index else np.nan
            )

    # Clean ratio columns and optional ratio defaults for descriptor math.
    for col in ["AN", "APN", "BN", "BPN", "BDPN", "ON"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Site-level totals.
    df["A_total_ratio"] = df.apply(
        lambda r: sum(
            ratio_for_formula(r, sym, ratio, required=(sym == "A"))
            for sym, ratio in A_SITE_FIELDS
            if clean_text(r.get(sym, ""))
        ),
        axis=1,
    )
    df["B_total_ratio"] = df.apply(
        lambda r: sum(
            ratio_for_formula(r, sym, ratio, required=(sym == "B"))
            for sym, ratio in B_SITE_FIELDS
            if clean_text(r.get(sym, ""))
        ),
        axis=1,
    )

    # Site-weighted atomic numbers and masses.
    def site_average(row: pd.Series, site_fields: Sequence[Tuple[str, str]], property_suffix: str) -> float:
        values = []
        weights = []
        for symbol_col, ratio_col in site_fields:
            if clean_text(row.get(symbol_col, "")):
                values.append(safe_float(row.get(f"{symbol_col}_{property_suffix}")))
                weights.append(ratio_for_formula(row, symbol_col, ratio_col, required=(symbol_col in {"A", "B"})))
        return weighted_average(values, weights)

    df["A_avg_Z"] = df.apply(lambda r: site_average(r, A_SITE_FIELDS, "Z"), axis=1)
    df["B_avg_Z"] = df.apply(lambda r: site_average(r, B_SITE_FIELDS, "Z"), axis=1)
    df["A_avg_mass"] = df.apply(lambda r: site_average(r, A_SITE_FIELDS, "Mass"), axis=1)
    df["B_avg_mass"] = df.apply(lambda r: site_average(r, B_SITE_FIELDS, "Mass"), axis=1)

    if any(c.endswith("_EN") for c in df.columns):
        df["A_avg_EN"] = df.apply(lambda r: site_average(r, A_SITE_FIELDS, "EN"), axis=1)
        df["B_avg_EN"] = df.apply(lambda r: site_average(r, B_SITE_FIELDS, "EN"), axis=1)
        df["EN_difference_B_minus_A"] = df["B_avg_EN"] - df["A_avg_EN"]

    # Formula mass estimate.
    def formula_mass(row: pd.Series) -> float:
        total = 0.0
        has_value = False
        for symbol_col, ratio_col, required in [
            ("A", "AN", True),
            ("AP", "APN", False),
            ("B", "BN", True),
            ("BP", "BPN", False),
            ("BDP", "BDPN", False),
            ("O", "ON", True),
        ]:
            symbol = clean_text(row.get(symbol_col, ""))
            if not symbol:
                continue
            mass = safe_float(row.get(f"{symbol_col}_Mass"))
            ratio = ratio_for_formula(row, symbol_col, ratio_col, required=required)
            if not np.isnan(mass) and not np.isnan(ratio):
                total += mass * ratio
                has_value = True
        return total if has_value else np.nan

    df["FormulaMass"] = df.apply(formula_mass, axis=1)

    # Simple structural ratios.
    cation_total = df["A_total_ratio"] + df["B_total_ratio"]
    df["O_to_cation_ratio"] = df["ON"] / cation_total.replace({0: np.nan})
    df["B_to_A_ratio"] = df["B_total_ratio"] / df["A_total_ratio"].replace({0: np.nan})

    # Binary flags for mixed-site chemistry.
    df["Mixed_A_site"] = (df["AP"].astype(str).str.len() > 0).astype(int)
    df["Mixed_B_site"] = ((df["BP"].astype(str).str.len() > 0) | (df["BDP"].astype(str).str.len() > 0)).astype(int)

    return df


# =============================================================================
# 10. SUMMARY TABLES AND PLOTTING HELPERS
# =============================================================================
# These functions create student-readable charts and summaries. Each chart is
# paired with short explanation text in the UI.

def summarize_by_element(df: pd.DataFrame, element_col: str, min_rows: int = 1) -> pd.DataFrame:
    """Summarize bubble and phase results by a selected element column."""

    if df.empty or element_col not in df.columns:
        return pd.DataFrame()

    subset = df[~df[element_col].isna() & (df[element_col].astype(str).str.strip() != "")].copy()
    if subset.empty:
        return pd.DataFrame()

    summary = (
        subset.groupby(element_col)
        .agg(
            compounds=("Formula", "count"),
            bubble_yes_rate=("BubbleYes", "mean"),
            avg_bubble_number=("BubN", "mean"),
            pure_rate=("PhaseN", lambda s: (s == 2).mean()),
            avg_formula_mass=("FormulaMass", "mean"),
        )
        .reset_index()
    )
    summary = summary[summary["compounds"] >= min_rows]
    summary["bubble_yes_rate"] = summary["bubble_yes_rate"] * 100
    summary["pure_rate"] = summary["pure_rate"] * 100

    return summary.sort_values(["bubble_yes_rate", "compounds"], ascending=[False, False])


def plot_bar_chart(data: pd.DataFrame, x_col: str, y_col: str, title: str, ylabel: str):
    """Return a clean matplotlib bar chart."""

    fig, ax = plt.subplots(figsize=(8, 4.2))
    plot_data = data[[x_col, y_col]].dropna().head(12)
    ax.bar(plot_data[x_col].astype(str), plot_data[y_col])
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_distribution(df: pd.DataFrame, column: str, title: str):
    """Return a simple count chart for labels such as phase or bubble response."""

    counts = df[column].fillna("missing").astype(str).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(counts.index, counts.values)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of compounds")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def numeric_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a correlation matrix from the numeric columns used in the app."""

    candidates = [
        "BubbleYes", "BubN", "PhaseN",
        "AN", "APN", "BN", "BPN", "BDPN", "ON",
        "A_avg_Z", "B_avg_Z", "A_avg_mass", "B_avg_mass",
        "FormulaMass", "O_to_cation_ratio", "B_to_A_ratio",
        "Mixed_A_site", "Mixed_B_site",
    ]
    candidates += [c for c in ["A_avg_EN", "B_avg_EN", "EN_difference_B_minus_A"] if c in df.columns]

    numeric_cols = [c for c in candidates if c in df.columns]
    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")

    if numeric.shape[1] < 2:
        return pd.DataFrame()

    return numeric.corr(numeric_only=True)


def plot_heatmap(corr: pd.DataFrame):
    """Return a beginner-friendly correlation heatmap figure."""

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)

    # Annotate cells only when the matrix is reasonably small.
    if corr.shape[0] <= 14:
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                value = corr.iloc[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)

    ax.set_title("Relationship Map: numeric features compared with each other", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation from -1 to +1")
    fig.tight_layout()
    return fig


def bubble_relationships(corr: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the strongest relationships with bubble response.

    This makes the heatmap easier for students who are not used to reading
    correlation matrices.
    """

    target_col = "BubbleYes" if "BubbleYes" in corr.columns else "BubN" if "BubN" in corr.columns else None
    if target_col is None:
        return pd.DataFrame()

    rel = corr[target_col].drop(labels=[target_col], errors="ignore").dropna()
    if rel.empty:
        return pd.DataFrame()

    out = rel.reset_index()
    out.columns = ["Feature", "Correlation with bubble result"]
    out["Strength"] = out["Correlation with bubble result"].abs()
    out = out.sort_values("Strength", ascending=False).drop(columns=["Strength"])
    return out.head(10)


# =============================================================================
# 11. MACHINE LEARNING HELPERS
# =============================================================================
# The ML section is intentionally simple and transparent. It is for generating
# hypotheses, not proving final scientific conclusions.

def feature_columns(use_phase: bool = True) -> Tuple[List[str], List[str]]:
    """
    Return numeric and categorical features used by the model.

    Future update point:
    Add or remove features here if the class wants a different ML experiment.
    """

    numeric = [
        "AN", "APN", "BN", "BPN", "BDPN", "ON",
        "A_avg_Z", "B_avg_Z", "A_avg_mass", "B_avg_mass",
        "FormulaMass", "O_to_cation_ratio", "B_to_A_ratio",
        "Mixed_A_site", "Mixed_B_site",
    ]
    if use_phase:
        numeric.append("PhaseN")

    categorical = ["A", "AP", "B", "BP", "BDP"]
    return numeric, categorical


def build_feature_matrix(
    df: pd.DataFrame,
    use_phase: bool = True,
    expected_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build one-hot encoded model features.

    If expected_columns is provided, the output is aligned to a trained model's
    column order. This is important when predicting a new single compound.
    """

    numeric_cols, categorical_cols = feature_columns(use_phase=use_phase)
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    numeric_part = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    categorical_part = pd.get_dummies(
        df[categorical_cols].fillna("").astype(str),
        prefix=categorical_cols,
        dummy_na=False,
    )

    features = pd.concat([numeric_part, categorical_part], axis=1)

    if expected_columns is not None:
        features = features.reindex(columns=expected_columns, fill_value=0)

    return features


@st.cache_data(show_spinner=False)
def train_ml_model(df: pd.DataFrame, use_phase: bool = True, random_state: int = 42) -> dict:
    """
    Train a Random Forest model to predict Bubble = yes.

    Returns a dictionary with the model, feature columns, accuracy, and feature
    importance table. If the data is not sufficient, returns an error message.
    """

    if not SKLEARN_AVAILABLE:
        return {"ok": False, "message": "scikit-learn is not installed. Install requirements.txt to use ML Lab."}

    model_df = df.dropna(subset=["BubbleYes"]).copy()
    model_df = model_df[model_df["Bub"].isin(["yes", "no", "maybe"])]

    if len(model_df) < 10:
        return {
            "ok": False,
            "message": "ML Lab needs at least 10 labeled compound rows. Upload more class data or use the app for cleaning/exploration first.",
        }

    if model_df["BubbleYes"].nunique() < 2:
        return {
            "ok": False,
            "message": "ML Lab needs both bubble=yes and bubble≠yes examples. The current data only has one class.",
        }

    X = build_feature_matrix(model_df, use_phase=use_phase)
    y = model_df["BubbleYes"].astype(int)

    # Use a stratified split when possible to preserve yes/no balance.
    stratify = y if y.value_counts().min() >= 2 else None
    test_size = 0.25 if len(model_df) >= 20 else 0.35

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=7,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    importance = pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)
    importance = importance[importance["Importance"] > 0].head(12)

    return {
        "ok": True,
        "model": model,
        "feature_columns": list(X.columns),
        "accuracy": float(accuracy),
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "importance": importance,
    }


def make_single_prediction_row(
    atomic: pd.DataFrame,
    en_table: pd.DataFrame,
    phase: str,
    a: str,
    an: float,
    ap: str,
    apn: float,
    b: str,
    bn: float,
    bp: str,
    bpn: float,
    bdp: str,
    bdpn: float,
    on: float,
) -> pd.DataFrame:
    """Create one cleaned/described row from ML Lab form input."""

    raw = pd.DataFrame(
        [
            {
                "GroupNumber": "Prediction",
                "Instructor": "",
                "Semester": "",
                "SourceRow": "",
                "Slot": 1,
                "FormulaRef": "",
                "A": a,
                "AN": an,
                "AP": ap,
                "APN": apn,
                "B": b,
                "BN": bn,
                "BP": bp,
                "BPN": bpn,
                "BDP": bdp,
                "BDPN": bdpn,
                "O": "O",
                "ON": on,
                "P": phase,
                "Bub": "maybe",
            }
        ]
    )
    clean = clean_and_encode_data(raw, atomic)
    described = add_chemical_descriptors(clean, atomic, en_table)
    return described


# =============================================================================
# 12. EXPORT HELPERS
# =============================================================================
# These helpers convert DataFrames to downloadable files.

@st.cache_data(show_spinner=False)
def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to UTF-8 CSV bytes."""

    return df.to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False)
def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    """Convert a DataFrame to an Excel file in memory."""

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return buffer.getvalue()


# =============================================================================
# 13. STREAMLIT APP
# =============================================================================
# The code below builds the actual app interface. The heavy data work is already
# done by the functions above, which keeps this section easier to read.

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
    uploaded_atomic = st.sidebar.file_uploader(
        "AtomicMass.csv (optional)",
        type=["csv"],
        help="Optional. The package already includes AtomicMass.csv.",
    )
    uploaded_en = st.sidebar.file_uploader(
        "PaulingEN.csv (optional)",
        type=["csv"],
        help="Optional. Add electronegativity descriptors if you have this file.",
    )

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
                <span class="step-pill">2 Clean</span>
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
    data_source = "Demo dataset"
    try:
        if uploaded_data is not None:
            raw_df = read_table_from_bytes(uploaded_data.getvalue(), uploaded_data.name)
            data_source = uploaded_data.name
        else:
            default_path = find_default_database()
            if default_path is not None:
                raw_df = read_local_table(str(default_path))
                data_source = default_path.name
            else:
                raw_df = make_demo_dataset()
                data_source = "Built-in demo dataset"
    except Exception as exc:
        st.error(f"Could not load class data: {exc}")
        st.stop()

    # -------------------------------------------------------------------------
    # Run the main data pipeline. Each step is cached for speed.
    # -------------------------------------------------------------------------
    long_df = normalize_to_long_format(raw_df)
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

    # -------------------------------------------------------------------------
    # Compact status strip.
    # -------------------------------------------------------------------------
    status_cols = st.columns(5)
    with status_cols[0]:
        st.metric("Data source", data_source)
    with status_cols[1]:
        st.metric("Raw rows", f"{len(raw_df):,}")
    with status_cols[2]:
        st.metric("Compound rows", f"{len(described_df):,}")
    with status_cols[3]:
        st.metric("Validation issues", f"{len(issues_df):,}")
    with status_cols[4]:
        st.metric("Bubble yes", f"{int(described_df.get('BubbleYes', pd.Series(dtype=int)).sum()):,}")

    # -------------------------------------------------------------------------
    # Main navigation. Each tab is ordered to match how a student should use it.
    # -------------------------------------------------------------------------
    tabs = st.tabs(
        [
            "🏁 Start Here",
            "✅ Check Data",
            "📊 Explore Results",
            "🧭 Relationship Map",
            "🤖 ML Lab",
            "➕ Add Compound",
            "⬇️ Export",
        ]
    )

    # -------------------------------------------------------------------------
    # TAB 1: START HERE
    # -------------------------------------------------------------------------
    with tabs[0]:
        st.subheader("What this app does")
        st.write(
            "This app helps CHEM 120 students turn split lab entries into formulas, "
            "check for data-entry mistakes, calculate chemistry descriptors, and look "
            "for patterns related to bubble response."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Step 1", "Upload", "Use the left sidebar to upload Combined_Data.xlsx.")
        with c2:
            metric_card("Step 2", "Check", "Fix missing elements, invalid labels, and strange ratios.")
        with c3:
            metric_card("Step 3", "Explore", "Use charts and ML to form testable hypotheses.")

        st.markdown("### Current data health")
        health_cols = st.columns(3)
        with health_cols[0]:
            style = "status-ok" if len(described_df) > 0 else "status-bad"
            st.markdown(
                f"""<div class="soft-card {style}">
                <b>Loaded compounds</b><br>
                <span class="helper-text">{len(described_df):,} compound rows are available for analysis.</span>
                </div>""",
                unsafe_allow_html=True,
            )
        with health_cols[1]:
            style = "status-ok" if len(issues_df) == 0 else "status-warn"
            st.markdown(
                f"""<div class="soft-card {style}">
                <b>Validation report</b><br>
                <span class="helper-text">{len(issues_df):,} possible data-entry issues were found.</span>
                </div>""",
                unsafe_allow_html=True,
            )
        with health_cols[2]:
            yes_count = int(described_df["BubbleYes"].sum()) if "BubbleYes" in described_df.columns else 0
            style = "status-ok" if yes_count > 0 else "status-warn"
            st.markdown(
                f"""<div class="soft-card {style}">
                <b>Bubble outcomes</b><br>
                <span class="helper-text">{yes_count:,} compounds are labeled bubble = yes.</span>
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

        st.markdown("### Preview of cleaned compounds")
        if described_df.empty:
            st.warning("No compound rows were found. Check whether the uploaded file uses recognizable column names like 1A, 1AN, 1B, 1BN, 1O, 1ON, 1P, and 1Bub.")
        else:
            st.dataframe(
                described_df[ordered_columns(described_df)].head(10),
                use_container_width=True,
                height=310,
            )

    # -------------------------------------------------------------------------
    # TAB 2: CHECK DATA
    # -------------------------------------------------------------------------
    with tabs[1]:
        st.subheader("Check Data")
        st.write(
            "This section catches common student-entry mistakes before the dataset is used for graphs or machine learning."
        )

        check_cols = st.columns([1, 1])
        with check_cols[0]:
            st.markdown("#### Validation issues")
            if issues_df.empty:
                st.success("No validation issues found.")
            else:
                st.warning(f"{len(issues_df):,} possible issues found. Review these before final analysis.")
                st.dataframe(issues_df, use_container_width=True, height=330)
        with check_cols[1]:
            st.markdown("#### Possible numeric outliers")
            if outlier_df.empty:
                st.success("No numeric outliers found at the current sensitivity.")
            else:
                st.info("Outliers are not automatically wrong. They are values worth double-checking.")
                st.dataframe(outlier_df, use_container_width=True, height=330)

        st.markdown("#### Cleaned compound table")
        display_cols = ordered_columns(described_df)
        st.dataframe(described_df[display_cols], use_container_width=True, height=420)

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

    # -------------------------------------------------------------------------
    # TAB 3: EXPLORE RESULTS
    # -------------------------------------------------------------------------
    with tabs[2]:
        st.subheader("Explore Results")
        st.write(
            "Use this section to compare compounds and look for patterns. These graphs are descriptive, not proof of causation."
        )

        # Student-friendly filters live in the tab so the sidebar stays simple.
        filter_cols = st.columns(3)
        with filter_cols[0]:
            semesters = sorted([x for x in described_df.get("Semester", pd.Series(dtype=str)).dropna().astype(str).unique() if x.strip()])
            selected_semesters = st.multiselect("Semester filter", semesters, default=semesters)
        with filter_cols[1]:
            instructors = sorted([x for x in described_df.get("Instructor", pd.Series(dtype=str)).dropna().astype(str).unique() if x.strip()])
            selected_instructors = st.multiselect("Instructor filter", instructors, default=instructors)
        with filter_cols[2]:
            min_rows = st.number_input("Minimum rows per element", min_value=1, max_value=50, value=1, step=1)

        filtered = described_df.copy()
        if selected_semesters:
            filtered = filtered[filtered["Semester"].astype(str).isin(selected_semesters)]
        if selected_instructors:
            filtered = filtered[filtered["Instructor"].astype(str).isin(selected_instructors)]

        if filtered.empty:
            st.warning("No rows match the selected filters.")
        else:
            dist_cols = st.columns(2)
            with dist_cols[0]:
                st.pyplot(plot_distribution(filtered, "BubbleLabel", "Bubble response counts"), use_container_width=True)
                st.caption("This shows how many compounds were recorded as yes, no, maybe, or missing.")
            with dist_cols[1]:
                st.pyplot(plot_distribution(filtered, "PhaseLabel", "Phase counts"), use_container_width=True)
                st.caption("This shows how many compounds were recorded as pure or impure.")

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
                    st.dataframe(a_summary.round(3), use_container_width=True, height=260)
            with chart_cols[1]:
                st.markdown("#### B-site trend")
                if b_summary.empty:
                    st.info("Not enough B-site data for the current filter.")
                else:
                    st.pyplot(
                        plot_bar_chart(b_summary, "B", "bubble_yes_rate", "B-site bubble yes rate", "Bubble yes rate (%)"),
                        use_container_width=True,
                    )
                    st.dataframe(b_summary.round(3), use_container_width=True, height=260)

            st.markdown("### Descriptor preview")
            descriptor_cols = [
                "Formula", "A_avg_Z", "B_avg_Z", "A_avg_mass", "B_avg_mass",
                "FormulaMass", "O_to_cation_ratio", "B_to_A_ratio", "BubbleLabel", "PhaseLabel",
            ]
            descriptor_cols = [c for c in descriptor_cols if c in filtered.columns]
            st.dataframe(filtered[descriptor_cols].round(4), use_container_width=True, height=300)

    # -------------------------------------------------------------------------
    # TAB 4: RELATIONSHIP MAP / HEATMAP
    # -------------------------------------------------------------------------
    with tabs[3]:
        st.subheader("Relationship Map")
        st.write(
            "This tab explains the heatmap in plain language so students can use it without already knowing statistics."
        )

        corr = numeric_correlation_table(described_df)

        if corr.empty:
            st.warning("Not enough numeric columns are available to build a relationship map.")
        else:
            explain_cols = st.columns([1.2, 1])
            with explain_cols[0]:
                st.pyplot(plot_heatmap(corr), use_container_width=True)
            with explain_cols[1]:
                st.markdown("#### How to read this heatmap")
                st.markdown(
                    """
                    Each square compares two numeric features.

                    - **+1.00** means the two features tend to increase together.
                    - **0.00** means there is little linear relationship.
                    - **-1.00** means one tends to increase while the other decreases.
                    - Look at the **BubbleYes** row/column to see what may relate to bubbling.
                    - This does **not** prove cause and effect. It only suggests patterns to investigate.
                    """
                )

                rel = bubble_relationships(corr)
                st.markdown("#### Strongest relationships with bubble result")
                if rel.empty:
                    st.info("Bubble relationship summary is not available yet.")
                else:
                    st.dataframe(rel.round(3), use_container_width=True, height=300)

            with st.expander("Example interpretation"):
                st.markdown(
                    """
                    If `B_avg_Z` has a positive value with `BubbleYes`, then compounds with a higher
                    average B-site atomic number tended to have more `bubble = yes` results in this dataset.

                    That does **not** mean atomic number causes bubbling. It means the pattern is worth
                    discussing and possibly testing with more experiments.
                    """
                )

    # -------------------------------------------------------------------------
    # TAB 5: ML LAB
    # -------------------------------------------------------------------------
    with tabs[4]:
        st.subheader("ML Lab")
        st.write(
            "The ML Lab trains a simple model to estimate whether a compound looks similar to past compounds labeled bubble = yes."
        )
        st.info("Use this as a hypothesis tool. It should not be treated as proof that a compound will work.")

        ml_result = train_ml_model(described_df, use_phase=use_phase_in_ml)

        if not ml_result.get("ok"):
            st.warning(ml_result.get("message", "ML model could not be trained."))
        else:
            ml_cols = st.columns(3)
            with ml_cols[0]:
                st.metric("Estimated test accuracy", f"{ml_result['accuracy']:.0%}")
            with ml_cols[1]:
                st.metric("Training rows", ml_result["training_rows"])
            with ml_cols[2]:
                st.metric("Testing rows", ml_result["testing_rows"])

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
            st.write("Enter a compound idea below. The model will estimate the chance of `bubble = yes` based on past data.")

            p1, p2, p3 = st.columns(3)
            with p1:
                pred_phase = st.selectbox("Phase", ["pure", "impure"], key="pred_phase")
                pred_a = st.text_input("A-site element", value="La", key="pred_a")
                pred_an = st.number_input("A-site ratio", value=2.0, min_value=0.0, step=0.5, key="pred_an")
                pred_ap = st.text_input("A′-site element", value="", key="pred_ap")
                pred_apn = st.number_input("A′ ratio", value=0.0, min_value=0.0, step=0.5, key="pred_apn")
            with p2:
                pred_b = st.text_input("B-site element", value="Ni", key="pred_b")
                pred_bn = st.number_input("B-site ratio", value=1.0, min_value=0.0, step=0.5, key="pred_bn")
                pred_bp = st.text_input("B′-site element", value="", key="pred_bp")
                pred_bpn = st.number_input("B′ ratio", value=0.0, min_value=0.0, step=0.5, key="pred_bpn")
            with p3:
                pred_bdp = st.text_input("B″-site element", value="", key="pred_bdp")
                pred_bdpn = st.number_input("B″ ratio", value=0.0, min_value=0.0, step=0.5, key="pred_bdpn")
                pred_on = st.number_input("Oxygen ratio", value=4.0, min_value=0.0, step=0.5, key="pred_on")

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
    # TAB 6: ADD COMPOUND
    # -------------------------------------------------------------------------
    with tabs[5]:
        st.subheader("Add Compound")
        st.write(
            "Use this form to enter a new compound one part at a time. The app rebuilds the formula and checks the entry."
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

        # When a form is submitted, save the preview in session_state. This makes
        # the later "Add to session dataset" button reliable across Streamlit reruns.
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

        # Show the saved preview until the user adds it or replaces it.
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
    # TAB 7: EXPORT
    # -------------------------------------------------------------------------
    with tabs[6]:
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

        st.markdown("### What should I submit or save?")
        st.markdown(
            """
            - Save **cleaned data** when you want an analysis-ready version of the class database.
            - Save the **validation report** when you need to correct student-entry mistakes.
            - Save the **outlier report** when you want to double-check unusual values.
            - The app does not overwrite the original Excel file automatically, which protects the master dataset.
            """
        )


if __name__ == "__main__":
    main()
