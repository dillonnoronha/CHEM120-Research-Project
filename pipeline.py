"""
CHEM 120 Catalyst Insight Studio — Data Pipeline
=================================================

All data loading, cleaning, validation, descriptor calculation, plotting helpers,
machine-learning helpers, and export helpers live here.

This module has no Streamlit UI calls inside any function body. The only reason
it imports streamlit is to use @st.cache_data on expensive functions so the app
stays fast when students switch tabs.

Sections
--------
1.  App configuration (constants shared across the whole project)
2.  Small utility functions
3.  Reference table loading
4.  Data loading
5.  Column mapping and reshaping
6.  Cleaning, label encoding, and formula reconstruction
7.  Validation and outlier detection
8.  Chemical descriptors
9.  Summary tables and plotting helpers
10. Machine-learning helpers
11. Export helpers
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

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
# Phase is normalized before encoding so older survey wording such as
# "pure phase/homogenous mixture" does not create messy graph labels.
# CHEM 120 officially uses impure -> 1 and pure -> 2; not made -> 0 is
# included to handle older/raw survey exports safely.
PHASE_MAP = {"not made": 0, "impure": 1, "pure": 2}
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
# 2. SMALL UTILITY FUNCTIONS
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


def normalize_phase_label(value: object) -> str:
    """
    Convert raw phase wording into a small set of clean categories.

    Why this exists:
    Older CHEM 120 spreadsheets sometimes use long phase descriptions such as
    "pure phase/homogenous mixture" or "did not make compound X/it melted to the
    crucible". If those raw strings are plotted directly, the Phase Counts graph
    becomes unreadable. This helper maps those long answers to clear labels.

    Returned labels:
        "pure"                -> clean/pure phase result
        "impure"              -> heterogeneous or mixed result
        "not made"            -> compound failed, melted, or was not produced
        "other/needs review"  -> value exists but does not match known wording
        ""                    -> missing/blank
    """

    label = clean_label(value)
    if not label:
        return ""

    # Remove punctuation/spaces so spelling variations are easier to match.
    key = normalize_key(label)

    # Handle failed/no-compound entries before checking for "pure" so phrases
    # like "did not make pure compound" do not accidentally map to pure.
    not_made_tokens = [
        "didnotmake",
        "notmade",
        "nocompound",
        "melted",
        "failed",
        "crucible",
    ]
    if any(token in key for token in not_made_tokens):
        return "not made"

    # Impure/heterogeneous wording from older Microsoft Forms exports.
    impure_tokens = [
        "impure",
        "heterogeneous",
        "heterogenous",
        "mixedphase",
        "mixtureofphases",
    ]
    if any(token in key for token in impure_tokens):
        return "impure"

    # Pure/homogeneous wording. The older data often spells homogeneous as
    # "homogenous", so both spellings are accepted.
    pure_tokens = [
        "pure",
        "purephase",
        "homogeneousmixture",
        "homogenousmixture",
        "homogeneous",
        "homogenous",
    ]
    if any(token in key for token in pure_tokens):
        return "pure"

    return "other/needs review"


def normalize_bubble_label(value: object) -> str:
    """
    Convert raw bubble responses into yes/no/maybe.

    The current survey uses exactly yes, no, or maybe. This function makes the
    app a little more tolerant of capitalization and older wording.
    """

    label = clean_label(value)
    if not label:
        return ""

    key = normalize_key(label)

    if key in {"yes", "y", "true", "1", "bubble", "bubbles"}:
        return "yes"
    if key in {"no", "n", "false", "2", "nobubble", "nobubbles"}:
        return "no"
    if key in {"maybe", "possibly", "possible", "unclear", "0"}:
        return "maybe"

    return "other/needs review"


def display_label(value: object) -> str:
    """Convert internal labels into clean labels for graphs and tables."""

    label = clean_label(value)
    if not label:
        return "Missing"

    replacements = {
        "pure": "Pure",
        "impure": "Impure",
        "not made": "Not made",
        "other/needs review": "Other / needs review",
        "yes": "Yes",
        "no": "No",
        "maybe": "Maybe",
    }
    return replacements.get(label, clean_text(value).title())


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


def ordered_columns(df: pd.DataFrame, preferred: Sequence[str] = FRONT_COLUMNS) -> List[str]:
    """Put important columns first, then append the rest."""

    first = [c for c in preferred if c in df.columns]
    rest = [c for c in df.columns if c not in first]
    return first + rest


# =============================================================================
# 3. REFERENCE TABLES
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
# 4. DATA LOADING
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
# 5. COLUMN MAPPING AND RESHAPING
# =============================================================================

def find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    """Return the first candidate column name that exists in columns."""

    normalized = {normalize_key(c): c for c in columns}
    for candidate in candidates:
        key = normalize_key(candidate)
        if key in normalized:
            return normalized[key]
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
def normalize_to_long_format(raw_df: pd.DataFrame) -> tuple:
    """
    Convert the uploaded wide spreadsheet into one row per compound.

    Example input:
        Group | 1A | 1AN | 1B | 1BN | 2A | 2AN | 2B | 2BN

    Example output:
        Group | Slot | A | AN | B | BN
        Group | 1    | ...
        Group | 2    | ...

    Returns (DataFrame, list[str]) where the list contains warnings about
    required columns that could not be matched. An empty list means all
    required columns were found.
    """

    if raw_df.empty:
        return pd.DataFrame(), []

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

    # Check that the required fields for slot 1 were all matched. If any are
    # missing it almost always means the Google Form column names changed and
    # infer_slot_column() needs new aliases added.
    required_fields = {
        "A":   "A-site element",
        "AN":  "A-site ratio",
        "B":   "B-site element",
        "BN":  "B-site ratio",
        "O":   "oxygen element",
        "ON":  "oxygen ratio",
        "P":   "phase",
        "Bub": "bubble response",
    }
    column_warnings: list = []
    first_slot = COMPOUND_SLOTS[0]
    for field, label in required_fields.items():
        if slot_map[first_slot].get(field) is None:
            column_warnings.append(
                f"Could not find a column for **{label}** (field `{first_slot}{field}`). "
                f"The uploaded file has these columns: {', '.join(f'`{c}`' for c in columns[:20])}"
                + (" …and more" if len(columns) > 20 else ".")
            )

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

    return pd.DataFrame(rows), column_warnings


# =============================================================================
# 6. CLEANING, LABEL ENCODING, AND FORMULA RECONSTRUCTION
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
    #
    # Phase receives special normalization because older/raw class spreadsheets
    # may contain long answers such as "pure phase/homogenous mixture" or
    # "did not make compound X/it melted to the crucible". Without this step the
    # phase count graph becomes crowded and unreadable.
    if "P" not in df.columns:
        df["P"] = ""
    if "Bub" not in df.columns:
        df["Bub"] = ""

    # Keep the original raw wording for troubleshooting and validation reports.
    df["P_raw"] = df["P"].apply(clean_text)
    df["Bub_raw"] = df["Bub"].apply(clean_text)

    # Store normalized labels in P/Bub so downstream charts and ML use clean data.
    df["P"] = df["P"].apply(normalize_phase_label)
    df["Bub"] = df["Bub"].apply(normalize_bubble_label)

    # If O is blank but ON is filled, assume O because the survey expects oxygen.
    if "O" in df.columns:
        df.loc[(df["O"] == "") & df["ON"].notna(), "O"] = "O"

    # Convert phase and bubble response into numeric labels used by charts/ML.
    df["PhaseN"] = df["P"].map(PHASE_MAP)
    df["BubN"] = df["Bub"].map(BUBBLE_MAP)
    df["BubbleYes"] = (df["Bub"] == "yes").astype(int)

    # Reconstruct formula after cleaning.
    df["Formula"] = df.apply(reconstruct_formula, axis=1)

    # Create clean student-friendly labels for display and plots.
    # These labels are short on purpose so graph tick labels do not overlap.
    df["PhaseLabel"] = df["P"].apply(display_label)
    df["BubbleLabel"] = df["Bub"].apply(display_label)

    return df


# =============================================================================
# 7. VALIDATION AND OUTLIER DETECTION
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

        # Allowed labels after normalization.
        #
        # "other/needs review" means the app could not confidently interpret the
        # raw entry. Students/instructors should review those rows manually.
        phase_label = clean_label(row.get("P"))
        bubble_label = clean_label(row.get("Bub"))

        if phase_label == "other/needs review" or phase_label not in PHASE_MAP:
            raw_phase = clean_text(row.get("P_raw", row.get("P", "")))
            add_issue(
                row,
                "P",
                f"Unrecognized phase label: '{raw_phase}'.",
                "Use pure, impure, or a recognized not-made/failed-compound wording.",
            )
        if bubble_label == "other/needs review" or bubble_label not in BUBBLE_MAP:
            raw_bubble = clean_text(row.get("Bub_raw", row.get("Bub", "")))
            add_issue(row, "Bub", f"Unrecognized bubble response: '{raw_bubble}'.", "Use exactly: yes, no, or maybe.")

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
# 8. CHEMICAL DESCRIPTORS
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
# 9. SUMMARY TABLES AND PLOTTING HELPERS
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


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    title: str,
    preferred_order: Optional[Sequence[str]] = None,
):
    """
    Return a readable count chart for labels such as phase or bubble response.

    IMPORTANT FIX:
    The first version of the app used vertical bars for Phase counts. Raw
    Microsoft Forms phase labels can be very long, so the x-axis became unreadable.
    This function now uses a HORIZONTAL bar chart. Horizontal labels have enough
    space, and phase labels are also normalized again inside this plotting helper
    as a safety net.

    Future developers:
    - Keep this as a horizontal chart for text-heavy categories.
    - If the survey adds new wording, update `normalize_phase_label()` above.
    """

    if column not in df.columns or df.empty:
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

    series = df[column].copy()

    # Safety net: if this is the Phase chart, collapse raw phrases like
    # "pure phase/homogenous mixture" before counting. This prevents the graph
    # from ever plotting long raw phase phrases.
    is_phase_chart = "phase" in title.lower() or column.lower() in {"p", "phase", "phaselabel", "p_raw"}
    is_bubble_chart = "bubble" in title.lower() or column.lower() in {"bub", "bubble", "bubblelabel", "bub_raw"}

    if is_phase_chart:
        series = series.apply(lambda value: display_label(normalize_phase_label(value)))
    elif is_bubble_chart:
        series = series.apply(lambda value: display_label(normalize_bubble_label(value)))
    else:
        series = series.apply(display_label)

    counts = series.replace("", "Missing").fillna("Missing").astype(str).value_counts()

    # Put common categories in a stable, student-friendly order. Any unexpected
    # categories still appear at the end so data problems remain visible.
    if preferred_order:
        ordered_labels = [label for label in preferred_order if label in counts.index]
        extra_labels = sorted([label for label in counts.index if label not in ordered_labels])
        labels = ordered_labels + extra_labels
        counts = counts.reindex(labels)
    else:
        counts = counts.sort_index()

    # Horizontal bars fix label overlap because category names are on the y-axis.
    fig_height = max(3.4, min(7.0, 0.55 * len(counts) + 1.4))
    fig, ax = plt.subplots(figsize=(8.5, fig_height))

    y_positions = np.arange(len(counts))
    ax.barh(y_positions, counts.values)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(counts.index)
    ax.invert_yaxis()

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of compounds")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)

    # Add numeric labels to the end of each bar for easier reading.
    max_count = max(counts.values) if len(counts) else 0
    for y, value in zip(y_positions, counts.values):
        ax.text(value + max(max_count * 0.01, 0.5), y, str(int(value)), va="center", fontsize=9)

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
# 10. MACHINE LEARNING HELPERS
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
    Train a Random Forest and a Logistic Regression model to predict Bubble = yes.

    Both models share the same train/test split so accuracy numbers are directly
    comparable. The RF model is used for predictions; LR serves as a sanity check.

    Overfitting warning is raised when RF accuracy is suspiciously high relative
    to LR, or when RF accuracy exceeds 85% on fewer than 100 labeled rows.

    Returns a dict with both models' results, or {"ok": False, "message": ...}.
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

    stratify = y if y.value_counts().min() >= 2 else None
    test_size = 0.25 if len(model_df) >= 20 else 0.35

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Random Forest — used for final predictions and feature importance.
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=7,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(X_train, y_train)
    rf_accuracy = accuracy_score(y_test, rf.predict(X_test))

    importance = pd.DataFrame(
        {"Feature": X.columns, "Importance": rf.feature_importances_}
    ).sort_values("Importance", ascending=False)
    importance = importance[importance["Importance"] > 0].head(12)

    # Logistic Regression — simpler model used as a sanity check.
    # StandardScaler is required because LR is sensitive to feature scale.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight="balanced",
    )
    lr.fit(X_train_scaled, y_train)
    lr_accuracy = accuracy_score(y_test, lr.predict(X_test_scaled))

    # Overfitting warning: RF >> LR on a small dataset is a red flag.
    overfit_warning = None
    if rf_accuracy - lr_accuracy > 0.15:
        overfit_warning = (
            f"Random Forest ({rf_accuracy:.0%}) is more than 15 points ahead of "
            f"Logistic Regression ({lr_accuracy:.0%}). On a small dataset this often "
            "means the RF memorized the training data. Treat feature importance and "
            "predictions with extra caution."
        )
    elif rf_accuracy > 0.85 and len(model_df) < 100:
        overfit_warning = (
            f"RF accuracy is {rf_accuracy:.0%} with only {len(model_df)} labeled rows. "
            "High accuracy on very small datasets is often a sign of overfitting. "
            "The Logistic Regression result is likely more reliable here."
        )

    return {
        "ok": True,
        "model": rf,
        "scaler": scaler,
        "feature_columns": list(X.columns),
        "rf_accuracy": float(rf_accuracy),
        "lr_accuracy": float(lr_accuracy),
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "importance": importance,
        "overfit_warning": overfit_warning,
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
# 11. EXPORT HELPERS
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
