"""
General Chemistry II Catalyst Insight Studio — Data Pipeline
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

# Build marker (bump to force Python to recompile if OneDrive serves a stale
# __pycache__/*.pyc with an older source mtime): 2026-07-01a
import difflib
import io
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
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

APP_TITLE = "General Chemistry II Catalyst Insight Studio"
APP_SUBTITLE = "Turn class lab entries into clean formulas, visual trends, and testable hypotheses."

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

# The survey stores up to four compounds per group. Each compound is stored in a
# "slot" such as 1A, 1AN, 2A, 2AN, etc.
COMPOUND_SLOTS = [1, 2, 3, 4]

# General Chemistry II project label conversions.
# Keeping these dictionaries near the top makes the app easy to adapt if the
# class changes its coding scheme later.
# Phase is normalized before encoding so older survey wording such as
# "pure phase/homogenous mixture" does not create messy graph labels.
# General Chemistry II officially uses impure -> 1 and pure -> 2; not made -> 0 is
# included to handle older/raw survey exports safely.
PHASE_MAP = {"not made": 0, "impure": 1, "pure": 2}
BUBBLE_MAP = {"maybe": 0, "yes": 1, "no": 2}

# Placeholder words students/templates type into an OPTIONAL element field to
# mean "there is no element here". Without this list these phrases leak into the
# reconstructed formula (e.g. "La2CoNo B' elementO4") or, worse, get silently
# read as a real symbol ("NO" -> Nobelium). They are normalized to blank.
# NOTE: "na" is deliberately NOT a marker because Na = sodium is a real element.
EMPTY_MARKERS = {
    "none", "nil", "null", "blank", "empty", "nan", "nothing", "no", "x", "n",
    "noelement", "noprime", "notapplicable", "doesnotapply",
}

# Optional element/ratio fields used for mixed A-site and B-site compounds.
A_SITE_FIELDS = [("A", "AN"), ("AP", "APN")]
B_SITE_FIELDS = [("B", "BN"), ("BP", "BPN"), ("BDP", "BDPN")]

# Columns that are useful to show first in student-facing tables.
FRONT_COLUMNS = [
    # Identifiers / who-entered-it first (left side of every table).
    "GroupNumber", "Instructor", "Name", "Email", "Members", "Semester", "Slot",
    # Then the actual compound info (formula and its parts).
    "Formula",
    "A", "AN", "AP", "APN", "B", "BN", "BP", "BPN", "BDP", "BDPN",
    "O", "ON", "P", "Bub", "PhaseN", "BubN",
]

# Personal / group-identifying columns. Instructor feedback (June 2026): the app
# still COLLECTS these (they stay in the data and in instructor downloads) but
# the student-facing UI never displays them.
PRIVATE_COLUMNS = ["Email", "Name", "Members"]

# ----- Shared chart styling (theme-aware) ------------------------------------
# Defaults are the dark "lab glass" palette; set_chart_theme("light") swaps the
# module-level values so every plot function picks up the active theme.
CHART_INK = "#dbe4f3"        # main text on charts
CHART_MUTED = "#8fa1bd"      # secondary text
CHART_GRID = "rgba(148, 163, 184, 0.14)"
CHART_ACCENT = "#38bdf8"     # sky
CHART_ACCENT_2 = "#8b5cf6"   # violet
CHART_HOVER_BG = "#111a2e"   # hover tooltip background
CHART_BAR_LOW = "#1c3a5e"    # low end of sequential bar colorscales
CHART_FONT = "Inter, -apple-system, 'Segoe UI', sans-serif"

_CHART_THEMES = {
    "dark": {
        "CHART_INK": "#dbe4f3", "CHART_MUTED": "#8fa1bd",
        "CHART_GRID": "rgba(148, 163, 184, 0.14)",
        "CHART_ACCENT": "#38bdf8", "CHART_ACCENT_2": "#8b5cf6",
        "CHART_HOVER_BG": "#111a2e", "CHART_BAR_LOW": "#1c3a5e",
    },
    "light": {
        "CHART_INK": "#24324a", "CHART_MUTED": "#5b6b85",
        "CHART_GRID": "rgba(15, 23, 42, 0.10)",
        "CHART_ACCENT": "#0284c7", "CHART_ACCENT_2": "#7c3aed",
        "CHART_HOVER_BG": "#ffffff", "CHART_BAR_LOW": "#bcd7f0",
    },
}


def set_chart_theme(mode: str) -> None:
    """Switch every chart's colors between 'dark' and 'light' (app theme toggle)."""

    globals().update(_CHART_THEMES["light" if mode == "light" else "dark"])

# Categorical palette used across all charts.
CHART_PALETTE = [
    "#38bdf8", "#fbbf24", "#34d399", "#f87171", "#a78bfa",
    "#22d3ee", "#f472b6", "#facc15", "#94a3b8", "#fb923c",
]

# Fixed colors for outcome labels so Yes/Pure are always green, etc.
OUTCOME_COLORS = {
    "Yes": "#34d399", "No": "#f87171", "Maybe": "#fbbf24",
    "Pure": "#34d399", "Impure": "#fbbf24", "Not made": "#f87171",
    "Missing": "#64748b", "Other / needs review": "#94a3b8",
}

# Passcode for Instructor mode (reveals the hidden contact columns inside the
# app). CHANGE THIS before sharing the app beyond the teaching team.
INSTRUCTOR_PASSCODE = "chemprofessor123"

# Shannon radius of O2- in six-fold coordination (Å), used by the Goldschmidt
# tolerance factor.
OXYGEN_RADIUS = 1.40


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
    Older General Chemistry II spreadsheets sometimes use long phase descriptions such as
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


def public_view(df: pd.DataFrame, extra_hidden: Sequence[str] = ()) -> pd.DataFrame:
    """
    Return a copy of df without personal/group-identifying columns.

    Every st.dataframe the app SHOWS should pass through this helper. The data
    itself is untouched — instructor downloads in the Export tab keep everything.
    """

    hidden = set(PRIVATE_COLUMNS) | set(extra_hidden)
    return df[[c for c in df.columns if c not in hidden]]


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


@st.cache_data(show_spinner=False)
def load_radii_table_from_bytes(file_bytes: Optional[bytes] = None, file_name: str = "") -> pd.DataFrame:
    """
    Load the Shannon ionic radii table (data/ShannonRadii.csv).

    Columns: Symbol, A_site_radius (12-coordinate, Å), B_site_radius
    (6-coordinate, Å). Values assume the most common oxidation state for
    perovskite chemistry (see the AssumedIon columns in the CSV) — a documented
    teaching approximation, since the spreadsheet doesn't record oxidation states.
    Returns an empty DataFrame when the file is missing so the app still runs.
    """

    if not file_bytes:
        local_path = DATA_DIR / "ShannonRadii.csv"
        if not local_path.exists():
            return pd.DataFrame(columns=["Symbol", "A_site_radius", "B_site_radius"])
        data = local_path.read_bytes()
    else:
        data = file_bytes

    radii = pd.read_csv(io.BytesIO(data))
    if "Symbol" not in radii.columns:
        return pd.DataFrame(columns=["Symbol", "A_site_radius", "B_site_radius"])

    radii["Symbol"] = radii["Symbol"].astype(str).str.strip()
    for col in ["A_site_radius", "B_site_radius"]:
        radii[col] = pd.to_numeric(radii.get(col), errors="coerce")
    return radii[["Symbol", "A_site_radius", "B_site_radius"]].dropna(subset=["Symbol"]).drop_duplicates("Symbol")


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

    # Treat "no element" placeholders as blank so they never reach the formula.
    if is_placeholder_symbol(text):
        return ""

    lower = text.lower()
    if lower in name_to_symbol:
        return name_to_symbol[lower]

    # Element symbols are usually one or two letters. This turns "FE" into "Fe".
    if len(text) <= 3 and text.isalpha():
        return text[0].upper() + text[1:].lower()

    # Leave unknown text visible so validation can flag it.
    return text


def is_placeholder_symbol(value: object) -> bool:
    """
    Return True when an element cell is really a "no element here" placeholder.

    Catches bare markers (none, nil, "NO", x), "no ... element" phrases such as
    "No B' element", and any multi-word/punctuated entry (no real element symbol
    contains a space or apostrophe). "Na"/"na" is preserved as sodium.
    """

    raw = clean_text(value)
    if not raw:
        return True

    key = normalize_key(raw)
    if key in EMPTY_MARKERS:
        return True
    if key.startswith("no") and "element" in key:
        return True
    # No periodic-table symbol contains a space or apostrophe.
    if " " in raw or "'" in raw or "/" in raw:
        return True
    return False


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
    Convert the uploaded spreadsheet into one row per compound.

    Handles two formats automatically:

    Wide format (one row per group, multiple compounds as column sets):
        Group | 1A | 1AN | 1B | 1BN | 2A | 2AN | 2B | 2BN

    Long format (one row per compound, already expanded):
        Group | Compound_Num | A | AN | B | BN | ...

    Returns (DataFrame, list[str]) where the list contains warnings about
    required columns that could not be matched. An empty list means all
    required columns were found.
    """

    if raw_df.empty:
        return pd.DataFrame(), []

    columns = list(raw_df.columns)
    _ncols = {normalize_key(c): c for c in columns}

    # Long-format detection: if the file already has direct column names like
    # A, B, AN, BN (no slot-number prefix), skip the wide-to-long expansion
    # and map each row directly as one compound record.
    if "a" in _ncols and "b" in _ncols:
        _field_aliases: Dict[str, List[str]] = {
            "FormulaRef": ["formularef", "formula", "fullformula", "compound", "referenceformula"],
            "A":    ["a", "asite", "asiteelement", "aelement"],
            "AN":   ["an", "aratio", "asiteratio", "aamount"],
            "AP":   ["ap", "aprime", "asiteprime", "aprimeelement"],
            "APN":  ["apn", "aprimeratio", "apratio", "asiteprimeratio"],
            "B":    ["b", "bsite", "bsiteelement", "belement"],
            "BN":   ["bn", "bratio", "bsiteratio", "bamount"],
            "BP":   ["bp", "bprime", "bsiteprime", "bprimeelement"],
            "BPN":  ["bpn", "bprimeratio", "bpratio", "bsiteprimeratio"],
            "BDP":  ["bdp", "bdprime", "bdoubleprime", "bsite2prime", "b2prime"],
            "BDPN": ["bdpn", "bdprimeratio", "bdoubleprimeratio", "b2primeratio"],
            "O":    ["o", "oxygen", "oxygenelement"],
            "ON":   ["on", "oratio", "oxygenratio", "oxygenamount"],
            "P":    ["p", "phase"],
            "PN":   ["pn", "phasen", "phasenumber"],
            "Bub":  ["bub", "bubble", "bubbleresponse", "h2bubble", "bubbles"],
            "BubN": ["bubn", "bubblen", "bubblenumber", "bubbleresponsen"],
        }
        _meta_aliases: Dict[str, List[str]] = {
            "GroupNumber": ["groupnumber", "groupid", "group", "team", "groupnumberid"],
            "Email":       ["email", "emailaddress"],
            "Name":        ["name", "studentname"],
            "Members":     ["members", "groupmembers"],
            "Instructor":  ["instructor", "instructorsection", "section"],
            "Semester":    ["semester", "semesteryear", "term", "year"],
        }

        field_map: Dict[str, Optional[str]] = {}
        for field, aliases in _field_aliases.items():
            for alias in aliases:
                if alias in _ncols:
                    field_map[field] = _ncols[alias]
                    break
            else:
                field_map[field] = None

        meta_map: Dict[str, Optional[str]] = {}
        for out_name, aliases in _meta_aliases.items():
            for alias in aliases:
                if alias in _ncols:
                    meta_map[out_name] = _ncols[alias]
                    break
            else:
                meta_map[out_name] = None

        compound_num_col = (
            _ncols.get("compoundnum")
            or _ncols.get("compoundnumber")
            or _ncols.get("compound")
        )

        rows: List[dict] = []
        for source_index, source_row in raw_df.iterrows():
            record: dict = {
                out: source_row.get(col, "") if col is not None else ""
                for out, col in meta_map.items()
            }
            record["SourceRow"] = int(source_index) + 2
            if compound_num_col and not is_blank(source_row.get(compound_num_col)):
                try:
                    record["Slot"] = int(float(source_row[compound_num_col]))
                except (ValueError, TypeError):
                    record["Slot"] = 1
            else:
                record["Slot"] = 1

            for field, col in field_map.items():
                record[field] = source_row.get(col, "") if col is not None else ""

            rows.append(record)

        return pd.DataFrame(rows), []

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


def split_member_names(text: object) -> List[str]:
    """Split a 'Members' cell into individual names.

    Handles commas, semicolons, ampersands, the word 'and', and newlines, e.g.
    'Justine Bailes, Lucas Chumacero, and Paloma Garcia' -> three names.
    """

    raw = clean_text(text)
    if not raw:
        return []
    parts = re.split(r",|;|&|\band\b|\n|/", raw)
    return [p.strip() for p in parts if p.strip()]


# Below this match score, the email is treated as NOT matching any member name
# (calibrated on real class data: genuine mismatches score well under 0.45).
EMAIL_MATCH_THRESHOLD = 0.45


def _name_email_score(local: str, name: str) -> float:
    """Similarity (0-1ish) between an email local part and one member name."""

    tokens = re.sub(r"[^a-z ]", "", name.lower()).split()
    if not tokens:
        return 0.0
    candidates = {
        "".join(tokens),                         # alexaalmanza
        tokens[0],                               # alexa
        tokens[-1],                              # almanza
        tokens[0][0] + tokens[-1],               # aalmanza
        tokens[-1] + tokens[0][0],               # almanzaa
        "".join(t[0] for t in tokens) + tokens[-1],
    }
    base = max(difflib.SequenceMatcher(None, local, c).ratio() for c in candidates)
    if len(tokens[-1]) >= 3 and tokens[-1] in local:   # last name appears in email
        base += 0.30
    if len(tokens[0]) >= 3 and tokens[0] in local:     # first name appears in email
        base += 0.20
    return base


def email_member_match(email: object, members: object) -> Tuple[str, float]:
    """Return (best-matching member name, match score) for an email + roster.

    Returns ("", 0.0) when there is no roster or no usable email local part.
    """

    names = split_member_names(members)
    local = re.sub(r"[^a-z]", "", clean_text(email).split("@")[0].lower())
    if not names or not local:
        return "", 0.0
    scored = [(name, _name_email_score(local, name)) for name in names]
    return max(scored, key=lambda pair: pair[1])


def best_name_for_email(email: object, members: object, fallback: object = "") -> str:
    """Pick the member name that best matches the submitter's email.

    Falls back to the existing name (or the first member) when nothing matches.
    """

    names = split_member_names(members)
    if not names:
        return clean_text(fallback)
    name, _score = email_member_match(email, members)
    if not name:
        return clean_text(fallback) or names[0]
    return name


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

    # Name / Members tidy-up.
    #   * The roster of group members can live in a "Members" column OR (in some
    #     exports) the whole group is dumped into the "Name" column. Use whichever
    #     actually holds a list.
    #   * Name  -> the single member whose name best matches the submitter email.
    #   * Members -> the full list, with that matched person listed FIRST.
    if "Members" in df.columns or "Name" in df.columns:
        emails = df["Email"] if "Email" in df.columns else pd.Series([""] * len(df), index=df.index)
        members_col = df["Members"] if "Members" in df.columns else pd.Series([""] * len(df), index=df.index)
        name_col = df["Name"] if "Name" in df.columns else pd.Series([""] * len(df), index=df.index)

        new_names: List[str] = []
        new_members: List[str] = []
        for email_value, members_value, name_value in zip(emails, members_col, name_col):
            roster = members_value if split_member_names(members_value) else name_value
            people = split_member_names(roster)
            if not people:
                new_names.append(clean_text(name_value))
                new_members.append("")
                continue
            matched = best_name_for_email(email_value, roster, fallback=people[0])
            ordered = [matched] + [p for p in people if p != matched]
            new_names.append(matched)
            new_members.append(", ".join(ordered))

        df["Name"] = new_names
        df["Members"] = new_members

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
    Check cleaned rows against the General Chemistry II data-entry rules.

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

    flagged_emails: set = set()
    for _, row in clean_df.iterrows():
        # Email vs members: flag when the email doesn't match anyone in Members
        # (once per group/email so it isn't repeated for every compound row).
        email_value = clean_text(row.get("Email", ""))
        members_value = row.get("Members", "")
        if "@" in email_value and split_member_names(members_value):
            group_key = (clean_text(row.get("GroupNumber", "")), email_value)
            if group_key not in flagged_emails:
                _matched, match_score = email_member_match(email_value, members_value)
                if match_score < EMAIL_MATCH_THRESHOLD:
                    flagged_emails.add(group_key)
                    add_issue(
                        row, "Email",
                        f"Email '{email_value}' does not clearly match anyone in the Members list.",
                        "Check that the submitter is listed in Members, or that the email is correct.",
                    )

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
def remove_invalid_element_rows(df: pd.DataFrame, atomic: pd.DataFrame) -> tuple:
    """
    Drop rows that contain an element symbol which is not on the periodic table.

    Placeholder text (e.g. "No B' element", "NO") is already blanked during
    cleaning, so it does NOT trigger removal here — only genuine non-elements
    like "BATU" do.

    Returns
    -------
    (clean_df, change_log)
        clean_df   : the dataframe with invalid-element rows removed.
        change_log : one row per problem found, with columns
                     Row, GroupNumber, Slot, Formula, Field, Value, Change.
                     Empty if nothing was removed.
    """

    log_columns = ["Row", "GroupNumber", "Slot", "Formula", "Field", "Value", "Change"]
    if df.empty:
        return df.copy(), pd.DataFrame(columns=log_columns)

    _name_to_symbol, valid_symbols = make_element_maps(atomic)
    element_fields = ["A", "AP", "B", "BP", "BDP", "O"]

    rows_to_drop: List[object] = []
    log: List[dict] = []

    for idx, row in df.iterrows():
        bad_fields = []
        for field in element_fields:
            value = clean_text(row.get(field, ""))
            if value and value not in valid_symbols:
                bad_fields.append((field, value))

        if bad_fields:
            rows_to_drop.append(idx)
            for field, value in bad_fields:
                log.append({
                    "Row": row.get("SourceRow", ""),
                    "GroupNumber": row.get("GroupNumber", ""),
                    "Slot": row.get("Slot", ""),
                    "Formula": row.get("Formula", ""),
                    "Field": field,
                    "Value": value,
                    "Change": f"Row removed — '{value}' is not a real element symbol.",
                })

    clean_df = df.drop(index=rows_to_drop).reset_index(drop=True)
    return clean_df, pd.DataFrame(log, columns=log_columns)


def suggest_elements(value: object, atomic: pd.DataFrame, n: int = 4) -> List[str]:
    """
    Suggest the most likely valid element symbols for a bad entry ("did you mean?").

    Combines three strategies, best matches first:
      1. Full-name typos   ("Nickle" -> Ni) via fuzzy match on element names.
      2. Symbol typos       ("BATU" -> Ba) via fuzzy match on symbols.
      3. Prefix heuristic   ("Nii" -> Ni) when the first 1-2 letters are a symbol.
    Returns an empty list when nothing is close (e.g. "Zzz").
    """

    text = clean_text(value)
    if not text:
        return []

    name_to_symbol, valid_symbols = make_element_maps(atomic)
    out: List[str] = []

    for name in difflib.get_close_matches(text.lower(), list(name_to_symbol.keys()), n=n, cutoff=0.6):
        symbol = name_to_symbol[name]
        if symbol not in out:
            out.append(symbol)

    for symbol in difflib.get_close_matches(text.title(), sorted(valid_symbols), n=n, cutoff=0.5):
        if symbol not in out:
            out.append(symbol)

    for k in (2, 1):
        prefix = text[:k].title()
        if prefix in valid_symbols and prefix not in out:
            out.append(prefix)

    return out[:n]


def valid_element_symbols(atomic: pd.DataFrame) -> List[str]:
    """Return the sorted list of valid element symbols (for fix dropdowns)."""

    _name_to_symbol, valid_symbols = make_element_maps(atomic)
    return sorted(valid_symbols)


def propose_element_fixes(long_df: pd.DataFrame, atomic: pd.DataFrame) -> pd.DataFrame:
    """
    Find element cells that are not valid symbols and attach suggestions.

    Runs on the long-format dataframe (raw element text) so it can be used BEFORE
    cleaning. Placeholder text (e.g. "No B' element") is treated as blank and is
    not reported here. Returns columns:
        SourceRow, Slot, GroupNumber, Field, Original, Suggestions (list[str]).
    """

    cols = ["SourceRow", "Slot", "GroupNumber", "Field", "Original", "Suggestions"]
    if long_df.empty:
        return pd.DataFrame(columns=cols)

    name_to_symbol, valid_symbols = make_element_maps(atomic)
    rows: List[dict] = []
    for _, row in long_df.iterrows():
        for field in ["A", "AP", "B", "BP", "BDP", "O"]:
            original = clean_text(row.get(field, ""))
            if not original:
                continue
            cleaned = clean_symbol(original, name_to_symbol)
            if cleaned and cleaned not in valid_symbols:
                rows.append({
                    "SourceRow": row.get("SourceRow", ""),
                    "Slot": row.get("Slot", ""),
                    "GroupNumber": row.get("GroupNumber", ""),
                    "Field": field,
                    "Original": original,
                    "Suggestions": suggest_elements(original, atomic),
                })
    return pd.DataFrame(rows, columns=cols)


def apply_element_fixes(long_df: pd.DataFrame, fixes: Dict[str, str]) -> tuple:
    """
    Replace element cells with user-chosen corrections before cleaning.

    `fixes` maps a cell key "SourceRow|Slot|Field" to the chosen valid symbol.
    Returns (fixed_long_df, corrections_log) where the log has columns
    Row, GroupNumber, Slot, Field, Original, Change.
    """

    cols = ["Row", "GroupNumber", "Slot", "Field", "Original", "Change"]
    if long_df.empty or not fixes:
        return long_df.copy(), pd.DataFrame(columns=cols)

    df = long_df.copy()
    log: List[dict] = []
    for idx, row in df.iterrows():
        key_base = f"{row.get('SourceRow', '')}|{row.get('Slot', '')}"
        for field in ["A", "AP", "B", "BP", "BDP", "O"]:
            new_symbol = fixes.get(f"{key_base}|{field}")
            if not new_symbol:
                continue
            original = clean_text(row.get(field, ""))
            if original and new_symbol != original:
                df.at[idx, field] = new_symbol
                log.append({
                    "Row": row.get("SourceRow", ""),
                    "GroupNumber": row.get("GroupNumber", ""),
                    "Slot": row.get("Slot", ""),
                    "Field": field,
                    "Original": original,
                    "Change": f"Corrected '{original}' → '{new_symbol}'.",
                })
    return df, pd.DataFrame(log, columns=cols)


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
def add_chemical_descriptors(clean_df: pd.DataFrame, atomic: pd.DataFrame, en_table: pd.DataFrame,
                             radii_table: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Add calculated chemistry descriptors.

    radii_table (optional) enables the Goldschmidt tolerance factor. When None,
    the bundled data/ShannonRadii.csv is loaded automatically.

    Future update point:
    Add new descriptors here, such as oxidation state estimates or group/period
    features.
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

    # ----- Added descriptors (#12) -------------------------------------------
    # Contrast between the A and B sites, captured as plain numbers.
    df["A_B_Z_diff"] = df["A_avg_Z"] - df["B_avg_Z"]
    df["A_B_mass_diff"] = df["A_avg_mass"] - df["B_avg_mass"]

    # Whole-compound averages, ratio-weighted across both cation sites.
    cation_ratio_total = (df["A_total_ratio"] + df["B_total_ratio"]).replace({0: np.nan})
    df["Avg_cation_Z"] = (
        df["A_avg_Z"] * df["A_total_ratio"] + df["B_avg_Z"] * df["B_total_ratio"]
    ) / cation_ratio_total
    df["Avg_cation_mass"] = (
        df["A_avg_mass"] * df["A_total_ratio"] + df["B_avg_mass"] * df["B_total_ratio"]
    ) / cation_ratio_total

    # How many distinct elements occupy each site (1 = simple, 2+ = doped/mixed).
    df["N_A_elements"] = (
        (df["A"].astype(str).str.len() > 0).astype(int)
        + (df["AP"].astype(str).str.len() > 0).astype(int)
    )
    df["N_B_elements"] = (
        (df["B"].astype(str).str.len() > 0).astype(int)
        + (df["BP"].astype(str).str.len() > 0).astype(int)
        + (df["BDP"].astype(str).str.len() > 0).astype(int)
    )
    df["Cation_count"] = df["N_A_elements"] + df["N_B_elements"]

    # Mixing fractions (#1): how much of each site is the *secondary* cation.
    # 0.0 = a single element on the site; 0.5 = a 50/50 mix; captures the fact
    # that A vs A' (or B vs B'/B'') can be present in DIFFERING amounts, which the
    # binary Mixed_*_site flags do not.
    def _dopant_fraction(row: pd.Series, main: Tuple[str, str], extras: Sequence[Tuple[str, str]]) -> float:
        main_ratio = ratio_for_formula(row, main[0], main[1], required=True)
        main_ratio = 0.0 if np.isnan(main_ratio) else main_ratio
        extra_ratio = sum(
            ratio_for_formula(row, sym, rat)
            for sym, rat in extras
            if clean_text(row.get(sym, ""))
        )
        total = main_ratio + extra_ratio
        return float(extra_ratio / total) if total > 0 else 0.0

    df["A_mix_fraction"] = df.apply(lambda r: _dopant_fraction(r, ("A", "AN"), [("AP", "APN")]), axis=1)
    df["B_mix_fraction"] = df.apply(
        lambda r: _dopant_fraction(r, ("B", "BN"), [("BP", "BPN"), ("BDP", "BDPN")]), axis=1)

    # ----- Goldschmidt tolerance factor --------------------------------------
    # t = (r_A + r_O) / (sqrt(2) * (r_B + r_O)) with ratio-weighted site radii.
    # t ~ 0.9-1.0 suggests an ideal cubic perovskite; below ~0.9 distorted;
    # above ~1.0 hexagonal tendencies. Radii come from data/ShannonRadii.csv
    # (typical oxidation states — a documented teaching approximation).
    if radii_table is None:
        radii_table = load_radii_table_from_bytes(None)

    if not radii_table.empty:
        radii_index = radii_table.set_index("Symbol")

        def _site_radius(row: pd.Series, fields: Sequence[Tuple[str, str]], col: str) -> float:
            values, weights = [], []
            for symbol_col, ratio_col in fields:
                symbol = clean_text(row.get(symbol_col, ""))
                if not symbol:
                    continue
                radius = (float(radii_index.loc[symbol, col])
                          if symbol in radii_index.index else np.nan)
                values.append(radius)
                weights.append(ratio_for_formula(row, symbol_col, ratio_col,
                                                 required=(symbol_col in {"A", "B"})))
            return weighted_average(values, weights)

        df["A_site_radius"] = df.apply(lambda r: _site_radius(r, A_SITE_FIELDS, "A_site_radius"), axis=1)
        df["B_site_radius"] = df.apply(lambda r: _site_radius(r, B_SITE_FIELDS, "B_site_radius"), axis=1)
        df["Tolerance_factor"] = (df["A_site_radius"] + OXYGEN_RADIUS) / (
            math.sqrt(2) * (df["B_site_radius"] + OXYGEN_RADIUS))

    # Median-fill sparse physics descriptors so ML's fillna(0) never injects a
    # nonsense zero (t = 0 or EN = 0 would be a strong, wrong signal).
    for sparse_col in ["Tolerance_factor", "A_site_radius", "B_site_radius",
                       "A_avg_EN", "B_avg_EN", "EN_difference_B_minus_A"]:
        if sparse_col in df.columns:
            median = pd.to_numeric(df[sparse_col], errors="coerce").median()
            if not np.isnan(median):
                df[sparse_col] = df[sparse_col].fillna(median)

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


def label_count_table(
    df: pd.DataFrame,
    column: str,
    kind: str = "auto",
    preferred_order: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Return bubble/phase response counts as a TABLE (not a chart).

    Replaces the old distribution bar charts (#16). Output columns:
        Response | Count | Percent
    Percent is share of all compounds, formatted as a string like "42.1%".
    """

    if column not in df.columns or df.empty:
        return pd.DataFrame(columns=["Response", "Count", "Percent"])

    series = df[column].copy()
    if kind == "phase":
        series = series.apply(lambda v: display_label(normalize_phase_label(v)))
    elif kind == "bubble":
        series = series.apply(lambda v: display_label(normalize_bubble_label(v)))
    else:
        series = series.apply(display_label)

    counts = series.replace("", "Missing").fillna("Missing").astype(str).value_counts()

    if preferred_order:
        ordered = [lbl for lbl in preferred_order if lbl in counts.index]
        extra = sorted([lbl for lbl in counts.index if lbl not in ordered])
        counts = counts.reindex(ordered + extra)

    total = int(counts.sum()) or 1
    out = counts.reset_index()
    out.columns = ["Response", "Count"]
    out["Count"] = out["Count"].astype(int)
    out["Percent"] = (out["Count"] / total * 100).map(lambda x: f"{x:.1f}%")
    return out


def _apply_chart_theme(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    """Apply the shared theme-aware look to a Plotly figure."""

    # title_text is ALWAYS set to a real string: leaving it undefined while other
    # title/template properties exist makes plotly.js render the literal word
    # "undefined" over the top-left of the chart.
    if title:
        fig.update_layout(title={"text": title,
                                 "font": {"size": 16, "color": CHART_INK, "family": CHART_FONT}})
    else:
        fig.update_layout(title_text="")

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": CHART_FONT, "color": CHART_INK, "size": 12},
        height=height,
        margin={"l": 10, "r": 10, "t": 48 if title else 16, "b": 10},
        hoverlabel={
            "bgcolor": CHART_HOVER_BG,
            "bordercolor": CHART_ACCENT,
            "font": {"family": CHART_FONT, "color": CHART_INK, "size": 12},
        },
        colorway=CHART_PALETTE,
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"color": CHART_MUTED}},
    )
    fig.update_xaxes(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID, linecolor=CHART_GRID,
                     tickfont={"color": CHART_MUTED})
    fig.update_yaxes(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID, linecolor=CHART_GRID,
                     tickfont={"color": CHART_MUTED})
    return fig


def plot_bar_chart(data: pd.DataFrame, x_col: str, y_col: str, title: str, ylabel: str) -> go.Figure:
    """
    Interactive horizontal bar chart (used for model feature importance).

    Horizontal bars keep long feature names readable; hover shows exact values.
    The x_col/y_col argument order is kept from the old matplotlib version so
    callers don't change: x_col = category names, y_col = numeric values.
    """

    plot_data = data[[x_col, y_col]].dropna().head(12).iloc[::-1]  # biggest ends up on top
    labels = plot_data[x_col].astype(str).map(friendly_label)
    values = plot_data[y_col].astype(float)

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={
                "color": values,
                "colorscale": [[0.0, CHART_BAR_LOW], [1.0, CHART_ACCENT]],
                "line": {"width": 0},
            },
            hovertemplate="<b>%{y}</b><br>" + ylabel + ": %{x:.3f}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text=ylabel, title_font={"color": CHART_MUTED, "size": 12})
    height = max(300, 30 * len(plot_data) + 110)
    return _apply_chart_theme(fig, title=title, height=height)


def numeric_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a correlation matrix from the numeric columns used in the app."""

    candidates = [
        "BubbleYes", "BubN", "PhaseN",
        "AN", "APN", "BN", "BPN", "BDPN", "ON",
        "A_avg_Z", "B_avg_Z", "A_avg_mass", "B_avg_mass",
        "FormulaMass", "O_to_cation_ratio", "B_to_A_ratio",
        "Mixed_A_site", "Mixed_B_site",
        "A_B_Z_diff", "A_B_mass_diff", "Avg_cation_Z", "Avg_cation_mass",
        "Cation_count", "N_B_elements", "A_mix_fraction", "B_mix_fraction",
    ]
    candidates += [c for c in ["A_avg_EN", "B_avg_EN", "EN_difference_B_minus_A",
                               "Tolerance_factor", "A_site_radius", "B_site_radius"] if c in df.columns]

    numeric_cols = [c for c in candidates if c in df.columns]
    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")

    if numeric.shape[1] < 2:
        return pd.DataFrame()

    return numeric.corr(numeric_only=True)


def friendly_label(name: str) -> str:
    """Turn an internal column name into a short human-readable heatmap label."""

    pretty = {
        "BubbleYes": "Bubbles (yes)",
        "BubN": "Bubble code",
        "PhaseN": "Phase (pure=2)",
        "AN": "A ratio", "BN": "B ratio", "ON": "O ratio",
        "APN": "A' ratio", "BPN": "B' ratio", "BDPN": "B'' ratio",
        "A_avg_Z": "A atomic #", "B_avg_Z": "B atomic #",
        "A_avg_mass": "A mass", "B_avg_mass": "B mass",
        "FormulaMass": "Formula mass",
        "O_to_cation_ratio": "O : cation", "B_to_A_ratio": "B : A",
        "Mixed_A_site": "Mixed A", "Mixed_B_site": "Mixed B",
        "A_B_Z_diff": "A-B atomic #", "A_B_mass_diff": "A-B mass",
        "Avg_cation_Z": "Avg cation #", "Avg_cation_mass": "Avg cation mass",
        "Cation_count": "# cations", "N_B_elements": "# B elements",
        "A_mix_fraction": "A mixing frac", "B_mix_fraction": "B mixing frac",
        "A_avg_EN": "A electroneg", "B_avg_EN": "B electroneg",
        "EN_difference_B_minus_A": "EN diff (B−A)",
        "A_total_ratio": "A-site total", "B_total_ratio": "B-site total",
        "Tolerance_factor": "Tolerance factor",
        "A_site_radius": "A ionic radius", "B_site_radius": "B ionic radius",
    }
    return pretty.get(name, name)


# Plain-English definition for every feature that can appear on the heatmap or
# in the model. Shown in the Heatmap tab glossary (instructor request: "can you
# all define the terms on the heat map?").
FEATURE_GLOSSARY: Dict[str, str] = {
    "BubbleYes": "1 if the compound bubbled (yes), otherwise 0. This is the outcome the class cares about.",
    "BubN": "Bubble response as a code: maybe = 0, yes = 1, no = 2.",
    "PhaseN": "Phase result as a code: not made = 0, impure = 1, pure = 2.",
    "AN": "How many of the main A-site element are in the formula (e.g. the 2 in La₂NiO₄).",
    "APN": "How many of the second A-site element (A′) are in the formula. 0 when the A-site is a single element.",
    "BN": "How many of the main B-site element are in the formula.",
    "BPN": "How many of the second B-site element (B′) are in the formula.",
    "BDPN": "How many of the third B-site element (B″) are in the formula.",
    "ON": "How many oxygen atoms are in the formula.",
    "A_avg_Z": "Average atomic number of the A-site, weighted by how much of each element is present.",
    "B_avg_Z": "Average atomic number of the B-site, weighted by how much of each element is present.",
    "A_avg_mass": "Average atomic mass of the A-site, weighted by the amounts of A and A′.",
    "B_avg_mass": "Average atomic mass of the B-site, weighted by the amounts of B, B′, and B″.",
    "FormulaMass": "Total mass of one formula unit — every element's atomic mass × its ratio, summed.",
    "O_to_cation_ratio": "Oxygen amount divided by the total cation amount (A-site + B-site). Higher = more oxygen-rich.",
    "B_to_A_ratio": "Total B-site amount divided by total A-site amount. 0.5 for La₂NiO₄ (1 B per 2 A).",
    "Mixed_A_site": "1 if the A-site contains more than one element (A and A′), otherwise 0.",
    "Mixed_B_site": "1 if the B-site contains more than one element (B′ or B″ present), otherwise 0.",
    "A_mix_fraction": "Share of the A-site taken by the SECOND element A′. 0 = single element, 0.5 = a 50/50 mix. Captures *how much* mixing, not just whether it happened.",
    "B_mix_fraction": "Share of the B-site taken by the extra elements (B′ + B″). 0 = single element, 0.5 = half the site.",
    "A_B_Z_diff": "A-site average atomic number minus B-site average atomic number — how different the two sites are.",
    "A_B_mass_diff": "A-site average mass minus B-site average mass.",
    "Avg_cation_Z": "Average atomic number across BOTH cation sites together, weighted by amounts.",
    "Avg_cation_mass": "Average atomic mass across both cation sites together, weighted by amounts.",
    "Cation_count": "How many distinct cation elements the compound has (A + A′ + B + B′ + B″ that are filled in).",
    "N_A_elements": "How many distinct elements are on the A-site (1 or 2).",
    "N_B_elements": "How many distinct elements are on the B-site (1, 2, or 3).",
    "A_avg_EN": "Average Pauling electronegativity of the A-site — how strongly its atoms pull electrons.",
    "B_avg_EN": "Average Pauling electronegativity of the B-site.",
    "EN_difference_B_minus_A": "B-site electronegativity minus A-site electronegativity — bigger = more ionic-character contrast between sites.",
    "Tolerance_factor": "Goldschmidt tolerance factor t = (r_A + r_O) / (√2 (r_B + r_O)) from Shannon ionic radii. t ≈ 0.9–1.0 fits an ideal cubic perovskite; lower = distorted, higher = hexagonal tendencies. Radii assume typical oxidation states; missing radii are filled with the class median.",
    "A_site_radius": "Ratio-weighted average Shannon ionic radius of the A-site (12-coordinate, Å).",
    "B_site_radius": "Ratio-weighted average Shannon ionic radius of the B-site (6-coordinate, Å).",
}


def glossary_table(features: Sequence[str]) -> pd.DataFrame:
    """Return a Term/Meaning table for the given internal feature names."""

    rows = []
    for name in features:
        rows.append({
            "Term": friendly_label(name),
            "Meaning": FEATURE_GLOSSARY.get(name, "Numeric descriptor calculated from the split formula."),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def mixing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize outcomes for single-element vs mixed-cation compounds.

    Directly answers the instructor question "when A/A′ or B/B′ are present in
    differing amounts, is that reflected anywhere?" — this table shows how many
    compounds mix cations on each site, how mixed they are on average, and how
    their bubble/purity rates compare with simple compounds.
    """

    needed = {"Mixed_A_site", "Mixed_B_site", "BubbleYes", "PhaseN"}
    if df.empty or not needed.issubset(df.columns):
        return pd.DataFrame()

    d = df.copy()
    a_mix = d["Mixed_A_site"].fillna(0).astype(int)
    b_mix = d["Mixed_B_site"].fillna(0).astype(int)

    def _bucket(a: int, b: int) -> str:
        if a and b:
            return "Mixed A & B sites"
        if a:
            return "Mixed A-site only"
        if b:
            return "Mixed B-site only"
        return "Single-element sites"

    d["MixCategory"] = [_bucket(a, b) for a, b in zip(a_mix, b_mix)]

    def _pct(sub_df: pd.DataFrame, column: str) -> float:
        """Mean of a 0-1 column as a percent, safe against missing/NaN."""
        if column not in sub_df.columns:
            return 0.0
        mean = pd.to_numeric(sub_df[column], errors="coerce").mean()
        return 0.0 if pd.isna(mean) else round(float(mean) * 100, 1)

    rows = []
    order = ["Single-element sites", "Mixed A-site only", "Mixed B-site only", "Mixed A & B sites"]
    for category in order:
        sub = d[d["MixCategory"] == category]
        if sub.empty:
            continue
        rows.append({
            "Compound type": category,
            "Compounds": int(len(sub)),
            "Bubble yes %": _pct(sub, "BubbleYes"),
            "Pure %": round(float((pd.to_numeric(sub["PhaseN"], errors="coerce") == 2).mean()) * 100, 1),
            "Avg A′ share of A-site %": _pct(sub, "A_mix_fraction"),
            "Avg B′/B″ share of B-site %": _pct(sub, "B_mix_fraction"),
        })
    return pd.DataFrame(rows)


def plot_heatmap(corr: pd.DataFrame) -> go.Figure:
    """
    Interactive correlation heatmap.

    Hover any cell to see the exact pair and correlation value; cells show the
    number when the matrix is small enough to stay readable.
    """

    n = len(corr.columns)
    labels_x = [friendly_label(c) for c in corr.columns]
    labels_y = [friendly_label(c) for c in corr.index]

    heat = go.Heatmap(
        z=np.round(corr.values, 2),
        x=labels_x,
        y=labels_y,
        zmin=-1, zmax=1,
        colorscale="RdBu", reversescale=True,   # blue = -1, red = +1 (matches old map)
        xgap=2, ygap=2,
        colorbar={
            "title": {"text": "Correlation", "font": {"color": CHART_MUTED}},
            "tickfont": {"color": CHART_MUTED},
            "thickness": 12, "outlinewidth": 0,
        },
        hovertemplate="<b>%{y} × %{x}</b><br>correlation = %{z:.2f}<extra></extra>",
    )
    fig = go.Figure(heat)

    if n <= 16:
        fig.update_traces(
            text=np.round(corr.values, 2), texttemplate="%{text:.2f}",
            textfont={"size": 10},
        )

    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-38)
    height = max(440, min(820, 40 * n + 190))
    return _apply_chart_theme(fig, height=height)


def bubble_relationships(corr: pd.DataFrame, restrict_to: Optional[Sequence[str]] = None,
                         top_n: int = 10) -> pd.DataFrame:
    """
    Extract the strongest relationships with bubble response.

    Parameters
    ----------
    restrict_to:
        When given, only these features are ranked. The Heatmap tab passes the
        user's current map selection so the table and the map always agree
        (instructor feedback: items appeared in this table but not on the map).
        When None, every numeric feature is ranked.

    Feature names are returned as the same friendly labels used on the heatmap.
    """

    target_col = "BubbleYes" if "BubbleYes" in corr.columns else "BubN" if "BubN" in corr.columns else None
    if target_col is None:
        return pd.DataFrame()

    rel = corr[target_col].drop(labels=[target_col, "BubN"], errors="ignore").dropna()
    if restrict_to is not None:
        keep = [f for f in restrict_to if f in rel.index and f != target_col]
        rel = rel.loc[keep]
    if rel.empty:
        return pd.DataFrame()

    out = rel.reset_index()
    out.columns = ["Feature", "Correlation with bubble result"]
    out["Feature"] = out["Feature"].map(friendly_label)
    out["Strength"] = out["Correlation with bubble result"].abs()
    out = out.sort_values("Strength", ascending=False).drop(columns=["Strength"])
    return out.head(top_n)


# =============================================================================
# 9b. PERIODIC TABLE VIEW, EXTRA CHARTS, NEIGHBORS, MERGE + REPORT HELPERS
# =============================================================================

def _build_periodic_positions() -> Dict[str, Tuple[int, int]]:
    """(row, column) grid position for every element, La/Ac series offset below."""

    positions: Dict[str, Tuple[int, int]] = {}

    def place(row: int, start_col: int, symbols: str) -> None:
        for offset, sym in enumerate(symbols.split()):
            positions[sym] = (row, start_col + offset)

    place(1, 1, "H"); positions["He"] = (1, 18)
    place(2, 1, "Li Be"); place(2, 13, "B C N O F Ne")
    place(3, 1, "Na Mg"); place(3, 13, "Al Si P S Cl Ar")
    place(4, 1, "K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr")
    place(5, 1, "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe")
    place(6, 1, "Cs Ba La"); place(6, 4, "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn")
    place(7, 1, "Fr Ra Ac"); place(7, 4, "Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og")
    place(9, 4, "Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu")   # lanthanides
    place(10, 4, "Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr")    # actinides
    return positions


PERIODIC_POSITIONS = _build_periodic_positions()


@st.cache_data(show_spinner=False)
def element_outcome_summary(df: pd.DataFrame, scope: str = "any") -> pd.DataFrame:
    """
    Per-element outcome stats for the periodic-table view.

    scope: "A" (A/A′ columns), "B" (B/B′/B″ columns), or "any" (all five).
    Each compound row counts once per element even if the element fills two slots.
    Returns columns: Symbol, Compounds, BubbleYesPct, PurePct.
    """

    scope_cols = {"A": ["A", "AP"], "B": ["B", "BP", "BDP"],
                  "any": ["A", "AP", "B", "BP", "BDP"]}.get(scope, ["A", "AP", "B", "BP", "BDP"])
    cols = [c for c in scope_cols if c in df.columns]
    if df.empty or not cols:
        return pd.DataFrame(columns=["Symbol", "Compounds", "BubbleYesPct", "PurePct"])

    work = df.reset_index(drop=True).copy()
    work["_row_id"] = work.index
    melted = work.melt(id_vars=["_row_id", "BubbleYes", "PhaseN"], value_vars=cols,
                       value_name="Symbol")
    melted["Symbol"] = melted["Symbol"].astype(str).str.strip()
    melted = melted[melted["Symbol"] != ""]
    melted = melted.drop_duplicates(subset=["_row_id", "Symbol"])
    if melted.empty:
        return pd.DataFrame(columns=["Symbol", "Compounds", "BubbleYesPct", "PurePct"])

    grouped = melted.groupby("Symbol").agg(
        Compounds=("_row_id", "count"),
        BubbleYesPct=("BubbleYes", lambda s: float(pd.to_numeric(s, errors="coerce").mean()) * 100),
        PurePct=("PhaseN", lambda s: float((pd.to_numeric(s, errors="coerce") == 2).mean()) * 100),
    ).reset_index()
    return grouped.round(1)


def plot_periodic_heat(summary: pd.DataFrame, value_col: str = "BubbleYesPct") -> go.Figure:
    """
    Periodic table colored by an outcome rate (default: bubble-yes %).

    Elements the class used are colored red → amber → green; unused elements are
    dim gray. Hover a colored tile for its counts and rates.
    """

    n_rows, n_cols = 10, 18
    z = [[None] * n_cols for _ in range(n_rows)]
    text = [[""] * n_cols for _ in range(n_rows)]
    custom = [[[0, 0.0, 0.0]] * n_cols for _ in range(n_rows)]

    stats = {row["Symbol"]: row for _, row in summary.iterrows()}
    unused_x, unused_y, unused_text = [], [], []

    for sym, (row, col) in PERIODIC_POSITIONS.items():
        r, c = row - 1, col - 1
        if sym in stats:
            s = stats[sym]
            z[r][c] = float(s[value_col])
            text[r][c] = sym
            custom[r][c] = [int(s["Compounds"]), float(s["BubbleYesPct"]), float(s["PurePct"])]
        else:
            unused_x.append(col)
            unused_y.append(row)
            unused_text.append(sym)

    value_title = "% bubbled" if value_col == "BubbleYesPct" else "% pure"
    fig = go.Figure(go.Heatmap(
        z=z, text=text, customdata=custom,
        x=list(range(1, n_cols + 1)), y=list(range(1, n_rows + 1)),
        zmin=0, zmax=100,
        colorscale=[[0.0, "#f87171"], [0.5, "#fbbf24"], [1.0, "#34d399"]],
        xgap=3, ygap=3, hoverongaps=False,
        texttemplate="%{text}", textfont={"size": 11, "color": "#0b1120"},
        colorbar={"title": {"text": value_title, "font": {"color": CHART_MUTED}},
                  "tickfont": {"color": CHART_MUTED}, "thickness": 12, "outlinewidth": 0},
        hovertemplate="<b>%{text}</b><br>%{customdata[0]} compounds"
                      "<br>bubble yes: %{customdata[1]:.0f}%"
                      "<br>pure: %{customdata[2]:.0f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=unused_x, y=unused_y, mode="text", text=unused_text,
        textfont={"size": 10, "color": "#3d4a63"}, hoverinfo="skip", showlegend=False,
    ))
    fig.update_xaxes(visible=False, range=[0.4, n_cols + 0.6])
    fig.update_yaxes(visible=False, range=[n_rows + 0.6, 0.4])
    return _apply_chart_theme(fig, height=430)


def plot_confusion(confusion: Dict[str, int]) -> go.Figure:
    """2×2 confusion-matrix heatmap for the bubble model's held-out test set."""

    z = [[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]]
    fig = go.Figure(go.Heatmap(
        z=z,
        x=["Predicted no", "Predicted yes"],
        y=["Actual no", "Actual yes"],
        colorscale=[[0.0, CHART_BAR_LOW], [1.0, CHART_ACCENT]],
        xgap=3, ygap=3, showscale=False,
        texttemplate="%{z}", textfont={"size": 16},
        hovertemplate="%{y} · %{x}: <b>%{z}</b><extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    return _apply_chart_theme(fig, height=280)


def plot_what_if(sweep: pd.DataFrame, x_label: str, current_value: float) -> go.Figure:
    """
    Line chart of predicted probabilities as ONE input is swept.

    Expects columns: value, bubble_prob, and optionally purity_prob.
    """

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sweep["value"], y=sweep["bubble_prob"], mode="lines+markers",
        name="Bubble probability", line={"color": CHART_ACCENT, "width": 3},
        marker={"size": 6},
        hovertemplate=x_label + " = %{x:g}<br>bubble: <b>%{y:.0%}</b><extra></extra>",
    ))
    if "purity_prob" in sweep.columns:
        fig.add_trace(go.Scatter(
            x=sweep["value"], y=sweep["purity_prob"], mode="lines+markers",
            name="Purity probability", line={"color": "#34d399", "width": 3, "dash": "dot"},
            marker={"size": 6},
            hovertemplate=x_label + " = %{x:g}<br>pure: <b>%{y:.0%}</b><extra></extra>",
        ))
    fig.add_vline(x=current_value, line_dash="dash", line_color="#94a3b8",
                  annotation_text="current", annotation_font_color="#94a3b8")
    fig.update_xaxes(title_text=x_label, title_font={"color": CHART_MUTED})
    fig.update_yaxes(range=[-0.02, 1.02], tickformat=".0%")
    fig.update_layout(legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0})
    fig = _apply_chart_theme(fig, height=380)
    fig.update_layout(margin={"t": 56})   # headroom so the legend never overlaps
    return fig


def nearest_neighbors(df: pd.DataFrame, pred_row: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    The k most compositionally similar past experiments to a proposed compound.

    Similarity = Euclidean distance in standardized descriptor space (composition
    only — phase is excluded; mass included since this is physical similarity,
    not model input). Returns Formula, Semester, Bubble, Phase, Distance.
    """

    feats, _ = feature_columns(use_phase=False, use_mass=True)
    feats = [c for c in feats if c in df.columns and c in pred_row.columns]
    if df.empty or not feats:
        return pd.DataFrame()

    base = df[feats].apply(pd.to_numeric, errors="coerce")
    means = base.mean()
    stds = base.std(ddof=0).replace({0: 1.0}).fillna(1.0)
    base_z = (base.fillna(means) - means) / stds

    target = pd.to_numeric(pred_row.iloc[0][feats], errors="coerce")
    target_z = (target.fillna(means) - means) / stds

    dist = np.sqrt(((base_z - target_z) ** 2).sum(axis=1))
    nearest = dist.nsmallest(int(k))

    out = pd.DataFrame({
        "Formula": df.loc[nearest.index, "Formula"] if "Formula" in df.columns else "",
        "Semester": df.loc[nearest.index, "Semester"] if "Semester" in df.columns else "",
        "Bubble": df.loc[nearest.index, "BubbleLabel"] if "BubbleLabel" in df.columns else "",
        "Phase": df.loc[nearest.index, "PhaseLabel"] if "PhaseLabel" in df.columns else "",
        "Distance": nearest.round(2),
    })
    return out.reset_index(drop=True)


def find_merge_duplicates(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """
    Rows in `new` that already appear in `existing` (New Semester merge check).

    Two rows match when group, semester, slot, composition, and outcomes are all
    identical after trimming/lowercasing. Returns the duplicate rows from `new`.
    """

    key_cols = [c for c in ["GroupNumber", "Semester", "Slot", "A", "AN", "B", "BN",
                            "ON", "P", "Bub"] if c in existing.columns and c in new.columns]
    if existing.empty or new.empty or len(key_cols) < 4:
        return pd.DataFrame()

    def _keys(frame: pd.DataFrame) -> pd.Series:
        parts = [frame[c].astype(str).str.strip().str.lower() for c in key_cols]
        out = parts[0]
        for p in parts[1:]:
            out = out + "|" + p
        return out

    existing_keys = set(_keys(existing))
    new_keys = _keys(new)
    return new[new_keys.isin(existing_keys)].copy()


def combine_long_tables(frames: Sequence[pd.DataFrame],
                        drop_exact_duplicates: bool = True) -> Tuple[pd.DataFrame, int]:
    """
    Concatenate several long-format tables (instructor Combine tab).

    Each student can download their own data from the app; this stitches those
    files back into ONE master table. When drop_exact_duplicates is True, rows
    that match on group, semester, slot, composition, and outcomes (after
    trimming/lowercasing) are kept once — so uploading overlapping files is safe.

    Returns (combined_df, n_duplicates_removed).
    """

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(), 0

    merged = pd.concat(frames, ignore_index=True)
    if not drop_exact_duplicates:
        return merged, 0

    key_cols = [c for c in ["GroupNumber", "Semester", "Slot", "A", "AN", "AP", "APN",
                            "B", "BN", "BP", "BPN", "BDP", "BDPN", "ON", "P", "Bub"]
                if c in merged.columns]
    if len(key_cols) < 4:
        return merged, 0

    normalized = merged[key_cols].astype(str).apply(lambda s: s.str.strip().str.lower())
    keep = ~normalized.duplicated(keep="first")
    return merged[keep].reset_index(drop=True), int((~keep).sum())


def _report_table(df: Optional[pd.DataFrame], max_rows: int = 12) -> str:
    """Render a DataFrame as report HTML (or a placeholder note)."""

    if df is None or df.empty:
        return "<p class='note'>Not available for this dataset.</p>"
    return df.head(max_rows).to_html(index=False, border=0, classes="tbl", justify="left")


def build_html_report(stats: Dict[str, object],
                      bubble_counts: Optional[pd.DataFrame],
                      phase_counts: Optional[pd.DataFrame],
                      links: Optional[pd.DataFrame],
                      mix: Optional[pd.DataFrame],
                      a_summary: Optional[pd.DataFrame],
                      b_summary: Optional[pd.DataFrame],
                      ml: Optional[dict],
                      purity: Optional[dict]) -> bytes:
    """
    Build a self-contained, print-friendly HTML lab report (no personal info —
    aggregates only). Open in a browser and print to PDF if needed.
    """

    def _pct(x: object) -> str:
        try:
            return f"{float(x):.0%}"
        except (TypeError, ValueError):
            return "—"

    ml_html = "<p class='note'>Bubble model not available.</p>"
    if ml and ml.get("ok"):
        cv_line = (f"<li>Cross-validated accuracy ({ml['cv_folds']}-fold): "
                   f"<b>{_pct(ml['cv_mean'])} ± {ml['cv_std']:.02f}</b></li>"
                   if ml.get("cv_mean") is not None else "")
        cm = ml.get("confusion", {})
        ml_html = f"""
        <ul>
          <li>Test accuracy: <b>{_pct(ml['rf_accuracy'])}</b> (always-guess baseline {_pct(ml['baseline_accuracy'])})</li>
          {cv_line}
          <li>Precision {_pct(ml['rf_precision'])} · Recall {_pct(ml['rf_recall'])} · F1 {_pct(ml['rf_f1'])}</li>
          <li>Confusion (test set): TP {cm.get('tp', '—')} · TN {cm.get('tn', '—')} ·
              FP {cm.get('fp', '—')} · FN {cm.get('fn', '—')}</li>
        </ul>
        <h3>Top model features</h3>
        {_report_table(ml.get('importance').assign(Feature=ml.get('importance')['Feature'].map(friendly_label)).round(3) if ml.get('importance') is not None else None)}
        """

    purity_html = "<p class='note'>Purity model not available.</p>"
    if purity and purity.get("ok"):
        cv_line = (f"<li>Cross-validated accuracy ({purity['cv_folds']}-fold): "
                   f"<b>{_pct(purity['cv_mean'])} ± {purity['cv_std']:.02f}</b></li>"
                   if purity.get("cv_mean") is not None else "")
        purity_html = f"""
        <ul>
          <li>Test accuracy: <b>{_pct(purity['accuracy'])}</b> (baseline {_pct(purity['baseline_accuracy'])})</li>
          {cv_line}
          <li>Precision (pure): {_pct(purity['precision'])}</li>
        </ul>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>General Chemistry II — Class Data Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #1a2233; margin: 2.2rem auto; max-width: 880px; line-height: 1.5; }}
  h1 {{ font-size: 1.7rem; margin-bottom: 0.2rem; }}
  h2 {{ font-size: 1.15rem; border-bottom: 2px solid #dbe4f3; padding-bottom: 0.25rem; margin-top: 2rem; }}
  h3 {{ font-size: 1rem; margin-bottom: 0.3rem; }}
  .sub {{ color: #5b6b85; margin-top: 0; }}
  .stats {{ display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 1rem 0; }}
  .stat b {{ font-size: 1.4rem; display: block; }}
  .stat span {{ color: #5b6b85; font-size: 0.85rem; }}
  table.tbl {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; margin: 0.4rem 0 1rem; }}
  table.tbl th {{ text-align: left; background: #eef3fb; padding: 0.4rem 0.6rem; }}
  table.tbl td {{ padding: 0.35rem 0.6rem; border-bottom: 1px solid #e4eaf5; }}
  .note {{ color: #5b6b85; font-style: italic; }}
  .disclaimer {{ margin-top: 2rem; padding: 0.8rem 1rem; background: #f5f8fd; border-left: 4px solid #7aa7e0; font-size: 0.85rem; color: #44526b; }}
</style></head><body>
<h1>General Chemistry II Catalyst Insight Studio — Class Data Report</h1>
<p class="sub">Generated {stats.get('generated', '')} · aggregates only, no student-identifying information</p>

<div class="stats">
  <div class="stat"><b>{stats.get('n_compounds', 0):,}</b><span>compounds</span></div>
  <div class="stat"><b>{stats.get('n_semesters', 0)}</b><span>semesters</span></div>
  <div class="stat"><b>{stats.get('bubble_rate', '—')}</b><span>bubbled (yes)</span></div>
  <div class="stat"><b>{stats.get('pure_rate', '—')}</b><span>came out pure</span></div>
  <div class="stat"><b>{stats.get('n_issues', 0)}</b><span>open data issues</span></div>
</div>

<h2>Response counts</h2>
<h3>Bubble response</h3>{_report_table(bubble_counts)}
<h3>Phase result</h3>{_report_table(phase_counts)}

<h2>Strongest links to bubbling (all features)</h2>{_report_table(links)}

<h2>Cation mixing (A/A′ and B/B′)</h2>{_report_table(mix)}

<h2>Element summaries</h2>
<h3>A-site elements (top by bubble-yes rate)</h3>{_report_table(a_summary)}
<h3>B-site elements (top by bubble-yes rate)</h3>{_report_table(b_summary)}

<h2>Bubble model</h2>{ml_html}
<h2>Purity model</h2>{purity_html}

<div class="disclaimer">These are descriptive statistics and hypothesis-generating models built from
student lab data. Correlation is not causation, and model predictions are not guarantees. Mass
descriptors: {stats.get('mass_note', 'per current sidebar setting')}.</div>
</body></html>"""
    return html.encode("utf-8")


# =============================================================================
# 10. MACHINE LEARNING HELPERS
# =============================================================================
# The ML section is intentionally simple and transparent. It is for generating
# hypotheses, not proving final scientific conclusions.

def feature_columns(use_phase: bool = True, use_mass: bool = True) -> Tuple[List[str], List[str]]:
    """
    Return numeric and categorical features used by the model.

    Future update point:
    Add or remove features here if the class wants a different ML experiment.
    """

    # Non-mass numeric features. Element identity is captured by atomic NUMBER
    # plus the mixing/ratio descriptors, so the model still "knows" the chemistry
    # without one-hot element columns.
    numeric = [
        "AN", "APN", "BN", "BPN", "BDPN", "ON",
        "A_avg_Z", "B_avg_Z", "O_to_cation_ratio", "B_to_A_ratio",
        "Mixed_A_site", "Mixed_B_site",
        "A_B_Z_diff", "Avg_cation_Z", "Cation_count", "N_B_elements",
        # Mixing fractions (#1): degree of A/A' and B/B' mixing.
        "A_mix_fraction", "B_mix_fraction",
        # Physics descriptors (July 2026): tolerance factor + electronegativity.
        # build_feature_matrix() drops any of these that aren't in the data.
        "Tolerance_factor", "A_site_radius", "B_site_radius",
        "A_avg_EN", "B_avg_EN", "EN_difference_B_minus_A",
    ]

    # Mass-based features are optional (#4). Atomic mass and atomic number are
    # strongly correlated, so the five mass features can dominate importance and
    # crowd out other signal. The "Include mass descriptors" toggle drops them.
    if use_mass:
        numeric += ["A_avg_mass", "B_avg_mass", "FormulaMass", "A_B_mass_diff", "Avg_cation_mass"]

    if use_phase:
        numeric.append("PhaseN")

    # Categorical element columns are intentionally removed (#3) to avoid the
    # model memorizing exact elements (overfitting on a small dataset).
    categorical: List[str] = []
    return numeric, categorical


def build_feature_matrix(
    df: pd.DataFrame,
    use_phase: bool = True,
    expected_columns: Optional[List[str]] = None,
    use_mass: bool = True,
) -> pd.DataFrame:
    """
    Build the numeric model feature matrix.

    If expected_columns is provided, the output is aligned to a trained model's
    column order. This is important when predicting a new single compound — and it
    means prediction stays correct even if use_mass differs, because the columns
    are reindexed to whatever the model was trained on.
    """

    numeric_cols, categorical_cols = feature_columns(use_phase=use_phase, use_mass=use_mass)
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    numeric_part = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # categorical_cols is normally empty now (element one-hots were removed, #3),
    # so only build dummies when categorical features actually exist. Calling
    # pd.get_dummies on a zero-column frame raises "No objects to concatenate".
    if categorical_cols:
        categorical_part = pd.get_dummies(
            df[categorical_cols].fillna("").astype(str),
            prefix=categorical_cols,
            dummy_na=False,
        )
        features = pd.concat([numeric_part, categorical_part], axis=1)
    else:
        features = numeric_part

    if expected_columns is not None:
        features = features.reindex(columns=expected_columns, fill_value=0)

    return features


@st.cache_data(show_spinner=False)
def train_ml_model(df: pd.DataFrame, use_phase: bool = True, use_mass: bool = True, random_state: int = 42) -> dict:
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

    X = build_feature_matrix(model_df, use_phase=use_phase, use_mass=use_mass)
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
    rf_preds = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_preds)
    rf_precision = precision_score(y_test, rf_preds, zero_division=0)
    rf_recall = recall_score(y_test, rf_preds, zero_division=0)
    rf_f1 = f1_score(y_test, rf_preds, zero_division=0)

    # Test-set confusion matrix: how the held-out predictions actually landed.
    cm = confusion_matrix(y_test, rf_preds, labels=[0, 1])
    confusion = {"tn": int(cm[0][0]), "fp": int(cm[0][1]),
                 "fn": int(cm[1][0]), "tp": int(cm[1][1])}

    # Stratified k-fold cross-validation: a more honest accuracy estimate than a
    # single split on a small dataset. Skipped when a class is too rare to fold.
    cv_mean = cv_std = None
    cv_folds = 5 if (y.value_counts().min() >= 5 and len(y) >= 50) else \
               3 if y.value_counts().min() >= 3 else 0
    if cv_folds:
        cv_model = RandomForestClassifier(
            n_estimators=150, max_depth=7, min_samples_leaf=2,
            random_state=random_state, n_jobs=-1, class_weight="balanced_subsample",
        )
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(cv_model, X, y, cv=skf, scoring="accuracy")
        cv_mean, cv_std = float(scores.mean()), float(scores.std())

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

    # Baseline: a naive model that always predicts the majority class.
    majority_class = int(y_train.mode()[0])
    baseline_preds = [majority_class] * len(y_test)
    baseline_accuracy = accuracy_score(y_test, baseline_preds)

    # Class balance across the full labeled dataset.
    yes_count = int((y == 1).sum())
    no_count = int((y == 0).sum())
    yes_rate = yes_count / len(y) if len(y) > 0 else 0.0

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
        "rf_precision": float(rf_precision),
        "rf_recall": float(rf_recall),
        "rf_f1": float(rf_f1),
        "lr_accuracy": float(lr_accuracy),
        "baseline_accuracy": float(baseline_accuracy),
        "yes_count": yes_count,
        "no_count": no_count,
        "yes_rate": float(yes_rate),
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "importance": importance,
        "overfit_warning": overfit_warning,
        "confusion": confusion,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "cv_folds": cv_folds,
    }


@st.cache_data(show_spinner=False)
def train_purity_model(df: pd.DataFrame, use_mass: bool = True, random_state: int = 42) -> dict:
    """
    Train a Random Forest to predict whether a compound comes out PURE (#14).

    Mirrors train_ml_model but the target is purity (pure = 1, impure/not made = 0).
    Phase is never used as an input here because it IS the thing being predicted.

    Returns a dict with results, or {"ok": False, "message": ...}.
    """

    if not SKLEARN_AVAILABLE:
        return {"ok": False, "message": "scikit-learn is not installed. Install requirements.txt to use ML Lab."}

    # Keep only rows whose phase was understood (pure / impure / not made).
    model_df = df[df["P"].isin(["pure", "impure", "not made"])].copy()
    if model_df.empty:
        return {"ok": False, "message": "No rows have a recognized phase label yet."}

    model_df["IsPure"] = (model_df["P"] == "pure").astype(int)

    if len(model_df) < 10:
        return {"ok": False, "message": "Purity model needs at least 10 rows with a known phase."}
    if model_df["IsPure"].nunique() < 2:
        return {"ok": False, "message": "Purity model needs both pure and not-pure examples."}

    # use_phase=False so PhaseN (the answer) is never fed in as a feature.
    X = build_feature_matrix(model_df, use_phase=False, use_mass=use_mass)
    y = model_df["IsPure"].astype(int)

    stratify = y if y.value_counts().min() >= 2 else None
    test_size = 0.25 if len(model_df) >= 20 else 0.35
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    rf = RandomForestClassifier(
        n_estimators=150, max_depth=7, min_samples_leaf=2,
        random_state=random_state, n_jobs=-1, class_weight="balanced_subsample",
    )
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)

    cv_mean = cv_std = None
    cv_folds = 5 if (y.value_counts().min() >= 5 and len(y) >= 50) else \
               3 if y.value_counts().min() >= 3 else 0
    if cv_folds:
        cv_model = RandomForestClassifier(
            n_estimators=150, max_depth=7, min_samples_leaf=2,
            random_state=random_state, n_jobs=-1, class_weight="balanced_subsample",
        )
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(cv_model, X, y, cv=skf, scoring="accuracy")
        cv_mean, cv_std = float(scores.mean()), float(scores.std())

    importance = pd.DataFrame(
        {"Feature": X.columns, "Importance": rf.feature_importances_}
    ).sort_values("Importance", ascending=False)
    importance = importance[importance["Importance"] > 0].head(12)

    majority_class = int(y_train.mode()[0])
    baseline_accuracy = accuracy_score(y_test, [majority_class] * len(y_test))

    pure_count = int((y == 1).sum())
    impure_count = int((y == 0).sum())
    pure_rate = pure_count / len(y) if len(y) else 0.0

    overfit_warning = None
    if accuracy > 0.85 and len(model_df) < 100:
        overfit_warning = (
            f"Purity accuracy is {accuracy:.0%} with only {len(model_df)} rows. "
            "High accuracy on a small dataset can mean overfitting — treat as a hint, not proof."
        )

    return {
        "ok": True,
        "model": rf,
        "feature_columns": list(X.columns),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "baseline_accuracy": float(baseline_accuracy),
        "pure_count": pure_count,
        "impure_count": impure_count,
        "pure_rate": float(pure_rate),
        "importance": importance,
        "overfit_warning": overfit_warning,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "cv_folds": cv_folds,
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
    radii_table: Optional[pd.DataFrame] = None,
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
    described = add_chemical_descriptors(clean, atomic, en_table, radii_table)
    return described


# =============================================================================
# 10b. DIMENSION REDUCTION (PCA / CLUSTERING)
# =============================================================================
# PCA squeezes the many numeric descriptors down to a 2-D map so compounds can be
# plotted and compared. It is an EXPLORATION tool: nearby points have similar
# composition. Outcomes (bubble / pure) do not separate strongly in this dataset,
# so the map is for spotting structure and outliers, not for prediction.

PCA_FEATURE_COLUMNS = [
    "AN", "BN", "ON", "A_avg_Z", "B_avg_Z", "A_avg_mass", "B_avg_mass",
    "FormulaMass", "O_to_cation_ratio", "B_to_A_ratio",
    "A_B_Z_diff", "A_B_mass_diff", "Avg_cation_Z", "Avg_cation_mass",
    "Cation_count", "N_B_elements", "A_mix_fraction", "B_mix_fraction",
    "Tolerance_factor", "A_site_radius", "B_site_radius",
    "A_avg_EN", "B_avg_EN", "EN_difference_B_minus_A",
]


@st.cache_data(show_spinner=False)
def compute_pca(df: pd.DataFrame, n_clusters: int = 0, random_state: int = 42) -> dict:
    """
    Project compounds onto their first two principal components.

    Parameters
    ----------
    n_clusters : int
        If >= 2, also run KMeans and label each point with a cluster id.

    Returns a dict: {ok, coords, explained, n_used, features, loadings, has_clusters}
    or {"ok": False, "message": ...} when PCA can't run.
    """

    if not SKLEARN_AVAILABLE:
        return {"ok": False, "message": "scikit-learn is not installed. Install requirements.txt to use this tab."}

    feats = [c for c in PCA_FEATURE_COLUMNS if c in df.columns]
    if df.empty or len(feats) < 2:
        return {"ok": False, "message": "Not enough numeric features to run PCA."}

    # Fill gaps with each column's mean, then drop any column that is still empty.
    data = df[feats].apply(pd.to_numeric, errors="coerce")
    data = data.fillna(data.mean(numeric_only=True)).dropna(axis=1, how="any")
    if data.shape[1] < 2 or len(data) < 3:
        return {"ok": False, "message": "Not enough complete rows/features to run PCA."}

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaled = StandardScaler().fit_transform(data.values)
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(scaled)

    out = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1]}, index=data.index)
    for out_col, src_col in [("Formula", "Formula"), ("Bubble", "BubbleLabel"), ("Phase", "PhaseLabel")]:
        out[out_col] = df.loc[data.index, src_col] if src_col in df.columns else ""

    if n_clusters and n_clusters >= 2 and len(out) >= n_clusters:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=int(n_clusters), n_init=10, random_state=random_state)
        out["Cluster"] = km.fit_predict(scaled).astype(str)

    loadings = pd.DataFrame(pca.components_[:2].T, index=data.columns, columns=["PC1", "PC2"])
    return {
        "ok": True,
        "coords": out,
        "explained": [float(v) for v in pca.explained_variance_ratio_[:2]],
        "n_used": len(out),
        "features": list(data.columns),
        "loadings": loadings,
        "has_clusters": "Cluster" in out.columns,
    }


def plot_pca_scatter(coords: pd.DataFrame, color_by: str = "Bubble") -> go.Figure:
    """
    Interactive 2-D PCA scatter, points colored by the chosen column.

    Hovering a point shows its formula and outcome — this is the fastest way for
    students to ask "what IS that odd point over there?".
    """

    fig = go.Figure()

    if color_by not in coords.columns:
        color_by = None

    if color_by is None:
        fig.add_trace(go.Scattergl(
            x=coords["PC1"], y=coords["PC2"], mode="markers",
            marker={"size": 9, "color": CHART_ACCENT, "opacity": 0.8},
            customdata=coords[["Formula"]].astype(str).values,
            hovertemplate="<b>%{customdata[0]}</b><br>PC1 %{x:.2f} · PC2 %{y:.2f}<extra></extra>",
        ))
    else:
        groups = coords[color_by].astype(str).replace("", "Missing").fillna("Missing")
        for i, label in enumerate(sorted(groups.unique())):
            sub = coords[groups == label]
            color = OUTCOME_COLORS.get(label, CHART_PALETTE[i % len(CHART_PALETTE)])
            hover_extra = f"{color_by}: {label}"
            fig.add_trace(go.Scattergl(
                x=sub["PC1"], y=sub["PC2"], mode="markers", name=str(label),
                marker={
                    "size": 9, "color": color, "opacity": 0.85,
                    "line": {"width": 1, "color": "rgba(10,15,28,0.9)"},
                },
                customdata=sub[["Formula"]].astype(str).values,
                hovertemplate="<b>%{customdata[0]}</b><br>PC1 %{x:.2f} · PC2 %{y:.2f}"
                              f"<extra>{hover_extra}</extra>",
            ))
        fig.update_layout(legend={
            "title": {"text": color_by, "font": {"color": CHART_MUTED}},
            "orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0,
        })

    fig.update_xaxes(title_text="PC1", title_font={"color": CHART_MUTED})
    fig.update_yaxes(title_text="PC2", title_font={"color": CHART_MUTED})
    fig = _apply_chart_theme(fig, height=520)
    fig.update_layout(margin={"t": 56})   # headroom so the legend never overlaps
    return fig


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
