# CHEM 120 Catalyst Insight Studio — Architecture

## Overview

Catalyst Insight Studio is a Streamlit web application that turns student chemistry lab data into clean compound records, visual trend analysis, and machine-learning-based hypotheses. The app separates UI concerns (Streamlit widgets and layout) from data processing, which keeps it portable and testable.

## File Organization

**`app.py`** (~1120 lines) — Streamlit user interface
- Page layout, CSS theming, widgets, tabs, forms
- Imports its functions from `pipeline.py`
- No data-processing logic lives here

**`pipeline.py`** (~1900 lines) — All data, ML, statistics, plotting, and export logic
- No Streamlit UI code inside function bodies (only `@st.cache_data` decorators)
- Organized into 11 numbered sections

## Data Flow

```
User uploads CSV/Excel  (or app auto-loads data/Combined_Data.xlsx)
       ↓
read_table_from_bytes()
       ↓
normalize_to_long_format()   [wide 1A/1AN/… → long; also detects already-long files]
       ↓
clean_and_encode_data()      [symbols, labels, formula, BubbleYes / BubbleYesNo]
       ↓
add_chemical_descriptors()   [atomic mass, Z, formula weight, ratios, mixed-site flags]
       ↓
split_quarantine()           [drop rows with CRITICAL errors from analysis]
       ↓
validate_compound_rows()     [error report, tagged Critical / Minor]
detect_numeric_outliers()    [z-score flags]
       ↓
[Explore: distributions, element trends, correlation heatmap]
[Relationship Map: numeric correlations, strongest links to bubbling]
[ML Lab: train models for bubbling AND purity, chi-squared, permutation importance, predict]
       ↓
Export: CSV / Excel download   ·   New Semester: merge a new file into the dataset
```

## Page Layout (`app.py`)

A hero header and health cards sit on top; everything else lives in **three main tabs** (the third nests the action tools as inner tabs).

1. **Sidebar:** file upload, outlier-sensitivity slider, quick-help
2. **Hero header:** title and workflow steps
3. **Health cards:** loaded compounds, validation summary
4. **Tab — ✅ Check Data:** validation issues (Critical/Minor), outliers, cleaned compound table
5. **Tab — 📊 Explore & Relationships:** count tables, A-site & B-site trends (bubbling/purity toggle), correlation heatmap
6. **Tab — 🤖 ML Lab & Tools:** ML Lab plus inner tabs **Add Compound · Export · New Semester**

Session state tracks manually added compounds in `st.session_state.manual_entries`.

## Core Sections in `pipeline.py`

| Section | Purpose |
|---------|---------|
| **1. App Config** | Constants: APP_TITLE, COMPOUND_SLOTS, PHASE_MAP, BUBBLE_MAP, TARGET_CONFIG, FRONT_COLUMNS |
| **2. Utilities** | Text cleaning, label normalization, numeric parsing |
| **3. Reference Tables** | Load AtomicMass.csv (and optional electronegativity table); build element lookup maps |
| **4. Data Loading** | Read CSV/Excel; find default database or generate demo dataset |
| **5. Column Mapping** | `normalize_to_long_format()` handles both wide (1A, 1AN…) and already-long files; `infer_slot_column()` fuzzy-matches survey columns |
| **6. Cleaning & Encoding** | Standardize symbols, reconstruct formulas, encode phase/bubble labels |
| **7. Validation & Outliers** | `classify_row_issues()` (severity-tagged), `validate_compound_rows()`, `split_quarantine()`, `detect_numeric_outliers()` |
| **8. Descriptors** | Weighted-average atomic number/mass per site, formula mass, structural ratios, mixed-site flags |
| **9. Summary & Plotting** | Aggregate by element, bar/distribution/heatmap charts, correlation table |
| **10. ML Helpers** | Features, one-hot matrix, `train_classification_model()`, `chi_squared_tests()`, single-row prediction |
| **11. Export** | DataFrames → CSV / Excel bytes |

## Key Data Structures

**Wide → Long** (`normalize_to_long_format`)
- Wide input: one row per group, up to 4 compounds (1A…4Bub). Long input (one row per compound) is detected automatically and passed through.
- Output: one row per compound with metadata and formula fields.

**Formula fields** (after cleaning): A/AN (+ optional AP/APN), B/BN (+ optional BP/BPN, BDP/BDPN), O/ON, P, Bub.

**Outcome columns:** `BubbleYes` (yes vs everything, used for element-rate charts), `BubbleYesNo` (yes vs no, "maybe" = NaN — used by the correlation map so it matches the ML), `PhaseN`, `BubN`.

**Descriptors:** A_avg_Z, B_avg_Z, A_avg_mass, B_avg_mass, FormulaMass, O_to_cation_ratio, B_to_A_ratio, Mixed_A_site, Mixed_B_site.

## ML Training (`train_classification_model`)

Predicts one of two targets, set by `TARGET_CONFIG`:
- **bubble** — bubble = yes vs no ("maybe" dropped)
- **purity** — pure vs impure ("not made" dropped; phase can never be a feature here)

For the chosen target it:
1. Builds the two-class dataset (`prepare_target`) and one-hot feature matrix.
2. Trains **Random Forest**, **Gradient Boosting** (`HistGradientBoostingClassifier`), and **Logistic Regression** (scaled, in a `Pipeline`).
3. Reports **5-fold stratified cross-validated** balanced accuracy and ROC-AUC (mean ± std) for each, plus single-split precision/recall/F1.
4. Picks the best tree model (RF vs GB) by **cross-validated balanced accuracy** as the predictor.
5. Computes **permutation importance** (fair across categorical/numeric features) on the best model.

`chi_squared_tests()` complements the ML with a statistical test of whether each element position is associated with the outcome (returns p-values).

## Caching Strategy

Heavy operations use `@st.cache_data`: file reading, normalization, cleaning, descriptors, validation, quarantine, outliers, model training, chi-squared, and export conversion. This keeps the UI responsive as students scroll and switch tabs.

## Future Extensions

- Add descriptors in `add_chemical_descriptors()` (ionic radius, tolerance factor, electronegativity difference)
- Add validation rules in `classify_row_issues()` (set each new issue's severity)
- Add or retune models / features in `train_classification_model()` and `feature_columns()`
- Add a new prediction target by extending `TARGET_CONFIG`
- Update survey column aliases in `infer_slot_column()` (and the long-format aliases in `normalize_to_long_format()`) if the Google Form changes
