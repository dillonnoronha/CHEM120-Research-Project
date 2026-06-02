# CHEM 120 Catalyst Insight Studio — Architecture

## Overview

Catalyst Insight Studio is a Streamlit web application that transforms student chemistry lab data into clean compound records, visual trend analysis, and machine-learning-based hypotheses. The app separates UI concerns (Streamlit widgets and layout) from data processing, ensuring portability and testability.

## File Organization

**`app.py`** (918 lines) — Streamlit user interface  
- Page layout, CSS theming, widget callbacks
- Imports ~50 functions from `pipeline.py`
- No data processing logic lives here

**`pipeline.py`** (1551 lines) — All data, ML, and plotting logic  
- No Streamlit UI code inside function bodies
- Uses `@st.cache_data` decorators to speed up expensive operations
- Organized into 11 sections (reference tables, data loading, cleaning, validation, descriptors, ML, visualization, export)

## Data Flow

```
User uploads CSV/Excel
       ↓
read_table_from_bytes()
       ↓
normalize_to_long_format() [wide → long]
       ↓
clean_and_encode_data() [symbols, labels, formula]
       ↓
add_chemical_descriptors() [atomic mass, Z, formula weight]
       ↓
validate_compound_rows() [error checking]
detect_numeric_outliers() [statistical flags]
       ↓
[Explore: plots, summaries, correlations]
[Hypothesize: train ML model, predict bubble=yes]
       ↓
Export: CSV or Excel download
```

## Page Layout (`app.py`)

1. **Sidebar:** File upload widgets, outlier sensitivity slider, phase/ML toggles
2. **Hero section:** Title and workflow steps  
3. **Status strip:** Data source, row counts, validation summary
4. **Check Data tab:** Validation issues, outliers, cleaned compound table
5. **Explore Results tab:** Distribution charts, element trends, correlation heatmap
6. **ML Lab tab:** Model accuracy, feature importance, single-compound predictions
7. **Add Compound tab:** Form to manually enter and validate one compound
8. **Export tab:** Download cleaned data, reports, outlier flags

Session state tracks manually added compounds in `st.session_state.manual_entries`.

## Core Modules in `pipeline.py`

| Section | Purpose |
|---------|---------|
| **App Config** | Constants: APP_TITLE, COMPOUND_SLOTS, PHASE_MAP, BUBBLE_MAP, FRONT_COLUMNS |
| **Utilities** | Text cleaning, label normalization, numeric parsing (safe_float, clean_symbol) |
| **Reference Tables** | Load AtomicMass.csv and optional PaulingEN.csv; build element lookup maps |
| **Data Loading** | Read CSV/Excel files; find default database or generate demo dataset |
| **Column Mapping** | `infer_slot_column()` fuzzy-matches student survey columns (1A, 1AN, 2B, etc.) to standard names |
| **Cleaning & Encoding** | Standardize element symbols, reconstruct chemical formulas, encode phase/bubble labels numerically |
| **Validation** | Check for missing fields, invalid elements, incorrect labels, non-numeric ratios |
| **Descriptors** | Compute weighted-average atomic numbers/masses per site, formula mass, cation/oxygen ratios, mixed-site flags |
| **Summary & Plotting** | Aggregate by element, create bar/distribution/heatmap charts, compute correlations |
| **ML Helpers** | Select features, build one-hot-encoded matrix, train Random Forest and Logistic Regression |
| **Export** | Convert DataFrames to CSV or Excel bytes for download |

## Key Data Structures

**Wide → Long** (normalize_to_long_format)
- Input: One row per student group, 4 potential compounds per group (columns 1A, 1AN, 1B, …, 4Bub)
- Output: One row per compound, with metadata and formula fields

**Formula Fields** (after cleaning)
- A, AN, AP, APN (A-site element and ratio; optional A′)
- B, BN, BP, BPN, BDP, BDPN (B-site element, ratio; up to three B variants)
- O, ON (oxygen element and ratio, always "O")
- P, Bub (phase and bubble response, encoded as PhaseN/BubN)

**Descriptors** (computed per compound)
- A_avg_Z, B_avg_Z (weighted atomic numbers by site)
- FormulaMass (sum of element masses × stoichiometry)
- O_to_cation_ratio, B_to_A_ratio (structural ratios)
- Mixed_A_site, Mixed_B_site (binary flags)

## ML Training (`train_ml_model`)

Predicts bubble=yes probability using two models:
- **Random Forest** (150 trees, max_depth=7) — final predictor, provides feature importance
- **Logistic Regression** — sanity check for overfitting; compared side-by-side

Both share the same 75/25 train/test split. Features include ratios, atomic properties, and one-hot-encoded element symbols.

## Caching Strategy

Heavy operations are cached with `@st.cache_data`:
- File reading and table loading
- Formula normalization
- Data cleaning and descriptor computation
- ML model training

This keeps the UI responsive when students click between tabs.

## Future Extensions

- Add new descriptors in `add_chemical_descriptors()` (ionic radius, tolerance factor, electronegativity difference)
- Add new validation rules in `validate_compound_rows()`
- Adjust ML features in `feature_columns()` or change models in `train_ml_model()`
- Update survey column aliases in `infer_slot_column()` if Google Form questions change
