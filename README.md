# CHEM 120 Catalyst Insight Studio

A student-friendly Streamlit app for the CHEM 120 research project.

## What the app does

The app helps students and instructors:

1. Upload CHEM 120 class data.
2. Convert the spreadsheet from wide format into one compound per row.
3. Rebuild formulas from A-site, B-site, oxygen, phase, and bubble fields.
4. Check for common data-entry mistakes.
5. Calculate chemical descriptors such as atomic number, atomic mass, formula mass, oxygen-to-cation ratio, and B-to-A ratio.
6. Explore bubble and phase trends with student-readable charts.
7. Explain the correlation heatmap in plain language.
8. Train a simple machine-learning model for hypothesis generation.
9. Add a new compound during the session and export the updated dataset.


## Latest fix: readable phase counts

The app now normalizes older phase wording before graphing. For example:

```text
pure phase/homogenous mixture -> Pure
homogenous mixture -> Pure
heterogeneous mixture -> Impure
did not make compound / melted to crucible -> Not made
blank -> Missing
```

This prevents the Phase Counts graph from becoming crowded with long overlapping labels when older class spreadsheets are uploaded.

## Run the app

Windows:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Or double-click:

```text
run_app_windows.bat
```

## Best file to upload

Upload the optimized class spreadsheet, usually:

```text
Combined_Data.xlsx
```

The app is designed to read columns like:

```text
1A, 1AN, 1AP, 1APN, 1B, 1BN, 1BP, 1BPN, 1BDP, 1BDPN, 1O, 1ON, 1P, 1Bub
2A, 2AN, ...
3A, 3AN, ...
4A, 4AN, ...
```

## Main tabs

### Start Here

A friendly overview that explains the purpose of the app and shows whether data loaded successfully.

### Check Data

Finds missing elements, invalid element symbols, wrong phase/bubble labels, missing ratios, and unusual numeric values.

### Explore Results

Shows bubble response and phase counts, plus A-site and B-site trend charts.

### Relationship Map

Shows a correlation heatmap and explains how to read it:
- +1 means two values tend to increase together.
- 0 means little linear relationship.
- -1 means one tends to increase while the other decreases.

### ML Lab

Trains a Random Forest model to estimate whether a compound is likely to have `bubble = yes`.
This is for hypothesis generation only.

### Add Compound

Lets a student enter A-site, B-site, oxygen, phase, and bubble values manually.
The app rebuilds the formula and can add it to the current session dataset.

### Export

Downloads cleaned data, validation report, outlier report, and Excel/CSV outputs.

## Important safety behavior

The app does not overwrite your original Excel file. This protects the master database.
Use the Export tab to save cleaned or updated data.


## Phase Counts graph fix

The Explore Results page intentionally groups long/raw phase answers into short categories before plotting:

- Pure
- Impure
- Not made
- Other / needs review
- Missing

This is handled in `normalize_phase_label()` and enforced again in `plot_distribution()` with a horizontal bar chart. Do not switch Phase Counts back to a vertical x-axis chart unless the labels are guaranteed to stay short.
