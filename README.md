# CHEM 120 Catalyst Insight Studio

Streamlit app for exploring CHEM 120 catalyst experiment data.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Windows shortcut: double-click `run_app_windows.bat`

## File structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — layout, widgets, tabs |
| `pipeline.py` | All data logic — cleaning, ML, exports |
| `data/Combined_Data.xlsx` | Main class database (auto-loaded if present) |
| `data/AtomicMass.csv` | Element reference table |

## Upload format

The app expects columns like `1A`, `1AN`, `1B`, `1BN`, `1O`, `1ON`, `1P`, `1Bub` (slots 1–4).
Upload `Combined_Data.xlsx` or any CHEM 120 survey export.

## Layout

The page scrolls through three sections — **Check Data → Explore Results → Relationship Map** — then four tabs at the bottom: **ML Lab | Add Compound | Export | New Semester**.

## ML Lab

- Predicts two outcomes: **bubbling** (yes vs no) and **purity** (pure vs impure). Pick the target with the toggle.
- Trains three models — **Random Forest, Gradient Boosting, Logistic Regression** — and reports **5-fold cross-validated** balanced accuracy and ROC-AUC (with ± spread), so scores are not at the mercy of one lucky split.
- Includes a **chi-squared test of independence** (is an element position statistically associated with the outcome?) and **permutation importance** (which features matter, without the mass bias of default tree importance).
- The single-compound predictor adapts its confidence to the model's cross-validated ROC-AUC — weak outcomes are shown as a rough lean, not a precise probability.

## Notes

- Original uploaded file is never overwritten. Use Export to save cleaned data.
- Rows with **critical** data-entry errors (missing/unknown element, missing/invalid required ratio) are automatically excluded from charts and ML. They still appear in the validation report.
- If the app shows a column-mapping warning after a Google Form change, see `DEVELOPER_NOTES.md → When the Google Form changes`.
- Use the **New Semester** tab to merge a new raw export into the existing dataset.

## Deploy

Push to GitHub, then deploy on [Streamlit Community Cloud](https://share.streamlit.io): point it at `app.py` on the `main` branch. Make sure `data/Combined_Data.xlsx` is committed so the app has data on startup.
