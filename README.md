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

The page scrolls through three sections — **Check Data → Explore Results → Relationship Map** — then three tabs at the bottom: **ML Lab | Add Compound | Export**.

## Notes

- Original uploaded file is never overwritten. Use Export to save cleaned data.
- If the app shows a column-mapping warning after a Google Form change, see `DEVELOPER_NOTES.md → When the Google Form changes`.
- ML Lab runs both Random Forest and Logistic Regression. If RF accuracy is much higher than LR on a small dataset, treat the results with caution.
