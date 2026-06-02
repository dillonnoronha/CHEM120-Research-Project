# CHEM 120 Data Compiler

This version adds a clean Streamlit app that fills the yellow auto-generated columns in the CHEM 120 Excel sheet.

## What it fills

### Atomic-number columns

The app converts element symbols into atomic numbers:

| Symbol column | Filled column |
|---|---|
| `A` | `ZA` |
| `AP` | `ZAP` |
| `B` | `ZB` |
| `BP` | `ZBP` |
| `BDP` | `ZBDP` |
| `O` | `ZO` |

It also supports the old wide-format naming pattern:

| Symbol column | Filled column |
|---|---|
| `1A` | `1ZA` |
| `2B` | `2ZB` |
| `3BDP` | `3ZBDP` |

### Optional result-code columns

The app can also fill:

| Text column | Filled column | Rule |
|---|---|---|
| `P` | `PN` | `impure = 1`, `pure = 2` |
| `Bub` | `BubN` | `maybe = 0`, `yes = 1`, `no = 2` |

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

Then upload the Excel file using the upload box in the top-left sidebar.

## Files

- `app.py` — Streamlit web interface
- `chem120_excel_processor.py` — Excel-processing logic
- `atomic_numbers.py` — periodic table lookup
- `requirements.txt` — Python dependencies

## Notes

- The uploaded Excel workbook formatting is preserved.
- Existing wrong values can be corrected if **Correct existing values** is enabled.
- Invalid element symbols are shown in a warning table instead of silently failing.
