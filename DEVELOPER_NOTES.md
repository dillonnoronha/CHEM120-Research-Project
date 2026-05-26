# Developer Notes

This file explains how the code is organized so future CHEM 120 students can update the app.

## Code structure

The app is organized into clear sections:

1. **App configuration**
   - App title, compound slots, phase/bubble mappings, and default columns.

2. **Page style**
   - `inject_css()` controls the modern UI look.

3. **Utility functions**
   - Helpers for cleaning text, parsing numbers, and formatting chemical ratios.

4. **Reference tables**
   - Loads `AtomicMass.csv`.
   - Optionally loads `PaulingEN.csv` for electronegativity descriptors.

5. **Data loading**
   - Reads uploaded CSV/Excel files.
   - Uses a demo dataset if no file is uploaded.

6. **Column mapping and reshaping**
   - `normalize_to_long_format()` converts the original wide spreadsheet into one compound per row.
   - `infer_slot_column()` is where you add support for new survey column names.

7. **Cleaning and formula reconstruction**
   - `clean_and_encode_data()` standardizes element symbols, ratios, phase, and bubble labels.
   - `reconstruct_formula()` rebuilds formulas such as `La2NiO4`.

8. **Validation and outlier detection**
   - `validate_compound_rows()` checks for missing/invalid entries.
   - `detect_numeric_outliers()` flags unusual numeric values.

9. **Chemical descriptors**
   - `add_chemical_descriptors()` calculates atomic number, atomic mass, formula mass, oxygen-to-cation ratio, B-to-A ratio, and mixed-site flags.
   - Add new chemistry descriptors here.

10. **Plotting helpers**
   - Chart functions are separate from the UI so they are easier to replace or improve.

11. **Machine learning helpers**
   - `train_ml_model()` trains the Random Forest model.
   - `feature_columns()` controls which features the model uses.
   - `build_feature_matrix()` prepares one-hot encoded features.

12. **Streamlit interface**
   - The bottom of `app.py` creates the sidebar, tabs, forms, charts, and downloads.

## Common updates

### Add a new descriptor

Edit:

```python
add_chemical_descriptors()
```

Example ideas:
- electronegativity difference
- tolerance factor
- ionic radius
- oxidation-state estimate

### Add a new validation rule

Edit:

```python
validate_compound_rows()
```

Example ideas:
- flag impossible oxygen ratios
- flag formulas with too many optional elements
- flag missing instructor/semester fields

### When the Google Form changes

If the department changes the Google Form wording, column names in the exported spreadsheet will change and the app will show a yellow warning banner listing exactly which fields it could not find.

**Step 1 — Get the new column names.**
Download a fresh export from the form (or look at the header row of the uploaded file listed in the warning).

**Step 2 — Open `infer_slot_column()` in `app.py`.**
Find the `field_aliases` dictionary near the top of the function. Each key is an internal field name; its value is a list of accepted lowercase column-name fragments.

```python
field_aliases = {
    "A":   ["a", "asite", "asiteelement", "aelement"],
    "AN":  ["an", "aratio", "asiteratio", "aamount"],
    "B":   ["b", "bsite", "bsiteelement", "belement"],
    "BN":  ["bn", "bratio", "bsiteratio", "bamount"],
    "O":   ["o", "oxygen", "oxygenelement"],
    "ON":  ["on", "oratio", "oxygenratio", "oxygenamount"],
    "P":   ["p", "phase"],
    "Bub": ["bub", "bubble", "bubbleresponse", "h2bubble", "bubbles"],
    # ... and so on
}
```

**Step 3 — Add the new column name fragment as a new alias.**
For example, if the form now exports a column called `1_AElement` for the A-site, add `"aelement"` (already there) or the new lowercase fragment to the `"A"` list.

**Step 4 — Test.**
Reload the app with the new export. The warning banner should disappear and compound rows should appear as normal.

**Required fields.** The app flags these eight as required for slot 1: `A`, `AN`, `B`, `BN`, `O`, `ON`, `P`, `Bub`. Optional fields (`AP`, `APN`, `BP`, `BPN`, `BDP`, `BDPN`) silently default to blank if not found.

### Support a new survey column name

Edit:

```python
infer_slot_column()
```

Add new aliases to `field_aliases`.

### Change model features

Edit:

```python
feature_columns()
```

### Change the UI style

Edit:

```python
inject_css()
```

## Performance notes

The app uses Streamlit caching:

```python
@st.cache_data(show_spinner=False)
```

Caching is used for file loading, reshaping, cleaning, descriptor calculations, validation, outlier detection, and export conversion.
This reduces reload time when students switch tabs or adjust filters.


## Phase graph fix

The phase graph is controlled by these functions in `app.py`:

```python
normalize_phase_label()
display_label()
plot_distribution()
```

`normalize_phase_label()` converts long raw survey answers into short categories before the graph is built. Update that function if future surveys add new phase wording.

Current phase categories:

```text
Pure
Impure
Not made
Other / needs review
Missing
```

This keeps the Phase Counts graph readable even when the uploaded spreadsheet contains long Microsoft Forms answers.
