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
