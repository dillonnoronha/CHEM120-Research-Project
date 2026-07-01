# Daily Report — July 1, 2026

**Project:** CHEM 120 Catalyst Insight Studio
**Author:** Dillon Noronha (with Claude)
**Branch:** `secondmain`

---

## 1. Instructor feedback implemented (Dr. Jaynes / Fry-Petit — all 8 items)

1. **Cation mixing reflected.** New "Cation mixing — A/A′ and B/B′" section in Explore showing how many compounds mix cations per site, their average secondary-cation share, and how bubble/purity rates compare (class data: 468 mixed-B compounds averaging a 38% B′ share). Mix fractions also feed the heatmap and both models.
2. **A, A′, B, B′ in prediction.** All supported (plus B″ under advanced), now via element dropdowns so typos are impossible.
3. **Heatmap terms defined.** A glossary under the map explains every term in plain English, matching whatever features are currently selected.
4. **Mass dominance addressed.** Mass descriptors now default **off** (sidebar toggle to compare); the Predict tab explains why.
5. **Table ↔ map consistency.** "Strongest links to bubbling" now ranks only the features shown on the map by default (toggle to search all features).
6. **PCA loadings sorted.** Loading table sorts by biggest absolute weight, on PC1 or PC2.
7. **Privacy.** Emails, names, and member lists are never displayed in the app. Still collected — full data remains in instructor Export downloads, plus a new "no contact info" CSV for sharing.
8. **All/None filter buttons** for the Semester and Instructor pill filters in Explore.

## 2. Flagship UI revamp

- Dark "lab glass" theme (`.streamlit/config.toml` + custom CSS): animated gradient hero with live dataset chips, glass KPI cards, glowing pill tabs, hover lifts, animated probability bars, formula subscripts (La₂NiO₄).
- **All charts switched from matplotlib to interactive Plotly** — hover any PCA point for its formula, hover heatmap cells for exact correlations, zoom everywhere.
- Requirements updated: `plotly>=5.18` added, `matplotlib` removed, `streamlit>=1.41`.

## 3. Ten new features added

1. **Goldschmidt tolerance factor** from a new curated `data/ShannonRadii.csv` (typical oxidation states, documented approximations). In the heatmap, both models, PCA, and shown with every prediction. Class data spans t = 0.79–1.00 (median 0.95). Hand-verified: La₂NiO₄ → t = 0.934.
2. **Electronegativity descriptors** via new `data/PaulingEN.csv` (90 elements) — A/B site averages and B−A difference.
3. **Periodic-table heat view** in Explore — elements colored by bubble-yes rate, gray = untried, A/B/any-site filter, hover for stats.
4. **What-if explorer** in Predict — sweep any ratio and plot how bubble/purity probability responds.
5. **Closest past experiments** — every prediction shows the 5 most similar compounds the class actually made and their outcomes.
6. **Cross-validation + confusion matrix** for the bubble model (CV for purity too) — more honest metrics on ~800 rows.
7. **One-click class report** in Export — print-ready HTML with stats, correlations, mixing, element tables, and both models. Aggregates only.
8. **Instructor mode** (sidebar 🔑) — passcode-gated view of the hidden contact columns. Passcode set in `pipeline.py` (`INSTRUCTOR_PASSCODE`); change it before sharing.
9. **Duplicate detection** in New Semester merges (same group + semester + composition + outcome), with one-click exclusion.
10. **Deployment guide updated** (`DEPLOYMENT.md`) — Streamlit Community Cloud steps plus a privacy checklist (private repo or scrubbed dataset).

## 4. Verification

- Full pipeline run on the real dataset (206 groups → 824 compound rows): cleaning, descriptors, correlation (32 features), mixing summary, privacy filter, glossary, neighbors, duplicate detection, report generation — all pass.
- Tolerance-factor math verified against hand calculation; nearest-neighbor search correctly surfaces the class's actual La₂NiO₄ entries for a La₂NiO₄ prediction.
- Syntax + import cross-checks pass on both modules. Local `pytest` run recommended after `pip install -r requirements.txt` (sandbox couldn't install scikit-learn).

## 5. Files changed today

| File | Change |
|---|---|
| `app.py` | Rewritten UI, all feedback items + new features wired in |
| `pipeline.py` | Plotly charts, tolerance/EN descriptors, CV/confusion, periodic view, neighbors, duplicates, report builder |
| `requirements.txt` | + plotly, − matplotlib, streamlit ≥ 1.41 |
| `.streamlit/config.toml` | New — dark theme base |
| `data/PaulingEN.csv` | New — Pauling electronegativities |
| `data/ShannonRadii.csv` | New — Shannon ionic radii (documented approximations) |
| `DEPLOYMENT.md` | Cloud deployment + privacy section |

## 6. To run

```
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## 7. Follow-up fixes (same day, second commit)

- **Stale-bytecode ImportError fixed** — first launch failed with `cannot import name 'INSTRUCTOR_PASSCODE'` because OneDrive served an old `__pycache__/pipeline.pyc`; the cache folder was cleared (delete `__pycache__` if it ever recurs).
- **UI spacing** — tab row no longer clips into the KPI cards; hero → cards → tabs are evenly spaced.
- **PCA legend bug fixed** — a literal "undefined" rendered over the Structure-tab legend (Plotly title left undefined while themed); charts now always set a real title string, and horizontal legends got headroom so nothing overlaps.
- **Light/dark mode switcher** — new ☀️ toggle at the top of the sidebar. Swaps the full look: CSS palette, every Plotly chart (via `set_chart_theme`), and Streamlit's native widget/dataframe colors at runtime. Dark stays the default.
- **Explore filters simplified** — the two All/None buttons are now a single toggle per filter ("✕ Clear all" ⇄ "✓ Select all").
- Branch `secondmain` updated with all of the above — push with `git push -u origin secondmain`.
