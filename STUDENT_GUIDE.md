# Student Guide

## Best order to use the app

The app loads `Combined_Data.xlsx` automatically (or upload your own file in the sidebar). Then work through the three main tabs:

1. **✅ Check Data** — confirm compounds loaded and review any issues.
2. **📊 Explore & Relationships** — compare A-site/B-site elements and view the relationship map.
3. **🤖 ML Lab & Tools** — train models and test compounds, plus the inner tools:
   - **Add Compound** — create a new cleaned entry.
   - **Export** — save cleaned files.
   - **New Semester** — merge a new semester's file into the dataset.

## What is bubble response?

Bubble response is the lab observation of whether bubbles were seen:
- `yes`
- `no`
- `maybe`

The ML Lab can predict two things: whether a compound **bubbles** (yes vs no) and whether it is **pure** (pure vs impure). Pick which one with the toggle at the top of ML Lab. Treat the results as hypotheses, not guarantees — the app shows how reliable each model is.

## What is the heatmap?

The heatmap compares numeric features:
- +1 means two things tend to go up together.
- 0 means no clear linear relationship.
- -1 means one goes up while the other goes down.

Look at the `BubbleYes` row or column to see which numeric features may be related to bubbling.
