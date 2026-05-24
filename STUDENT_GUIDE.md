# Student Guide

## Best order to use the app

1. Upload `Combined_Data.xlsx`.
2. Open **Start Here** and confirm compounds loaded.
3. Open **Check Data** and review any issues.
4. Open **Explore Results** to compare A-site and B-site elements.
5. Open **Relationship Map** to look at numeric relationships.
6. Open **ML Lab** to test possible compounds as hypotheses.
7. Use **Add Compound** to create a new cleaned entry.
8. Use **Export** to save cleaned files.

## What is bubble response?

Bubble response is the lab observation of whether bubbles were seen:
- `yes`
- `no`
- `maybe`

The ML Lab predicts whether the result may be `bubble = yes`.

## What is the heatmap?

The heatmap compares numeric features:
- +1 means two things tend to go up together.
- 0 means no clear linear relationship.
- -1 means one goes up while the other goes down.

Look at the `BubbleYes` row or column to see which numeric features may be related to bubbling.
