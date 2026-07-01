# Deploying CHEM 120 Catalyst Insight Studio

This guide covers how a school can put the app online so students and instructors
just open a link — no installing Python.

## Is Streamlit a good choice for a school?

Yes. Streamlit is a strong fit here:

- **Free hosting.** Streamlit Community Cloud hosts public app repos for free.
- **Zero install for students.** They open a URL in any browser.
- **Pure Python.** The whole app is `app.py` + `pipeline.py`, easy for the next
  student team to maintain.
- **Auto-redeploy.** Push to GitHub and the live app updates within a minute.

Limits to know: Community Cloud apps are public by default (anyone with the link
can view), they sleep after inactivity and take ~30s to wake, and they run on
modest resources. For a class dataset of a few thousand rows that is plenty. If
the data must stay private, use the password option below or host internally.

## Option A — Streamlit Community Cloud (recommended for a school)

1. Create a free account at https://share.streamlit.io using a GitHub login.
2. Put this project in a GitHub repository (the instructor can own it). Make sure
   these files are included: `app.py`, `pipeline.py`, `requirements.txt`,
   `.streamlit/config.toml` (the theme), and the `data/` folder with
   `AtomicMass.csv`, `PaulingEN.csv`, `ShannonRadii.csv`, and `Combined_Data.xlsx`.
3. In Streamlit Cloud click **New app**, pick the repo, set the main file to
   `app.py`, and click **Deploy**.
4. Share the resulting URL with the class.

To update the app later, edit the files and `git push` — the live app rebuilds
automatically.

**Optional password.** To keep it semi-private, in the app's Streamlit Cloud
**Settings → Secrets** add a value and gate the app with a simple password check,
or restrict viewers to specific emails under the app's sharing settings.

### ⚠️ Privacy before deploying

`Combined_Data.xlsx` contains student emails, names, and group members. The app
never *displays* them (outside Instructor mode), but anyone who can see a public
GitHub repo can open the raw file. Before deploying publicly, either:

1. **Use a private GitHub repo** (Streamlit Cloud can deploy from private repos), or
2. **Ship a scrubbed dataset**: open the app locally, go to **Export → Cleaned
   data — no contact info (CSV)**, save that file as the repo's dataset instead,
   and keep the full spreadsheet offline.

Also change `INSTRUCTOR_PASSCODE` in `pipeline.py` before sharing the app —
the default value is not a secret.

## Option B — Run locally on one computer (no internet)

Good for a single lab machine or testing.

```
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Then open the URL it prints (usually http://localhost:8501). On Windows you can
also double-click `run_app_windows.bat`.

## Option C — Host on a school server

If the school has its own server, run the same command behind the school network
and optionally put it behind the institution's single sign-on. This keeps the data
fully private.

## Loading the class data

The app loads data in this priority order:

1. A file the user uploads in the sidebar.
2. `data/Combined_Data.xlsx` shipped with the app.
3. A small built-in demo dataset (only if neither of the above exists).

`data/Combined_Data.xlsx` is now included, so the app loads the real class data on
startup. To roll forward a new semester, use the **New Semester** tab to merge the
new export, download the combined file, and replace `data/Combined_Data.xlsx` with
it (then push to GitHub if using Option A).
