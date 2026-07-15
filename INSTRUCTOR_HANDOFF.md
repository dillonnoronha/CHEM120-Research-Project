# Instructor Handoff — General Chemistry II Catalyst Insight Studio

Everything you need to keep the app running without the original student team.
Written for a non-programmer; the few command-line steps are copy/paste.

---

## 1. What this is

A Streamlit web app that takes the class's perovskite lab spreadsheet, cleans it,
and turns it into interactive charts, a correlation heatmap, a periodic-table
view, and two machine-learning models (will it bubble? will it be pure?).
Students use it in the browser; nothing is installed on their machines.

## 2. The pieces

| File / folder | What it is |
|---|---|
| `app.py` | The user interface (this is the "main file path" for deployment) |
| `pipeline.py` | All data cleaning, chemistry math, and ML. Settings live at the top — including `INSTRUCTOR_PASSCODE` |
| `data/Combined_Data.xlsx` | The class dataset the app loads on startup |
| `data/AtomicMass.csv`, `PaulingEN.csv`, `ShannonRadii.csv` | Reference tables (atomic data, electronegativity, ionic radii) |
| `.streamlit/config.toml` | Colors/theme (light by default; in-app toggle for dark) |
| `requirements.txt` | The Python packages the app needs |
| `DEPLOYMENT.md` | Detailed hosting options + privacy checklist |
| `DEVELOPER_NOTES.md` | For a future student maintainer — includes "when the Google Form changes" |
| `STUDENT_GUIDE.md` | How students use each tab |

The code lives on GitHub: **github.com/dillonnoronha/CHEM120-Research-Project**.
Whoever maintains the app needs access to that repository.

## 3. The link students use

The app is hosted on **Streamlit Community Cloud**. After deploying (see
DEPLOYMENT.md, Option A — main file path is `app.py`), the app gets a permanent
URL of the form:

    https://<app-name>.streamlit.app

That URL is the student link — bookmark it and put it in the syllabus/Canvas.
You can see or customize it at https://share.streamlit.io → your app → Settings.
Students never need a password; the 🔑 Instructor passcode is only for the
teaching team (it reveals the hidden contact columns — see Section 6).

Two behaviors to expect: the app **sleeps** after a few days without visitors
(first visitor waits ~30 seconds while it wakes), and it **redeploys itself**
automatically within a minute whenever the GitHub repository is updated.

## 4. Each new semester (the one recurring task)

1. Open the app → **📥 New Semester** tab.
2. Upload the semester's raw Excel export. The app expands it, warns about
   duplicates, and shows a merge preview.
3. Click **Download merged Combined_Data.xlsx**.
4. Replace `data/Combined_Data.xlsx` in the GitHub repository with that file
   (on github.com: open the file → pencil/upload → commit). The live app
   updates itself.

## 5. Running it on your own computer (optional)

With Python installed (python.org, 3.11 or newer), from the project folder:

    py -m pip install -r requirements.txt
    py -m streamlit run app.py

or double-click `run_app_windows.bat`. The app opens at http://localhost:8501.

## 6. Privacy & the instructor passcode

The app never displays student emails, names, or group members — anywhere.
That data stays in the spreadsheet and in the full Export downloads.
To see it inside the app: sidebar → **🔑 Instructor mode** → passcode → Unlock.
The passcode is the `INSTRUCTOR_PASSCODE` line near the top of `pipeline.py`
(change it there; anyone with the repo can read it, so treat it as a
convenience latch, not security). Before making the repository public, read
the **Privacy before deploying** section of DEPLOYMENT.md.

## 7. If something breaks

| Symptom | Fix |
|---|---|
| `ImportError: cannot import name …` right after files changed | Delete the `__pycache__` folder in the project and rerun (stale Python cache, common with OneDrive) |
| `ModuleNotFoundError` | Rerun `py -m pip install -r requirements.txt` |
| "Column mapping problems" warning after a form change | The survey's column names changed — see DEVELOPER_NOTES.md → "When the Google Form changes" |
| Live app shows an error after editing files on GitHub | Open share.streamlit.io → the app → "Manage app" → view logs / click Reboot |
| App seems ancient/stale in the browser | Hard-refresh (Ctrl+F5); on Cloud, Reboot from Manage app |

## 8. Finding a future maintainer

Any student comfortable with Python can maintain this. Point them at
DEVELOPER_NOTES.md and ARCHITECTURE.md first. The app is two files on purpose:
UI in `app.py`, everything else in `pipeline.py`, both heavily commented, with
tests in `tests/` (`py -m pytest`).
