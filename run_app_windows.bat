@echo off
title CHEM 120 Catalyst Insight Studio
echo.
echo Starting CHEM 120 Catalyst Insight Studio...
echo.
py -m pip install -r requirements.txt
echo.
py -m streamlit run app.py
pause
