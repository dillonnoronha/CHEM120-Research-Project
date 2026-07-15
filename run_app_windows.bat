@echo off
title General Chemistry II Catalyst Insight Studio
echo.
echo Starting General Chemistry II Catalyst Insight Studio...
echo.
py -m pip install -r requirements.txt
echo.
py -m streamlit run app.py
pause
