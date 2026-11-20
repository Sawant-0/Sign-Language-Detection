@echo off
cd /d "%~dp0"
title Sign Language Detection - SignSense
echo ========================================================
echo   Starting SignSense Sign Language Detection Studio...
echo ========================================================
echo.
if exist ".venv\Scripts\streamlit.exe" (
    ".venv\Scripts\streamlit.exe" run app.py
) else (
    echo Python virtual environment not found in .venv.
    echo Trying system streamlit...
    streamlit run app.py
)
pause
