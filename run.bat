@echo off
cd /d "%~dp0"
if not exist .venv (
  echo [1/3] Creating virtual environment...
  python -m venv .venv
)
call .venv\Scripts\activate.bat
echo [2/3] Installing packages...
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt
echo [3/3] Starting app - your browser will open. Press Ctrl+C here to stop.
python -m streamlit run app.py
pause
