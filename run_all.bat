@echo off
REM One-command setup and run (Windows). Keys go in the .env file in this folder.
cd /d "%~dp0"
if not exist .env (
  copy .env.example .env >nul
  echo A .env file was created. Paste your keys after the = signs, save, close Notepad, then run this again.
  notepad .env
  exit /b 1
)
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
pip install -q -r requirements.txt || exit /b 1
python -m capstone.check_keys
python -m capstone.fetch_data --start 2015-01-01 || exit /b 1
python -m capstone.fetch_overnight || exit /b 1
python -m capstone.train || exit /b 1
python -m capstone.predict || exit /b 1
streamlit run app.py
