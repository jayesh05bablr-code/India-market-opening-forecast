#!/usr/bin/env bash
# One-command setup and run (Mac/Linux). Keys go in the .env file in this folder.
set -e
cd "$(dirname "$0")"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "A .env file was created. Paste your keys after the = signs in .env, save, then run this again."
  exit 1
fi
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q -r requirements.txt
python -m capstone.check_keys
python -m capstone.fetch_data --start 2015-01-01
python -m capstone.fetch_overnight
python -m capstone.train
python -m capstone.predict
streamlit run app.py
