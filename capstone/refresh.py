"""Retrain and re-predict from the dashboard, so uploads need no terminal commands.

Runs `python -m capstone.train` then `python -m capstone.predict` in a fresh
process (same Python as the dashboard). Their output is echoed to the terminal
running Streamlit and also returned, so the dashboard can show it.
"""
from __future__ import annotations

import os
import subprocess
import sys

from .paths import DATA, FUNDAMENTALS_FILE, MODELS, PROCESSED, ROOT

MODEL_FILE = MODELS / "gap_model.joblib"
PREDICTIONS_FILE = PROCESSED / "predictions.csv"
# Inputs the model reads besides prices: when one changes, the model is out of date.
MODEL_INPUTS = [FUNDAMENTALS_FILE, DATA / "stock_map.csv", DATA / "sector_links.csv"]


def stale() -> bool:
    """True when an input the model uses was saved after the model was trained."""
    if not MODEL_FILE.exists():
        return False  # never trained here: the first run is done with run_all
    built = MODEL_FILE.stat().st_mtime
    if not PREDICTIONS_FILE.exists() or PREDICTIONS_FILE.stat().st_mtime < built:
        return True  # model retrained but predict did not finish
    return any(p.exists() and p.stat().st_mtime > built for p in MODEL_INPUTS)


def _echo(text: str) -> None:
    try:
        print(text, end="", flush=True)
    except UnicodeEncodeError:  # a Windows console without UTF-8 cannot print ₹
        print(text.encode("ascii", "replace").decode(), end="", flush=True)


def _run(module: str) -> tuple[bool, str]:
    _echo(f"\n[auto-update] python -m {module}\n")
    # On Windows a piped child writes in the ANSI code page (cp1252), which has no ₹ and crashes
    # predict's printout; force UTF-8 for the child.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    proc = subprocess.Popen([sys.executable, "-m", module], cwd=ROOT, stdout=subprocess.PIPE, env=env,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    lines = []
    for line in proc.stdout:
        _echo(line)
        lines.append(line)
    return proc.wait() == 0, "".join(lines)


def retrain_and_predict() -> tuple[bool, str]:
    """Train then predict. Returns (ok, combined output)."""
    ok, log = _run("capstone.train")
    if ok:
        ok, log2 = _run("capstone.predict")
        log += log2
    _echo(f"[auto-update] {'done' if ok else 'FAILED'}\n")
    return ok, log
