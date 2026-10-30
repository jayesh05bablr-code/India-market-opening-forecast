"""Load API keys from a local .env file in the project folder.

Each line looks like NAME=value. Values in .env win over old environment
variables, so a stale or placeholder key set earlier with setx can't hide them.
The .env file stays on your computer: it is git-ignored and never printed.
"""
from __future__ import annotations

import os

from .paths import ROOT

ENV_FILE = ROOT / ".env"


def load(path=ENV_FILE) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip().strip('"').strip("'")
        if name and value:
            os.environ[name] = value
