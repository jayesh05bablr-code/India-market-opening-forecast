"""Check which API keys are set and whether each one works.

    python -m capstone.check_keys

Prints only whether a key is set and its length, never the key itself.
"""
from __future__ import annotations

import os

from . import exchangerate_client as exr
from . import twelvedata_client as td
from .env import ENV_FILE
from .http_json import ApiError

KEYS = ["TWELVEDATA_API_KEY", "EXCHANGERATE_API_KEY", "TEJ_API_KEY", "ALPHAVANTAGE_API_KEY"]


def main() -> None:
    print(f".env file: {'found' if ENV_FILE.exists() else 'not found'} at {ENV_FILE}")
    for name in KEYS:
        value = os.environ.get(name, "")
        note = ""
        if value.lower().startswith(("your", "paste")):
            note = "  <- this is placeholder text, paste the real key"
        print(f"{name}: {'set, ' + str(len(value)) + ' characters' if value else 'not set'}{note}")
    print()
    if td.has_key():
        try:
            df = td._time_series("SPY", 2)
            print(f"Twelve Data works: SPY closed at {df['close'].iloc[-1]:.2f} on {df['date'].iloc[-1]:%d %b %Y}")
        except ApiError as exc:
            print(f"Twelve Data failed: {exc}")
    try:
        live = exr.latest()
        print(f"exchangerate.dev works: USD/INR {live['rate']:.4f}")
    except ApiError as exc:
        print(f"exchangerate.dev failed: {exc}")


if __name__ == "__main__":
    main()
