"""Download or top up daily NSE OHLCV for every symbol in data/symbols.txt.

    python -m capstone.fetch_data --start 2015-01-01

Re-running only fetches dates after the last row already saved.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import pandas as pd

from . import tej_client
from .paths import RAW, load_symbols

COLUMNS = ["date", "open", "high", "low", "close", "prev_close", "volume", "turnover", "trades"]


def update_symbol(symbol: str, start: str) -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{symbol}.csv"
    old = pd.read_csv(path, parse_dates=["date"]) if path.exists() else None
    if old is not None and len(old):
        start = (old["date"].max().date() + timedelta(days=1)).isoformat()
    if start > date.today().isoformat():
        return 0
    rows = tej_client.ohlcv(symbol, start=start, end=date.today().isoformat())
    if not rows:
        return 0
    new = pd.DataFrame(rows)
    new = new[[c for c in COLUMNS if c in new.columns]]
    new["date"] = pd.to_datetime(new["date"])
    df = new if old is None else pd.concat([old, new])
    df = df.drop_duplicates("date", keep="last").sort_values("date")
    df.to_csv(path, index=False)
    return len(new)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2015-01-01", help="first date for a new symbol")
    args = parser.parse_args()
    for symbol in load_symbols():
        try:
            n = update_symbol(symbol, args.start)
            print(f"{symbol}: +{n} rows")
        except tej_client.TejError as exc:
            print(f"{symbol}: skipped ({exc})")


if __name__ == "__main__":
    main()
