"""1-minute NSE candles from the internet, for the Intraday scalping tab.

Source order:
1. Yahoo Finance chart data (no key; about the last 7 days of 1-minute bars).
2. Twelve Data, if TWELVEDATA_API_KEY is set and your plan covers NSE.

Candles are kept in data/intraday/<SYMBOL>.csv and new fetches are merged in,
so the history for backtests grows each day you fetch. Times are IST.
"""
from __future__ import annotations

import os
import urllib.parse

import pandas as pd

from .http_json import ApiError, get_json
from .paths import DATA

INTRADAY_DIR = DATA / "intraday"
COLS = ["datetime", "open", "high", "low", "close", "volume"]
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{}?interval=1m&range={}"
# Yahoo turns away the default Python user agent.
BROWSER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def parse_yahoo(body: dict) -> pd.DataFrame:
    result = (body.get("chart") or {}).get("result") or []
    if not result:
        err = (body.get("chart") or {}).get("error") or {}
        raise ApiError(f"Yahoo: {err.get('description') or 'no data in the response'}")
    r = result[0]
    stamps = r.get("timestamp") or []
    q = (r.get("indicators", {}).get("quote") or [{}])[0]
    df = pd.DataFrame({"datetime": pd.to_datetime(stamps, unit="s", utc=True),
                       **{c: q.get(c, [None] * len(stamps)) for c in COLS[1:]}})
    df["datetime"] = df["datetime"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return df.dropna(subset=["open", "high", "low", "close"])


def fetch_yahoo(symbol: str, days: int = 7) -> pd.DataFrame:
    ticker = urllib.parse.quote(f"{symbol}.NS")
    body = get_json(YAHOO_URL.format(ticker, f"{days}d"), f"Yahoo {symbol}", BROWSER, retries=2)
    return parse_yahoo(body)


def fetch_twelvedata(symbol: str) -> pd.DataFrame:
    key = os.environ.get("TWELVEDATA_API_KEY")
    if not key:
        raise ApiError("TWELVEDATA_API_KEY is not set")
    query = {"symbol": symbol, "exchange": "NSE", "interval": "1min", "outputsize": 2000,
             "timezone": "Asia/Kolkata", "order": "asc"}
    url = "https://api.twelvedata.com/time_series?" + urllib.parse.urlencode(query)
    body = get_json(url, f"Twelve Data {symbol} 1min", {"Authorization": f"apikey {key}"}, retries=1)
    if body.get("status") == "error":
        raise ApiError(f"Twelve Data: {str(body.get('message', ''))[:160].replace(key, '***')}")
    df = pd.DataFrame(body.get("values") or [])
    if df.empty:
        raise ApiError("Twelve Data: no data")
    df["datetime"] = pd.to_datetime(df["datetime"])
    for c in COLS[1:]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    return df[COLS]


def load(symbol: str) -> pd.DataFrame:
    path = INTRADAY_DIR / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame(columns=COLS)
    return pd.read_csv(path, parse_dates=["datetime"])


def save(symbol: str, new: pd.DataFrame) -> pd.DataFrame:
    new = new.reindex(columns=COLS)
    old = load(symbol)
    both = pd.concat([old, new], ignore_index=True) if len(old) else new.copy()
    both["datetime"] = pd.to_datetime(both["datetime"])
    both = both.drop_duplicates("datetime", keep="last").sort_values("datetime")
    # Keep market hours only (Yahoo can add a stray pre-open or post-close bar).
    t = both["datetime"].dt.hour * 60 + both["datetime"].dt.minute
    both = both[(t >= 9 * 60 + 15) & (t < 15 * 60 + 30)]
    INTRADAY_DIR.mkdir(parents=True, exist_ok=True)
    both.to_csv(INTRADAY_DIR / f"{symbol}.csv", index=False)
    return both


def update(symbol: str) -> tuple[str, int]:
    """Fetch and store 1-minute candles. Returns (source used or error text, rows now stored)."""
    errors = []
    for name, fetch in (("Yahoo Finance", fetch_yahoo), ("Twelve Data", fetch_twelvedata)):
        try:
            df = fetch(symbol)
            if len(df):
                return name, len(save(symbol, df))
        except ApiError as exc:
            errors.append(str(exc))
    return "failed: " + " | ".join(errors), len(load(symbol))


PREOPEN_FILE = INTRADAY_DIR / "preopen.csv"


def save_preopen(df: pd.DataFrame) -> None:
    """Keep NSE pre-open opening prices, one row per stock per day."""
    old = pd.read_csv(PREOPEN_FILE) if PREOPEN_FILE.exists() else pd.DataFrame()
    both = pd.concat([old, df], ignore_index=True).drop_duplicates(["date", "symbol"], keep="last")
    INTRADAY_DIR.mkdir(parents=True, exist_ok=True)
    both.to_csv(PREOPEN_FILE, index=False)


def preopen(day) -> pd.DataFrame:
    if not PREOPEN_FILE.exists():
        return pd.DataFrame(columns=["symbol", "prev_close", "open", "gap_pct", "date"])
    df = pd.read_csv(PREOPEN_FILE)
    return df[df["date"] == pd.Timestamp(day).strftime("%Y-%m-%d")]


def opening_range(minutes: pd.DataFrame, day, or_minutes: int) -> dict | None:
    """Open, high and low of the first `or_minutes` after 09:15 on `day`, once all of them have printed."""
    start = pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=15)
    end = start + pd.Timedelta(minutes=or_minutes)
    g = minutes[(minutes["datetime"] >= start) & (minutes["datetime"] < end)]
    if len(g) == 0 or g["datetime"].max() < end - pd.Timedelta(minutes=1):
        return None
    return {"open": float(g["open"].iloc[0]), "high": float(g["high"].max()), "low": float(g["low"].min())}


if __name__ == "__main__":
    # python -m capstone.intraday            -> fetch every stock in data/stock_map.csv's prediction list
    # python -m capstone.intraday INFY TCS   -> fetch just these
    import sys

    from .paths import PROCESSED
    syms = [a.upper() for a in sys.argv[1:]]
    if not syms:
        preds = PROCESSED / "predictions.csv"
        syms = list(pd.read_csv(preds)["symbol"]) if preds.exists() else []
    for sym in syms:
        src, n = update(sym)
        m = load(sym)
        last = m["datetime"].max() if len(m) else "-"
        print(f"{sym}: {src}, {n} minutes stored, latest {last} -> {INTRADAY_DIR / (sym + '.csv')}")
