"""Minimal Twelve Data client for US daily bars and daily FX.

The key is read from TWELVEDATA_API_KEY and sent in the Authorization
header, so it never appears in a URL, log or file. Free keys allow about
8 calls a minute and 800 a day, and return up to 5000 days per call, so one
run fetches full history.
"""
from __future__ import annotations

import os
import time
import urllib.parse

import pandas as pd

from .http_json import ApiError, get_json

BASE_URL = os.environ.get("TWELVEDATA_BASE_URL", "https://api.twelvedata.com")
MIN_SECONDS_BETWEEN_CALLS = 8  # stays under 8 calls a minute
_last_call = 0.0


class TwelveDataError(ApiError):
    pass


class RateLimited(TwelveDataError):
    pass


def has_key() -> bool:
    return bool(os.environ.get("TWELVEDATA_API_KEY"))


def _time_series(symbol: str, outputsize: int, retry: bool = True) -> pd.DataFrame:
    global _last_call
    key = os.environ.get("TWELVEDATA_API_KEY")
    if not key:
        raise TwelveDataError("TWELVEDATA_API_KEY is not set")
    wait = MIN_SECONDS_BETWEEN_CALLS - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    query = {"symbol": symbol, "interval": "1day", "outputsize": outputsize, "order": "asc"}
    url = BASE_URL.rstrip("/") + "/time_series?" + urllib.parse.urlencode(query)
    body = get_json(url, f"Twelve Data {symbol}", {"Authorization": f"apikey {key}"},
                    error=TwelveDataError)
    _last_call = time.time()
    if body.get("status") == "error":
        msg = str(body.get("message", "")).replace(key, "***")[:200]
        if body.get("code") == 429 or "credits" in msg.lower():
            if retry and "minute" in msg.lower():
                time.sleep(61)  # per-minute limit: wait it out once
                return _time_series(symbol, outputsize, retry=False)
            raise RateLimited(msg)
        raise TwelveDataError(f"HTTP {body.get('code')}: {msg}")
    values = body.get("values") or []
    if not values:
        raise TwelveDataError("no price data in the response")
    df = pd.DataFrame(values).rename(columns={"datetime": "date"})
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def daily(symbol: str, full: bool = True) -> pd.DataFrame:
    """US stock/ETF daily bars: date, open, high, low, close, volume."""
    return _time_series(symbol, 5000 if full else 30)


def fx_daily(from_ccy: str = "USD", to_ccy: str = "INR", full: bool = True) -> pd.DataFrame:
    """Daily FX bars: date, open, high, low, close (rupees per dollar for USD/INR)."""
    return _time_series(f"{from_ccy}/{to_ccy}", 5000 if full else 30)
