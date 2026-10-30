"""Minimal Alpha Vantage client for US daily bars and daily FX.

The key is read from the ALPHAVANTAGE_API_KEY environment variable and is
never printed or saved. Free keys allow about 25 calls a day and 5 a minute,
and return only the latest 100 days ("compact"), so fetch_overnight caches
every series and tops it up.
"""
from __future__ import annotations

import os
import time
import urllib.parse

import pandas as pd

from .http_json import ApiError, get_json

BASE_URL = os.environ.get("ALPHAVANTAGE_BASE_URL", "https://www.alphavantage.co/query")
MIN_SECONDS_BETWEEN_CALLS = 13  # stays under 5 calls a minute
_last_call = 0.0


class AlphaVantageError(ApiError):
    pass


class RateLimited(AlphaVantageError):
    pass


def _query(params: dict[str, str]) -> dict:
    global _last_call
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        raise AlphaVantageError("ALPHAVANTAGE_API_KEY is not set")
    wait = MIN_SECONDS_BETWEEN_CALLS - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    url = BASE_URL + "?" + urllib.parse.urlencode({**params, "apikey": key})
    body = get_json(url, f"Alpha Vantage {params['function']}", error=AlphaVantageError)
    _last_call = time.time()
    # Limits and plan messages come back as HTTP 200 with one of these keys.
    for k in ("Note", "Information", "Error Message"):
        if k in body:
            # Alpha Vantage repeats the key inside some messages; never print it.
            msg = str(body[k]).replace(key, "***")
            if "rate limit" in msg.lower() or "per day" in msg.lower() or "frequency" in msg.lower():
                raise RateLimited(msg[:200])
            raise AlphaVantageError(msg[:200])
    return body


def _to_frame(series: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(series, orient="index")
    df.columns = [c.split(". ", 1)[-1] for c in df.columns]  # "4. close" -> "close"
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = pd.to_datetime(df.index)
    return df.sort_index().rename_axis("date").reset_index()


def _series(body: dict, key: str) -> pd.DataFrame:
    if not body.get(key):
        # Name only the response's keys so the cause shows without echoing anything sensitive.
        raise AlphaVantageError(f"no price data in the response (keys: {', '.join(body) or 'none'})")
    return _to_frame(body[key])


def daily(symbol: str, full: bool = False) -> pd.DataFrame:
    """US stock/ETF daily bars: date, open, high, low, close, volume (not split-adjusted)."""
    body = _query({"function": "TIME_SERIES_DAILY", "symbol": symbol,
                   "outputsize": "full" if full else "compact"})
    return _series(body, "Time Series (Daily)")


def fx_daily(from_ccy: str = "USD", to_ccy: str = "INR", full: bool = True) -> pd.DataFrame:
    """Daily FX bars: date, open, high, low, close (rupees per dollar for USD/INR)."""
    body = _query({"function": "FX_DAILY", "from_symbol": from_ccy, "to_symbol": to_ccy,
                   "outputsize": "full" if full else "compact"})
    return _series(body, "Time Series FX (Daily)")
