"""Minimal exchangerate.dev client for USD/INR.

The key is read from EXCHANGERATE_API_KEY and sent as a Bearer header; it is
never printed or saved. Free plans cover about the last 30 days of history,
so long history comes from Alpha Vantage FX_DAILY (see fetch_overnight).
"""
from __future__ import annotations

import os
import urllib.parse
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from .http_json import ApiError, get_json

BASE_URL = os.environ.get("EXCHANGERATE_BASE_URL", "https://api.exchangerate.dev")


class ExchangeRateError(ApiError):
    pass


def _get(path: str, query: dict[str, str]) -> dict:
    headers = {}
    key = os.environ.get("EXCHANGERATE_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    url = BASE_URL.rstrip("/") + path + "?" + urllib.parse.urlencode(query)
    return get_json(url, f"exchangerate.dev {path.split('/')[2]}", headers, error=ExchangeRateError)


def latest(base: str = "USD", symbol: str = "INR") -> dict:
    """{'rate': 84.1, 'as_of': '2026-10-23T02:45:00Z', 'session': 'open'}"""
    body = _get(f"/v1/latest/{base}", {"symbols": symbol})
    as_of = (body.get("effective_at") or {}).get(symbol) or body.get("timestamp")
    return {"rate": body["rates"][symbol], "as_of": as_of, "session": body.get("market_session")}


def daily_history(days: int = 30, base: str = "USD", symbol: str = "INR") -> pd.DataFrame:
    """Daily closes for the last `days` days: date, close. Empty if the plan refuses."""
    end = datetime.now(timezone.utc).replace(microsecond=0)
    start = end - timedelta(days=days)
    rows, cursor = [], None
    while True:
        query = {"base": base, "symbols": symbol, "interval": "1d",
                 "from": start.isoformat().replace("+00:00", "Z"),
                 "to": end.isoformat().replace("+00:00", "Z")}
        if cursor:
            query["cursor"] = cursor
        body = _get("/v1/timeseries", query)
        for row in body.get("data") or []:
            rate = (row.get("rates") or {}).get(symbol)
            if rate is not None:
                rows.append({"date": row["timestamp"][:10], "close": rate})
        cursor = body.get("next_cursor")
        if not body.get("has_more") or not cursor:
            break
    df = pd.DataFrame(rows, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def on_date(day: date, base: str = "USD", symbol: str = "INR") -> float | None:
    body = _get(f"/v1/{day.isoformat()}/{base}", {"symbols": symbol})
    if body.get("is_forward_filled"):
        return None
    return (body.get("rates") or {}).get(symbol)
