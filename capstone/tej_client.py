"""Minimal TejHQ client for NSE end-of-day OHLCV.

Uses only the standard library. The API key is optional for OHLCV and is read
from the TEJ_API_KEY environment variable; it is never logged or written.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = os.environ.get("TEJ_BASE_URL", "https://api.tejhq.dev")
RETRYABLE = {429, 500, 502, 503, 504}


class TejError(RuntimeError):
    pass


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/json", "User-Agent": "nse-capstone/0.1"}
    key = os.environ.get("TEJ_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _get(path: str, query: dict[str, str] | None = None, retries: int = 4) -> dict:
    url = BASE_URL.rstrip("/") + path
    if query:
        url += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v})
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers=_headers())
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in RETRYABLE and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise TejError(f"GET {path} failed with HTTP {exc.code}") from None
        except urllib.error.URLError as exc:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise TejError(f"GET {path} failed: {exc.reason}") from None
    raise TejError(f"GET {path} failed after {retries} retries")


def ohlcv(symbol: str, start: str | None = None, end: str | None = None,
          exchange: str = "nse") -> list[dict]:
    """Daily rows: date, open, high, low, close, last, prev_close, volume, turnover, trades."""
    body = _get(f"/v1/ohlcv/{exchange.lower()}/{symbol.upper()}", {"from": start, "to": end})
    return body.get("data") or []
