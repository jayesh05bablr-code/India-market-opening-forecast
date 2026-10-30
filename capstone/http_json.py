"""Shared GET-JSON helper with retries for the API clients.

Error messages name only the endpoint label, never the URL, because some
APIs (Alpha Vantage) take the key as a query parameter.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

RETRYABLE = {429, 500, 502, 503, 504}


class ApiError(RuntimeError):
    network = False  # True when the host could not be reached at all


def get_json(url: str, label: str, headers: dict[str, str] | None = None,
             retries: int = 3, error: type[ApiError] = ApiError) -> dict:
    headers = {"Accept": "application/json", "User-Agent": "nse-capstone/0.2", **(headers or {})}
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in RETRYABLE and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            detail = ""
            try:
                detail = json.loads(exc.read().decode("utf-8")).get("error", "")
            except Exception:
                pass
            hint = " (the API key is missing or wrong)" if exc.code in (401, 403) else ""
            raise error(f"{label} failed with HTTP {exc.code}{hint} {detail}".strip()) from None
        except urllib.error.URLError as exc:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            err = error(f"{label} failed: {exc.reason}. Is the domain blocked by your network?")
            err.network = True
            raise err from None
    raise error(f"{label} failed after {retries} retries")
