"""Download or top up overnight inputs: US ETFs and ADRs, and USD/INR
history (Twelve Data, or Alpha Vantage if no Twelve Data key), plus
exchangerate.dev for the latest USD/INR days.

    python -m capstone.fetch_overnight

Keys come from TWELVEDATA_API_KEY (or ALPHAVANTAGE_API_KEY) and
EXCHANGERATE_API_KEY. Run it in the morning after the US close (after about
02:30 IST) and before predict.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import alphavantage_client as av
from . import exchangerate_client as exr
from . import twelvedata_client as td
from .http_json import ApiError


def _provider():
    """Twelve Data when its key is set (full history on a free key), else Alpha Vantage."""
    return (td, "twelvedata", "Twelve Data") if td.has_key() else (av, "alphavantage", "Alpha Vantage")
from .overnight import FX_PAIR, us_tickers
from .paths import FX_DIR, SYNTHETIC_FLAG, US_DIR, load_symbols


def _merge_save(path, new: pd.DataFrame, source: str) -> int:
    if new.empty or "close" not in new.columns:
        return 0  # never write a file without prices
    new = new.assign(source=source)
    old = pd.read_csv(path, parse_dates=["date"]) if path.exists() else None
    before = 0 if old is None else len(old)
    df = new if old is None else pd.concat([old, new])
    df = df.drop_duplicates("date", keep="last").sort_values("date")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return len(df) - before


def _last_us_close(now: datetime) -> datetime:
    """Latest weekday 21:30 UTC at or before now: safely after the 16:00 New York
    close in both summer and winter time, so the day's bar is final."""
    t = now.replace(hour=21, minute=30, second=0, microsecond=0)
    if t > now:
        t -= timedelta(days=1)
    while t.weekday() >= 5:
        t -= timedelta(days=1)
    return t


def _fresh(path) -> bool:
    """True if the file was saved after the latest US close (nothing new to fetch).

    A run during the US session can save a part-day bar, so it is refetched later."""
    if not path.exists():
        return False
    modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    return modified >= _last_us_close(datetime.now(timezone.utc))


def update_us(tickers: list[str]) -> None:
    api, source, name = _provider()
    for ticker in tickers:
        path = US_DIR / f"{ticker}.csv"
        if _fresh(path):
            print(f"{ticker}: already up to date")
            continue
        try:
            try:
                df = api.daily(ticker, full=not path.exists())
            except ApiError as exc:
                if api is td or exc.network or isinstance(exc, api.RateLimited) or "not set" in str(exc):
                    raise
                df = api.daily(ticker, full=False)  # Alpha Vantage "full" needs a premium key
            print(f"{ticker}: +{_merge_save(path, df, source)} rows")
        except api.RateLimited as exc:
            print(f"{name} limit reached ({exc}). Cached data kept; run again later.")
            return
        except ApiError as exc:
            print(f"{ticker}: skipped ({exc})")
            if exc.network or "not set" in str(exc):
                return


def update_fx() -> None:
    path = FX_DIR / f"{FX_PAIR}.csv"
    api, source, name = _provider()
    try:
        df = api.fx_daily("USD", "INR", full=True)
        print(f"{FX_PAIR} ({name}): +{_merge_save(path, df[['date', 'close']], source)} rows")
    except ApiError as exc:
        print(f"{FX_PAIR} history from {name} skipped ({exc})")
    try:
        recent = exr.daily_history(days=30)
        if path.exists() and len(recent):
            # Only add days after the long history, so two sources never alternate.
            last = pd.read_csv(path, parse_dates=["date"])["date"].max()
            recent = recent[recent["date"] > last]
        print(f"{FX_PAIR} (exchangerate.dev): +{_merge_save(path, recent, 'exchangerate.dev')} rows")
    except exr.ExchangeRateError as exc:
        print(f"{FX_PAIR} recent days from exchangerate.dev skipped ({exc})")
    try:
        live = exr.latest()
        live["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        (FX_DIR / "USDINR_live.json").write_text(json.dumps(live, indent=2))
        print(f"USD/INR now: {live['rate']:.4f} (as of {live['as_of']})")
    except exr.ExchangeRateError as exc:
        print(f"Live USD/INR skipped ({exc})")


def main() -> None:
    if SYNTHETIC_FLAG.exists():
        print(f"Warning: {SYNTHETIC_FLAG.name} is present. Delete data/raw, data/us, data/fx and that "
              "file before mixing real data with the synthetic sample.")
    update_us(us_tickers(load_symbols()))
    update_fx()


if __name__ == "__main__":
    main()
