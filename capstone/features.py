"""Turn raw OHLCV + fundamentals into one model-ready table.

Every price feature for day t uses only data known at day t's close; the
overnight features (overnight.py) add what the US market and the rupee did
before the next open. The target is the overnight gap: next day's open
divided by today's close, minus one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import fundamentals, overnight
from .paths import PROCESSED, RAW, load_symbols

# Overnight moves beyond this are almost always splits, bonuses or data errors.
MAX_ABS_GAP = 0.20

PRICE_FEATURES = [
    "ret_1", "ret_5", "ret_20", "intraday_ret", "gap_today", "range_pct",
    "close_pos", "vol_ratio_20", "volatility_20", "dist_sma_20", "dow",
    "mkt_ret_1", "mkt_gap_today",
]
FUNDAMENTAL_FEATURES = ["log_mcap", "pe", "pb", "eps_ttm", "roe_pct", "debt_to_equity",
                        "promoter_pct", "fii_pct", "dii_pct"]
OVERNIGHT_FEATURES = overnight.OVERNIGHT_FEATURES
FEATURES = PRICE_FEATURES + OVERNIGHT_FEATURES + FUNDAMENTAL_FEATURES


def load_prices(symbols: list[str] | None = None) -> pd.DataFrame:
    frames = []
    for symbol in symbols or load_symbols():
        path = RAW / f"{symbol}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"])
            df["symbol"] = symbol
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No price files in {RAW}. Run python -m capstone.fetch_data first.")
    return pd.concat(frames, ignore_index=True)


def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    prev_close = g["close"].shift(1)
    g["ret_1"] = g["close"] / prev_close - 1
    g["ret_5"] = g["close"] / g["close"].shift(5) - 1
    g["ret_20"] = g["close"] / g["close"].shift(20) - 1
    g["intraday_ret"] = g["close"] / g["open"] - 1
    g["gap_today"] = g["open"] / prev_close - 1
    g["range_pct"] = (g["high"] - g["low"]) / g["close"]
    span = (g["high"] - g["low"]).replace(0, np.nan)
    g["close_pos"] = (g["close"] - g["low"]) / span
    g["vol_ratio_20"] = g["volume"] / g["volume"].rolling(20).mean()
    g["volatility_20"] = g["ret_1"].rolling(20).std()
    g["dist_sma_20"] = g["close"] / g["close"].rolling(20).mean() - 1
    g["next_date"] = g["date"].shift(-1)
    g["next_open"] = g["open"].shift(-1)
    g["target_gap"] = g["next_open"] / g["close"] - 1
    return g


def build(symbols: list[str] | None = None, save: bool = True) -> pd.DataFrame:
    prices = load_prices(symbols)
    df = pd.concat([_per_symbol(g) for _, g in prices.groupby("symbol")], ignore_index=True)
    df["dow"] = df["date"].dt.dayofweek
    # Market-wide mood: the average move across all symbols that day.
    daily = df.groupby("date")[["ret_1", "gap_today"]].mean()
    daily.columns = ["mkt_ret_1", "mkt_gap_today"]
    df = df.merge(daily, left_on="date", right_index=True, how="left")
    df = overnight.attach(df)
    df = fundamentals.attach(df, fundamentals.load())
    df["log_mcap"] = np.log(df["market_cap_cr"])
    # Drop corporate-action days so splits do not look like crashes.
    bad = (df["target_gap"].abs() > MAX_ABS_GAP) | (df["gap_today"].abs() > MAX_ABS_GAP)
    df = df[~bad].replace([np.inf, -np.inf], np.nan)
    if save:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED / "features.csv", index=False)
    return df
