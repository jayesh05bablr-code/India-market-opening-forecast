"""Company fundamentals behind one interface.

Today the only source is a CSV (data/fundamentals.csv, same columns as
data/fundamentals_template.csv), which you can fill from a Bloomberg export,
Tracxn, Screener.in or annual reports. To add a live source later, write
another function that returns the same columns and use it in load().
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .paths import FUNDAMENTALS_FILE

NUMERIC = ["market_cap_cr", "pe", "pb", "eps_ttm", "roe_pct", "debt_to_equity",
           "promoter_pct", "fii_pct", "dii_pct"]


def from_csv(path: Path = FUNDAMENTALS_FILE) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "as_of", *NUMERIC])
    df = pd.read_csv(path, parse_dates=["as_of"])
    df["symbol"] = df["symbol"].str.upper().str.strip()
    for col in NUMERIC:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    # Rows may hold only some columns (e.g. a shareholding upload), so carry each
    # column's latest known value forward within a company.
    df = df.sort_values(["symbol", "as_of"])
    df[NUMERIC] = df.groupby("symbol")[NUMERIC].ffill()
    return df


def load() -> pd.DataFrame:
    return from_csv()


def attach(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Join each price row to the latest fundamentals known on that date.

    merge_asof looks backward only, so a quarter's numbers never leak into
    days before they were published (set as_of to the results date).
    """
    if fundamentals.empty:
        for col in NUMERIC:
            prices[col] = np.nan
        return prices
    out = pd.merge_asof(
        prices.sort_values("date"),
        fundamentals.sort_values("as_of")[["symbol", "as_of", *NUMERIC]],
        left_on="date", right_on="as_of", by="symbol", direction="backward",
    ).drop(columns="as_of")
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)
