"""What happened overnight between India's close and the next open.

NSE closes at 15:30 IST. The US session dated the same day opens at 19:00 or
20:00 IST and closes at 01:30 or 02:30 IST, before NSE opens at 09:15. So for
an Indian close on day d and next open on day n, the new information is every
US session dated d up to n-1. All moves below are measured from the last close
before d to the last close before n, which uses only data available before
the next open.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .paths import FX_DIR, SECTOR_LINKS_FILE, STOCK_MAP_FILE, US_DIR

MARKET_ETF = "SPY"   # S&P 500
TECH_ETF = "QQQ"     # Nasdaq 100
SECTOR_ETFS = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLB": "Materials",
    "XLI": "Industrials", "XLC": "Communication Services", "XLU": "Utilities",
    "XLRE": "Real Estate",
}
FX_PAIR = "USDINR"

OVERNIGHT_FEATURES = [
    "us_mkt_ret", "us_tech_ret", "us_sector_ret", "us_sector_rel",
    "adr_ret", "usdinr_chg", "fx_effect", "us_sessions",
]


def stock_map() -> pd.DataFrame:
    stocks = pd.read_csv(STOCK_MAP_FILE, dtype=str).fillna("")
    links = pd.read_csv(SECTOR_LINKS_FILE)
    stocks["symbol"] = stocks["symbol"].str.upper().str.strip()
    return stocks.merge(links, on="india_sector", how="left")


def us_tickers(symbols: list[str]) -> list[str]:
    adrs = stock_map().query("symbol in @symbols")["adr"]
    return [MARKET_ETF, TECH_ETF, *SECTOR_ETFS, *sorted(a for a in adrs.unique() if a)]


def load_close(name: str, folder=US_DIR) -> pd.Series:
    path = folder / f"{name}.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(path, parse_dates=["date"])
    if df.empty or not {"date", "close"} <= set(df.columns):
        print(f"Ignoring {path.name}: it has no prices. Delete it and run fetch_overnight again.")
        return pd.Series(dtype=float)
    df = df.dropna(subset=["close"])
    return df.drop_duplicates("date", keep="last").set_index("date")["close"].sort_index()


def load_fx() -> pd.Series:
    return load_close(FX_PAIR, FX_DIR)


def window_return(close: pd.Series, start: pd.Series, end: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Move from the last close before `start` to the last close before `end`.

    Returns (return, number of sessions in the window). A window with no
    session (a US holiday) is a 0% move. NaN where the series does not cover
    the window, so missing data is never mistaken for a quiet night.
    """
    n = len(start)
    if close.empty:
        return np.full(n, np.nan), np.full(n, np.nan)
    dates = close.index.values
    vals = close.to_numpy(dtype=float)
    i0 = np.searchsorted(dates, pd.to_datetime(start).values, side="left") - 1
    i1 = np.searchsorted(dates, pd.to_datetime(end).values, side="left") - 1
    ok = (i0 >= 0) & (i1 >= i0) & (pd.to_datetime(start).values <= dates[-1])
    ret = np.where(ok, vals[np.clip(i1, 0, None)] / vals[np.clip(i0, 0, None)] - 1, np.nan)
    return ret, np.where(ok, i1 - i0, np.nan)


def attach(df: pd.DataFrame) -> pd.DataFrame:
    """Add OVERNIGHT_FEATURES to rows with date (India close) and next_date.

    Rows without a next_date (the latest close) assume the next weekday.
    """
    df = df.copy()
    start = df["date"]
    end = df["next_date"].fillna(df["date"] + pd.offsets.BDay(1))
    df["us_mkt_ret"], df["us_sessions"] = window_return(load_close(MARKET_ETF), start, end)
    df["us_tech_ret"], _ = window_return(load_close(TECH_ETF), start, end)
    df["usdinr_chg"], _ = window_return(load_fx(), start, end)

    smap = stock_map().set_index("symbol")
    df = df.drop(columns=[c for c in ("india_sector",) if c in df.columns])
    df["india_sector"] = df["symbol"].map(smap["india_sector"])
    etf = df["symbol"].map(smap["us_etf"])
    adr = df["symbol"].map(smap["adr"]).replace("", np.nan)
    exposure = df["symbol"].map(smap["fx_exposure"]).astype(float)

    df["us_sector_ret"] = np.nan
    df["adr_ret"] = np.nan
    for ticker in etf.dropna().unique():
        rows = etf == ticker
        df.loc[rows, "us_sector_ret"], _ = window_return(load_close(ticker), start[rows], end[rows])
    for ticker in adr.dropna().unique():
        rows = adr == ticker
        df.loc[rows, "adr_ret"], _ = window_return(load_close(ticker), start[rows], end[rows])
    df["us_sector_rel"] = df["us_sector_ret"] - df["us_mkt_ret"]
    # Positive usdinr_chg means more rupees per dollar, i.e. the rupee weakened.
    df["fx_effect"] = exposure.fillna(0) * df["usdinr_chg"]
    return df


def sector_moves(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """US market, Nasdaq and sector ETF moves over one overnight window."""
    rows = []
    names = {MARKET_ETF: "S&P 500", TECH_ETF: "Nasdaq 100", **SECTOR_ETFS}
    for ticker, name in names.items():
        ret, n = window_return(load_close(ticker), pd.Series([start]), pd.Series([end]))
        rows.append({"ticker": ticker, "name": name, "ret": ret[0], "sessions": n[0],
                     "is_sector": ticker in SECTOR_ETFS})
    return pd.DataFrame(rows)


def latest_session(ticker: str = MARKET_ETF) -> pd.Timestamp | None:
    close = load_close(ticker)
    return None if close.empty else close.index.max()
