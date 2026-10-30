"""Opening-range scalping planner for the first minutes after NSE opens (09:15).

Educational tool, not trading advice. The rule:
1. Watch the opening range: the high and low of the first `or_minutes`.
2. Long if price breaks above the range high (short if it breaks below the
   low), only in the direction of the bias (the model's expected gap, or the
   actual opening gap).
3. Stop at the other side of the range; target = R x the risk.
4. Close any open trade at 09:15 + `window` minutes.

Without minute data, the range is estimated from daily volatility scaled to
minutes (sqrt-time, times an opening boost because the first minutes are the
most volatile). Upload 1-minute candles to backtest the rule for real.
"""
from __future__ import annotations

import csv
import io
import math
import re

import numpy as np
import pandas as pd

SESSION_MINUTES = 375  # 09:15 to 15:30


def charges(buy_value: float, sell_value: float, brokerage_per_order: float = 20.0) -> float:
    """Approximate NSE intraday equity charges for one round trip (both legs), in ₹."""
    brokerage = min(brokerage_per_order, 0.0003 * buy_value) + min(brokerage_per_order, 0.0003 * sell_value)
    stt = 0.00025 * sell_value
    exchange = 0.0000297 * (buy_value + sell_value)
    sebi = 0.000001 * (buy_value + sell_value)
    stamp = 0.00003 * buy_value
    gst = 0.18 * (brokerage + exchange + sebi)
    return brokerage + stt + exchange + sebi + stamp + gst


def estimate_range(price: float, daily_vol: float, or_minutes: int, open_boost: float) -> float:
    """Expected high-low span of the first `or_minutes` (expected range of a random walk ≈ 1.6σ√t)."""
    sigma_min = daily_vol / math.sqrt(SESSION_MINUTES)
    return 1.6 * sigma_min * math.sqrt(or_minutes) * open_boost * price


def plan_row(symbol: str, open_: float, or_high: float, or_low: float, bias: float, capital: float,
             risk_pct: float, reward: float, leverage: float, buffer_pct: float,
             brokerage: float) -> dict:
    side = "Long" if bias > 0 else "Short" if bias < 0 else "Either"
    buf = open_ * buffer_pct
    long_entry, short_entry = or_high + buf, or_low - buf
    risk_long, risk_short = long_entry - or_low, or_high - short_entry
    rows = {}
    for s, entry, stop, risk in (("Long", long_entry, or_low, risk_long),
                                 ("Short", short_entry, or_high, risk_short)):
        if side not in (s, "Either") or risk <= 0:
            continue
        target = entry + reward * risk if s == "Long" else entry - reward * risk
        qty = int(min(risk_pct * capital / risk, capital * leverage / entry))
        value = qty * entry
        fees_t = charges(value, qty * target, brokerage) if s == "Long" else charges(qty * target, value, brokerage)
        fees_s = charges(value, qty * stop, brokerage) if s == "Long" else charges(qty * stop, value, brokerage)
        rows[s] = {"symbol": symbol, "side": s, "trigger": entry, "stop": stop, "target": target,
                   "qty": qty, "exposure": value, "risk_per_share": risk,
                   "net_profit_at_target": qty * reward * risk - fees_t,
                   "net_loss_at_stop": -qty * risk - fees_s,
                   "charges": fees_t,
                   "breakeven_move": fees_t / qty if qty else np.nan}
    return rows


_ALIASES = {
    "datetime": ["datetime", "date time", "timestamp", "time stamp", "candle time", "date and time",
                 "trade time", "ts"],
    "date": ["date", "trade date", "day"],
    "time": ["time", "minute", "time ist", "candle"],
    "open": ["open", "open price", "o", "opening price"],
    "high": ["high", "high price", "h"],
    "low": ["low", "low price", "l"],
    "close": ["close", "close price", "c", "ltp", "last", "last price", "closing price"],
    "volume": ["volume", "vol", "qty", "quantity", "traded qty"],
}
_LOOK = {a: k for k, al in _ALIASES.items() for a in al}


def _key(c) -> str:
    c = re.sub(r"\(.*?\)", " ", str(c).lower())
    return re.sub(r"[^a-z]+", " ", c).strip()


def _frames(name: str, data: bytes) -> list[pd.DataFrame]:
    if name.lower().endswith((".xlsx", ".xls")):
        return list(pd.read_excel(io.BytesIO(data), sheet_name=None, header=None).values())
    text = data.decode("utf-8-sig", errors="replace")
    sep = ";" if text.count(";") > text.count(",") else ("\t" if text.count("\t") > text.count(",") else ",")
    rows = list(csv.reader(io.StringIO(text), delimiter=sep))  # title lines may have fewer fields
    width = max((len(r) for r in rows), default=0)
    return [pd.DataFrame([r + [None] * (width - len(r)) for r in rows]).replace("", None)]


def read_minutes(name: str, data: bytes) -> pd.DataFrame:
    """1-minute candles. Headers are matched loosely (Date/Time, Timestamp, Open Price, LTP ...),
    may sit below a title block, and any sheet of an Excel file can hold them."""
    seen = []
    for raw in _frames(name, data):
        raw = raw.dropna(how="all").dropna(axis=1, how="all")
        for h in raw.index[:15]:  # header row: the first one naming open, high, low and close
            names = [_LOOK.get(_key(c), _key(c)) for c in raw.loc[h]]
            if {"open", "high", "low", "close"} <= set(names):
                break
        else:
            seen.append(", ".join(str(c) for c in raw.iloc[0].tolist()[:10]))
            continue
        df = raw.loc[raw.index > h].copy()
        df.columns = names
        df = df.loc[:, ~pd.Index(names).duplicated()]
        if "datetime" not in df:
            if {"date", "time"} <= set(df.columns):
                df["datetime"] = df["date"].astype(str).str[:10] + " " + df["time"].astype(str)
            elif "date" in df:
                df["datetime"] = df["date"]
            elif "time" in df:
                df["datetime"] = df["time"]
            else:
                raise ValueError("no date/time column found. Name one Datetime, Timestamp, or Date + Time. "
                                 f"Columns seen: {', '.join(map(str, names))}")
        dt = df["datetime"]
        if pd.api.types.is_numeric_dtype(dt) or dt.astype(str).str.fullmatch(r"\d{10}(\.0)?").all():
            dt = pd.to_datetime(pd.to_numeric(dt), unit="s", utc=True)  # epoch seconds
        else:
            dt = pd.to_datetime(dt.astype(str), errors="coerce", dayfirst=not dt.astype(str).str.match(r"^\d{4}-").all())
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        df["datetime"] = dt
        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
        out = df.dropna(subset=["datetime", "open", "high", "low", "close"])
        if out.empty:
            raise ValueError("found the columns but could not read any dates or prices; check the date format")
        if (out["datetime"].dt.hour == 0).all():
            raise ValueError("the dates have no time of day; 1-minute candles need a time (e.g. 2026-10-27 09:15)")
        need = ["datetime", "open", "high", "low", "close"]
        return out.sort_values("datetime")[need].reset_index(drop=True)
    raise ValueError("no Open, High, Low and Close columns found. First row seen: " + " / ".join(seen[:2]))


def read_preopen(name: str, data: bytes) -> pd.DataFrame | None:
    """NSE pre-open market download (MW-Pre-Open-Market-*.csv): one final opening price per stock.
    Returns symbol, prev_close, open, gap_pct and date, or None if the file is not in that format."""
    try:
        raw = _frames(name, data)[0]
    except Exception:
        return None
    raw = raw.dropna(how="all")
    head = [str(c).strip().upper() for c in raw.iloc[0]]
    if "SYMBOL" not in head or "FINAL PRICE" not in head:
        return None
    df = raw.iloc[1:].copy()
    df.columns = head

    def num(col):
        return pd.to_numeric(df[col].astype(str).str.replace(",", "").str.strip()
                             .replace({"-": None, "": None}), errors="coerce")
    out = pd.DataFrame({"symbol": df["SYMBOL"].astype(str).str.strip().str.upper(),
                        "prev_close": num("PREV. CLOSE") if "PREV. CLOSE" in df else np.nan,
                        "open": num("FINAL PRICE")}).dropna(subset=["open"])
    out["gap_pct"] = (out["open"] / out["prev_close"] - 1) * 100
    m = re.search(r"(\d{1,2}-[A-Za-z]{3}-\d{4})", name)
    day = pd.to_datetime(m.group(1)) if m else pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    out["date"] = day.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)


def backtest(minutes: pd.DataFrame, or_minutes: int, window: int, reward: float, buffer_pct: float,
             bias_mode: str, risk_rupees: float, brokerage: float) -> pd.DataFrame:
    """One opening-range trade per day at most; returns a row per day with the outcome."""
    out = []
    prev_close = None
    for day, g in minutes.groupby(minutes["datetime"].dt.date):
        start = pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=15)
        g = g[(g["datetime"] >= start) & (g["datetime"] < start + pd.Timedelta(minutes=window))]
        rng = g[g["datetime"] < start + pd.Timedelta(minutes=or_minutes)]
        after = g[g["datetime"] >= start + pd.Timedelta(minutes=or_minutes)]
        day_close = minutes.loc[minutes["datetime"].dt.date == day, "close"].iloc[-1]
        if rng.empty or after.empty:
            prev_close = day_close
            continue
        open_, hi, lo = rng["open"].iloc[0], rng["high"].max(), rng["low"].min()
        gap = 0.0 if prev_close is None else open_ / prev_close - 1
        prev_close = day_close
        allowed = {"Gap direction only": ("Long",) if gap > 0 else ("Short",) if gap < 0 else (),
                   "Both directions": ("Long", "Short")}[bias_mode]
        buf = open_ * buffer_pct
        trade = None
        for _, bar in after.iterrows():
            if trade is None:
                if "Long" in allowed and bar["high"] >= hi + buf:
                    e = hi + buf
                    trade = {"side": "Long", "entry": e, "stop": lo, "target": e + reward * (e - lo)}
                elif "Short" in allowed and bar["low"] <= lo - buf:
                    e = lo - buf
                    trade = {"side": "Short", "entry": e, "stop": hi, "target": e - reward * (hi - e)}
                if trade is None:
                    continue
            # Same bar may hit both: count the stop first (conservative).
            if trade["side"] == "Long":
                if bar["low"] <= trade["stop"]:
                    trade["exit"], trade["how"] = trade["stop"], "stop"
                elif bar["high"] >= trade["target"]:
                    trade["exit"], trade["how"] = trade["target"], "target"
            else:
                if bar["high"] >= trade["stop"]:
                    trade["exit"], trade["how"] = trade["stop"], "stop"
                elif bar["low"] <= trade["target"]:
                    trade["exit"], trade["how"] = trade["target"], "target"
            if "exit" in trade:
                break
        if trade is None:
            out.append({"date": day, "gap_pct": gap * 100, "side": "No trade", "result_r": 0.0, "net_pnl": 0.0})
            continue
        if "exit" not in trade:
            trade["exit"], trade["how"] = after["close"].iloc[-1], "time"
        risk = abs(trade["entry"] - trade["stop"])
        qty = int(risk_rupees / risk) if risk > 0 else 0
        sign = 1 if trade["side"] == "Long" else -1
        gross = sign * (trade["exit"] - trade["entry"]) * qty
        buy, sell = (trade["entry"], trade["exit"]) if sign > 0 else (trade["exit"], trade["entry"])
        net = gross - (charges(qty * buy, qty * sell, brokerage) if qty else 0)
        out.append({"date": day, "gap_pct": gap * 100, "side": trade["side"], "exit_by": trade["how"],
                    "entry": trade["entry"], "exit": trade["exit"], "qty": qty,
                    "result_r": sign * (trade["exit"] - trade["entry"]) / risk if risk else 0.0,
                    "net_pnl": net})
    return pd.DataFrame(out)
