"""Position-sizing planner for the dashboard's "Invest" tab.

Educational tool for the capstone, not investment advice. The model
forecasts the opening price, which is already priced in once the market
opens, so the plan assumes you buy at the expected open and manages risk
from there:

- stop loss  = entry - k x ATR(14)       (ATR = average daily range)
- target     = entry + reward x (entry - stop)
- size       = the strategy's split, capped so a stop-out loses at most
               risk_pct of the capital per stock
- expected profit comes from a backtest on each stock's own history: enter
  at the open, exit at the stop, the target or the close after N days,
  whichever comes first (a day touching both counts as a stop).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STRATEGIES = {
    "Equal risk": "Each stock gets money in inverse proportion to how volatile it is.",
    "Favour historical edge": "Tilts money toward stocks whose stop/target plan made money in the "
                              "backtest; stocks that lost get nothing.",
    "Equal amount": "Same rupee amount in every stock.",
}


def atr(prices: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = prices["close"].shift(1)
    tr = pd.concat([prices["high"] - prices["low"], (prices["high"] - prev).abs(),
                    (prices["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def backtest(prices: pd.DataFrame, stop_mult: float, reward: float, hold_days: int,
             lookback: int = 500) -> dict:
    """Win rate and average return per trade of the stop/target plan on past days."""
    p = prices.sort_values("date").reset_index(drop=True)
    a = atr(p).shift(1)  # ATR known before the entry day's open
    o, h, l, c = (p[k].to_numpy() for k in ("open", "high", "low", "close"))
    rets, wins = [], 0
    start = max(15, len(p) - lookback - hold_days)
    for i in range(start, len(p) - hold_days):
        if not np.isfinite(a.iloc[i]):
            continue
        entry = o[i]
        stop = entry - stop_mult * a.iloc[i]
        target = entry + reward * (entry - stop)
        exit_price = c[i + hold_days - 1]
        for j in range(i, i + hold_days):
            if l[j] <= stop:
                exit_price = stop
                break
            if h[j] >= target:
                exit_price = target
                break
        r = exit_price / entry - 1
        rets.append(r)
        wins += r > 0
    if not rets:
        return {"trades": 0, "win_rate": np.nan, "avg_ret": np.nan}
    return {"trades": len(rets), "win_rate": wins / len(rets), "avg_ret": float(np.mean(rets))}


def _cap(w: pd.Series, max_w: float) -> pd.Series:
    """Cap each weight at max_w and hand the excess to the uncapped ones; any rest stays cash."""
    w = w.clip(lower=0)
    if w.sum() == 0:
        return w
    w = w / w.sum()
    for _ in range(10):
        over = w > max_w
        if not over.any():
            break
        excess = (w[over] - max_w).sum()
        w[over] = max_w
        free = ~over & (w > 0)
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w.clip(upper=max_w)


def plan(preds: pd.DataFrame, prices: dict[str, pd.DataFrame], capital: float, strategy: str,
         risk_pct: float, stop_mult: float, reward: float, hold_days: int, max_weight: float) -> pd.DataFrame:
    rows = []
    for _, r in preds.iterrows():
        p = prices.get(r["symbol"])
        if p is None or len(p) < 30:
            continue
        a = float(atr(p).iloc[-1])
        vol = float(p["close"].pct_change().tail(20).std())
        bt = backtest(p, stop_mult, reward, hold_days)
        entry = float(r["pred_open"])
        rows.append({"symbol": r["symbol"], "sector": r.get("india_sector", ""), "entry": entry,
                     "pred_gap_pct": r["pred_gap_pct"], "atr": a, "vol": vol,
                     "stop": entry - stop_mult * a, "target": entry + reward * stop_mult * a, **bt})
    df = pd.DataFrame(rows).set_index("symbol")
    if df.empty:
        return df
    if strategy == "Equal amount":
        raw = pd.Series(1.0, index=df.index)
    elif strategy == "Favour historical edge":
        raw = (df["avg_ret"].clip(lower=0) / df["vol"]).fillna(0)
    else:
        raw = 1 / df["vol"]
    df["weight"] = _cap(raw, max_weight)
    df["stop_pct"] = df["stop"] / df["entry"] - 1
    # Risk cap: losing at the stop must cost at most risk_pct of capital.
    df["max_by_risk"] = risk_pct * capital / df["stop_pct"].abs()
    value = np.minimum(df["weight"] * capital, df["max_by_risk"])
    df["shares"] = np.floor(value / df["entry"]).astype(int)
    df["invest"] = df["shares"] * df["entry"]
    df["loss_at_stop"] = df["shares"] * (df["stop"] - df["entry"])
    df["profit_at_target"] = df["shares"] * (df["target"] - df["entry"])
    df["expected_profit"] = df["invest"] * df["avg_ret"]
    df["limited_by_risk"] = df["max_by_risk"] < df["weight"] * capital
    return df.reset_index()
