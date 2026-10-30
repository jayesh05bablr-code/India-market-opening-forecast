"""Turn overnight moves and model output into plain-language reasons.

Attribution is a what-if: for each driver group we re-run the model with that
group set to "no move" and call the difference its contribution. It shows
what the model reacted to; it is approximate (the parts need not add up
exactly, the rest is shown as Indian factors) and it is not proof of cause.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DRIVERS = {
    "US market": ["us_mkt_ret", "us_tech_ret"],
    "US sector": ["us_sector_ret", "us_sector_rel"],
    "ADR": ["adr_ret"],
    "Rupee": ["usdinr_chg", "fx_effect"],
}
OTHER = "Indian factors"


def pct(x: float, digits: int = 2) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:+.{digits}f}%"


def size_word(x: float) -> str:
    a = abs(x)
    return "sharply" if a >= 0.015 else "solidly" if a >= 0.007 else "slightly" if a >= 0.002 else "barely"


def verb(x: float, up: str = "rose", down: str = "fell") -> str:
    return up if x >= 0 else down


def move(x: float) -> str:
    """'rose 0.94%' / 'fell 0.94%' / 'was flat'."""
    if pd.isna(x):
        return "has no data"
    if abs(x) < 0.0005:
        return "was flat"
    return f"{verb(x)} {abs(x) * 100:.2f}%"


def contributions(model, X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per-row contribution of each driver group to the predicted gap."""
    full = model.predict(X[cols])
    out = pd.DataFrame(index=X.index)
    for name, group in DRIVERS.items():
        group = [c for c in group if c in cols]
        if not group:
            out[name] = 0.0
            continue
        neutral = X[cols].copy()
        # Only neutralise values that exist; a stock with no ADR gets 0 for "ADR".
        for c in group:
            neutral[c] = np.where(neutral[c].isna(), np.nan, 0.0)
        out[name] = full - model.predict(neutral)
    out[OTHER] = full - out[list(DRIVERS)].sum(axis=1)
    return out


def market_story(moves: pd.DataFrame, fx_chg: float, fx_level: float | None,
                 preds: pd.DataFrame, links: pd.DataFrame, us_date: str | None) -> list[str]:
    """Bullet points for the whole market, most important first."""
    lines = []
    mkt = moves.set_index("ticker")
    spx, ndx = mkt.loc["SPY", "ret"], mkt.loc["QQQ", "ret"]
    if mkt.loc["SPY", "sessions"] == 0:
        lines.append("US markets were closed overnight, so there is no fresh signal from Wall Street.")
    elif pd.notna(spx):
        when = f" (session of {us_date})" if us_date else ""
        lines.append(f"Wall Street{when}: the S&P 500 {size_word(spx)} {move(spx)} "
                     f"and the Nasdaq 100 {move(ndx)}. "
                     + ("A positive US close usually sets a positive tone for Indian openings."
                        if spx > 0.002 else "A weak US close usually weighs on Indian openings."
                        if spx < -0.002 else "A flat US close gives little direction to India."))

    sectors = moves[moves["is_sector"]].dropna(subset=["ret"]).sort_values("ret", ascending=False)
    if len(sectors):
        top = ", ".join(f"{r.name} {pct(r.ret)}" for r in sectors.head(3).itertuples())
        bottom = ", ".join(f"{r.name} {pct(r.ret)}" for r in sectors.tail(3).iloc[::-1].itertuples())
        lines.append(f"Strongest US sectors: {top}. Weakest: {bottom}.")

    # One line per Indian sector held, saying what its US twin did.
    held = preds.groupby("india_sector")["symbol"].apply(lambda s: ", ".join(s)).to_dict()
    for sector, names in held.items():
        link = links[links["india_sector"] == sector]
        if link.empty:
            continue
        link = link.iloc[0]
        r = mkt["ret"].get(link["us_etf"], np.nan)
        if pd.isna(r):
            continue
        effect = "a tailwind" if r > 0.002 else "a headwind" if r < -0.002 else "neutral"
        lines.append(f"US {link['us_sector']} {move(r)}, {effect} for Indian {sector} "
                     f"({names}). {link['why_us']}.")

    if pd.notna(fx_chg):
        level = f" to ₹{fx_level:.2f} per dollar" if fx_level else ""
        if abs(fx_chg) < 0.0005:
            lines.append(f"The rupee was steady{level}, so currency is not a big factor today.")
        else:
            weak = fx_chg > 0
            winners = sorted({s for s in held if _exposure(links, s) > 0})
            losers = sorted({s for s in held if _exposure(links, s) < 0})
            if not weak:
                winners, losers = losers, winners
            txt = (f"The rupee {'weakened' if weak else 'strengthened'} {abs(fx_chg) * 100:.2f}%{level}. "
                   + ("A weaker rupee helps exporters that earn dollars and hurts importers, and can "
                      "prompt foreign investors to sell." if weak else
                      "A stronger rupee cuts import costs and tends to attract foreign buying, but trims "
                      "exporters' rupee earnings."))
            if winners:
                txt += f" Helped here: {', '.join(winners)}."
            if losers:
                txt += f" Hurt here: {', '.join(losers)}."
            lines.append(txt)

    if len(preds):
        avg = preds["pred_gap_pct"].mean() / 100
        up = int((preds["pred_gap_pct"] > 0).sum())
        lines.append(f"Putting it together, the model expects your {len(preds)} stocks to open "
                     f"{pct(avg)} on average, with {up} of {len(preds)} opening higher.")
    return lines


def _exposure(links: pd.DataFrame, sector: str) -> float:
    row = links[links["india_sector"] == sector]
    return 0.0 if row.empty else float(row["fx_exposure"].iloc[0])


def mood(avg_gap: float) -> str:
    if avg_gap > 0.004:
        return "Gap-up"
    if avg_gap > 0.001:
        return "Mildly positive"
    if avg_gap < -0.004:
        return "Gap-down"
    if avg_gap < -0.001:
        return "Mildly negative"
    return "Flat"


def stock_reason(row: pd.Series) -> str:
    """One paragraph for one stock; row has prediction, inputs and contrib_* columns."""
    gap = row["pred_gap_pct"] / 100
    head = (f"{row['symbol']} is expected to open {pct(gap)} at about ₹{row['pred_open']:,.2f} "
            f"(likely range ₹{row['pred_open_low']:,.2f} to ₹{row['pred_open_high']:,.2f}).")
    parts = []
    contrib = {k: row.get(f"contrib_{k}", 0.0) for k in [*DRIVERS, OTHER]}
    for name, c in sorted(contrib.items(), key=lambda kv: -abs(kv[1])):
        if abs(c) < 0.0003:
            continue
        effect = f"adding about {pct(c)}" if c > 0 else f"taking off about {pct(-c)[1:]}"
        if name == "US market":
            parts.append(f"the S&P 500 {move(row['us_mkt_ret'])} overnight, {effect}")
        elif name == "US sector" and pd.notna(row.get("us_sector_ret")):
            parts.append(f"its US twin sector ({row['us_sector']}, {row['us_etf']}) "
                         f"{move(row['us_sector_ret'])}, {effect}")
        elif name == "ADR" and pd.notna(row.get("adr_ret")):
            parts.append(f"its US-listed ADR ({row['adr']}) {move(row['adr_ret'])}, {effect}")
        elif name == "Rupee" and pd.notna(row.get("usdinr_chg")):
            way = "weakened" if row["usdinr_chg"] > 0 else "strengthened"
            exp = row.get("fx_exposure", 0) or 0
            kind = "an exporter" if exp > 0 else "an importer" if exp < 0 else "mostly domestic"
            parts.append(f"the rupee {way} {abs(row['usdinr_chg']) * 100:.2f}% and {row['symbol']} is {kind}, {effect}")
        elif name == OTHER:
            parts.append(f"its own recent price action and Indian market trend, {effect}")
    if not parts:
        return head + " No single factor stands out; the model sees a quiet night."
    return head + " Main reasons: " + "; ".join(parts[:4]) + "."
