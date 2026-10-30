"""Write a SYNTHETIC sample of Indian prices, US ETFs/ADRs and USD/INR so the
whole overnight pipeline and dashboard can be tested without network access.

    python -m tests.make_synthetic_overnight

The Indian opening gaps are built from the fake US and rupee moves with known
weights, so the model has something real to find. These numbers are not
market data and say nothing about real accuracy. Before using real data,
delete data/raw, data/us, data/fx and data/SYNTHETIC_DATA.
"""
import numpy as np
import pandas as pd

from capstone import overnight
from capstone.paths import FX_DIR, RAW, SYNTHETIC_FLAG, US_DIR, load_symbols

rng = np.random.default_rng(7)
START, END = "2022-01-03", "2026-10-22"
# Planted weights: how much of each overnight move shows up in the Indian open.
W_MKT, W_SECTOR_REL, W_ADR, W_FX = 0.30, 0.25, 0.40, 0.50


def drop_holidays(dates: pd.DatetimeIndex, per_year: int) -> pd.DatetimeIndex:
    n = int(len(dates) / 252 * per_year)
    return dates.delete(rng.choice(len(dates), n, replace=False))


def save(path, dates, close):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": dates, "close": close, "source": "synthetic"}).to_csv(path, index=False)


symbols = load_symbols()
smap = overnight.stock_map().set_index("symbol")

us_dates = drop_holidays(pd.bdate_range(START, END), 9)
mkt = rng.normal(0.0004, 0.010, len(us_dates))
save(US_DIR / "SPY.csv", us_dates, 450 * np.exp(np.cumsum(mkt)))
save(US_DIR / "QQQ.csv", us_dates, 380 * np.exp(np.cumsum(1.2 * mkt + rng.normal(0, 0.004, len(us_dates)))))
sector_ret = {}
for etf in overnight.SECTOR_ETFS:
    beta = rng.uniform(0.7, 1.3)
    sector_ret[etf] = beta * mkt + rng.normal(0, 0.008, len(us_dates))
    save(US_DIR / f"{etf}.csv", us_dates, 100 * np.exp(np.cumsum(sector_ret[etf])))
for sym in symbols:
    adr = smap["adr"].get(sym, "")
    if adr:
        etf = smap.loc[sym, "us_etf"]
        r = 0.6 * sector_ret[etf] + 0.3 * mkt + rng.normal(0, 0.009, len(us_dates))
        save(US_DIR / f"{adr}.csv", us_dates, 20 * np.exp(np.cumsum(r)))

fx_dates = pd.bdate_range(START, END)
fx = 75 * np.exp(np.cumsum(rng.normal(0.00012, 0.003, len(fx_dates))))
save(FX_DIR / "USDINR.csv", fx_dates, fx)

in_dates = drop_holidays(pd.bdate_range(START, END), 14)
RAW.mkdir(parents=True, exist_ok=True)
for sym in symbols:
    n = len(in_dates)
    frame = pd.DataFrame({"symbol": sym, "date": in_dates, "next_date": in_dates[1:].append(pd.DatetimeIndex([pd.NaT]))})
    feats = overnight.attach(frame).fillna({"adr_ret": 0, "us_sector_rel": 0, "fx_effect": 0, "us_mkt_ret": 0})
    gap = (W_MKT * feats["us_mkt_ret"] + W_SECTOR_REL * feats["us_sector_rel"]
           + W_ADR * feats["adr_ret"] + W_FX * feats["fx_effect"]).to_numpy() + rng.normal(0, 0.003, n)
    close = np.empty(n)
    open_ = np.empty(n)
    close[0] = open_[0] = rng.uniform(500, 3000)
    for t in range(1, n):
        open_[t] = close[t - 1] * (1 + gap[t - 1])
        close[t] = open_[t] * np.exp(rng.normal(0, 0.011))
    prev = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    pd.DataFrame({"date": in_dates, "open": open_, "high": high, "low": low, "close": close,
                  "prev_close": prev, "volume": rng.integers(1e5, 1e6, n)}).to_csv(RAW / f"{sym}.csv", index=False)

SYNTHETIC_FLAG.write_text("Synthetic sample data from tests/make_synthetic_overnight.py. Not market data.\n")
print(f"Wrote SYNTHETIC prices for {len(symbols)} stocks, {len(list(US_DIR.glob('*.csv')))} US series and USD/INR")
