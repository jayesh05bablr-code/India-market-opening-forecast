"""Write SYNTHETIC random-walk prices into data/raw so the pipeline can be
tested without network access. Delete data/raw before using real data.
These numbers are not market data and say nothing about model accuracy.
"""
import numpy as np
import pandas as pd

from capstone.paths import RAW, load_symbols

rng = np.random.default_rng(0)
dates = pd.bdate_range("2022-01-03", "2026-10-15")
RAW.mkdir(parents=True, exist_ok=True)
for symbol in load_symbols():
    close = 1000 * np.exp(np.cumsum(rng.normal(0, 0.015, len(dates))))
    prev = np.r_[close[0], close[:-1]]
    open_ = prev * (1 + rng.normal(0, 0.005, len(dates)))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, len(dates)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, len(dates)))
    vol = rng.integers(1e5, 1e6, len(dates))
    pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close,
                  "prev_close": prev, "volume": vol}).to_csv(RAW / f"{symbol}.csv", index=False)
print(f"Wrote synthetic prices for {len(load_symbols())} symbols to {RAW}")
