"""Predict tomorrow's opening price for every symbol from its latest close.

    python -m capstone.predict

Best run in the morning after the US close (about 02:30 IST) and after
fetch_data and fetch_overnight, so the overnight US and rupee moves are in.
Writes data/processed/predictions.csv (with reasons), overnight_story.json,
and appends to predictions_log.csv.
"""
from __future__ import annotations

import json

import joblib
import pandas as pd

from . import explain, features, overnight
from .paths import FX_DIR, MODELS, PROCESSED

LOG_COLUMNS = ["symbol", "as_of_close", "close", "pred_gap_pct", "pred_open", "pred_open_low", "pred_open_high"]


def _live_fx() -> float | None:
    path = FX_DIR / "USDINR_live.json"
    return json.loads(path.read_text())["rate"] if path.exists() else None


def main() -> None:
    bundle = joblib.load(MODELS / "gap_model.joblib")
    cols = bundle["features"]
    df = features.build(save=False)
    latest = df.sort_values("date").groupby("symbol").tail(1).copy()
    gap = bundle["model"].predict(latest[cols])
    latest["pred_gap_pct"] = gap * 100
    latest["pred_open"] = latest["close"] * (1 + gap)
    latest["pred_open_low"] = latest["close"] * (1 + gap + bundle["band"]["q10"])
    latest["pred_open_high"] = latest["close"] * (1 + gap + bundle["band"]["q90"])
    contrib = explain.contributions(bundle["model"], latest, cols)
    for name in contrib.columns:
        latest[f"contrib_{name}"] = contrib[name]

    as_of = latest["date"].max()
    us_last = overnight.latest_session()
    complete = us_last is not None and us_last >= as_of
    if not complete:
        print(f"Note: no US session after {as_of:%Y-%m-%d} in data/us yet. Run fetch_overnight after "
              "the US close (about 02:30 IST) for the overnight signal.")
    smap = overnight.stock_map().set_index("symbol")
    for c in ["us_etf", "us_sector", "adr", "fx_exposure"]:
        latest[c] = latest["symbol"].map(smap[c])
    latest["india_sector"] = latest["india_sector"].fillna("Unmapped")

    out = latest.rename(columns={"date": "as_of_close"})
    out["reason"] = out.apply(explain.stock_reason, axis=1)
    out["as_of_close"] = out["as_of_close"].dt.strftime("%Y-%m-%d")
    keep = LOG_COLUMNS + ["india_sector", "us_etf", "us_sector", "adr", "fx_exposure",
                          *[c for c in features.OVERNIGHT_FEATURES if c in out.columns],
                          *[f"contrib_{n}" for n in contrib.columns], "reason"]
    out = out[keep]
    out[LOG_COLUMNS[2:]] = out[LOG_COLUMNS[2:]].round(2)

    end = as_of + pd.offsets.BDay(1)
    moves = overnight.sector_moves(as_of, end)
    fx_chg = float(latest["usdinr_chg"].iloc[0]) if latest["usdinr_chg"].notna().any() else float("nan")
    fx_level = _live_fx()
    if fx_level is None and not overnight.load_fx().empty:
        fx_level = float(overnight.load_fx().iloc[-1])
    links = pd.read_csv(overnight.SECTOR_LINKS_FILE)
    story = {
        "as_of_close": f"{as_of:%Y-%m-%d}",
        "next_open": f"{end:%Y-%m-%d}",
        "us_session": None if us_last is None else f"{us_last:%a %d %b %Y}",
        "overnight_complete": bool(complete),
        "usdinr": fx_level,
        "usdinr_chg": fx_chg,
        "mood": explain.mood(out["pred_gap_pct"].mean() / 100),
        "lines": explain.market_story(moves, fx_chg, fx_level, out, links,
                                      None if us_last is None else f"{us_last:%a %d %b}"),
        "moves": moves.to_dict(orient="records"),
    }

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROCESSED / "predictions.csv", index=False)
    (PROCESSED / "overnight_story.json").write_text(json.dumps(story, indent=2, default=float))
    log = PROCESSED / "predictions_log.csv"
    out[LOG_COLUMNS].to_csv(log, mode="a", header=not log.exists(), index=False)
    print(f"Expected opening: {story['mood']}")
    print("\n".join(f"- {line}" for line in story["lines"]))
    print()
    print(out[LOG_COLUMNS].to_string(index=False))


if __name__ == "__main__":
    main()
