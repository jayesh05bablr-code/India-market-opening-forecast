"""Train the gap model with walk-forward validation and compare it to baselines.

    python -m capstone.train

Prints MAE in rupees and percent, and direction accuracy, for each fold.
When overnight US/rupee data is present it also trains the same model
without those features, so the report shows how much they add.
Saves the final model (trained on all history) to models/gap_model.joblib.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import features
from .paths import MODELS

N_FOLDS = 5
MIN_TRAIN_DAYS = 250


def make_model() -> HistGradientBoostingRegressor:
    # Handles missing fundamentals natively; shallow trees resist noise.
    return HistGradientBoostingRegressor(
        loss="absolute_error", max_depth=3, learning_rate=0.05,
        max_iter=300, min_samples_leaf=50, l2_regularization=1.0, random_state=0,
    )


def scores(df: pd.DataFrame, gap_pred: np.ndarray) -> dict[str, float]:
    pred_open = df["close"].to_numpy() * (1 + gap_pred)
    actual = df["next_open"].to_numpy()
    err = np.abs(pred_open - actual)
    true_dir = np.sign(df["target_gap"].to_numpy())
    moved = true_dir != 0
    return {
        "mae_rupees": float(err.mean()),
        "mape_pct": float((err / actual).mean() * 100),
        "direction_acc_pct": float((np.sign(gap_pred)[moved] == true_dir[moved]).mean() * 100),
    }


def usable_features(df: pd.DataFrame, candidates: list[str] | None = None) -> list[str]:
    # A column with fewer than two distinct values (e.g. fundamentals not filled in yet, or
    # only known after this training window) carries no signal and crashes the binning.
    return [f for f in (candidates or features.FEATURES) if df[f].nunique(dropna=True) >= 2]


def walk_forward(df: pd.DataFrame, all_cols: list[str]) -> pd.DataFrame:
    dates = np.sort(df["date"].unique())
    test_days = (len(dates) - MIN_TRAIN_DAYS) // N_FOLDS
    if test_days < 20:
        raise ValueError(f"Only {len(dates)} trading days; need at least {MIN_TRAIN_DAYS + 20 * N_FOLDS}.")
    rows = []
    for k in range(N_FOLDS):
        split = dates[MIN_TRAIN_DAYS + k * test_days]
        stop = dates[min(MIN_TRAIN_DAYS + (k + 1) * test_days, len(dates) - 1)]
        train = df[df["date"] < split]
        test = df[(df["date"] >= split) & (df["date"] < stop)]
        cols = usable_features(train, all_cols)
        model = make_model().fit(train[cols], train["target_gap"])
        candidates = {
            "model": model.predict(test[cols]),
            "baseline_open_eq_close": np.zeros(len(test)),
            "baseline_mean_gap": np.full(len(test), train["target_gap"].mean()),
        }
        base_cols = [c for c in cols if c not in features.OVERNIGHT_FEATURES]
        if base_cols != cols:
            no_overnight = make_model().fit(train[base_cols], train["target_gap"])
            candidates["model_without_overnight"] = no_overnight.predict(test[base_cols])
        for name, pred in candidates.items():
            rows.append({"fold": k + 1, "test_start": str(split)[:10], "method": name, **scores(test, pred)})
    return pd.DataFrame(rows)


def main() -> None:
    df = features.build().dropna(subset=["target_gap"])
    cols = usable_features(df)
    skipped = sorted(set(features.FEATURES) - set(cols))
    if skipped:
        print(f"Skipping empty features: {', '.join(skipped)}")
    report = walk_forward(df, cols)
    pd.set_option("display.width", 120)
    print(report.round(3).to_string(index=False))
    summary = report.groupby("method")[["mae_rupees", "mape_pct", "direction_acc_pct"]].mean()
    print("\nAverage over folds:\n", summary.round(3).to_string())

    model = make_model().fit(df[cols], df["target_gap"])
    # In-sample residuals understate error, so the band uses the last fold's out-of-sample spread.
    cutoff = pd.Timestamp(report["test_start"].iloc[-1])
    before, last = df[df["date"] < cutoff], df[df["date"] >= cutoff]
    hold_cols = usable_features(before, cols)
    holdout_model = make_model().fit(before[hold_cols], before["target_gap"])
    resid = last["target_gap"] - holdout_model.predict(last[hold_cols])
    band = {"q10": float(resid.quantile(0.10)), "q90": float(resid.quantile(0.90))}
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": cols, "band": band}, MODELS / "gap_model.joblib")
    report.to_csv(MODELS / "walk_forward_report.csv", index=False)
    (MODELS / "band.json").write_text(json.dumps(band, indent=2))
    print(f"\nSaved model to {MODELS / 'gap_model.joblib'}")


if __name__ == "__main__":
    main()
