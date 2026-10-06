"""The splits and the confident-call check every forecaster in this project is held to.

Train on what ends by TRAIN_END, choose on what falls in VAL_START..VAL_END, retrain on everything up
to VAL_END and score once from TEST_START. Scored origins sit SCORE_STRIDE trading days apart, so
20-day outcomes do not overlap.
"""
import pandas as pd

from backend.models.core.backend import default_rng, xp

TRAIN_END, VAL_START, VAL_END, TEST_START = "2021-12-31", "2022-01-01", "2023-12-31", "2024-01-01"
SCORE_STRIDE = 21
CALL_RATE = 0.10           # share of validation forecasts that become "up" calls


def selective(val_rows, test_rows, horizons, rate=CALL_RATE, draws=2000, seed=0, key="p_up"):
    """Precision of "up" calls on the test window, with the cut on `key` taken from validation.
    The 95% range resamples whole origin dates, because stocks on one date move together."""
    v, t = pd.DataFrame(val_rows), pd.DataFrame(test_rows)
    rng = default_rng(seed)
    out = []
    for h in horizons:
        vh, th = v[v["horizon"] == h], t[t["horizon"] == h]
        cut = float(vh[key].quantile(1 - rate))
        calls = th[th[key] >= cut]
        by_date = calls.assign(hit=calls["y"] > 0).groupby("date")["hit"].agg(["sum", "count"]).to_numpy()
        boot = [by_date[i, 0].sum() / by_date[i, 1].sum()
                for i in rng.integers(0, len(by_date), (draws, len(by_date)))] if len(by_date) else []
        out.append({"horizon": h, "cut": round(cut, 4), "calls": int(len(calls)), "of": int(len(th)),
                    "call_rate": round(len(calls) / max(1, len(th)), 4),
                    "precision": round(float((calls["y"] > 0).mean()), 4) if len(calls) else None,
                    "precision_lo": round(float(xp.percentile(boot, 2.5)), 4) if len(by_date) else None,
                    "precision_hi": round(float(xp.percentile(boot, 97.5)), 4) if len(by_date) else None,
                    "base_rate": round(float((th["y"] > 0).mean()), 4)})
    return out
