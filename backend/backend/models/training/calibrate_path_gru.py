#!/usr/bin/env python3
"""Make the path GRU's 80% range hold 80%, and test it the way banks backtest a value-at-risk model.

Trained on likelihood, the GRU's 10-day 80% range held 83.0% of test outcomes: too wide, so the "1 in 10 ends
lower" label on its low path overstated how far prices fall. Conformal calibration fixes the width without
retraining: the multiple of the GRU's own spread that 80% of known outcomes fell within becomes the range.

Which outcomes it is fitted on is the question, and it is measured. Fitted once on the validation years it
gave 75.1% on the test years -- volatility there was not volatility here. Refitted at every test date on the
trailing year of outcomes already known by then, it follows the regime. The policy served is the one whose
test coverage lands nearest 80%, with:

- a date-resampled 95% range, because stocks on one date move together;
- Kupiec's proportion-of-failures test (the Basel backtest), on one stock-date per date so the independence
  it assumes is not broken by stocks moving together.

    python -m backend.models.training.calibrate_path_gru
"""
import json
import math

import pandas as pd

from backend.models.core.backend import default_rng, xp
from backend.models.learning.protocol import SCORE_STRIDE, TEST_START, VAL_END, VAL_START
from backend.models.networks import path_gru as G
from backend.models.training import train_path_gru as T
from backend.paths import ARTIFACTS

WEIGHTS = ARTIFACTS / "path_gru.npz"
LEVEL = 0.8
# provisional: a year of known outcomes per refit, long enough for a stable 80th percentile (about 2,000 rows)
TRAILING_DAYS = 365
# an origin's 10-day outcome is known about two calendar weeks later
KNOWN_AFTER_DAYS = 20


def spreads(mean, log_sd, Y, k):
    """(standardised distance of each outcome from the centre, cumulative spread) over the first k days."""
    centre = mean[:, :k].sum(1)
    spread = xp.sqrt((xp.exp(log_sd[:, :k]) ** 2).sum(1))
    return xp.abs(Y[:, :k].sum(1) - centre) / spread, spread


def kupiec_p(hits, n, level=LEVEL):
    """Kupiec's unconditional-coverage likelihood-ratio test: the chance of a miss rate this far from 1 - level."""
    misses, expected = n - hits, 1 - level
    observed = misses / n
    if observed in (0.0, 1.0):
        return 0.0
    ratio = -2 * ((n - misses) * math.log(1 - expected) + misses * math.log(expected)
                  - (n - misses) * math.log(1 - observed) - misses * math.log(observed))
    return math.erfc(math.sqrt(ratio / 2))          # chi-square with one degree of freedom


def date_range(dates, covered, draws=2000, seed=0):
    frame = pd.Series(covered.astype(float)).groupby(dates)
    total, count = frame.sum().to_numpy(), frame.count().to_numpy()
    pick = default_rng(seed).integers(0, len(total), (draws, len(total)))
    boot = total[pick].sum(1) / count[pick].sum(1)
    return round(float(xp.percentile(boot, 2.5)), 4), round(float(xp.percentile(boot, 97.5)), 4)


def one_per_date(rows, seed=0):
    """One row index per date, chosen at random, so a coverage test sees independent outcomes."""
    by_date = {}
    for i, row in enumerate(rows):
        by_date.setdefault(row["date"], []).append(i)
    rng = default_rng(seed)
    return xp.asarray([held[int(rng.integers(0, len(held)))] for held in by_date.values()])


def main():
    from backend.database.sources.panel import index_constituents, universe_panel
    import yfinance as yf
    model, record = G.load(WEIGHTS)
    panel = universe_panel("nifty500", "10y")
    market = yf.Ticker("^NSEI").history(period="10y", auto_adjust=True)["Close"]
    market.index = market.index.tz_localize(None).normalize()
    scored = [s for s in index_constituents("nifty200")["symbol"] if s in panel.close.columns]
    Xv, Yv, rv = T.samples(panel, scored, market, VAL_START, VAL_END, SCORE_STRIDE)
    Xs, Ys, rs = T.samples(panel, scored, market, TEST_START, None, SCORE_STRIDE)
    mean_v, sd_v = T.predict(model, Xv)
    mean_s, sd_s = T.predict(model, Xs)
    dates = [r["date"] for r in rs]
    single = one_per_date(rs)
    origins = pd.to_datetime([r["date"] for r in rv + rs])
    test_origins = pd.to_datetime(dates)
    lag, window = pd.Timedelta(days=KNOWN_AFTER_DAYS), pd.Timedelta(days=TRAILING_DAYS)
    calibration, out = {}, []
    for k in range(1, G.HORIZON + 1):
        distance_v, _ = spreads(mean_v, sd_v, Yv, k)
        distance_s, _ = spreads(mean_s, sd_s, Ys, k)
        known = xp.concatenate([distance_v, distance_s])
        rolling = xp.empty(len(distance_s))
        for day in sorted(set(test_origins)):
            usable = (origins <= day - lag) & (origins > day - lag - window)
            rolling[test_origins == day] = xp.quantile(known[usable], LEVEL)
        fixed = float(xp.quantile(distance_v, LEVEL))
        latest = origins.max()
        calibration[k] = round(float(xp.quantile(known[origins > latest - lag - window], LEVEL)), 4)
        for name, cut in (("raw", T.Z80), ("validation-fitted", fixed), ("rolling 12 months", rolling)):
            covered = distance_s <= cut
            out.append({"day": k, "range": name, "z": round(float(xp.mean(cut)), 4),
                        "cover80": round(float(covered.mean()), 4), "cover80_lo_hi": date_range(dates, covered),
                        "kupiec_p_one_per_date": round(kupiec_p(int(covered[single].sum()), len(single)), 4),
                        "dates": len(single)})
    for row in out:
        if row["day"] in T.REPORT_DAYS:
            print(f"day {row['day']:2} {row['range']:18} z {row['z']:.3f}  held {row['cover80']:.1%} "
                  f"(95% range {row['cover80_lo_hi'][0]:.1%}-{row['cover80_lo_hi'][1]:.1%})  "
                  f"Kupiec p {row['kupiec_p_one_per_date']:.3f} over {row['dates']} dates")
    record["interval_z80"] = {str(k): z for k, z in calibration.items()}
    record["calibration"] = {"method": f"conformal, refitted on the trailing {TRAILING_DAYS} days of known outcomes",
                             "level": LEVEL, "fitted_through": str(origins.max().date()), "test": out}
    G.save(WEIGHTS, model, record)
    print(f"saved the calibration into {WEIGHTS}")
    print(json.dumps(calibration))


if __name__ == "__main__":
    main()
