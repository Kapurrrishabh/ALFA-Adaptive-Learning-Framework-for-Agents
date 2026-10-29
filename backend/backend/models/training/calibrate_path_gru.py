#!/usr/bin/env python3
"""Make the path GRU's 80% range hold 80%, and test it only on outcomes its weights never trained on.

Trained on likelihood, the GRU's 10-day 80% range held 84.6% of test outcomes (the 20 dates scored to
2026-09-21): too wide, so the "1 in 10 ends lower" label on its low path overstated how far prices fall.
Conformal calibration fixes the width without retraining: the multiple of the GRU's own spread that 80% of
known outcomes fell within becomes the range.

The served weights were retrained on everything up to VAL_END, so only test outcomes are out of sample. At
each test date the multiple is refitted on the trailing year of test outcomes known by then, and a date is
scored only once a full year of them exists. The range fitted once on the validation years is kept for
comparison, but it is fitted on residuals the weights trained on, so its shortfall (77.2%) is not only a
change of regime. Every range carries a date-resampled 95% range and its number of dates, because stocks on
one date move together. Kupiec's test on one stock per date is not used: at 20 dates it passes 65% to 95%.

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


def rolling_cut(distance, origins):
    """Each origin's multiple, refitted on the trailing TRAILING_DAYS of outcomes known by then; NaN until a full
    window of them is known. `distance` must hold only outcomes the weights never trained on."""
    lag, window = pd.Timedelta(days=KNOWN_AFTER_DAYS), pd.Timedelta(days=TRAILING_DAYS)
    days = sorted(set(origins[origins >= origins.min() + lag + window]))
    if not days:
        raise ValueError(f"origins {origins.min().date()} to {origins.max().date()} never have {TRAILING_DAYS} days "
                         f"of known outcomes behind them; score a longer test period")
    cut = xp.full(len(distance), xp.nan)
    for day in days:
        usable = (origins <= day - lag) & (origins > day - lag - window)
        cut[origins == day] = xp.quantile(distance[usable], LEVEL)
    return cut


def main():
    from backend.database.sources.panel import index_constituents, universe_panel
    import yfinance as yf
    model, record = G.load(WEIGHTS)
    panel = universe_panel("nifty500", "10y")
    market = yf.Ticker("^NSEI").history(period="10y", auto_adjust=True)["Close"]
    market.index = market.index.tz_localize(None).normalize()
    scored = [s for s in index_constituents("nifty200")["symbol"] if s in panel.close.columns]
    Xv, Yv, _ = T.samples(panel, scored, market, VAL_START, VAL_END, SCORE_STRIDE)
    Xs, Ys, rs = T.samples(panel, scored, market, TEST_START, None, SCORE_STRIDE)
    mean_v, sd_v = T.predict(model, Xv)
    mean_s, sd_s = T.predict(model, Xs)
    origins = pd.to_datetime([r["date"] for r in rs])
    lag, window = pd.Timedelta(days=KNOWN_AFTER_DAYS), pd.Timedelta(days=TRAILING_DAYS)
    calibration, out = {}, []
    for k in range(1, G.HORIZON + 1):
        distance_v, _ = spreads(mean_v, sd_v, Yv, k)
        distance_s, _ = spreads(mean_s, sd_s, Ys, k)
        rolling = rolling_cut(distance_s, origins)
        full_year = ~xp.isnan(rolling)
        fixed = float(xp.quantile(distance_v, LEVEL))
        latest = origins.max()
        calibration[k] = round(float(xp.quantile(distance_s[origins > latest - lag - window], LEVEL)), 4)
        for name, cut in (("raw", T.Z80), ("validation-fitted", fixed), ("rolling 12 months", rolling[full_year])):
            covered = distance_s[full_year] <= cut
            out.append({"day": k, "range": name, "z": round(float(xp.mean(cut)), 4),
                        "cover80": round(float(covered.mean()), 4), "cover80_lo_hi": date_range(origins[full_year], covered),
                        "dates": origins[full_year].nunique(), "forecasts": int(full_year.sum())})
    for row in out:
        if row["day"] in T.REPORT_DAYS:
            print(f"day {row['day']:2} {row['range']:18} z {row['z']:.3f}  held {row['cover80']:.1%} "
                  f"(95% range {row['cover80_lo_hi'][0]:.1%}-{row['cover80_lo_hi'][1]:.1%}) "
                  f"over {row['forecasts']} forecasts on {row['dates']} dates")
    record["interval_z80"] = {str(k): z for k, z in calibration.items()}
    record["calibration"] = {"method": f"conformal, refitted on the trailing {TRAILING_DAYS} days of known test outcomes",
                             "level": LEVEL, "fitted_through": str(origins.max().date()), "test": out}
    G.save(WEIGHTS, model, record)
    print(f"saved the calibration into {WEIGHTS}")
    print(json.dumps(calibration))


if __name__ == "__main__":
    main()
