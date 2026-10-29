#!/usr/bin/env python3
"""Does the gold fan's 10-day 80% range hold 80%? The chart draws it; nothing had tested it at 10 days.

ALFA's return generator was scored on next-day returns, where it beat GARCH(1,1)-t. The fan on the chart is
ten of its days chained, and the record says so rather than claiming the one-day result for it. This backtests
the chained fan the way calibrate_path_gru.py backtests the GRU: twelve large caps, an origin every 21 trading
days from the test start, the 10th and 90th percentile of 500 drawn paths against the close ten days later.
The RiskMetrics band (the app's grey band) is scored on the same rows, so the two can be compared directly.

    python -m backend.models.training.backtest_alfa_fan
"""
import json
import math

from backend.advisory.calculators.performance import FORECAST_STOCKS
from backend.models.core.backend import xp
from backend.models.data import returns
from backend.models.learning.protocol import SCORE_STRIDE, TEST_START
from backend.models.serving import alfa_fan
from backend.models.training import train_path_gru as T
from backend.models.training.calibrate_path_gru import date_range, kupiec_p, one_per_date

OUT = alfa_fan.BACKTEST
DAYS = alfa_fan.STEPS


def main():
    import pandas as pd
    from backend.database.sources.panel import universe_panel
    panel = universe_panel("nifty500", "10y")
    rows = []
    for symbol in [s.split(".")[0] for s in FORECAST_STOCKS]:
        close = panel.close[symbol].dropna()
        sd = T.ewma_sd(returns.log_returns(close.to_numpy(dtype=float)))
        start = close.index.searchsorted(pd.Timestamp(TEST_START))
        for at in range(start, len(close) - DAYS, SCORE_STRIDE):
            fan = alfa_fan.fan(close.iloc[:at + 1])
            later = float(close.iloc[at + DAYS])
            band = T.Z80 * float(sd[at - 1]) * math.sqrt(DAYS)
            last = float(close.iloc[at])
            rows.append({"symbol": symbol, "date": str(close.index[at].date()), "later": later,
                         "alfa_in": fan["q10"][-1] <= later <= fan["q90"][-1],
                         "band_in": last * math.exp(-band) <= later <= last * math.exp(band),
                         "alfa_width": math.log(fan["q90"][-1] / fan["q10"][-1]), "band_width": 2 * band})
        print(f"{symbol}: {sum(r['symbol'] == symbol for r in rows)} origins", flush=True)
    dates = [r["date"] for r in rows]
    single = one_per_date(rows)
    out = {}
    for name in ("alfa", "band"):
        covered = xp.asarray([r[f"{name}_in"] for r in rows])
        out[name] = {"cover80": round(float(covered.mean()), 4), "cover80_lo_hi": date_range(dates, covered),
                     "kupiec_p_one_per_date": round(kupiec_p(int(covered[single].sum()), len(single)), 4),
                     "mean_log_width": round(float(xp.mean([r[f"{name}_width"] for r in rows])), 4)}
        print(f"{name:5} held {out[name]['cover80']:.1%} (95% range {out[name]['cover80_lo_hi'][0]:.1%}-"
              f"{out[name]['cover80_lo_hi'][1]:.1%}), Kupiec p {out[name]['kupiec_p_one_per_date']:.3f}, "
              f"width {out[name]['mean_log_width']:.3f}")
    OUT.write_text(json.dumps({"days": DAYS, "rows": len(rows), "dates": len(single), "from": TEST_START, **out}, indent=1))


if __name__ == "__main__":
    main()
