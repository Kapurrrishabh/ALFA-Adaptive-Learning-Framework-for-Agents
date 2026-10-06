#!/usr/bin/env python3
"""The paper's two data figures, rebuilt from the served model and the recorded measurements.

The fan is drawn as of 2 June 2025 by whichever drawer `backend/models/scenarios.load` serves, with the path
AAPL actually took after that date laid over it -- a date inside the test period, so the model had never
seen it. The learning curve is the B6 table from docs/BUILD_QUEUE.md, copied here as data.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.agent import finance  # noqa: E402
from backend.models import scenarios  # noqa: E402
from selfagent.data import prices  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
INK, MUTED, BLUE, ORANGE = "#1f2937", "#6b7280", "#1d4ed8", "#c2410c"
AS_OF, STEPS = "2025-06-02", 60

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})


def fan():
    drawer = scenarios.load(ROOT / "artifacts/returns.npz")
    market = finance.Market(ROOT / "data/prices", scenarios=drawer)
    taken, close, drawn = market.draw_paths("AAPL", STEPS, 2000, np.random.default_rng(0), AS_OF)
    dates, bars = prices.load_bars(market.paths["AAPL"])
    start = prices.index_on_or_before(dates, AS_OF)
    real = bars[start : start + STEPS + 1, 3]
    priced = scenarios.priced(close, drawn)
    steps = np.arange(STEPS + 1)
    band = lambda q: np.concatenate([[close], np.quantile(priced, q, axis=0)])  # noqa: E731

    figure, axis = plt.subplots(figsize=(3.45, 2.1))
    axis.fill_between(steps, band(0.05), band(0.95), color=BLUE, alpha=0.15, linewidth=0)
    axis.fill_between(steps, band(0.25), band(0.75), color=BLUE, alpha=0.30, linewidth=0)
    axis.plot(steps, band(0.5), color=BLUE, linewidth=1.2)
    axis.plot(steps[: len(real)], real, color=INK, linewidth=1.2)
    inside = ((real[1:] >= band(0.05)[1 : len(real)]) & (real[1:] <= band(0.95)[1 : len(real)])).mean()
    axis.text(STEPS + 1, band(0.95)[-1], "90%", color=BLUE, va="center", fontsize=7)
    axis.text(STEPS + 1, band(0.75)[-1], "50%", color=BLUE, va="center", fontsize=7)
    axis.text(STEPS + 1, real[-1], "realised", color=INK, va="center", fontsize=7)
    axis.set_xlim(0, STEPS + 12)
    axis.set_xlabel(f"sessions after {taken}")
    axis.set_ylabel("AAPL close (USD)")
    figure.tight_layout(pad=0.3)
    figure.savefig(OUT / "fan.pdf")
    print(f"fan: drawn by {drawer.name} as of {taken}; realised path inside the 90% band on {inside:.0%} "
          f"of {len(real) - 1} sessions")


def learning_curve():
    rows = np.array([10, 20, 40, 60, 80, 118])
    oracle = np.array([12.9, 12.0, 9.5, 8.5, 9.3, 9.6])
    agent = np.array([13.3, 10.4, 9.5, 9.0, 7.7, 6.3])
    figure, axis = plt.subplots(figsize=(3.45, 2.0))
    axis.plot(rows, oracle, color=BLUE, marker="o", markersize=3, linewidth=1.2, label="oracle labels, 60% bar")
    axis.plot(rows, agent, color=ORANGE, marker="s", markersize=3, linewidth=1.2, label="agent labels, 80% bar")
    axis.axhline(16.7, color=BLUE, linestyle=":", linewidth=0.9)
    axis.axhline(10.4, color=ORANGE, linestyle=":", linewidth=0.9)
    axis.text(119, 16.7, "fixed cut, 60%", color=MUTED, fontsize=6.5, va="bottom", ha="right")
    axis.text(119, 10.4, "fixed cut, 80%", color=MUTED, fontsize=6.5, va="bottom", ha="right")
    axis.set_xlabel("feedback rows in the log")
    axis.set_ylabel("miss (points)")
    axis.set_ylim(0, 19)
    axis.legend(frameon=False, fontsize=6.5, loc="lower left")
    figure.tight_layout(pad=0.3)
    figure.savefig(OUT / "learning_curve.pdf")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fan()
    learning_curve()
