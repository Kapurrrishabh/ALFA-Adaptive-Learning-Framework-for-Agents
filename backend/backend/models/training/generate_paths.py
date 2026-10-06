#!/usr/bin/env python3
"""Write synthetic price paths for one instrument: new data, drawn from bars up to an as-of date.

The drawer is whichever `backend/models/scenarios.load` serves -- the return generator if its file records
beating GARCH, GARCH otherwise -- so the data written here comes from the same model the app draws with.
A sidecar JSON records what produced it: the drawer, the checkpoint's digest, the seed and the as-of date.
Synthetic rows that cannot be traced to their model are indistinguishable from real ones a year later.

The paths are for stress tests and for training other models on more histories than one market has lived
through. They are not forecasts of direction, and the sidecar says so.
"""
from backend.paths import ARTIFACTS, DATA

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from backend.models.agent import finance  # noqa: E402
from backend.models.serving import registry, scenarios  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol")
    parser.add_argument("--prices", default=str(DATA / "prices"))
    parser.add_argument("--checkpoint", default=str(ARTIFACTS / "returns.npz"))
    parser.add_argument("--as-of", default=None, help="last bar the paths may read; the last on file if unset")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--paths", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(DATA / "synthetic"))
    args = parser.parse_args()

    drawer = scenarios.load(Path(args.checkpoint))
    market = finance.Market(Path(args.prices), scenarios=drawer)
    held = args.symbol.upper()
    if held not in market.paths:
        raise SystemExit(f"no price file for {args.symbol} under {args.prices}")
    taken, close, drawn = market.draw_paths(held, args.steps, args.paths, np.random.default_rng(args.seed),
                                            args.as_of)
    priced = scenarios.priced(close, drawn)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{held}_{taken}_{drawer.name}_seed{args.seed}"
    with open(f"{stem}.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "step", "close", "log_return"])
        for path in range(args.paths):
            for step in range(args.steps):
                writer.writerow([path, step + 1, f"{priced[path, step]:.6f}", f"{drawn[path, step]:.8f}"])
    Path(f"{stem}.json").write_text(json.dumps(dict(
        symbol=held, as_of=str(taken), start_close=close, steps=args.steps, paths=args.paths, seed=args.seed,
        drawer=drawer.describe(), checkpoint=args.checkpoint, digest=registry.digest(args.checkpoint),
        synthetic=True, direction=scenarios.DIRECTION), indent=2, sort_keys=True))
    print(f"{args.paths} paths x {args.steps} steps of {held} from {taken} (close {close:.2f}) by "
          f"{drawer.name} -> {stem}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
