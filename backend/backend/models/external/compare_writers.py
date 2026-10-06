"""Which open language model writes the analyst note most faithfully?

The same fact sheets (24 of the most-traded Nifty 200 stocks) go to every candidate, one model
loaded at a time. A note counts as faithful only if `advisory.analyst.check` passes it: no figure
that is not in its facts, and no claim about direction. Ties go to the faster, smaller model.

    python -m backend.models.external.compare_writers
"""
import argparse
import gc
import json
import logging
import resource
import time
from argparse import Namespace

from backend.advisory import analyst
from backend.database.sources.provider import DataUnavailable
from backend.models.external.language import OpenLM
from backend.paths import DATA

CANDIDATES = ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen3-1.7B", "HuggingFaceTB/SmolLM3-3B", "microsoft/Phi-4-mini-instruct")
INPUTS = DATA / "cache" / "analyst_inputs.json"     # raw model outputs, so wording changes need no recompute
OUT = DATA / "cache" / "writer_comparison.json"


def build_facts(n: int):
    from backend.api.hub import Hub
    from backend.cli import DEFAULT_DB, _orchestrator
    hub = Hub(_orchestrator(Namespace(db=DEFAULT_DB, demo=False)))
    p = hub.panel()
    traded = (p.close * p.volume).iloc[-250:].mean()
    names = [s for s in hub.names().query("nifty200")["symbol"] if s in traded.index]
    sheets = []
    for sym in sorted(names, key=lambda s: -traded[s]):
        if len(sheets) == n:
            break
        try:
            t, v = hub.technical(sym, bars=21), hub.verdict(sym)
            sheets.append({"symbol": sym, "close": t["close"][-1], "as_of": t["as_of"], "alfa": hub.ai_forecast(sym, "alfa"),
                           "gru": hub.ai_forecast(sym, "gru"), "band": t["cone"][-1],
                           "verdict": {"stop": t["stop"], "rank": v["momentum_rank"], "universe": v["universe_size"],
                                       "verdict": v["verdict"]}})
        except DataUnavailable as exc:      # e.g. a recent listing has too little history for the models
            logging.info("skip %s: %s", sym, exc)
            continue
        logging.info("facts for %s", sym)
    INPUTS.write_text(json.dumps(sheets, ensure_ascii=False, default=float))
    return sheets


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stocks", type=int, default=24)
    ap.add_argument("--models", nargs="*", default=list(CANDIDATES))
    ap.add_argument("--rescore", action="store_true", help="re-check the saved drafts with the current checker")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    raw = json.loads(INPUTS.read_text()) if INPUTS.exists() else build_facts(args.stocks)
    sheets = [analyst.facts(r["symbol"], r["close"], r["as_of"], r["alfa"], r["gru"], r["band"], r["verdict"]) for r in raw]
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    if args.rescore:
        by_stock = {f["stock"]: f for f in sheets}
        for r in results.values():
            for n in r["notes"]:
                n["rejected"] = analyst.check(n["draft"], by_stock[n["stock"]])
            r["faithful"] = sum(not n["rejected"] for n in r["notes"])
            r["invented_figure"] = sum(any("figure" in x for x in n["rejected"]) for n in r["notes"])
            r["claimed_direction"] = sum(any("direction" in x for x in n["rejected"]) for n in r["notes"])
        OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))
        args.models = []
    for model_id in args.models:
        writer, rows, started = OpenLM(model_id), [], time.time()
        for f in sheets:
            t0 = time.time()
            out = analyst.write(f, writer)
            rows.append({"stock": f["stock"], "seconds": round(time.time() - t0, 1), "rejected": out["draft_rejected"],
                         "draft": out["draft"]})
        faithful = sum(not r["rejected"] for r in rows)
        results[model_id] = {"faithful": faithful, "of": len(rows),
                             "invented_figure": sum(any("figure" in x for x in r["rejected"]) for r in rows),
                             "claimed_direction": sum(any("direction" in x for x in r["rejected"]) for r in rows),
                             "seconds_per_note": round((time.time() - started) / len(rows), 1),
                             "peak_rss_gb_so_far": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 1),
                             "notes": rows}
        logging.info("%s: %d/%d faithful, %.1fs per note", model_id, faithful, len(rows), results[model_id]["seconds_per_note"])
        OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))
        del writer
        gc.collect()
    for m, r in results.items():
        print(f"{m:40s} faithful {r['faithful']}/{r['of']}  invented {r['invented_figure']}  direction {r['claimed_direction']}  "
              f"{r['seconds_per_note']}s/note")


if __name__ == "__main__":
    main()
