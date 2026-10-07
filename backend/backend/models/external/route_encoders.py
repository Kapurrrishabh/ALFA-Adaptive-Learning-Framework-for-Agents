"""Does a pretrained sentence encoder route the agent's questions better than ALFA's own? Measured, not assumed.

The same questions and the same gate fit as models/training/route.py, which stays free of deep-learning
libraries: every phrasing, a chat-register rendering of each trained one, the held-out frames in sentence
shapes the router never saw, and twelve questions that are not ours at all. A scorer is worth serving only if
it routes more questions correctly and still refuses the out-of-domain ones at the same precision bar.

    python -m backend.models.external.route_encoders --save route_gate_bge.json
"""
import argparse
from pathlib import Path

from backend.models.agent import Router, finance, with_coverage
from backend.models.data import advisory
from backend.models.external import sentence
from backend.models.training import route
from backend.paths import ARTIFACTS


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wanted", type=float, default=0.97)
    ap.add_argument("--weights", type=float, nargs="*", default=(0.1, 0.25, 0.5),
                    help="how much term coverage to add to the pretrained cosine")
    ap.add_argument("--save", default="", help="where to write the last scorer's gate, if it wins")
    args = ap.parse_args()
    known = finance.examples(True)
    asked = route.rows(0)
    texts = [text for _, _, text in known]
    scorers = route.scorers(ARTIFACTS, "advisory_combined.npz", known, route.COVERAGE_WEIGHT)
    scorers = [(name, score) for name, score in scorers if name == "combined"]
    cosine = sentence.similarity(texts)
    scorers.append((f"pretrained ({sentence.ENCODER_ID})", cosine))
    for weight in args.weights:
        scorers.append((f"pretrained + coverage x{weight}", with_coverage(texts, cosine, weight)))
    results = {}
    for name, score in scorers:
        router = Router(known, score)
        measured = route.measure(router, asked)
        pool = measured + [("out of domain", False, router.route(text).margin) for text in route.OUT_OF_DOMAIN]
        gate, precision, coverage, _ = route._fit(pool, args.wanted, 0)
        refused, survived = route.refused_out_of_domain(router, gate)
        right = sum(ok for _, ok, _ in measured)
        print(f"\n{name}: {right}/{len(measured)} routed right ({right / len(measured):.1%})")
        for axis in route.AXES:
            picked = [ok for kind, ok, _ in measured if kind == axis]
            print(f"  {axis:14s} {sum(picked):>3}/{len(picked):<3} ({sum(picked) / len(picked):.0%})")
        print(f"  gate for {args.wanted:.0%}: cut {gate.cut:.3f}, held-out precision {precision:.1%}, "
              f"answers {coverage:.1%}, refuses {refused}/{len(route.OUT_OF_DOMAIN)} out of domain")
        for text, label, margin in survived:
            print(f"    SERVED {text!r} as {label} at {margin:.3f}")
        results[name] = (gate, right, coverage, refused)
    if args.save:
        name = scorers[-1][0]
        results[name][0].save(ARTIFACTS / args.save)
        print(f"\n{name} gate -> {ARTIFACTS / args.save}")


if __name__ == "__main__":
    main()
