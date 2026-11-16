"""Does a pretrained sentence encoder route the agent's questions better than ALFA's own? Measured, not assumed.

The same questions and the same gate fit as models/training/route.py, which stays free of deep-learning
libraries: every phrasing, a chat-register rendering of each trained one, the held-out frames in sentence
shapes the router never saw, and twelve questions that are not ours at all. A scorer is worth serving only if
it routes more questions correctly and still refuses the out-of-domain ones at the same precision bar.

    python -m backend.models.external.route_encoders --save route_gate_bge.json
    python -m backend.models.external.route_encoders --wanted 0.99 --weights --facts
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from backend.models.agent import Router, finance, with_coverage
from backend.models.data import advisory
from backend.models.external import sentence
from backend.models.training import route
from backend.paths import ARTIFACTS, DATA


# An old question the figures do not carry, placed on the facts route, is placed right: answering it is the route's purpose
ANSWERED_BY_FACTS = {(finance.UNANSWERABLE, finance.FACTS)}
# Typed price, sell and facts questions, many with facts words, by someone who did not pick the router's words: a dev
# half the router was tuned on and a test half only reported
HELD_OUT = Path(__file__).with_name("routing_heldout.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wanted", type=float, default=0.97)
    ap.add_argument("--weights", type=float, nargs="*", default=(0.1, 0.25, 0.5),
                    help="how much term coverage to add to the pretrained cosine")
    ap.add_argument("--save", default="", help="where to write the last scorer's gate, if it wins")
    ap.add_argument("--facts", action="store_true",
                    help="route as served with the facts route on, and measure that label on the stock Q&A set's "
                         "held-out rows")
    args = ap.parse_args()
    known = finance.examples(True, facts=args.facts)
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
        router = finance.FactsFirst(known, score) if args.facts else Router(known, score)
        measured = route.measure(router, asked, ANSWERED_BY_FACTS if args.facts else ())
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
    if args.facts:
        name, score = scorers[-1]
        print(f"\nfacts label, {name}")
        facts_label(finance.FactsFirst(known, score), results[name][0], asked)
    if args.save:
        name = scorers[-1][0]
        results[name][0].save(ARTIFACTS / args.save)
        print(f"\n{name} gate -> {ARTIFACTS / args.save}")


def facts_label(router, gate, asked):
    """Precision and coverage of the facts label past `gate` on each held-out split of the stock Q&A set, and how many
    of the short templates' test split, which no router reads, and of HELD_OUT are placed on their label past it.

    Its questions are routed as served, with only the symbol taken out. The old intents' questions and the
    out-of-domain ones are what it must not take, but for those in ANSWERED_BY_FACTS, so its precision is read against them.
    """
    others = [(intent, router.route(text)) for intent, _, text in asked] + \
        [("out of domain", router.route(text)) for text in route.OUT_OF_DOMAIN]
    taken = Counter(intent for intent, placed in others
                    if placed.label == finance.FACTS and gate.answers([placed.margin])[0])
    also = sum(n for intent, n in taken.items() if (intent, finance.FACTS) in ANSWERED_BY_FACTS)
    wrong = sum(taken.values()) - also
    print(f"  old or out-of-domain questions answered as facts: {sum(taken.values())}/{len(others)} {dict(taken)}, "
          f"{also} of them rightly")
    for split in ("validation", "test"):
        rows = finance.stock_questions(split)
        placed = [router.route(finance.without_subject(row["question"], row["symbol"])) for row in rows]
        answered = [p.label for p in placed if gate.answers([p.margin])[0]]
        right = answered.count(finance.FACTS)
        precision = (right + also) / (right + also + wrong) if right + also + wrong else float("nan")
        print(f"  {split}: {right}/{len(rows)} answered as facts ({right / len(rows):.1%}), precision {precision:.1%}, "
              f"refused {len(rows) - len(answered)}, answered as another intent "
              f"{dict(Counter(label for label in answered if label != finance.FACTS))}")
    placed = [(label, router.route(finance.without_subject(question.format(t=route.TICKER), route.TICKER)))
              for label, question in finance.short_templates("test")]
    past = [(label, p.label) for label, p in placed if gate.answers([p.margin])[0]]
    print(f"  short templates, test split: {sum(a == b for a, b in past)}/{len(placed)} placed right past the cut, "
          f"{sum(a != b for a, b in past)} wrong past it")
    held_out(router, gate, served_resolve())


def held_out(router, gate, resolve):
    """Each half of HELD_OUT placed past `gate`, and how many questions the word rule names wrongly go to facts."""
    for half, rows in json.loads(HELD_OUT.read_text(encoding="utf-8")).items():
        texts = [finance.without_subject(row["question"], ticker) if (ticker := resolve(row["question"]))
                 else row["question"] for row in rows]
        placed = [(row["label"], finance.asks_facts(text), router.route(text)) for row, text in zip(rows, texts)]
        past = [(label, named, p.label) for label, named, p in placed if gate.answers([p.margin])[0]]
        named = sum(n for label, n, _ in placed if label != finance.FACTS)
        print(f"  held-out {half}: {sum(a == b for a, _, b in past)}/{len(rows)} placed right past the cut, "
              f"{sum(a != b for a, _, b in past)} wrong past it; the word rule names {named} of the "
              f"{sum(label != finance.FACTS for label, _, _ in placed)} other questions, and "
              f"{sum(n and a != finance.FACTS and b == finance.FACTS for a, n, b in past)} of them go to facts past "
              "the cut")


def served_resolve():
    """The share a question names, read as the live agent reads it, with no price ever fetched."""
    from backend.models.agent.run.ask import listed_names, nse_shares

    def unfetched(ticker, path):
        raise RuntimeError(f"route_encoders reads the price files on disk only, and {ticker} has none")
    return finance.Market(DATA / "prices", symbols=DATA / "raw/sec_edgar/company_tickers.json",
                          listed=listed_names(), shares=nse_shares(), fetch=unfetched).resolve


if __name__ == "__main__":
    main()
