#!/usr/bin/env python3
"""Type a question, get an answer or a named refusal. No network, one process, one checkpoint.

This is C1's gate: the four stages the architecture names -- route, assemble, generate, screen -- wired
into one call. Everything it uses was measured somewhere else and is loaded here rather than chosen: the
routing cut comes from route.py, the answer cut and the calibrator from the feedback log the way chat.py
refits them, and the evidence is the price files on disk with the same as-of rule the training snapshots
obey.

Run it with no arguments and it asks the questions that reach each of the five ways a turn can stop, so
the refusals are visible next to the answers instead of being taken on trust. A refusal names its stage;
none of them is a generic apology, because "i could not understand you" and "i do not hold that figure"
are different faults with different fixes.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import models  # noqa: E402
from backend.agent import Agent, Router, combined, finance  # noqa: E402
from chat import adapt  # noqa: E402
from check_answers import load_model  # noqa: E402
from selfagent.learn import FeedbackLog  # noqa: E402
from selfagent.learn.abstain import Abstainer  # noqa: E402

# One question per stage that can stop a turn, plus two that should get through. Kept here rather than in
# a README so the demo and the claim about it cannot drift apart.
DEMO = (
    "how has AAPL been doing over the last month ?",
    "hows msft been doin lately",
    "is AAPL overbought right now ?",
    "how risky is AAPL over the next week ?",
    "how is TSLA doing ?",
    "write me a poem about the sea .",
)


def build(artifacts, dataset, checkpoint, prices, gate, log, wanted, warmup, price_head):
    """The served agent, with every threshold loaded from what solved for it."""
    tokenizer, model, config = load_model(artifacts, dataset, checkpoint)
    known = finance.examples()
    router = Router(known, combined(
        [text for _, _, text in known], model, tokenizer, config.max_text_length))
    with FeedbackLog(log) as feedback:
        abstainer, calibrator, learned = adapt(feedback, wanted, warmup)
    advisor = models.load(Path(price_head)) if price_head else None
    return Agent(finance, finance.Market(Path(prices), advisor), router, Abstainer.load(gate), tokenizer,
                 model, config, abstainer, calibrator,
                 np.random.default_rng(config.seed)), abstainer, learned


def show(turn):
    print(f"\n> {turn.question}")
    if turn.intent:
        rewritten = f", asked as {turn.asked!r}" if turn.asked else ""
        print(f"  routed to {turn.intent} (margin {turn.margin:.3f}){rewritten}")
    if turn.evidence:
        print(f"  evidence  {turn.evidence}")
    if turn.answer and turn.answer != turn.served:
        print(f"  wrote     {turn.answer!r}")
    stated = "" if turn.stated is None else f", stated {turn.stated:.0%} likely right"
    sure = "" if turn.confidence != turn.confidence else f" (confidence {turn.confidence:.4f}{stated})"
    print(f"  {'ANSWER' if turn.spoke else 'QUIET '}    {turn.served}{sure}"
          + ("" if turn.spoke else f"  [{turn.because}]"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*", help="one question; omit for the demo set")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--dataset", default="advisory_combined")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--prices", default="data/prices")
    parser.add_argument("--gate", default="artifacts/route_gate.json",
                        help="the routing cut route.py solved for")
    parser.add_argument("--log", default="artifacts/chat.sqlite",
                        help="the feedback log the answer cut and the calibrator are refitted from")
    parser.add_argument("--price-head", default="artifacts/price_head.npz",
                        help="the artifact the week-ahead outlook comes from; empty leaves the risk "
                             "intent with nothing to quote, which is a refusal not a guess")
    parser.add_argument("--as-of", default=None, help="the last date a snapshot may read")
    parser.add_argument("--wanted", type=float, default=0.6, help="the stated chance of being right")
    parser.add_argument("--warmup", type=int, default=10,
                        help="judged rows before a solved cut replaces the hand-picked one")
    args = parser.parse_args()

    agent, abstainer, learned = build(Path(args.artifacts), args.dataset, args.checkpoint, args.prices,
                                      args.gate, args.log, args.wanted, args.warmup, args.price_head)
    # `expected` rather than `wanted`: the 60% bar is not reachable on this checkpoint, so the fit returns
    # the most precise cut its coverage floor allows and reports the shortfall. Printing the bar alone
    # would claim a precision nothing measured.
    solved = (f"solved for {abstainer.wanted:.0%}, expects {abstainer.expected:.0%} on "
              f"{abstainer.coverage:.0%}" if learned else "still hand-picked, too little feedback")
    advisor = agent.market.advisor
    print(f"routing cut {agent.gate.cut:.3f} for {agent.gate.wanted:.0%} routing precision; "
          f"answer cut {abstainer.cut:.5f} ({solved})")
    # Printed because the artifact, not this script, decides whether the head or the arithmetic answers.
    print(f"week-ahead outlook from {advisor.version if advisor else 'nothing loaded'}")

    questions = [" ".join(args.question)] if args.question else DEMO
    turns = [agent.answer(question, as_of=args.as_of) for question in questions]
    for turn in turns:
        show(turn)
    spoke = sum(turn.spoke for turn in turns)
    print(f"\nspoke on {spoke} of {len(turns)}; quiet because "
          + ", ".join(sorted({turn.because for turn in turns if not turn.spoke})))


if __name__ == "__main__":
    sys.exit(main())
