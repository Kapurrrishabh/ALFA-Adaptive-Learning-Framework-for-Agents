#!/usr/bin/env python3
"""Which question is this, and how sure is the router? Measured before it is served.

Misrouting is the largest single group of wrong answers: of the 95 the agent judge called wrong, 14 were
buy advice answering an overbought question, 13 performance answering an outlook or volatility one, 6
drawdown answering an outlook one. The figure guard catches none of them, because every digit in a
wrong-intent answer is supported by the evidence it was handed. So the intent needs its own guard, and
this fits it.

Two scorers, because which one wins is a measurement and not an opinion. On intent among real questions
they tie: BM25 over the known phrasings, with no weights at all, gets 94.3%, and so does a cosine
against the served checkpoint's encoder. Both place a chat-register question as well as a clean one
(100% each), and both fail on sentence shape -- 71% and 68% on a frame training never saw -- which is
the blocker probe_intents.py already measured on the encoder alone.

On knowing a question is *not* one of ours they are not close, and that is what the gate is for. The
encoder's margin separates: at a 97% bar it refuses all 12 out-of-domain questions and still answers
88.5% of real ones at 97.9% routing precision. BM25's margin does not, because a query's unmatched words
cost it nothing -- "write me a poem about the sea" matches "tell me about the recent price action in X"
at a margin of 5.55, wider than most genuine questions score. So the served router is the semantic one,
reading the encoder the decoder already shares. The 12 out-of-domain questions are a small hand-written
set: enough to separate the two scorers, not enough to quote a rejection rate from.

The encoder to route with is the served checkpoint's, not the pretrained one it started from, and the
gap is small but real: 68% against 66% on novel frames, and 12/12 out-of-domain refused at 88.5%
coverage against 82.4%. Task fine-tuning cost the shape axis nothing here because `--freeze-encoder`
held every transformer block; only the four embedding tensors moved.

The gate is fitted the way B4 fits the answer cut: a stated bar on precision, solved for the margin that
delivers it, on half the questions and quoted on the other half. Half rather than by axis, because a cut
fitted on the axes that never fail is placed at the lowest margin there is and refuses nothing.

Out-of-domain questions have no right intent, so the only thing worth counting for them is whether the
turn ends in silence -- a margin under the cut, or the `unsupported` intent, both of which say nothing
rather than guess. They are in the fit for the same reason: fitted on real questions alone the bar is
already met by answering all of them, the cut lands at the lowest margin present, and the gate refuses
nothing at all.

With --payoff the decoder is loaded too and the point of the whole exercise is measured directly: the same
held-out question, asked once as typed and once rewritten into the trained phrasing the router picked,
scored on exact match against each arm's own correct answer. A misroute counts as a miss in the rewritten
arm, because a fluent answer to the wrong question is the failure this is trying to buy out of.

**It pays, and it is the largest frozen-weight win in the repo.** Over 200 held-out wordings, exact match
goes 33.5% as typed to 73.5% rewritten, with the weights loaded once and never touched. The semantic
router earns its keep here too, 73.5% against BM25's 65.5%, entirely through misrouting less on wordings
training never saw -- 47 against 65 -- which the tie on the full phrasing set hides because trained
phrasings dominate it and both place those perfectly.

What is left is nearly all routing. Of the 26.5% still missed, 23.5 points are the 47 misroutes and about
3 are the decoder getting a correctly-rewritten question wrong. So the ceiling on this approach is the
router's accuracy on novel sentence shapes, not the decoder's wording, and that is the number the next
piece of work has to move.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.agent import Router, combined, finance, lexical, semantic  # noqa: E402
from backend.agent.router import COVERAGE_WEIGHT  # noqa: E402
from check_answers import answer_rows, load_model  # noqa: E402
from prepare_advisory import OUTLOOK_CONCENTRATION  # noqa: E402
from selfagent import pretrained  # noqa: E402
from selfagent.data import advisory, prices  # noqa: E402
from selfagent.data.encode import build_sources  # noqa: E402
from selfagent.learn.abstain import Abstainer, precision_at  # noqa: E402
from selfagent.models import GroundedGenerator  # noqa: E402
from selfagent.tokenizer.vocab import CLS_ID, PAD_ID, SEP_ID  # noqa: E402
from selfagent.tokenizer.wordpiece import WordPiece, pretokenize  # noqa: E402

CASUAL = "casual"

# The pool now carries paraphrase shapes, so `frame` measures an unseen frame inside a family the pool
# does cover, and `shape` is the one measuring a family it does not. Read them as two different questions.
SHAPE = "shape"

# The cross the other two miss, and where a typed question actually lives: a shape the pool does not
# carry, typed the way somebody types it. `casual` renders trained shapes the pool holds verbatim and
# scores 100%, which read alone says register is solved when only register on familiar shapes is.
CASUAL_SHAPE = "casual shape"
AXES = (advisory.TRAINED, "word", "frame", "word and frame", CASUAL, SHAPE, CASUAL_SHAPE)

# One ticker for every question. The intent has to be read off the wording, and it is then stripped back
# out exactly as serving strips it, so the router is measured on the text it is actually given.
TICKER = "AAPL"

# Questions the agent holds no evidence for and no intent about. Kept deliberately close to finance in
# vocabulary -- "portfolio", "account", "tax" -- because an out-of-domain question about the weather is
# not the one that gets served by mistake.
OUT_OF_DOMAIN = (
    "what is the weather tomorrow ?",
    "how do i reset my password ?",
    "can you transfer money to my account ?",
    "what is the capital of france ?",
    "book me a flight to delhi .",
    "how much tax do i owe this year ?",
    "please close my demat account .",
    "who won the match last night ?",
    "write me a poem about the sea .",
    "what is my portfolio worth ?",
    "call my broker for me .",
    "translate this into hindi .",
)


def rows(seed):
    """(intent, axis, text) for every phrasing plus a chat-register rendering of each trained one.

    The ticker is stripped from all of them, because that is exactly what the router is given at serving
    time and leaving one in would make every question about another instrument slightly unfamiliar.
    """
    rng = np.random.default_rng(seed)
    asked = []
    for intent in advisory.INTENTS:
        for phrasing in range(advisory.phrasings(intent)):
            axis = advisory.novelty(intent, phrasing)
            asked.append((intent, axis, _asked(intent, phrasing)))
            if axis == advisory.TRAINED:
                asked.append((intent, CASUAL, _asked(intent, phrasing, rng)))
        for frame in advisory.held_out_paraphrases(intent):
            asked.append((intent, SHAPE, finance.without_subject(frame.format(t=TICKER), TICKER)))
            # Roughed up from the frame, not the finished question, so the ticker survives to be stripped.
            asked.append((intent, CASUAL_SHAPE,
                          finance.without_subject(advisory.casual(frame, rng).format(t=TICKER), TICKER)))
    return asked


def _asked(intent, phrasing, rng=None):
    return finance.without_subject(advisory.ask(intent, TICKER, phrasing, rng), TICKER)


def scorers(artifacts, checkpoint, known, weight):
    """The scorers to compare, each as (name, score function over the known phrasings)."""
    texts = [text for _, _, text in known]
    built = [("lexical", lexical(texts, pretokenize))]
    if checkpoint:
        tokenizer = WordPiece.load(artifacts / "tokenizer.json")
        config, weights = pretrained.load(artifacts / checkpoint)
        model = GroundedGenerator(config)
        # Only the encoder is loaded, whatever else the checkpoint holds: a router reads questions and
        # never writes an answer, so the decoder's weights would be dead here.
        model.load_pretrained_encoder(weights)
        model.eval()
        built.append(("semantic", semantic(texts, model, tokenizer, config.max_text_length)))
        built.append(("combined", combined(texts, model, tokenizer, config.max_text_length, weight)))
    return built


def measure(router, asked):
    """(axis, did it get the intent, margin) for every question, in the order asked."""
    routed = [(axis, router.route(text), intent) for intent, axis, text in asked]
    return [(axis, route.label == intent, route.margin) for axis, route, intent in routed]


def refused_out_of_domain(router, gate):
    """How many out-of-domain questions end in silence, and which ones do not."""
    survived = []
    for text in OUT_OF_DOMAIN:
        route = router.route(text)
        if bool(gate.answers([route.margin])[0]) and route.label != "unsupported":
            survived.append((text, route.label, route.margin))
    return len(OUT_OF_DOMAIN) - len(survived), survived


def held_out_questions(price_dir, router, rng, limit):
    """(typed question, rewritten question, evidence, gold answer, gold for the rewrite) per row.

    Built from snapshots rather than from the saved .npy splits, because the rewrite changes which
    phrasing is asked and the answer form rotates with the phrasing. Reading the stored gold would score
    the rewritten arm against another wording's answer and report a loss that is only bookkeeping.
    """
    plan = [(path, intent, phrasing)
            for path in sorted(Path(price_dir).glob("*.csv"))
            for intent in advisory.INTENTS
            for phrasing in range(advisory.phrasings(intent))
            if advisory.is_held_out(intent, phrasing)]
    # Sampled across every instrument rather than taken in order: the first few files in a row would
    # measure four snapshots and call it two hundred.
    picked = sorted(plan[index] for index in rng.permutation(len(plan))[:limit])

    built, at, taken = [], None, None
    for path, intent, phrasing in picked:
        if at != path:
            at, taken = path, _snapshot(path, rng)
        if taken is None:
            continue
        ticker, facts, shown, evidence = taken
        typed, gold = advisory.row(intent, ticker, facts, shown, phrasing)
        route = router.route(finance.without_subject(typed, ticker))
        # A misroute rewrites into another intent's wording, and then no answer to it is right.
        rewritten, instead = (advisory.row(intent, ticker, facts, shown, route.key)
                              if route.label == intent else (typed, None))
        built.append((typed, rewritten, evidence, gold, instead))
    return built


def _snapshot(path, rng):
    """(ticker, facts, text, evidence) at the last bar, or None where there is too little history.

    Confidences are drawn rather than read off the price head, the same as prepare_advisory does it: the
    decoder's job is to phrase whatever the head reports, so it should not depend on one checkpoint.
    """
    try:
        _, bars = prices.load_bars(path)
    except ValueError:
        return None
    if len(bars) < advisory.BARS_NEEDED:
        return None
    ticker = path.stem.upper()
    facts = advisory.snapshot(bars, len(bars) - 1)
    shown = advisory.as_text(facts)
    shown.update(advisory.risk_outlook(rng.dirichlet([OUTLOOK_CONCENTRATION] * 3)))
    return ticker, facts, shown, advisory.render_evidence(ticker, shown)


def as_split(tokenizer, rows, question_of, answer_of, length, budget):
    """One arm of the comparison as the (source, keep, target) arrays the scorer already reads."""
    source = np.zeros((len(rows), 1, length), dtype=np.uint16)
    keep = np.zeros((len(rows), 1, length), dtype=np.uint8)
    target = np.full((len(rows), budget), PAD_ID, dtype=np.uint16)
    for index, row in enumerate(rows):
        source[index], keep[index] = build_sources(
            tokenizer.encode(question_of(row))[: advisory.QUESTION_TOKENS],
            [tokenizer.encode(row[2])], 1, length)
        # A misrouted row has no correct answer at all; it keeps the typed gold so the array is well
        # formed and is scored as a miss regardless.
        ids = [CLS_ID] + tokenizer.encode(answer_of(row) or row[3])[: budget - 2] + [SEP_ID]
        target[index, : len(ids)] = ids
    return source, keep, target


def measure_payoff(artifacts, dataset, checkpoint, price_dir, router, limit, batch_size, seed):
    """Exact match on held-out wordings, asked as typed against asked as the router rewrites them."""
    tokenizer, model, config = load_model(artifacts, dataset, checkpoint)
    rng = np.random.default_rng(seed)
    rows = held_out_questions(price_dir, router, rng, limit)
    misrouted = sum(instead is None for *_, instead in rows)
    chosen = np.arange(len(rows))

    print(f"\npayoff on {len(rows)} held-out wordings, {misrouted} of them misrouted")
    for name, question_of, answer_of, blame_misroutes in (
            ("as typed  ", lambda row: row[0], lambda row: row[3], False),
            ("rewritten ", lambda row: row[1], lambda row: row[4], True)):
        split = as_split(tokenizer, rows, question_of, answer_of,
                         config.max_text_length, config.max_answer_length)
        scored = answer_rows(model, tokenizer, split, chosen, np.random.default_rng(seed),
                             batch_size=batch_size)
        # A misroute is only the rewritten arm's fault: the typed arm still asked the right question.
        right = [wanted.split() == answer.split() and not (blame_misroutes and rows[i][4] is None)
                 for i, (_, _, wanted, answer, _) in enumerate(scored)]
        print(f"  {name} exact match {sum(right)}/{len(right)} ({np.mean(right):.1%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--checkpoint", default="advisory_combined.npz",
                        help="encoder for the semantic scorer; empty to measure lexical alone")
    # Well above the 90% the router reaches by answering everything: a bar under that rate is met by the
    # lowest margin present, the cut lands there, and the gate refuses nothing. 97% is where the semantic
    # margin rejects every out-of-domain question, and asking for 99% only costs coverage.
    parser.add_argument("--wanted", type=float, default=0.97,
                        help="the routing precision the saved gate is solved for")
    parser.add_argument("--bars", type=float, nargs="+", default=(0.95, 0.97, 0.99),
                        help="the bars the table sweeps, to show the curve the cut is placed on")
    parser.add_argument("--seed", type=int, default=0, help="seeds the chat-register rewrites")
    parser.add_argument("--save", default="route_gate.json",
                        help="where the fitted gate is written; empty to fit without saving")
    parser.add_argument("--payoff", action="store_true",
                        help="also load the decoder and score held-out wordings rewritten against typed")
    parser.add_argument("--dataset", default="advisory_combined", help="the decoder --payoff scores")
    parser.add_argument("--prices", default="data/prices")
    parser.add_argument("--pool", choices=("wide", "trained"), default="wide",
                        help="'trained' drops the routing paraphrases: the control the fix is read against")
    parser.add_argument("--weight", type=float, default=COVERAGE_WEIGHT,
                        help="how much of the combined score is term coverage rather than cosine")
    parser.add_argument("--payoff-rows", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    known = finance.examples(args.pool == "wide")
    asked = rows(args.seed)
    print(f"{len(known)} known phrasings over {len(advisory.INTENTS)} intents, "
          f"{len(asked)} questions asked, {len(OUT_OF_DOMAIN)} of them out of domain\n")

    fitted = {}
    for name, score in scorers(artifacts, args.checkpoint, known, args.weight):
        router = Router(known, score)
        measured = measure(router, asked)
        overall = np.mean([right for _, right, _ in measured])
        print(f"{name}: {overall:.1%} of questions routed to the right intent "
              f"(chance {1 / len(advisory.INTENTS):.0%})")
        for axis in AXES:
            picked = [right for kind, right, _ in measured if kind == axis]
            print(f"  {axis:14s} {sum(picked):>3}/{len(picked):<3} ({np.mean(picked):.0%})")

        # Out-of-domain rows join the fit as rows that must not be answered. Without them the bar is met
        # by answering everything, so the cut lands at the lowest margin there is and refuses nothing.
        pool = measured + [("out of domain", False, router.route(text).margin)
                           for text in OUT_OF_DOMAIN]
        print(f"  {'asked for':>11}{'cut':>9}{'held-out right':>17}{'coverage':>11}{'refused':>12}")
        for wanted in sorted(set(args.bars) | {args.wanted}):
            gate, precision, coverage, base = _fit(pool, wanted, args.seed)
            refused, survived = refused_out_of_domain(router, gate)
            print(f"  {wanted:>11.0%}{gate.cut:>9.3f}{precision:>16.1%}{coverage:>11.1%}"
                  f"{refused:>9}/{len(OUT_OF_DOMAIN)}")
            if wanted != args.wanted:
                continue
            fitted[name] = gate
            print(f"    answering everything on the same rows: {base:.1%}")
            for text, label, margin in survived:
                print(f"    SERVED {text!r} as {label} at a margin of {margin:.3f}")

        if args.payoff:
            measure_payoff(artifacts, args.dataset, args.checkpoint, args.prices, router,
                           args.payoff_rows, args.batch_size, args.seed)

    if args.save:
        name = next(n for n in ("combined", "semantic", "lexical") if n in fitted)
        fitted[name].save(artifacts / args.save)
        print(f"\n{name} gate -> {artifacts / args.save}")


def _fit(measured, wanted, seed):
    """(gate, precision, coverage, base) with the cut solved on half the rows and read off the other.

    Halved at random rather than by axis: the axes that never fail put the cut at the lowest margin
    there is, and a gate that refuses nothing is not a gate.
    """
    order = np.random.default_rng(seed).permutation(len(measured))
    half = len(measured) // 2
    margins = np.array([margin for _, _, margin in measured])
    right = np.array([is_right for _, is_right, _ in measured], dtype=bool)
    gate = Abstainer.fit(margins[order[:half]], right[order[:half]], wanted)
    held = order[half:]
    return (gate, *precision_at(margins[held], right[held], gate.cut), right[held].mean())


if __name__ == "__main__":
    sys.exit(main())
