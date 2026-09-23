#!/usr/bin/env python3
"""One conversation turn at a time, with the adaptation refitted after every turn.

This is the thesis loop run in the small: ask, answer, judge, adapt, and show what moved. Nothing here
computes a gradient. What changes between turn 1 and turn 20 is the abstention cut and the calibrator
behind the stated chance of being right, both solved from the feedback log while the transformer's
weights are loaded once and never touched.

**What adapts is when the agent speaks, not what it says.** Best-of-k reranking was measured in B5 and
does not pay on this model — eight samples give 1.77 distinct answers — so there is nothing for feedback
to change about the wording. The wording comes from frozen weights; the judgement about whether it is
worth saying comes from the log. That is a smaller claim than "the agent learns to answer better", and
it is the one the measurements support.

Two deliberate limits:

  the questions come from the dataset, not from a person. A typed question arrives with no evidence, and
  deciding which passages to put in front of the decoder is C2's retriever and C1's context assembler.
  Guessing at it here would leave two rules for assembling context and the answer would depend on which
  script you ran.

  the log is its own file — artifacts/chat.sqlite, not the feedback.sqlite that B2 through B6 were
  measured on — so replaying a session cannot move a published number.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_answers import answer_rows, load_model  # noqa: E402
from fit_abstention import HAND_PICKED  # noqa: E402
from selfagent.data import advisory  # noqa: E402
from selfagent.learn import ORACLE, FeedbackLog  # noqa: E402
from selfagent.learn.abstain import Abstainer  # noqa: E402
from selfagent.learn.calibrate import Calibrator  # noqa: E402
from selfagent.learn.teacher import OracleTeacher  # noqa: E402
from train_generator import load_split  # noqa: E402


def adapt(feedback, wanted, warmup, labeller=ORACLE):
    """(abstainer, calibrator, learned) to serve the next turn with, refitted from the log as it stands.

    Below `warmup` judged rows, or before the log holds both a right and a wrong answer, the hand-picked
    cut is served and `learned` says so. A threshold solved from three rows is a coincidence, and serving
    one while calling it learned is the one thing this loop must not do.

    One labeller per fit: a typed question has no gold answer, so the server's judge is the agent rather
    than the oracle, and a cut fitted across both would be fitted to two different standards at once.
    """
    rows = feedback.rows(labeller=labeller)
    right = np.array([row["is_right"] for row in rows], dtype=bool)
    if len(rows) < warmup or not right.any() or right.all():
        return Abstainer(HAND_PICKED), None, False
    confidence = np.array([row["confidence"] for row in rows])
    return Abstainer.fit(confidence, right, wanted), Calibrator().fit(confidence, right), True


def converse(model, tokenizer, split, chosen, rng, feedback, wanted, warmup, batch_size):
    """Run the turns in order, adapting between each one, and report the session."""
    oracle = OracleTeacher(advisory.UNSUPPORTED)
    abstainer, calibrator, learned = adapt(feedback, wanted, warmup)
    opening, spoke, spoke_right, right = abstainer.cut, 0, 0, 0

    for turn, row in enumerate(chosen, start=1):
        question, evidence, gold, answer, sure = answer_rows(
            model, tokenizer, split, np.array([row]), rng, False, batch_size)[0]
        # The cut in force is the one fitted before this turn, never one fitted on the turn it decides.
        speaks = bool(abstainer.answers([sure])[0])
        is_right, why = oracle.judge(question, evidence, answer, gold)
        feedback.append(datetime.now(timezone.utc).isoformat(timespec="seconds"), question, evidence,
                        answer, sure, is_right, oracle.name, why, turn=turn, spoke=speaks)
        spoke += speaks
        spoke_right += speaks and is_right
        right += is_right

        stated = f"{calibrator(sure):.0%} likely right" if calibrator else "no stated chance yet"
        print(f"\nturn {turn}  {question}")
        print(f"  {'answered:' if speaks else 'stayed quiet, would have said:'} {answer}")
        print(f"  confidence {sure:.4f} -> {stated};  oracle: {why}")
        before = abstainer
        abstainer, calibrator, learned = adapt(feedback, wanted, warmup)
        moved = (f"solved for {wanted:.0%}, expects {abstainer.expected:.0%} on "
                 f"{abstainer.coverage:.0%}" if learned else "still hand-picked, too little feedback")
        print(f"  cut {before.cut:.5f} -> {abstainer.cut:.5f} ({moved});  spoke on {spoke}/{turn}, "
              f"right on {spoke_right}")

    return dict(turns=len(chosen), spoke=spoke, spoke_right=spoke_right, right=right,
                opening=opening, cut=abstainer.cut, learned=learned, abstainer=abstainer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--dataset", default="advisory")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--split", default="unseen")
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--log", default="artifacts/chat.sqlite")
    parser.add_argument("--wanted", type=float, default=0.6, help="the stated chance of being right")
    parser.add_argument("--warmup", type=int, default=10,
                        help="judged rows before a solved cut replaces the hand-picked one")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    tokenizer, model, config = load_model(artifacts, args.dataset, args.checkpoint)
    split = load_split(artifacts, args.dataset, args.split)
    rng = np.random.default_rng(config.seed)
    chosen = rng.permutation(len(split[0]))[: args.turns]

    with FeedbackLog(args.log) as feedback:
        started = len(feedback)
        session = converse(model, tokenizer, split, chosen, rng, feedback, args.wanted, args.warmup,
                           args.batch_size)
        grew = len(feedback) - started

    print(f"\n{session['turns']} turns, log {started} -> {started + grew} rows (+{grew})")
    quiet = session["turns"] - session["spoke"]
    if session["spoke"]:
        print(f"  spoke on {session['spoke']}, right on {session['spoke_right']} "
              f"({session['spoke_right'] / session['spoke']:.1%}); answering all {session['turns']} "
              f"would have been right on {session['right']} "
              f"({session['right'] / session['turns']:.1%})")
    print(f"  stayed quiet on {quiet}, and the cut moved {session['opening']:.5f} -> "
          f"{session['cut']:.5f}"
          + ("" if session["learned"] else " — never left the hand-picked value at this session length"))


if __name__ == "__main__":
    main()
