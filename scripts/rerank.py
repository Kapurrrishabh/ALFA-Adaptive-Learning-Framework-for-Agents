#!/usr/bin/env python3
"""Sample k answers per question, pick one with the weights frozen, and report what the picking bought.

Five numbers, because the other four are what make the picking columns readable:

  single sample             what the model does today, the 31.0% the gate is set against
  best of k by confidence   ranking on the model's own confidence alone
  best of k by rank()       the same, after grounded candidates are put above invented ones
  most repeated in best tier  self-consistency, which uses no confidence at all
  any candidate right       the ceiling: no picker can choose a right answer that was never sampled

If the ceiling is barely above the single sample then sampling is the problem and no picker will fix it,
and that is worth knowing before building a better one. The distinct-answers line says whether that is
what happened: at the served sampler this model draws 1.54 distinct answers out of 8.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_answers import answer_rows, load_model  # noqa: E402
from selfagent.data import advisory  # noqa: E402
from selfagent.learn.rank import consensus, rank  # noqa: E402
from selfagent.learn.teacher import OracleTeacher  # noqa: E402
from selfagent.models.generator import ANSWER_TEMPERATURE, ANSWER_TOP_P  # noqa: E402
from train_generator import load_split  # noqa: E402


def sample_candidates(model, tokenizer, split, chosen, rng, trials, batch_size, temperature, top_p):
    """`trials` independent samples of every chosen row, as a list of per-trial answer lists."""
    return [answer_rows(model, tokenizer, split, chosen, rng, False, batch_size, temperature, top_p)
            for _ in range(trials)]


def score(drawn, oracle):
    """Each strategy's accuracy over the rows, plus how often the tiers overruled the confidence.

    That last one separates a flat result from a broken one: if the tiers never overrule, the ranking
    and confidence columns are the same number because they made the same picks, not because the
    reordering was tried and did not help.
    """
    rows = len(drawn[0])
    counts = dict(single=0, confident=0, ranked=0, consensus=0, ceiling=0, overruled=0, distinct=0,
                  rescued=0, spoiled=0)
    for row in range(rows):
        question, evidence, gold = drawn[0][row][:3]
        answers = [trial[row][3] for trial in drawn]
        confidences = [trial[row][4] for trial in drawn]
        right = [oracle.judge(question, evidence, answer, gold)[0] for answer in answers]
        surest = int(np.argmax(confidences))
        top = rank(answers, evidence, confidences, advisory.UNSUPPORTED)[0]

        counts["single"] += right[0]
        counts["confident"] += right[surest]
        counts["ranked"] += right[top]
        voted = right[consensus(answers, evidence, advisory.UNSUPPORTED)]
        counts["consensus"] += voted
        counts["ceiling"] += any(right)
        counts["overruled"] += top != surest
        counts["distinct"] += len(set(answers))
        # The two counts a difference of accuracies hides. A net gain of four rows out of two hundred
        # reads very differently at 4 rescued and 0 spoiled than at 20 and 16.
        counts["rescued"] += voted and not right[0]
        counts["spoiled"] += right[0] and not voted
    return {name: count / rows for name, count in counts.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--dataset", default="advisory")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--split", default="unseen")
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    # Ranking needs the k candidates to differ. At the served pair the model draws 1.54 distinct answers
    # out of 8, so these are the knobs that decide whether there is anything to rank at all.
    parser.add_argument("--temperature", type=float, default=ANSWER_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=ANSWER_TOP_P)
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    tokenizer, model, config = load_model(artifacts, args.dataset, args.checkpoint)
    split = load_split(artifacts, args.dataset, args.split)
    rng = np.random.default_rng(config.seed)
    chosen = rng.permutation(len(split[0]))[: args.rows]

    drawn = sample_candidates(model, tokenizer, split, chosen, rng, args.candidates, args.batch_size,
                              args.temperature, args.top_p)
    measured = score(drawn, OracleTeacher(advisory.UNSUPPORTED))

    print(f"\n{args.split}, {len(chosen)} rows, {args.candidates} candidates each "
          f"at temperature {args.temperature} top-p {args.top_p}")
    for name, label in (("single", "single sample"), ("confident", "best of k by confidence"),
                        ("ranked", "best of k by rank()"), ("consensus", "most repeated in best tier"),
                        ("ceiling", "any candidate right")):
        gain = 100 * (measured[name] - measured["single"])
        print(f"  {label:<26}{measured[name]:>7.1%}" + (f"{gain:>+8.1f} pts" if gain else ""))
    print(f"  the tiers overruled the confidence on {measured['overruled']:.1%} of rows")
    print(f"  {measured['distinct']:.2f} distinct answers per row out of {args.candidates} sampled")
    rescued, spoiled = (round(measured[name] * len(chosen)) for name in ("rescued", "spoiled"))
    print(f"  consensus rescued {rescued} rows the first sample got wrong and spoiled {spoiled} it got "
          f"right, so the gain rests on {rescued + spoiled} rows")


if __name__ == "__main__":
    main()
