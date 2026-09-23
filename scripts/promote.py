#!/usr/bin/env python3
"""Measure a candidate checkpoint and promote it only if its own numbers meet the standards.

Nothing ships on a promise here. This generates answers on held-out rows, counts them with
`check_answers.tally` -- the same counting every generation number in this project is quoted from -- and
hands the counts to `backend.models.registry`, which refuses the promotion and says why when S14's
faithfulness bar or the generation comparison against the promoted checkpoint is not met.

Both splits are measured because they answer different questions. `unseen` is the reworded one and is
what the standards are judged on; `casual` is the register the agent is asked in, and a candidate that
wins on the first and collapses on the second is a candidate the record has to show.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.models import registry  # noqa: E402
from check_answers import answer_rows, load_model, report, tally  # noqa: E402
from selfagent.models.generator import ANSWER_TEMPERATURE, ANSWER_TOP_P  # noqa: E402
from train_generator import load_split  # noqa: E402


def measure(artifacts, dataset, checkpoint, splits, rows, batch_size):
    """(the tally per split, how the answers were sampled) for one checkpoint."""
    tokenizer, model, config = load_model(artifacts, dataset, checkpoint)
    measured = {}
    for name in splits:
        split = load_split(artifacts, dataset, name)
        # Seeded from the config so the rows and the sampling are the ones check_answers reports on.
        rng = np.random.default_rng(config.seed)
        chosen = rng.permutation(len(split[0]))[:rows]
        scored = answer_rows(model, tokenizer, split, chosen, rng, False, batch_size)
        print(f"\n{checkpoint or dataset} on {dataset}'s {name} split")
        counts = tally(scored)
        report(counts)
        measured[name] = {"dataset": dataset, **counts}
    sampler = (f"nucleus temperature {ANSWER_TEMPERATURE}, top_p {ANSWER_TOP_P}, "
               f"seed {config.seed}")
    return measured, sampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--dataset", default="advisory_casual",
                        help="the dataset whose splits the candidate is measured on")
    parser.add_argument("--checkpoint", default="advisory_combined.npz")
    parser.add_argument("--splits", nargs="+", default=[registry.STANDARD_SPLIT, "casual"])
    parser.add_argument("--rows", type=int, default=200,
                        help="held-out rows per split; 200 is what every published number here used")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--role", default=registry.GENERATOR)
    parser.add_argument("--registry", default=str(registry.DEFAULT_PATH))
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    measured, sampler = measure(artifacts, args.dataset, args.checkpoint, args.splits, args.rows,
                                args.batch_size)
    try:
        entry = registry.promote(args.role, artifacts / args.checkpoint, measured, sampler,
                                 args.registry)
    except registry.Refused as refused:
        print(f"\nREFUSED  {refused}")
        return 1
    print(f"\npromoted {entry['artifact']} ({entry['digest']}) to {args.role} in {args.registry}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
