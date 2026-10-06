#!/usr/bin/env python3
"""Read pairs of ranked answers, say which is better without being told, and score the models against it.

The 22,379 pairs training runs on are labelled by StackExchange votes, and a vote count on a ten-year-old
thread partly records who happened to see the thread. Nothing in this repository has ever checked whether
those labels are any good. This is that check, and it is the only judgement here a human actually makes.

Two passes, like scripts/label_feedback.py, because a verdict has to be replayable:

  1. --ask   samples pairs, decodes them to text, and writes them to artifacts/preference_wanted.jsonl
             as an unlabelled "a" and "b" -- with the ranked order hidden and the sides shuffled per pair,
             so the judgement cannot be read off the layout.
  2. --tell  reads the verdicts back from artifacts/preference_given.jsonl, reports how often they agree
             with the votes, and scores each named checkpoint against both sets of labels.

The second number is the one that answers the project's question. Held-out agreement with the votes says
preference training learned the labels it was given; agreement with a reader who never saw those labels
says it learned something about answers. Those can come apart, and if they do the votes were the problem.

Measured, on 60 held-out pairs at seed 0: the reader agrees with the votes on 40 of them, 66.7%. So the
labels training runs on are real but noisy, and a third of them are a judgement a careful reader would not
make. 60 pairs pins that to about six points either way, which is enough to say the labels are neither
random nor clean and not enough to say more.

The pairs are decoded back from the token ids rather than re-read from the corpus. What the judge sees is
then exactly what the model sees -- same truncation, same vocabulary, same lost characters -- so a
disagreement is about the answers and not about text one of them never got.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ablate_generator import evidence_starts  # noqa: E402
from selfagent import pretrained  # noqa: E402
from selfagent.learn.teacher import PairJudge, pair_key  # noqa: E402
from selfagent.models import GroundedGenerator  # noqa: E402
from selfagent.tokenizer.vocab import CLS_ID, PAD_ID, SEP_ID  # noqa: E402
from selfagent.tokenizer.wordpiece import WordPiece  # noqa: E402
from train_generator import load_split  # noqa: E402
from train_preference import PARTS, score  # noqa: E402

WANTED = "preference_wanted.jsonl"
GIVEN = "preference_given.jsonl"
CACHE = "preference_verdicts.jsonl"


def sampled(split, rows, seed):
    """Which rows go in front of the judge. Seeded, because --tell has to rebuild the same sample to know
    which row each verdict belongs to, and storing that mapping is one more file to fall out of step."""
    return np.random.default_rng(seed).permutation(len(split[0]))[:rows]


def texts(tokenizer, split, rows):
    """(question, chosen, rejected) for each row, decoded from the ids the model is trained on."""
    source, _, chosen_ids, rejected_ids = split
    starts = evidence_starts(source[rows])
    read = []
    for offset, row in enumerate(rows):
        whole = source[row, 0]
        question = tokenizer.decode([i for i in whole[: starts[offset, 0]] if i != PAD_ID])
        answers = tuple(tokenizer.decode([i for i in ids[row] if i not in (PAD_ID, CLS_ID, SEP_ID)])
                        for ids in (chosen_ids, rejected_ids))
        read.append((question, *answers))
    return read


def ask(artifacts, dataset, split, rows, seed):
    tokenizer = WordPiece.load(artifacts / "tokenizer.json")
    loaded = load_split(artifacts, dataset, split, PARTS)
    judge = PairJudge(artifacts / CACHE)
    for question, chosen, rejected in texts(tokenizer, loaded, sampled(loaded, rows, seed)):
        judge.judge(question, chosen, rejected)
    written = judge.write_requests(artifacts / WANTED)
    print(f"{rows} pairs sampled from {dataset}/{split}, {written} awaiting a verdict in "
          f"{artifacts / WANTED}")
    if written:
        print(f'Read each and write {{"key": ..., "better": "a"|"b", "why": ...}} per line to '
              f"{artifacts / GIVEN}, then re-run with --tell")


def judged(artifacts, dataset, split, rows, seed):
    """(row indices, what the judge said about each, what the votes said). One entry per verdict on file."""
    tokenizer = WordPiece.load(artifacts / "tokenizer.json")
    loaded = load_split(artifacts, dataset, split, PARTS)
    given = artifacts / GIVEN
    if not given.exists():
        raise SystemExit(f"no verdicts at {given}; run --ask first and judge the pairs it writes")
    verdicts = {row["key"]: row
                for row in map(json.loads, filter(str.strip, given.read_text().splitlines()))}

    judge = PairJudge(artifacts / CACHE)
    chosen = sampled(loaded, rows, seed)
    kept, agrees, missing = [], [], 0
    for row, (question, taken, refused) in zip(chosen, texts(tokenizer, loaded, chosen)):
        key = pair_key(question, taken, refused)
        # A verdict already in the cache was recorded by an earlier --tell. Writing it again would count
        # one judgement twice in every rate computed from this file.
        settled = judge.judge(question, taken, refused)[0]
        if settled is None:
            if key not in verdicts:
                missing += 1
                continue
            judge.record(key, verdicts[key]["better"], verdicts[key].get("why", ""))
            settled = judge.judge(question, taken, refused)[0]
        kept.append(row)
        agrees.append(settled)
    if missing:
        print(f"{missing} of {len(chosen)} sampled pairs have no verdict yet and are left out")
    return np.array(kept), np.array(agrees, dtype=bool), loaded


def tell(artifacts, dataset, split, rows, seed, checkpoints, batch_size):
    kept, agrees, loaded = judged(artifacts, dataset, split, rows, seed)
    if not len(kept):
        raise SystemExit("no pairs carry a verdict, so there is nothing to measure")
    print(f"\n{len(kept)} pairs judged; the reader agrees with the votes on {agrees.mean():.1%} of them")

    subset = tuple(array[kept] for array in loaded)
    for name in checkpoints:
        config, weights = pretrained.load(artifacts / name)
        model = GroundedGenerator(config)
        model.load_state_dict(weights)
        model.eval()
        taken, refused = score(model, subset, batch_size)
        prefers_chosen = taken > refused
        print(f"  {name}: agrees with the votes {prefers_chosen.mean():.1%}, "
              f"with the reader {(prefers_chosen == agrees).mean():.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--dataset", default="preference")
    parser.add_argument("--split", default="validation",
                        help="held out by default: a label the model trained on cannot test it")
    parser.add_argument("--rows", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0, help="fixes which pairs are sampled, both passes")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoints", nargs="+", default=["generator.npz", "preference.npz"],
                        help="scored in the order given; the frozen one first is what makes it a before")
    parser.add_argument("--ask", action="store_true", help="sample pairs and write them out blind")
    parser.add_argument("--tell", action="store_true", help="read verdicts back and score the models")
    args = parser.parse_args()
    if args.ask == args.tell:
        raise SystemExit("pass exactly one of --ask or --tell")

    artifacts = Path(args.artifacts)
    if args.ask:
        ask(artifacts, args.dataset, args.split, args.rows, args.seed)
    else:
        tell(artifacts, args.dataset, args.split, args.rows, args.seed, args.checkpoints, args.batch_size)


if __name__ == "__main__":
    main()
