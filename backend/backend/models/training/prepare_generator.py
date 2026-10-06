#!/usr/bin/env python3
"""Build the grounded training set for the generator: question, retrieved passages, answer.

The passages are the point. A decoder trained to produce an answer with nothing in front of it learns
to recall, and a 5M-parameter model recalling finance figures invents them. Trained with the evidence
visible it learns to stitch and paraphrase, which it can actually do.

The retrieved passages never include the thread the answer came from. That exclusion is the whole
integrity of this dataset: the collected threads are 145 MB of the pretraining corpus, so without it
the index hands back the target verbatim, the model learns to copy one passage, and every faithfulness
check passes while the model has learned nothing.

`--preference` writes the same rows with two answers instead of one -- the answer a site's voters ranked
clearly above another, and the one they ranked below it -- for the preference training in
`selfagent/learn/preference.py`. One script for both because the source side is the whole of the work and
is identical: same pool, same exclusion, same encoding. A separate builder would be a second chance to
ground a preference pair differently from the supervised row it has to be comparable to.

`--grounded-only` drops a row whose answer states a figure its own retrieved passages do not, and the
measurement behind it is the reason the generator is unfaithful. Over 4,000 sampled rows, 2,390 state at
least one figure and only 116 of those -- 4.9% -- have every figure present in their passages. So 2,274
rows, 56.9% of the set, teach the model to state a number its evidence does not contain, and the 59.4%
unsupported-figure rate it produces is that dataset learned correctly. No objective over these rows can
fix it: the check has to happen here, where the target is chosen, and the same rule the server screens
with is what does the checking. On the full build it keeps 30,436 of 62,155 rows and drops 31,719.
Retrained on what it keeps, the generator's answers with an invented figure fell from 41.0% to 11.3% on 300
held-out rows neither model trained on, while its supported figures rose from 52 to 61.
"""
from backend.paths import ARTIFACTS, DATA

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from backend.models.guardrails import guardrails  # noqa: E402
from backend.models.core.config import ModelConfig  # noqa: E402
from backend.models.data import qa_pairs  # noqa: E402
from backend.models.data.encode import build_sources  # noqa: E402
from backend.models.data.qa_pairs import PASSAGES, QUESTION_TOKENS  # noqa: E402
from backend.models.networks.retriever import BM25  # noqa: E402
from backend.models.core.tokenizer.vocab import CLS_ID, PAD_ID, SEP_ID  # noqa: E402
from backend.models.core.tokenizer.wordpiece import WordPiece, pretokenize  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", default=str(DATA / "qa"))
    parser.add_argument("--artifacts", default=str(ARTIFACTS))
    parser.add_argument("--passages", type=int, default=PASSAGES,
                        help="passages retrieved per question")
    parser.add_argument("--validation", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=0, help="cap the pairs, for a smoke run")
    parser.add_argument("--preference", action="store_true",
                        help="write two ranked answers per row instead of one endorsed answer")
    parser.add_argument("--grounded-only", action="store_true",
                        help="drop rows whose answer states a figure its retrieved passages do not; "
                             "56.9% of them do, which is what taught the model to invent numbers")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    tokenizer = WordPiece.load(artifacts / "tokenizer.json")
    length = ModelConfig.max_text_length
    answer_budget = ModelConfig.max_answer_length

    print("reading threads")
    questions, answers = qa_pairs.load_threads(Path(args.qa))
    pairs = list(qa_pairs.supervised_pairs(questions, answers, tokenizer.encode, answer_budget))
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"  {len(pairs):,} supervised pairs")

    # The pool is the answers themselves, which is what a passage worth retrieving looks like here. It stays
    # the endorsed answers under --preference too: what a row is grounded in should not depend on which
    # answers happen to have been ranked.
    pool_text = [answer for _, _, answer in pairs]
    pool_owner = [key for key, _, _ in pairs]

    names = ("chosen", "rejected") if args.preference else ("target",)
    if args.preference:
        judged = [(key, question,
                   tuple(qa_pairs.truncate_to_sentences(text, tokenizer.encode, answer_budget)
                         for text in (chosen, rejected)))
                  for key, question, chosen, rejected in qa_pairs.preference_pairs(questions, answers)]
        # Dropped when either side loses everything to the budget: preferring a written answer over an
        # empty one is a judgement about length, and the loss would have nothing to score.
        kept = [row for row in judged if all(row[2])]
        print(f"  {len(kept):,} judged pairs, {len(judged) - len(kept):,} dropped as too long")
        examples = kept[: args.limit] if args.limit else kept
    else:
        examples = [(key, question, (answer,)) for key, question, answer in pairs]

    print(f"indexing {len(pool_text):,} passages")
    started = time.monotonic()
    index = BM25(pool_text, pretokenize, owners=pool_owner)
    print(f"  {len(index.postings):,} terms in {time.monotonic() - started:.0f}s")

    passage_ids = [tokenizer.encode(text)[: length - QUESTION_TOKENS - 3] for text in pool_text]
    # Decoded once here rather than per row: the check has to read the passages the model is actually
    # shown, and the truncation above is what makes those different from pool_text.
    passage_text = [tokenizer.decode(ids) for ids in passage_ids] if args.grounded_only else []

    print("retrieving and encoding")
    started = time.monotonic()
    sources = np.zeros((len(examples), args.passages, length), dtype=np.uint16)
    keeps = np.zeros((len(examples), args.passages, length), dtype=np.uint8)
    targets = {name: np.full((len(examples), answer_budget), PAD_ID, dtype=np.uint16) for name in names}
    faithful = np.ones(len(examples), dtype=bool)
    grounded = 0

    for row, (key, question, texts) in enumerate(examples):
        found = index.search(question, args.passages, exclude_owner=key)
        grounded += bool(found)
        sources[row], keeps[row] = build_sources(
            tokenizer.encode(question)[:QUESTION_TOKENS],
            [passage_ids[hit] for hit in found],
            args.passages,
            length,
        )
        # The answer is bracketed so the decoder learns where an answer starts and where it stops.
        for name, text in zip(names, texts):
            ids = [CLS_ID] + tokenizer.encode(text)[: answer_budget - 2] + [SEP_ID]
            targets[name][row, : len(ids)] = ids
            if args.grounded_only:
                # The truncated answer against the truncated passages, so the row is judged on exactly
                # what the model would be shown and asked to write.
                faithful[row] &= not guardrails.unsupported_figures(
                    tokenizer.decode(ids[1:-1]), " ".join(passage_text[hit] for hit in found))
        if row and row % 5000 == 0:
            print(f"  {row:,}/{len(examples):,} in {time.monotonic() - started:.0f}s")

    print(f"  {grounded:,} of {len(examples):,} rows retrieved at least one passage")

    eligible = np.flatnonzero(faithful)
    if args.grounded_only:
        print(f"  {len(eligible):,} of {len(examples):,} rows state no figure their passages do not; "
              f"{len(examples) - len(eligible):,} dropped as unfaithful")
        if not len(eligible):
            raise SystemExit("every row states a figure its passages do not, so --grounded-only would "
                             "write an empty dataset; check that retrieval returned anything at all")
    held = min(args.validation, len(eligible) // 10)
    order = np.random.default_rng(ModelConfig.seed).permutation(eligible)
    split = {"train": order[held:], "validation": order[:held]}
    # Its own prefix, so a filtered build never overwrites the set the served checkpoint was trained on.
    stem = ("preference" if args.preference else "generator") + ("_grounded" if args.grounded_only else "")
    for name, rows in split.items():
        np.save(artifacts / f"{stem}_{name}_source.npy", sources[rows])
        np.save(artifacts / f"{stem}_{name}_keep.npy", keeps[rows])
        for answer, array in targets.items():
            np.save(artifacts / f"{stem}_{name}_{answer}.npy", array[rows])
        print(f"  {name}: {len(rows):,} rows -> {artifacts}/")


if __name__ == "__main__":
    sys.exit(main())
