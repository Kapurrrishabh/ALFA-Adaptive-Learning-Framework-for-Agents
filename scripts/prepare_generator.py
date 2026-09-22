#!/usr/bin/env python3
"""Build the grounded training set for the generator: question, retrieved passages, answer.

The passages are the point. A decoder trained to produce an answer with nothing in front of it learns
to recall, and a 5M-parameter model recalling finance figures invents them. Trained with the evidence
visible it learns to stitch and paraphrase, which it can actually do.

The retrieved passages never include the thread the answer came from. That exclusion is the whole
integrity of this dataset: the collected threads are 145 MB of the pretraining corpus, so without it
the index hands back the target verbatim, the model learns to copy one passage, and every faithfulness
check passes while the model has learned nothing.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from selfagent.config import ModelConfig  # noqa: E402
from selfagent.data import qa_pairs  # noqa: E402
from selfagent.data.encode import build_sources  # noqa: E402
from selfagent.models.retriever import BM25  # noqa: E402
from selfagent.tokenizer.vocab import CLS_ID, PAD_ID, SEP_ID  # noqa: E402
from selfagent.tokenizer.wordpiece import WordPiece, pretokenize  # noqa: E402

# Of the 128-token window, the question gets the smaller share: it is one question, and the passages
# are what the answer has to be built out of.
QUESTION_TOKENS = 48


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", default="data/qa")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--passages", type=int, default=4, help="passages retrieved per question")
    parser.add_argument("--validation", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=0, help="cap the pairs, for a smoke run")
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

    # The pool is the answers themselves, which is what a passage worth retrieving looks like here.
    pool_text = [answer for _, _, answer in pairs]
    pool_owner = [key for key, _, _ in pairs]

    print(f"indexing {len(pool_text):,} passages")
    started = time.monotonic()
    index = BM25(pool_text, pretokenize, owners=pool_owner)
    print(f"  {len(index.postings):,} terms in {time.monotonic() - started:.0f}s")

    passage_ids = [tokenizer.encode(text)[: length - QUESTION_TOKENS - 3] for text in pool_text]

    print("retrieving and encoding")
    started = time.monotonic()
    sources = np.zeros((len(pairs), args.passages, length), dtype=np.uint16)
    keeps = np.zeros((len(pairs), args.passages, length), dtype=np.uint8)
    targets = np.full((len(pairs), answer_budget), PAD_ID, dtype=np.uint16)
    grounded = 0

    for row, (key, question, answer) in enumerate(pairs):
        found = index.search(question, args.passages, exclude_owner=key)
        grounded += bool(found)
        sources[row], keeps[row] = build_sources(
            tokenizer.encode(question)[:QUESTION_TOKENS],
            [passage_ids[hit] for hit in found],
            args.passages,
            length,
        )
        # The answer is bracketed so the decoder learns where an answer starts and where it stops.
        ids = [CLS_ID] + tokenizer.encode(answer)[: answer_budget - 2] + [SEP_ID]
        targets[row, : len(ids)] = ids
        if row and row % 5000 == 0:
            print(f"  {row:,}/{len(pairs):,} in {time.monotonic() - started:.0f}s")

    print(f"  {grounded:,} of {len(pairs):,} rows retrieved at least one passage")

    held = min(args.validation, len(pairs) // 10)
    order = np.random.default_rng(ModelConfig.seed).permutation(len(pairs))
    split = {"train": order[held:], "validation": order[:held]}
    for name, rows in split.items():
        np.save(artifacts / f"generator_{name}_source.npy", sources[rows])
        np.save(artifacts / f"generator_{name}_keep.npy", keeps[rows])
        np.save(artifacts / f"generator_{name}_target.npy", targets[rows])
        print(f"  {name}: {len(rows):,} rows -> {artifacts}/")


if __name__ == "__main__":
    sys.exit(main())
