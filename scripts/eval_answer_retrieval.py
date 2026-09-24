#!/usr/bin/env python3
"""Can the index find the answer to a question, when it holds one? The first labelled retrieval number.

Every other retrieval measurement here is a proxy and says so. C2's is *find the rest of the document*,
which is weak supervision. C9's relevance is a hand reading of 20 pairs. This one has ground truth nobody
here wrote: the asker accepted an answer, or the site voted one up, and that answer is in the index under
its thread's key. So the query is the question as its asker typed it, and a hit is a chunk of that
thread's own endorsed answer, found among every other answer in the corpus.

Three numbers, and the third is the one to quote:

  present   the share of sampled questions whose answer is in the index at all -- the ceiling no
            ranking can beat, and the number the old thread-blob index lost on
  recall@k  among those, how often the top k holds it -- ranking quality, measured where it can be
  found@k   present x recall@k, which is what a user asking a collected question actually gets

**What it does not claim.** An answer often restates the question it answers, so some of this is lexical
overlap the retriever gets for free. It is still the served task -- somebody types a question, the index
has to put the right passage first -- and it is the same task for every index compared here, which is
what makes the comparison fair even where the absolute number is generous.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.retrieval import LEXICAL, Hybrid, keys_by_thread, load  # noqa: E402
from eval_retrieval import recall  # noqa: E402
from selfagent.data.qa_pairs import load_threads, question_text  # noqa: E402
from selfagent.tokenizer.wordpiece import pretokenize  # noqa: E402


def labelled(chunks, manifest, qa_root, count, rng):
    """([(question, the chunk indices that answer it)], how many questions were sampled to get them).

    Sampled from every collected question, not only from the ones whose answer survived the budget:
    conditioning the sample on what was indexed would hide exactly the failure that matters, which is a
    question whose answer is not in the corpus at all.
    """
    held = {}
    for index, chunk in enumerate(chunks):
        held.setdefault(chunk.document, []).append(index)
    keys = keys_by_thread(manifest)
    questions, _ = load_threads(Path(qa_root))

    asked = []
    picked = [questions[position] for position in rng.permutation(len(questions))]
    picked = [question for question in picked if question_text(question).strip()][:count]
    for question in picked:
        targets = held.get(keys.get((question["site"], question["id"]), ""), [])
        if targets:
            asked.append((question_text(question), set(targets)))
    return asked, len(picked)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--index", default="reference_index.npz")
    parser.add_argument("--manifest", default="data/manifest.jsonl")
    parser.add_argument("--qa", default="data/qa")
    parser.add_argument("--questions", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    chunks, _ = load(Path(args.artifacts) / args.index)
    rng = np.random.default_rng(args.seed)
    asked, sampled = labelled(chunks, args.manifest, args.qa, args.questions, rng)
    present = len(asked)
    documents = len({chunk.document for chunk in chunks})
    print(f"{len(chunks)} chunks over {documents} documents in {args.index}, {sampled} questions sampled")
    print(f"  present     {present}/{sampled} ({present / sampled:.1%}) have their endorsed answer indexed")
    if not asked:
        print("  nothing to rank: no sampled question's answer is in this index")
        return 0

    # Lexical only, because that is what serves. The arms were compared in C2 and the vector one lost.
    index = Hybrid(chunks, pretokenize)
    for top_k in (1, args.top_k):
        found, reciprocal, returned = recall(index, asked, top_k, LEXICAL)
        print(f"  recall@{top_k:<5}{found:>8.1%}   MRR {reciprocal:.3f}   "
              f"found@{top_k} {found * present / sampled:.1%}   {returned:.1f}/{top_k} returned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
