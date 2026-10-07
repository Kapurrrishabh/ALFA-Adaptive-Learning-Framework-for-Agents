"""Does a cross-encoder put the endorsed answer first more often than BM25 alone? Measured on labelled questions.

The labels are models/training/eval_answer_retrieval.py's: a Stack Exchange question whose asker accepted an
answer, asked by its title the way a user types, with a hit being a chunk of that answer. Recall@1 is the
number that matters, because the reference path quotes the first passage it keeps.

    python -m backend.models.external.rerank_eval --questions 300
"""
import argparse
import time
from pathlib import Path

from backend.knowledge_base.retrieval import ARMS, HYBRID, Hybrid, load
from backend.models.core.backend import default_rng, xp
from backend.models.core.tokenizer.wordpiece import pretokenize
from backend.models.external import sentence
from backend.models.external.embed_index import vectors_for
from backend.models.training.eval_answer_retrieval import labelled, title_only
from backend.paths import ARTIFACTS, DATA


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--questions", type=int, default=300)
    ap.add_argument("--pool", type=int, default=50, help="BM25 candidates the cross-encoder reorders")
    args = ap.parse_args()
    chunks, _ = load(ARTIFACTS / "reference_index.npz")
    vectors = xp.asarray(vectors_for(chunks), dtype=xp.float32)
    index = Hybrid(chunks, pretokenize, vectors, lambda text: sentence.embed([text])[0], centre=False)
    asked, sampled = labelled(chunks, DATA / "manifest.jsonl", DATA / "qa", args.questions, default_rng(0), title_only)
    print(f"{len(asked)} of {sampled} sampled questions have their endorsed answer indexed")
    ranked, scores, started = {}, [], time.time()
    for question, targets in asked:
        for arm in ARMS:
            pool = index.search(question, args.pool, arm=arm)
            ranked.setdefault(arm, []).append(pool)
            relevance = sentence.relevance(question, [chunks[i].text for i in pool])
            order = [pool[i] for i in xp.argsort(-relevance)] if len(pool) else []
            ranked.setdefault(f"{arm} then cross-encoder", []).append(order)
            if arm == HYBRID:
                scores.append((float(relevance.max()) if len(pool) else float("-inf"), bool(order) and order[0] in targets))
    print(f"three pools of {args.pool}, each reranked, in {(time.time() - started) / len(asked):.2f}s per question")
    for name, lists in ranked.items():
        at = [xp.mean([bool(set(r[:k]) & t) for r, (_, t) in zip(lists, asked)]) for k in (1, 5)]
        print(f"  {name:32} recall@1 {at[0]:.1%}   recall@5 {at[1]:.1%}")
    # how the top passage's score separates a right first passage from a wrong one, for a refusal cut
    right = xp.asarray([s for s, ok in scores if ok])
    wrong = xp.asarray([s for s, ok in scores if not ok])
    for cut in (-2.0, 0.0, 2.0, 4.0):
        kept_right, kept_wrong = (right >= cut).sum(), (wrong >= cut).sum()
        print(f"  quote only when the top score >= {cut:+.0f}: {kept_right + kept_wrong}/{len(scores)} quoted, "
              f"{kept_right / max(kept_right + kept_wrong, 1):.1%} of those right")


if __name__ == "__main__":
    main()
