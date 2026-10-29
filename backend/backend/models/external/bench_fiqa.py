"""The agent's retrieval measured on BEIR FiQA-2018, the standard finance question-answering retrieval benchmark.

Our own measurement (rerank_eval.py) uses Stack Exchange questions whose asker accepted an answer, which is a
fair test of our index but a number nobody else reports. FiQA is the same kind of task -- personal-finance
questions, answers from Stack Exchange and Reddit -- with published scores to stand against: nDCG@10 on its
648 test questions, BM25 0.236 and BM25 reranked by a cross-encoder 0.347 (Thakur et al., BEIR, 2021), and
bge-small-en-v1.5 0.403 (its MTEB report). Only the method is measured here: FiQA's corpus is not served.

    python -m backend.models.external.bench_fiqa
"""
import argparse
import json
import math
import time

from backend.knowledge_base.retrieval import Hybrid
from backend.knowledge_base.retrieval.chunk import Chunk
from backend.models.core.backend import xp
from backend.models.core.tokenizer.wordpiece import pretokenize
from backend.models.external import sentence
from backend.paths import ARTIFACTS

PUBLISHED = {"BM25 (BEIR)": 0.236, "BM25 + cross-encoder (BEIR)": 0.347, "bge-small-en-v1.5 (MTEB)": 0.403}
OUT = ARTIFACTS / "external" / "bench_fiqa.json"


def load_fiqa():
    """(document ids, document texts, {query id: text}, {query id: {relevant document ids}}) for the test split."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    corpus = pd.read_parquet(hf_hub_download("BeIR/fiqa", "corpus/corpus-00000-of-00001.parquet", repo_type="dataset"))
    queries = pd.read_parquet(hf_hub_download("BeIR/fiqa", "queries/queries-00000-of-00001.parquet", repo_type="dataset"))
    qrels = pd.read_csv(hf_hub_download("BeIR/fiqa-qrels", "test.tsv", repo_type="dataset"), sep="\t")
    relevant = {}
    for qid, did, score in qrels[["query-id", "corpus-id", "score"]].itertuples(index=False):
        if score > 0:
            relevant.setdefault(str(qid), set()).add(str(did))
    texts = {str(q): t for q, t in zip(queries["_id"], queries["text"]) if str(q) in relevant}
    ids = [str(d) for d in corpus["_id"]]
    docs = [" ".join(f"{title} {text}".split()) for title, text in zip(corpus["title"].fillna(""), corpus["text"].fillna(""))]
    return ids, docs, texts, relevant


def ndcg_at_10(ranked, relevant):
    gain = sum(1 / math.log2(rank + 2) for rank, doc in enumerate(ranked[:10]) if doc in relevant)
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(relevant), 10)))
    return gain / ideal


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N test questions (0 for all)")
    ap.add_argument("--no-instruction", action="store_true", help="embed questions without bge's query instruction")
    args = ap.parse_args()
    ids, docs, queries, relevant = load_fiqa()
    if args.limit:
        queries = dict(list(queries.items())[:args.limit])
    print(f"{len(docs)} documents, {len(queries)} test questions")
    started = time.time()
    vectors = sentence.embed(docs)
    print(f"embedded the corpus in {(time.time() - started) / 60:.1f} min")
    chunks = [Chunk(text, [], doc_id, "") for doc_id, text in zip(ids, docs)]
    query = (lambda q: sentence.embed([q])[0]) if args.no_instruction else sentence.embed_query
    index = Hybrid(chunks, pretokenize, vectors.astype(xp.float32), query, centre=False)
    arms = {"BM25 (ours)": [], "dense bge-small": [], "BM25 then cross-encoder": [], "dense then cross-encoder": [],
            "hybrid then cross-encoder": []}
    started = time.time()
    for qid, question in queries.items():
        rel = relevant[qid]
        pools = {name: index.search(question, args.pool, arm=arm) for name, arm in
                 (("lexical", "lexical"), ("vector", "vector"), ("hybrid", "hybrid"))}
        arms["BM25 (ours)"].append(ndcg_at_10([ids[i] for i in pools["lexical"]], rel))
        arms["dense bge-small"].append(ndcg_at_10([ids[i] for i in pools["vector"]], rel))
        for name, pool in (("BM25 then cross-encoder", pools["lexical"]), ("dense then cross-encoder", pools["vector"]),
                           ("hybrid then cross-encoder", pools["hybrid"])):
            scores = sentence.relevance(question, [docs[i] for i in pool])
            arms[name].append(ndcg_at_10([ids[pool[i]] for i in xp.argsort(-scores)], rel) if len(pool) else 0.0)
    print(f"scored in {(time.time() - started) / 60:.1f} min\n")
    result = {name: round(float(xp.mean(values)), 4) for name, values in arms.items()}
    for name, value in PUBLISHED.items():
        print(f"  {name:32} {value:.3f}  (published)")
    for name, value in result.items():
        print(f"  {name:32} {value:.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_name(f"bench_fiqa_{sentence.RERANKER_ID.split('/')[-1]}.json").write_text(json.dumps({"benchmark": "BEIR FiQA-2018 test, nDCG@10", "questions": len(queries),
                               "query_instruction": not args.no_instruction,
                               "encoder": sentence.ENCODER_ID, "reranker": sentence.RERANKER_ID, "pool": args.pool,
                               "published": PUBLISHED, "ours": result}, indent=1))


if __name__ == "__main__":
    main()
