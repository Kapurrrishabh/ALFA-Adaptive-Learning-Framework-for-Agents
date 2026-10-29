"""Pretrained sentence models for the agent: an encoder for routing and a cross-encoder for passages.

ALFA's own encoder was trained only to fill in masked words, so it never learned to push unlike sentences
apart: unrelated passages sit at cosine 0.925 (knowledge_base/retrieval/index.py), and the router's margin
is thin on wordings it never saw. A model trained to tell sentences apart is what both need. These live
here with the other open-source models and are handed to the agent, which still imports no deep-learning
library itself.
"""
from __future__ import annotations

import os
from functools import lru_cache

from backend.config import TORCH_DEVICE
from backend.models.core.backend import xp

ENCODER_ID = os.environ.get("STOCKINTEL_ENCODER", "BAAI/bge-small-en-v1.5")
RERANKER_ID = os.environ.get("STOCKINTEL_RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2")


@lru_cache(maxsize=1)
def encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(ENCODER_ID, device=TORCH_DEVICE)


@lru_cache(maxsize=1)
def reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_ID, device=TORCH_DEVICE)


def embed(texts):
    """Unit vectors, one row per text, so a dot product is a cosine."""
    return xp.asarray(encoder().encode(list(texts), normalize_embeddings=True, batch_size=64))


# what bge v1.5 is told a short question is for when it searches long passages; passages and the router's
# like-for-like comparisons take no instruction (BAAI's model card)
QUERY_INSTRUCTION = os.environ.get("STOCKINTEL_QUERY_INSTRUCTION",
                                   "Represent this sentence for searching relevant passages: ")


def embed_query(text):
    """One question as a search vector, with the instruction the encoder was trained to see on queries."""
    return embed([QUERY_INSTRUCTION + text])[0]


def closeness(question, passages):
    """Cosine between the question, embedded as a search, and each passage."""
    return embed(passages) @ embed_query(question)


def similarity(texts):
    """A router score function over `texts`: one cosine per known text for each question asked."""
    known = embed(texts)
    return lambda text: known @ embed([text])[0]


def relevance(question, passages):
    """How well each passage answers the question, as the cross-encoder's logit (higher is better)."""
    if not passages:
        return xp.asarray([])
    return xp.asarray(reranker().predict([(question, passage) for passage in passages]))


# provisional: candidates the cross-encoder reorders; 50 is what rerank_eval.py measured
POOL = 50


def reranked(index, pool=POOL):
    """A ranker for the reference path: the index's search by meaning, reordered by the cross-encoder.

    Measured on 252 labelled questions (models/external/rerank_eval.py): the endorsed answer came first
    20.6% of the time against 12.3% for BM25, and was in the top five 37.3% against 17.9%.
    """
    def rank(query, top_k, as_of=None):
        found = index.search(query, pool, as_of, arm="vector")
        if not found:
            return []
        scores = relevance(query, [index.cite(where).text for where in found])
        return [found[where] for where in xp.argsort(-scores)][:top_k]
    return rank
