"""Pretrained sentence models for the agent: an encoder for routing and search, a cross-encoder it is measured against,
and an entailment model that checks a language model's reply against its passages.

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
# Not settable like the two above: phrase.ENTAILED was measured on this model's scale. Not Vectara's HHEM either, which
# is the independent judge eval_user_questions grades the same replies with. 142M parameters: 568 MB on disk, about
# 570 MB more resident once it has scored a reply; loaded at start for a writer with news to check (ask.entails_from).
# Not FactCG-DeBERTa-v3-Large, LLM-AggreFact's best under 1B: on the 298 replies ENTAILED was read on it refused 58 of
# 166 good ones to this model's 36, served as many bad ones, and took 7 to 13 times as long per reply.
NLI_ID = "cross-encoder/nli-deberta-v3-small"


@lru_cache(maxsize=1)
def encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(ENCODER_ID, device=TORCH_DEVICE)


@lru_cache(maxsize=1)
def reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_ID, device=TORCH_DEVICE)


@lru_cache(maxsize=1)
def nli():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(NLI_ID, device=TORCH_DEVICE)


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


def entailment(pairs):
    """For each (premise, hypothesis), the chance the premise entails the hypothesis."""
    if not pairs:
        return xp.asarray([])
    model = nli()
    return xp.asarray(model.predict(pairs, apply_softmax=True)[:, model.config.label2id["entailment"]])
