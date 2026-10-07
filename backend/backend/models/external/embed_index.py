"""Embed every passage of the reference index with the pretrained sentence encoder, for a search by meaning.

BM25 cannot find an answer that shares no words with the question, and a reranker can only reorder what
BM25 found. Saved as float16 beside the index, with the passage count checked on load, so vectors for an
older index are refused rather than paired with the wrong passages.

    python -m backend.models.external.embed_index
"""
import argparse
import time

from backend.knowledge_base.retrieval import load
from backend.models.core.backend import xp
from backend.models.external import sentence
from backend.paths import ARTIFACTS

VECTORS = ARTIFACTS / "reference_vectors_bge.npy"


def vectors_for(chunks, path=VECTORS):
    """The saved vectors for `chunks`, or an error naming the command that rebuilds them."""
    if not path.exists():
        raise FileNotFoundError(f"no passage vectors at {path}; build them with python -m backend.models.external.embed_index")
    held = xp.load(path, mmap_mode="r")
    if len(held) != len(chunks):
        raise ValueError(f"{path} holds {len(held)} vectors for an index of {len(chunks)} passages; rebuild it")
    return held


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch", type=int, default=2048)
    args = ap.parse_args()
    chunks, _ = load(ARTIFACTS / "reference_index.npz")
    out = xp.lib.format.open_memmap(VECTORS.with_suffix(".partial.npy"), mode="w+", dtype=xp.float16,
                                    shape=(len(chunks), sentence.encoder().get_sentence_embedding_dimension()))
    started = time.time()
    for start in range(0, len(chunks), args.batch):
        out[start:start + args.batch] = sentence.embed([c.text for c in chunks[start:start + args.batch]])
        done = min(start + args.batch, len(chunks))
        print(f"{done}/{len(chunks)} passages, {(time.time() - started) / done * (len(chunks) - done) / 60:.0f} min left",
              flush=True)
    out.flush()
    del out
    VECTORS.with_suffix(".partial.npy").replace(VECTORS)
    print(f"saved {VECTORS}")


if __name__ == "__main__":
    main()
