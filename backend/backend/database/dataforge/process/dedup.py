"""Deduplication at two levels, because filings repeat themselves in two different ways.

Paragraph level catches boilerplate: the same safe-harbour and accounting-policy paragraphs appear
in thousands of filings. Document level catches near-duplicates: this year's 10-K is often 90%+
identical to last year's. Both matter, and nominal corpus size badly overstates the effective
size until they are removed.
"""

import hashlib
import re

import numpy as np

# Bottom line for the banding scheme: a pair with Jaccard similarity s collides in some band with
# probability 1 - (1 - s^ROWS)^BANDS. With 8 rows and 16 bands that crosses 50% near s = 0.72,
# which is where a re-filed annual report sits and an independently written one does not.
PERMUTATIONS = 128
BANDS = 16
ROWS_PER_BAND = PERMUTATIONS // BANDS
SHINGLE_WORDS = 5

# Hashes are truncated to 32 bits so the permutation arithmetic stays inside uint64 without
# overflowing. Collisions at 32 bits are far below the noise floor of a similarity threshold.
_HASH_BITS = 32
_PRIME = (1 << 31) - 1
_SHINGLE_CHUNK = 4096

_WORD = re.compile(r"\w+")


class ParagraphFilter:
    """Keeps the first occurrence of each paragraph and drops every repeat."""

    def __init__(self):
        self._seen = set()
        self.dropped = 0

    def keep_new(self, text):
        kept = []
        for paragraph in text.split("\n\n"):
            digest = hashlib.sha1(" ".join(paragraph.lower().split()).encode()).digest()[:16]
            if digest in self._seen:
                self.dropped += 1
                continue
            self._seen.add(digest)
            kept.append(paragraph)
        return "\n\n".join(kept)


def _shingle_hashes(text):
    words = _WORD.findall(text.lower())
    if len(words) < SHINGLE_WORDS:
        return np.empty(0, dtype=np.uint64)
    shingles = {
        " ".join(words[start : start + SHINGLE_WORDS])
        for start in range(len(words) - SHINGLE_WORDS + 1)
    }
    return np.fromiter(
        (
            int.from_bytes(hashlib.sha1(s.encode()).digest()[: _HASH_BITS // 8], "big")
            for s in shingles
        ),
        dtype=np.uint64,
        count=len(shingles),
    )


class NearDuplicateIndex:
    """MinHash + LSH. `add_if_new` returns False when a similar document was already added."""

    def __init__(self, seed=0):
        generator = np.random.default_rng(seed)
        # Random affine permutations of the hash space, the standard MinHash construction.
        self._multipliers = generator.integers(1, _PRIME, PERMUTATIONS, dtype=np.uint64)
        self._offsets = generator.integers(0, _PRIME, PERMUTATIONS, dtype=np.uint64)
        self._buckets = [{} for _ in range(BANDS)]
        self.dropped = 0

    def _signature(self, text):
        hashes = _shingle_hashes(text)
        if hashes.size == 0:
            return None
        # A large filing has tens of thousands of shingles, so the permuted matrix is built in
        # chunks and reduced as it goes rather than all at once.
        signature = np.full(PERMUTATIONS, _PRIME, dtype=np.uint64)
        for start in range(0, hashes.size, _SHINGLE_CHUNK):
            chunk = hashes[start : start + _SHINGLE_CHUNK, None]
            permuted = (chunk * self._multipliers + self._offsets) % _PRIME
            signature = np.minimum(signature, permuted.min(axis=0))
        return signature

    def add_if_new(self, text):
        signature = self._signature(text)
        if signature is None:
            return False

        keys = [
            hashlib.sha1(
                signature[band * ROWS_PER_BAND : (band + 1) * ROWS_PER_BAND].tobytes()
            ).digest()[:12]
            for band in range(BANDS)
        ]
        if any(key in bucket for key, bucket in zip(keys, self._buckets)):
            self.dropped += 1
            return False

        for key, bucket in zip(keys, self._buckets):
            bucket[key] = True
        return True
