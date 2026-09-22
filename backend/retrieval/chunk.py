"""Cutting a document into passages the encoder can read whole.

Sized in tokens measured by the tokenizer, not in characters or words: the window is a token budget, and
a character estimate is wrong by a factor of three between a table of figures and a paragraph of prose.
The ids are kept on the chunk because the embedder needs them anyway -- tokenizing 18,000 filings twice
is the most expensive mistake available here.

Chunks overlap by whole sentences. A figure and the thing it measures often sit either side of a
boundary, and without overlap both chunks lose the pair.

Cleaning is deliberately only whitespace: extracted text still carries filing boilerplate, and the prose
filter that would drop it lives in `dataforge/process/clean.py`, which `backend/` must not import. What
this does instead is refuse a chunk too short to be a passage, which is what most boilerplate is.
"""

import re
import unicodedata

# A sentence end, or a blank line. Extracted text wraps mid-sentence, so a newline alone is not a break.
_BREAK = re.compile(r"(?<=[.!?])\s+|\n\s*\n")
_SPACE = re.compile(r"\s+")

# Below this a chunk is a heading or a filename, not a passage worth retrieving. Provisional: what would
# settle it is recall against a real relevance set, which S13 is the row for.
MIN_TOKENS = 16


class Chunk:
    """One passage: its text, its token ids, and the document and date it inherits."""

    __slots__ = ("text", "ids", "document", "day")

    def __init__(self, text, ids, document, day):
        self.text = text
        self.ids = ids
        self.document = document
        self.day = day

    def __repr__(self):
        return f"Chunk({self.document!r}, {self.day!r}, {len(self.ids)} tokens, {self.text[:40]!r})"


def sentences(text):
    """The document as whitespace-collapsed sentences, empty ones dropped."""
    return [piece for piece in (_SPACE.sub(" ", part).strip() for part in _BREAK.split(text)) if piece]


def deduplicate(chunks):
    """The chunks with later exact repeats dropped, keeping the first -- which is the earliest dated.

    Measured at 5.4% of a 46,601-chunk index: a standing paragraph like "the minutes for each regularly
    scheduled meeting" appears in 43 press releases. Retrieving the same passage five times fills the
    generator's window with one passage's information, and every duplicate is also a wasted embedding.

    Exact after case and punctuation only. Near-duplicates are a further ~11% and catching them needs the
    MinHash index in `dataforge/process/dedup.py`, which `backend/` cannot import -- so they are a
    recorded limitation rather than a second copy of that algorithm.
    """
    kept, seen = [], set()
    for chunk in chunks:
        folded = unicodedata.normalize("NFKD", chunk.text).lower()
        folded = " ".join(part for part in re.sub(r"[^a-z0-9 ]", " ", folded).split())
        if folded in seen:
            continue
        seen.add(folded)
        kept.append(chunk)
    return kept


def chunks(document, tokenizer, size, overlap=1, min_tokens=MIN_TOKENS):
    """`Chunk`s covering one `Document`, each at most `size` tokens, overlapping by `overlap` sentences.

    A single sentence longer than the budget is truncated to it rather than dropped: in a filing that is
    usually a list of figures run together by the extractor, and the first `size` tokens of it are still
    the passage a question about it would match.
    """
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    built, held = [], []

    def close():
        total = sum(len(ids) for _, ids in held)
        if total >= min_tokens:
            built.append(Chunk(" ".join(text for text, _ in held),
                               [i for _, ids in held for i in ids], document.key, document.day))

    for sentence in sentences(document.text):
        ids = tokenizer.encode(sentence)[:size]
        if held and sum(len(kept) for _, kept in held) + len(ids) > size:
            close()
            # Carried as whole sentences, not as tokens: a chunk starting mid-sentence reads as noise to
            # an encoder trained on whole ones. Dropped entirely when the next sentence needs the room.
            held = held[-overlap:] if overlap else []
            if sum(len(kept) for _, kept in held) + len(ids) > size:
                held = []
        held.append((sentence, ids))
    close()
    return built
