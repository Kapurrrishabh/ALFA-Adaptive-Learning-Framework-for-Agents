"""WordPiece trainer and encoder.

Written here rather than taken from a library because S11 forbids the deep-learning stack, and
because the vocabulary has to carry finance's shapes: currency, percentages and numbers survive
pre-tokenization as whole pieces instead of being shredded into punctuation.

Merges are learned over a word-frequency table rather than over the text, so training cost depends
on how many distinct words the corpus has, not on its 475 MB. Non-initial pieces carry WordPiece's
`##` prefix, which whole-word masking later reads to find where each word begins.
"""

import heapq
import json
import re
from collections import Counter

from .normalize import FORM_TOKEN_PATTERN, NORMALIZATION_FINGERPRINT, normalize
from .vocab import CONTINUATION, SPECIAL_TOKENS, UNK, UNK_ID

# Numbers keep their separators, decimal point, currency and percent sign; letters group into
# words; anything else stands alone. Splitting "$1.2bn" into rubble is the failure to avoid.
# A number has to end in a digit, percent or nothing: allowing it to end on a separator glued the
# sentence comma onto "2023," and gave every year two spellings.
_PRETOKEN = re.compile(FORM_TOKEN_PATTERN + r"|\$?\d(?:[\d,]*\d)?(?:\.\d+)?%?|[a-z]+|[^\sa-z]")

# Words seen once are mostly extraction noise. They distort merge statistics more than they inform
# them, and they still encode afterwards through their characters.
_MINIMUM_WORD_FREQUENCY = 2

# A word longer than this is not a word. PDF extraction produces run-together lines that would
# otherwise dominate the longest-match search.
_MAXIMUM_WORD_CHARACTERS = 32

# Characters are kept on their own total frequency, not on their words'. Every distinct percentage
# ("6.50%") occurs once, so the word cut above removed every word holding a '%' and the character
# vanished from the vocabulary entirely. Measured over the corpus: '##%' occurs 20,492 times, while
# this threshold drops the 212 one-off extraction artifacts and keeps 199 real symbols.
_MINIMUM_CHARACTER_FREQUENCY = 10


def pretokenize(text):
    """Normalised, lowercased words, numbers and single symbols, in order.

    Normalisation lives here so training and inference cannot drift apart: a corpus canonicalised
    at preparation time but a user's question left raw would encode the same words differently.
    """
    return _PRETOKEN.findall(normalize(text.lower()))


def _as_symbols(word):
    """A word as its starting character followed by `##`-prefixed continuations."""
    return [word[0]] + [CONTINUATION + character for character in word[1:]]


def _joined(first, second):
    return first + second.removeprefix(CONTINUATION)


def _pairs(symbols):
    return Counter(zip(symbols, symbols[1:]))


def _apply_merge(symbols, first, second, merged):
    output, index = [], 0
    while index < len(symbols):
        if index + 1 < len(symbols) and symbols[index] == first and symbols[index + 1] == second:
            output.append(merged)
            index += 2
        else:
            output.append(symbols[index])
            index += 1
    return output


def count_words(lines):
    """Word frequencies over an iterable of text, so the corpus is never held in memory."""
    frequencies = Counter()
    for line in lines:
        frequencies.update(
            word for word in pretokenize(line) if len(word) <= _MAXIMUM_WORD_CHARACTERS
        )
    return frequencies


def _frequent_characters(word_frequencies):
    """Characters worth a vocabulary slot, counted across every occurrence of every word."""
    totals = Counter()
    for word, count in word_frequencies.items():
        for symbol in _as_symbols(word):
            totals[symbol] += count
    return {
        symbol
        for symbol, total in totals.items()
        if total >= _MINIMUM_CHARACTER_FREQUENCY
    }


def _learn_pieces(word_frequencies, target_pieces, log):
    """Pieces beyond the single characters, in the order they were merged.

    A lazy heap rather than a scan for the best pair: there are millions of distinct pairs and
    thousands of merges, and rescanning every pair per merge is the difference between a minute
    and a day.
    """
    words = [_as_symbols(word) for word in word_frequencies]
    frequencies = list(word_frequencies.values())

    pair_counts = Counter()
    holders = {}
    for index, symbols in enumerate(words):
        for pair, occurrences in _pairs(symbols).items():
            pair_counts[pair] += occurrences * frequencies[index]
            holders.setdefault(pair, set()).add(index)

    # Ties broken by the pair itself so a rerun on the same corpus yields the same vocabulary.
    heap = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)

    learned = []
    while heap and len(learned) < target_pieces:
        negative_count, pair = heapq.heappop(heap)
        if pair_counts.get(pair) != -negative_count:
            continue  # A later merge changed this count; the live entry is still in the heap.

        first, second = pair
        merged = _joined(first, second)
        learned.append(merged)

        for index in sorted(holders.pop(pair, ())):
            frequency = frequencies[index]
            before = _pairs(words[index])
            words[index] = _apply_merge(words[index], first, second, merged)
            after = _pairs(words[index])

            for changed in set(before) | set(after):
                delta = (after[changed] - before[changed]) * frequency
                if not delta:
                    continue
                pair_counts[changed] += delta
                if pair_counts[changed] <= 0:
                    del pair_counts[changed]
                    holders.pop(changed, None)
                    continue
                holders.setdefault(changed, set()).add(index)
                heapq.heappush(heap, (-pair_counts[changed], changed))

        if len(learned) % 1000 == 0:
            log(f"  {len(learned)} pieces learned")

    return learned


class WordPiece:
    """Maps text to token ids, greedy longest match first."""

    def __init__(self, pieces):
        self.pieces = list(pieces)
        self.ids = {piece: index for index, piece in enumerate(self.pieces)}
        if len(self.ids) != len(self.pieces):
            raise ValueError("vocabulary contains a duplicate piece")
        self._longest = max(len(piece) for piece in self.pieces)
        self._cache = {}

    @property
    def vocab_size(self):
        return len(self.pieces)

    @property
    def continuation_ids(self):
        """Ids that continue a word. Whole-word masking needs this to group pieces."""
        return frozenset(
            index for index, piece in enumerate(self.pieces) if piece.startswith(CONTINUATION)
        )

    def encode_word(self, word):
        """Pieces of one word, or [UNK] if any part of it is unrepresentable."""
        cached = self._cache.get(word)
        if cached is not None:
            return cached

        ids, start, prefix = [], 0, ""
        while start < len(word):
            for end in range(min(len(word), start + self._longest), start, -1):
                candidate = prefix + word[start:end]
                if candidate in self.ids:
                    ids.append(self.ids[candidate])
                    start, prefix = end, CONTINUATION
                    break
            else:
                ids = [UNK_ID]
                break

        self._cache[word] = ids
        return ids

    def encode(self, text):
        ids = []
        for word in pretokenize(text):
            ids.extend(self.encode_word(word))
        return ids

    def decode(self, ids):
        """Best-effort text, for eyeballing what the model saw."""
        out = []
        for token_id in ids:
            piece = self.pieces[token_id]
            if piece.startswith(CONTINUATION):
                out.append(piece.removeprefix(CONTINUATION))
            else:
                out.append(" " + piece)
        return "".join(out).strip()

    @classmethod
    def train(cls, lines, vocab_size, log=print):
        """Learns a vocabulary of exactly `vocab_size` pieces, or fewer if the text is small."""
        frequencies = count_words(lines)
        log(f"  {len(frequencies):,} distinct words")
        frequent = {
            word: count
            for word, count in frequencies.items()
            if count >= _MINIMUM_WORD_FREQUENCY
        }
        log(f"  {len(frequent):,} seen at least {_MINIMUM_WORD_FREQUENCY} times")

        characters = sorted(_frequent_characters(frequencies))
        log(f"  {len(characters):,} characters kept")
        reserved = len(SPECIAL_TOKENS) + len(characters)
        if reserved > vocab_size:
            raise ValueError(
                f"vocab_size {vocab_size} cannot hold {len(SPECIAL_TOKENS)} special tokens and "
                f"{len(characters)} characters; raise it to at least {reserved}"
            )

        learned = _learn_pieces(frequent, vocab_size - reserved, log)
        return cls(list(SPECIAL_TOKENS) + characters + learned)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {"pieces": self.pieces, "normalization": NORMALIZATION_FINGERPRINT},
                handle,
                ensure_ascii=False,
            )

    @staticmethod
    def matches_normalization(path):
        """Whether a saved vocabulary was learned under the rules this code normalises with.

        A caller that can re-learn should ask before loading, because for it a mismatch means
        rebuild rather than fail; load() keeps raising for everyone downstream who cannot.
        """
        with open(path, encoding="utf-8") as handle:
            return json.load(handle).get("normalization") == NORMALIZATION_FINGERPRINT

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as handle:
            stored = json.load(handle)
        if stored.get("normalization") != NORMALIZATION_FINGERPRINT:
            raise ValueError(
                f"vocabulary at {path} was learned under normalisation rules "
                f"{stored.get('normalization')}, but this code normalises to "
                f"{NORMALIZATION_FINGERPRINT}; re-run scripts/prepare_pretrain.py"
            )
        loaded = cls(stored["pieces"])
        if loaded.pieces[UNK_ID] != UNK:
            raise ValueError(f"vocabulary at {path} does not start with the special tokens")
        return loaded
