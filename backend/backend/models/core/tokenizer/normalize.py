"""Finance spellings canonicalised before the text is split into pieces.

Every rule here is kept because it was counted in the corpus, not because it sounded useful. Counts
are over an 8.5 MB sample drawn evenly from all nine sources, so the whole corpus holds about 53
times each figure.

Measured and deliberately dropped: ISO and slash dates (21 and 12 occurrences), NSE ticker suffixes
(0), cashtags (2), and the forms 20-F, 6-K, 13F and S-1 (7 between them). Digit bucketing was
dropped too — 60% of number words occur exactly once, but their digits are already in the character
vocabulary, so replacing them with a magnitude marker would take arithmetic away from an agent that
has to read "revenue rose 12 percent" for no measured gain.
"""

import hashlib
import re

# The pre-tokenizer split "10-k" into three pieces, so the model never saw a form name as one thing
# and answered "filed, 1, 8, 2, k" when asked for one. 197 occurrences. The hyphen is required: all
# ten bare "10k"/"8k" in the sample were sums of money, and "$10k" must not become a form name.
_FORM = re.compile(r"\b(10-k|10-q|8-k)\b")

# The pre-tokenizer splits letters from digits, so a normalised "form10k" would come straight back
# apart. It is handed this pattern to keep the form whole, declared next to the rule that writes it.
FORM_TOKEN_PATTERN = r"form\d{1,2}[kq]"

# 4,639 occurrences. Dropping the separators leaves every large number with one spelling, not two.
# Two-digit groups are accepted because India writes lakh and crore as 12,50,000: 629 numbers in the
# sample, against 2 comma-separated lists the rule wrongly joins. The lookarounds matter more — a
# pattern anchored on \b matched "50,000" inside "12,50,000" and left a stray comma behind, which
# corrupted 664 numbers into a third spelling instead of collapsing them onto one.
_GROUPED_NUMBER = re.compile(r"(?<![\d,])\d{1,3}(?:,\d{2,3})+(?![\d,])")

# "rs" is how 2,002 of the 2,060 rupee amounts in the sample are written, so the rarer spellings
# normalise onto it rather than onto the sign. The trailing dot goes because "rs.45" otherwise
# tokenizes as three pieces.
_RUPEE = re.compile(r"\brs\b\.?|\binr\b|₹")
_DOLLAR = re.compile(r"\bus\$|\busd\b")

# A vocabulary learned under different rules encodes the same text differently and nothing
# downstream would notice. Hashing the patterns means this guard updates itself when a rule changes.
NORMALIZATION_FINGERPRINT = hashlib.sha256(
    "|".join(
        [expression.pattern for expression in (_FORM, _GROUPED_NUMBER, _RUPEE, _DOLLAR)]
        + [FORM_TOKEN_PATTERN]
    ).encode()
).hexdigest()[:12]


def normalize(text):
    """Canonical finance spellings. Takes lowercased text and returns lowercased text."""
    text = _FORM.sub(lambda match: "form" + match.group().replace("-", ""), text)
    text = _GROUPED_NUMBER.sub(lambda match: match.group().replace(",", ""), text)
    text = _RUPEE.sub("rs ", text)
    return _DOLLAR.sub("$", text)
