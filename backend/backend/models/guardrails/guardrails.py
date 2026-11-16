"""Check an answer against the evidence it was given, before anyone reads it.

The generator is a phraser: every figure it states is supposed to be one the evidence already states.
That claim is only worth something if it is checked, because the failure it guards against is the one
that looks fine — a fluent sentence with an invented number in it. Measured on the Stack Exchange
task, 69% of what an answer had to say was absent from its input, and the model duly made it up.

Pure string work, deliberately. It runs on whatever the model produced without re-running the model,
so it can sit in front of serving and in a test over the training set alike.
"""

import functools
import re

from backend.knowledge_base.lessons import STOP
from backend.knowledge_base.retrieval.chunk import sentences
from backend.models.core.tokenizer.wordpiece import pretokenize

# Signed decimals and percentages, with thousands separators, which is every shape as_text produces.
# The separated form is tried first and needs a real group, so "5 ," keeps its comma as punctuation
# while a plain "1158.70" still matches whole instead of splitting into 115 and 8.70.
_FIGURE = re.compile(r"[+-]?\d+(?:,\d{3})+(?:\.\d+)?%?|[+-]?\d+(?:\.\d+)?%?")
# The digits in an evidence key name ("; rsi_14 73", decoded "; rsi _ 14 73") are a window, not a value, so they
# back only an answer's window ("14 day", "the last 5"): a wrong rsi of 50 passed on "sma_50". Free text is untouched.
_KEY = re.compile(r"(?:^|; )[a-z]+ ?_ ?(\d+)")
_WINDOW = re.compile(r"(?<=last |past )(\d+)(?![\w%]|[.,]\d)|"
                     r"(?<![\w.,])(\d+)(?=(?: and \d+)? ?-? ?(?:trading )?days?\b)", re.I)
_NAMED_FIGURE = re.compile(r"_\d+(?![\d%]|[.,]\d)|" + _FIGURE.pattern)


def figures(text):
    """Every figure the text states, in order. The denominator for an unsupported-figure rate."""
    return _FIGURE.findall(text)


def unsupported_figures(answer, evidence):
    """Figures the answer states that the evidence does not, in the order they appear.

    Both sides are read by the same extractor, which is what makes `unsupported_figures(text, text)`
    empty for every text. Searching the evidence as a string instead did not: a figure ending a sentence
    there is followed by a full stop, and the bounded search that stopped 18.2 passing for 18.25 also
    refused 2017 against evidence reading "for 2017." -- so a quote of the evidence failed against it.
    Comparing extracted figures keeps the 18.25 case, because that is not a figure the evidence states.

    An unsigned answer figure is supported by a signed evidence one of the same magnitude, because the
    model writes its minus sign as a separate token and the extractor then reads "- 5.6%" as unsigned
    while the evidence states "-5.6%". A *signed* figure still has to match exactly, so an answer that
    flips the sign on a grounded number is still caught.

    Thousands separators are compared away: the tokenizer drops them, so the model writes "1339.70" for a
    facts passage's "1,339.70", and 9 of 461 stock Q&A answers were refused for that alone.
    """
    stated = {figure.replace(",", "") for figure in figures(_KEY.sub(" ", evidence))}
    magnitudes = {figure.lstrip("+-") for figure in stated}
    windows = set(_KEY.findall(evidence)) | magnitudes
    return [figure.lstrip("_") for figure in _windows_named(answer)
            if (figure[1:] not in windows if figure[:1] == "_"
                else figure.replace(",", "") not in stated
                and (figure[:1] in "+-" or _magnitude(figure) not in magnitudes))]


BRACKETED = re.compile(r"\s*\(([^()]*)\)")
# a run of capitalised words, which may hold "of"; "Reserve Bank of India (RBI)" is one name with two spellings
_NAME = re.compile(r"[A-Z][\w&-]*(?:(?: of)? [A-Z][\w&-]*)*")
_ALIAS = re.compile(rf"({_NAME.pattern}) \(([A-Z][A-Z&]+)\)")
_MONTHS = frozenset("jan january feb february mar march apr april may jun june jul july aug august sep sept september "
                    "oct october nov november dec december".split())
_MONTH_ORDER = "jan feb mar apr may jun jul aug sep oct nov dec".split()
# A day as ISO or as a reader writes it. figures() reads "2026-03-31" as 2026, -03 and -31, and a reason listing those
# pieces reached the user, so a check reads each day whole, by its value
_MONTH = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DAY = re.compile(rf"\b(\d{{4}})-(\d\d)-(\d\d)\b|\b(\d{{1,2}}) ({_MONTH})\.?,? (\d{{4}})\b|"
                  rf"\b({_MONTH})\.? (\d{{1,2}}),? (\d{{4}})\b", re.I)


def days(text):
    """Each day `text` writes, in ISO: "7 Oct 2026", "October 7, 2026" and "2026-10-07" are one day."""
    found = []
    for day in _DAY.finditer(text):
        group = day.groups()
        year, month, dom = group[:3] if group[0] else (group[5], group[4], group[3]) if group[3] \
            else (group[8], group[6], group[7])
        month = int(month) if month.isdigit() else _MONTH_ORDER.index(month[:3].lower()) + 1
        found.append(f"{year}-{month:02}-{int(dom):02}")
    return found


def undated(text):
    """`text` without the days it writes, so its other figures can be read on their own."""
    return _DAY.sub(" ", text)


def unbound_figures(answer, passages):
    """(figure, name) for each figure the answer ties to a name no passage states it with, sentence by sentence.

    A figure is tied to the nearest name before it, or after it when none is before. Only names and figures some
    passage holds are read: a figure none holds is `unsupported_figures`' to catch, and reading names none holds
    refused 2 good replies of 298 to catch 1 bad, so "HCLTech shares lost 1.2%" is left to the entailment model. That
    model does not tie a number to what it measures: "TCS shares fell 1.2%" passed where 1.2% was the Nifty's fall.
    """
    held = [passage.lower() for passage in passages]
    unbound = []
    for sentence in sentences(answer):
        aliases = dict(_ALIAS.findall(sentence))
        sentence = BRACKETED.sub("", sentence)
        names = []
        for found in _NAME.finditer(sentence):
            spellings = _spellings(found.group(0), aliases.get(found.group(0)))
            if spellings and any(_says(passage, spellings) for passage in held):
                names.append((found.start(), spellings))
        for found in _FIGURE.finditer(sentence):
            before = [spellings for start, spellings in names if start < found.start()]
            tied = before[-1] if before else names[0][1] if names else ()
            if tied and _unsaid(found.group(0), tied, held):
                unbound.append((found.group(0), tied[0]))
    return unbound


def _unsaid(figure, spellings, held):
    """Whether some passage of `held` states `figure` and none of those says it by one of `spellings`."""
    stating = [passage for passage in held if _magnitude(figure) in map(_magnitude, figures(passage))]
    return bool(stating) and not any(_says(passage, spellings) for passage in stating)


def _spellings(run, alias):
    """A name's spellings, without the everyday words and months a capital can start: "The Sensex" is "Sensex". A name of
    three words or more is also spelt by its initials: "Reserve Bank of India" is "RBI"."""
    words = run.split()
    while words and _everyday(words[0]):
        words.pop(0)
    while words and _everyday(words[-1]):
        words.pop()
    initials = "".join(word[0] for word in words if word != "of")
    return tuple(filter(None, [" ".join(words), alias, initials if len(initials) > 2 else ""]))


def _everyday(word):
    """"The", "Oct" or a dangling "of", but not "IT" or "US", which are names in capitals."""
    return word.lower() in STOP | _MONTHS | {"of"} and not (len(word) > 1 and word.isupper())


def _says(passage, spellings):
    return any(re.search(rf"(?<!\w){re.escape(spelling.lower())}(?!\w)", passage) for spelling in spellings)


def _magnitude(figure):
    return figure.lstrip("+-").replace(",", "")


# a metric's plain meaning, set off by commas right after it: "p / b , the price against ... , was 3.81 x"
_ASIDE = re.compile(r" , [^,;]*? , ")


def mislabelled_figures(answer, evidence, metrics):
    """(figure, metric) for each figure the answer gives a metric that no clause of the evidence stating it names.

    `metrics` are spelling groups, ("Return on equity", "ROE"), read on both texts as `pretokenize` reads them. A figure
    is a metric's when it is the first after it, past a meaning set off by commas and any clauses without a figure, and
    no other metric comes first: in "a 1.7% free cash flow yield" after the dividend yield, 1.7% is not the dividend's.
    Tied as `unbound_figures` ties names, nearest first, 272 of 2,752 written answers were refused; this refuses 5.
    Read only to the next comma, it spoke ROE's 20.2% as ROIC in "roic, profit on ..., was light, was 20.2%".
    """
    spelt, named = _metrics(metrics)
    held = [" ".join(pretokenize(clause)) for clause in evidence.split(";")]
    mislabelled = []
    for sentence in sentences(" ".join(pretokenize(answer))):
        past = 0
        for metric in named.finditer(sentence):
            if metric.start() < past:
                continue
            aside = _ASIDE.match(sentence, metric.end())
            past = aside.end() if aside and not _FIGURE.search(aside.group(0)) else metric.end()
            clause = next((clause for clause in sentence[past:].split(";", 1)[0].split(",")
                           if named.search(clause) or _FIGURE.search(clause)), "")
            figure, spellings = _FIGURE.search(clause), spelt[metric.group(0)]
            if figure and not named.search(clause) and _unsaid(figure.group(0), spellings, held):
                mislabelled.append((figure.group(0), spellings[0]))
    return mislabelled


def own_meanings(answer, metrics):
    """The metrics `answer` gives a meaning of its own, set off by commas after it: "dividend yield , spare cash ... ,".

    The reply explains its terms in TERM_MEANINGS' words, and a written one contradicted them: dividend yield given free
    cash flow yield's meaning, then the right one in the block below.
    """
    spelt, named = _metrics(metrics)
    said = " ".join(pretokenize(answer))
    return [spelt[metric.group(0)][0] for metric in named.finditer(said)
            if (aside := _ASIDE.match(said, metric.end())) and not _FIGURE.search(aside.group(0))]


def metrics_named(text, metrics):
    """The spelling groups of `metrics` that `text` names, in the order it first names them, read as the label guard."""
    _, named = _metrics(metrics)
    groups = {" ".join(pretokenize(spelling)): group for group in metrics for spelling in group}
    return list(dict.fromkeys(groups[found.group(0)] for found in named.finditer(" ".join(pretokenize(text)))))


@functools.lru_cache(maxsize=4)
def _metrics(metrics):
    """({spelling: its metric's spellings}, a pattern finding any, longest first), as `pretokenize` reads them."""
    spelt = {}
    for group in metrics:
        read = tuple(" ".join(pretokenize(spelling)) for spelling in group)
        spelt.update(dict.fromkeys(read, read))
    longest = sorted(spelt, key=len, reverse=True)
    return spelt, re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, longest)) + r")(?!\w)")


def _windows_named(text):
    """The figures `text` states, with every window written "_14"."""
    return _NAMED_FIGURE.findall(_WINDOW.sub(r"_\1\2", re.sub(r"_ (?=\d)", "_", text)))


def is_refusal(text, refusal):
    """Opening words rather than equality, so a model that refuses and then keeps talking still counts.

    Shared because a scorer and a labeller that disagreed about what a refusal is would produce an
    abstention rate and a feedback label that cannot be compared.
    """
    return " ".join(refusal.split()[:7]) in text


def screen(answer, evidence, refusal):
    """(text to serve, unsupported figures). The refusal is returned whenever the list is non-empty.

    Both halves are returned rather than one: swapping in a refusal silently would hide exactly the
    rate this exists to measure.
    """
    unsupported = unsupported_figures(answer, evidence)
    return (refusal if unsupported else answer), unsupported


def snapshot_values(evidence):
    """{name: value} from a snapshot's "name value ; name value" evidence line."""
    found = {}
    for part in evidence.split(" ; "):
        name, _, value = part.partition(" ")
        if name and value:
            found[name] = value
    return found


# The answer forms' "above both", "below both" and "between its 20 and 50 day averages", and "above the 20 and 50"
_TREND = re.compile(r"\b(above|below|between) (?:both\b|(?:its|the) 20 and 50\b)")
TREND_FIGURES = ("close", "sma_20", "sma_50")


def trend_position(close, sma_20, sma_50):
    """Where the price is against its 20- and 50-day averages, in the trend words: "above", "below" or "between"."""
    return "above" if close > max(sma_20, sma_50) else "below" if close < min(sma_20, sma_50) else "between"


def wrong_trend(answer, evidence):
    """(what the answer says, where the price is) for each trend phrase the evidence's close and averages contradict.

    No guard read words, so 54 spoken answers on outside shares said "between" for a price above or below both averages,
    or the reverse, every figure right. A trend said of evidence with no averages is wrong too, where the price is None.
    """
    shown = snapshot_values(evidence)
    truth = (trend_position(*(float(shown[name]) for name in TREND_FIGURES))
             if all(name in shown for name in TREND_FIGURES) else None)
    return [(said, truth) for said in _TREND.findall(answer.lower()) if said != truth]


def other_shares(answer, evidence, names, slots):
    """The shares `answer` names that `evidence` does not, in order: those of `names`, else symbols written "<name>.NS",
    else a word where an answer form puts its instrument that neither the forms nor the evidence use.

    Outside the 101 trained stocks the generator named another share in 349 of 700 answers ("maruti. ns" for
    IPCALAB.NS), every figure supported. Both sides are read as `pretokenize` reads them, so "m & m. ns" is M&M.NS.
    `slots` is (the forms' words, the words before their instrument, the words after it): "lin has returned" names LIN,
    a word too short for `names`, and without the slots 14 such answers on outside shares were served at the 0.971 cut.
    """
    said, shown = pretokenize(answer), pretokenize(evidence)
    spelled = _spelled(names)
    longest = max(map(len, spelled), default=1)
    stated = set(_runs(shown, longest))
    held = {shown[end - 1] for end in _symbol_ends(shown)}
    # a symbol's suffix alone names no share: "ns reads 47" was served for CAMS.NS, its name lost
    named = set(shown) - {shown[end + 1] for end in _symbol_ends(shown)}
    suffixes = {end + 1 for end in _symbol_ends(said)}
    words, before, after = slots
    placed = [_word_from(said, at + step, step, suffixes) for at, word in enumerate(said) for step in (1, -1)
              if word in (before if step == 1 else after)]
    return list(dict.fromkeys([spelled[run] for run in _runs(said, longest) if run in spelled and run not in stated]
                              or [f"{said[end - 1]}.ns" for end in _symbol_ends(said) if said[end - 1] not in held]
                              or [word for word in placed if word and word not in words and word not in named]))


def _word_from(tokens, at, step, suffixes):
    """The first word from `at` on, stepping by `step` past figures, punctuation and symbols' `suffixes`: "am" in
    "am , is below", "cams" in "cams. ns reads"."""
    while 0 <= at < len(tokens) and (not tokens[at].isalpha() or at in suffixes):
        at += step
    return tokens[at] if 0 <= at < len(tokens) else ""


@functools.lru_cache(maxsize=4)
def _spelled(names):
    return {tuple(pretokenize(name)): name for name in names}


def _runs(tokens, longest):
    return [tuple(tokens[start:start + size]) for start in range(len(tokens))
            for size in range(1, min(longest, len(tokens) - start) + 1)]


def _symbol_ends(tokens):
    """Where each symbol written "<name>.NS" ends, at its full stop."""
    return [at for at in range(1, len(tokens) - 1) if tokens[at] == "." and tokens[at + 1] == "ns"]


# a model that claims direction is wrong by the project's own measurements, so the draft is refused
DIRECTION = re.compile(r"\b(will|is likely to|is expected to|should|could)\s+(rise|fall|go up|go down|increase|decrease|"
                       r"climb|drop|rally|decline|gain)|\b(bullish|bearish|upside target|price target|upward trend|"
                       r"downward trend|uptrend|downtrend|positive bias|negative bias)\b", re.I)
# terms a writer may reach for that the note never uses; adding one means it added a claim
NEW_TERMS = re.compile(r"\b(granger|sharpe|sortino|rsi|macd|bollinger|fibonacci|moving average|support level|"
                       r"resistance|p/e|earnings|dividend|beta|alpha|regression|neural|deep learning|sentiment)\b", re.I)
# praise the records do not support
OVERCLAIM = re.compile(r"\b(reliab\w*|trust\w*|accurate|better than random|strong evidence|"
                       r"performing well|confident(ly)?)\b", re.I)


# a cause the source never gives: "infosys is falling because it is a technology company" passed every
# figure check, since a reason invented out of true facts states no new number
REASON = re.compile(r"\b(because|due to|as a result of|driven by|owing to|on account of|thanks to|on the back of)\b",
                    re.I)


def added_claims(text, source):
    """Why `text` says more than `source` does: a direction call, praise, or a finance term it never uses.

    Compared against the source rather than judged alone, so a rewrite may keep a call the draft made
    ("turbulent" from the week-ahead model) and still may not add one.
    """
    reasons = []
    for pattern, what in ((DIRECTION, "it claims a direction"), (OVERCLAIM, "it praises a model beyond its record"),
                          (REASON, "it gives a reason the source does not")):
        made = {m.group(0).lower() for m in pattern.finditer(text)} - {m.group(0).lower() for m in pattern.finditer(source)}
        if made:
            reasons.append(f"{what}: “{sorted(made)[0]}”")
    added = {t.lower() for t in NEW_TERMS.findall(text)} - {t.lower() for t in NEW_TERMS.findall(source)}
    if added:
        reasons.append(f"it adds terms the source does not use: {', '.join(sorted(added))}")
    return reasons
