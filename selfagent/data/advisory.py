"""Advisory question, evidence and answer triples built from price history.

This is the task the generator is actually for. The Stack Exchange set asked it to recall facts that
appeared nowhere in its input — measured, only 31% of an answer's content words were present — so it
learned to ignore the evidence and produce fluent invention. Here every figure in the answer is one
the evidence states, in the same characters, so the answer is reachable by copying and a guardrail
can check it afterwards by search.

Each question comes in many phrasings and each answer in several forms, because a single template per
fact teaches a lookup table rather than language. Six phrasings per intent was not enough: the encoder
placed the five it trained on correctly and the sixth at chance, so it had memorised the strings. The
phrasings are now built from frames and interchangeable words, and what is held out is whole words
rather than whole templates — a reserved word appears in no training question at all, so placing it
takes knowing what it means.

Some rows deliberately ask for a figure the evidence does not carry, and their answer is a refusal.
Without them a model trained only on answerable rows assumes the evidence always holds the answer
and fabricates when it does not: training on answerable rows alone moved held-out perplexity on
ungrounded questions from 60 to 310.

Pure functions. The caller supplies the bars and, where the outlook is being discussed, the price
head's own probabilities; nothing here loads a file or runs a model.
"""

import re

from . import indicators

# One format per fact, applied once, so the characters in the answer are the characters in the
# evidence. A figure rendered two ways cannot be copied and cannot be checked.
_FORMATS = {
    "close": "{:.2f}",
    "return_5d": "{:+.1%}",
    "return_20d": "{:+.1%}",
    "volatility_20d": "{:.1%}",
    "rsi_14": "{:.0f}",
    "sma_20": "{:.2f}",
    "sma_50": "{:.2f}",
    "drawdown_60d": "{:.1%}",
    "atr_14": "{:.2f}",
}

UNSUPPORTED = "i do not have that in the evidence for this question, so i cannot answer it."

# Advisory questions are one short sentence, unlike the Stack Exchange ones, so the evidence gets the
# rest of the window: a full snapshot is 78 tokens and truncating it would cut a figure in half. A
# clean question reaches 15 tokens and a misspelling costs two more, measured at 19 over every frame.
QUESTION_TOKENS = 24

BARS_NEEDED = 61

RISK_NAMES = ("calm", "normal", "turbulent")


def snapshot(bars, end):
    """As-of facts for bar `end`. Reads no bar after it, so a window cannot see its own future."""
    closes = bars[: end + 1, 3]
    highs = bars[: end + 1, 1]
    lows = bars[: end + 1, 2]
    return {
        "close": float(closes[-1]),
        "return_5d": indicators.total_return(closes, 5),
        "return_20d": indicators.total_return(closes, 20),
        "volatility_20d": indicators.realized_volatility(closes, 20),
        "rsi_14": indicators.relative_strength_index(closes, 14),
        "sma_20": indicators.simple_moving_average(closes, 20),
        "sma_50": indicators.simple_moving_average(closes, 50),
        "drawdown_60d": indicators.max_drawdown(closes, 60),
        "atr_14": indicators.average_true_range(highs, lows, closes, 14),
    }


def as_text(facts):
    """The formatted figure for each fact, which both the evidence and the answer then quote."""
    return {name: _FORMATS[name].format(value) for name, value in facts.items()}


def render_evidence(ticker, shown):
    """The passage the encoder reads. Flat name-value lines, because that is what is checkable."""
    lines = [f"ticker {ticker}"] + [f"{name} {value}" for name, value in shown.items()]
    return " ; ".join(lines)


def risk_outlook(probabilities, names=RISK_NAMES):
    """The head's own call, as evidence lines, so the answer quotes it instead of inventing it.

    The confidence belongs in the evidence for the same reason every other figure does: an answer
    stating a number it was never shown is exactly what the guardrail exists to catch.
    """
    best = max(range(len(probabilities)), key=lambda i: probabilities[i])
    return {"outlook": names[best], "outlook_confidence": f"{probabilities[best]:.0%}"}


# Tokenising splits a name like return_20d into "return _ 20 d", so a slot is matched piece by piece
# with optional spaces rather than as one literal.
_SLOT_PIECES = re.compile(r"[a-z]+|\d+|_")


def slot_names(shown):
    """Each evidence key standing in for its own figure, to hand `row` in place of the figures.

    The copy-trained generator does learn to copy, but it slips a digit: it answered +3.3% where the
    evidence said +3.7%, which is a plausible, unfalsifiable, wrong number in front of someone making
    a trade. Naming the fact instead and substituting the value in `fill` costs the model nothing it
    was good at and makes faithfulness structural rather than a rate to keep watching.
    """
    return {name: name for name in shown}


def fill(text, shown):
    """The served answer: every fact the model named replaced by that fact's figure.

    Longest name first, so outlook_confidence is not half eaten by outlook.
    """
    for name in sorted(shown, key=len, reverse=True):
        pieces = r"\s*".join(re.escape(piece) for piece in _SLOT_PIECES.findall(name))
        text = re.sub(rf"(?<![a-z\d]){pieces}(?![a-z\d])", lambda _, v=shown[name]: v, text)
    return text


def _trend_words(facts):
    if facts["close"] > facts["sma_20"] > facts["sma_50"]:
        return "above both its 20 and 50 day averages"
    if facts["close"] < facts["sma_20"] < facts["sma_50"]:
        return "below both its 20 and 50 day averages"
    return "between its 20 and 50 day averages"


def _rsi_words(value):
    if value >= 70.0:
        return "overbought territory"
    if value <= 30.0:
        return "oversold territory"
    return "neither overbought nor oversold"


# No integer appears in a question or answer unless the evidence carries it, which the key names do
# for 5, 14, 20, 50 and 60. A phrase like "over the past 3 weeks" would be unsupported on sight.
#
# Questions are built from frames and interchangeable content words rather than written out one by one.
# With six wordings per intent the encoder placed 83% of the ones it trained on correctly and the
# reserved one at chance, so it had learned the strings themselves.
_FRAMES = {
    "performance": (
        "how has {t} been doing over the last month ?",
        "what has {t} done recently ?",
        "give me a summary of how {t} is trading .",
        "has {t} gone up or down lately ?",
        "where does {t} stand after the last few weeks ?",
        "tell me about the recent price action in {t} .",
        "how is {t} {w} ?",
        "how has {t} been {w} ?",
        "i want to know whether {t} is {w} well .",
    ),
    "overbought": (
        "is {t} overbought right now ?",
        "what does the rsi say about {t} ?",
        "is {t} oversold ?",
        "how does momentum look on {t} ?",
        "is {t} {w} at these levels ?",
        "does {t} look {w} to you ?",
        "would you say {t} is {w} ?",
        "is there any sign {t} is {w} ?",
    ),
    "volatility": (
        "how volatile is {t} at the moment ?",
        "how much does {t} move around ?",
        "what is volatility like on {t} ?",
        "how wide are the daily swings in {t} ?",
        "how big are the daily moves in {t} ?",
        "how much does {t} bounce around ?",
        "is {t} a {w} name ?",
        "would you call {t} {w} ?",
        "tell me whether {t} is {w} or not .",
    ),
    "drawdown": (
        "how far is {t} off its recent high ?",
        "what is the worst drop in {t} lately ?",
        "how deep was the last fall in {t} ?",
        "what drawdown has {t} taken ?",
        "how much has {t} given back from its peak ?",
        "has {t} had a {w} stretch recently ?",
        "has it been a {w} run for {t} ?",
        "i am wondering if {t} has been {w} .",
    ),
    "risk": (
        "how risky is {t} over the next week ?",
        "what should i expect from {t} next week ?",
        "what does your model say about risk in {t} ?",
        "how much turbulence do you expect in {t} ?",
        "read the week ahead for {t} .",
        "is {t} likely to be {w} from here ?",
        "do you expect {t} to be {w} next week ?",
        "give me a sense of whether {t} stays {w} .",
    ),
    "buy": (
        "should i buy {t} ?",
        "is {t} a good buy right now ?",
        "would you go long {t} ?",
        "is this a good entry on {t} ?",
        "do you like {t} here ?",
        "should i be buying or selling {t} ?",
        "would you {w} {t} at this price ?",
        "is {t} one to {w} here ?",
        "i am thinking i should {w} {t} .",
    ),
    "unsupported": (
        "what did {t} earn last quarter ?",
        "who is the chief executive of {t} ?",
        "how many shares of {t} are outstanding ?",
        "what do analysts rate {t} ?",
        "who are the main competitors of {t} ?",
        "when does {t} next report results ?",
        "what sector is {t} in ?",
        "where is {t} headquartered ?",
        "what is the {w} for {t} ?",
        "do you know the {w} for {t} ?",
        "i need the {w} on {t} .",
    ),
}

# One frame per intent is reserved too, and it reuses only trained words. Holding out a word alone is
# too easy -- "is X overheated at these levels" differs from a trained question by one word, so surface
# similarity places it without knowing what the word means. Reserving both axes separates which of the
# two a failure comes from, and their intersection is the case a real user actually types.
_HELD_OUT_FRAMES = {
    "performance": "i want to know whether {t} is {w} well .",
    "overbought": "is there any sign {t} is {w} ?",
    "volatility": "tell me whether {t} is {w} or not .",
    "drawdown": "i am wondering if {t} has been {w} .",
    "risk": "give me a sense of whether {t} stays {w} .",
    "buy": "i am thinking i should {w} {t} .",
    "unsupported": "i need the {w} on {t} .",
}

# Both polarities of a word ask the same question: "would you call X quiet" and "would you call X wild"
# are each a volatility question, and the reading answers both.
_WORDS = {
    "performance": ("doing", "performing", "holding up", "shaping up", "faring", "getting on"),
    "overbought": ("stretched", "extended", "overdone", "exhausted", "overheated", "toppy"),
    "volatility": ("choppy", "jumpy", "quiet", "calm", "wild", "restless", "twitchy", "steady"),
    "drawdown": ("bad", "rough", "poor", "painful", "torrid", "grim"),
    "risk": ("bumpy", "settled", "lively", "rough", "eventful", "smooth"),
    "buy": ("buy", "hold", "add to", "get into", "own", "avoid", "touch", "pick up"),
    "unsupported": (
        "dividend yield", "price target", "market capitalisation", "price to earnings ratio",
        "revenue growth", "debt load", "book value", "free cash flow",
    ),
}

# A reserved word appears in no training question at all, so placing one correctly takes knowing what it
# means rather than having seen it. Each is an ordinary synonym of a word that is trained, which is why
# failing on them is evidence about the encoder's English and not about the difficulty of the question.
_HELD_OUT_WORDS = {
    "performance": ("getting on",),
    "overbought": ("toppy",),
    "volatility": ("twitchy",),
    "drawdown": ("torrid",),
    "risk": ("eventful",),
    "buy": ("pick up",),
    "unsupported": ("free cash flow",),
}

TRAINED = "trained"

# Routing-only paraphrases, and the fix for the largest weakness left: the router places a sentence shape
# it has never seen 68% of the time, which is 23.5 of the 26.5 points of wrong answers. Every frame above
# is a direct question, so an indirect request, a fragment or an imperative has nothing near it to match.
# What the router is short of is shapes to match against, not weights — fine-tuning the encoder for this
# measured 68% -> 32%. These are matched against and never trained on, and a question they catch is
# rewritten into the intent's canonical phrasing, so what the decoder reads does not change.
_PARAPHRASES = {
    "performance": (
        "recent performance on {t} ?",
        "any idea how {t} has been trading ?",
        "could you tell me how {t} has been doing ?",
        "walk me through the recent move in {t} .",
        "i would like to know how {t} has traded lately .",
        "{t} price action lately ?",
        "catch me up on {t} .",
        "anything to say about how {t} is trading ?",
    ),
    "overbought": (
        "rsi on {t} ?",
        "any idea whether {t} is overbought ?",
        "could you check momentum on {t} ?",
        "show me where the rsi sits on {t} .",
        "i would like to know if {t} is overbought .",
        "{t} momentum reading ?",
        "talk me through the rsi on {t} .",
        "anything in the rsi for {t} ?",
    ),
    "volatility": (
        "volatility on {t} ?",
        "any idea how much {t} moves around ?",
        "could you tell me how volatile {t} is ?",
        "walk me through the daily swings in {t} .",
        "i would like to know how volatile {t} is .",
        "{t} daily range ?",
        "give me the volatility picture on {t} .",
        "anything unusual in how much {t} moves ?",
    ),
    "drawdown": (
        "drawdown on {t} ?",
        "any idea how far {t} is off its high ?",
        "could you tell me the worst fall in {t} ?",
        "show me how much {t} has dropped from its peak .",
        "i would like to know the deepest fall in {t} .",
        "{t} off its high ?",
        "talk me through the last drop in {t} .",
        "anything to say about how far {t} has fallen ?",
    ),
    "risk": (
        "week ahead on {t} ?",
        "any idea how risky {t} looks next week ?",
        "could you tell me what to expect from {t} next week ?",
        "walk me through the risk in {t} for the week ahead .",
        "i would like to know how risky {t} is next week .",
        "{t} risk for the coming week ?",
        "give me your read on {t} for next week .",
        "anything to watch in {t} this coming week ?",
    ),
    "buy": (
        "buy {t} ?",
        "any idea whether i should buy {t} ?",
        "could you tell me if {t} is worth buying ?",
        "talk me through whether to buy {t} .",
        "i would like to know if i should buy {t} .",
        "{t} a buy here ?",
        "give me your view on buying {t} .",
        "anything stopping me from buying {t} ?",
    ),
    "unsupported": (
        "earnings on {t} ?",
        "any idea what {t} earned last quarter ?",
        "could you tell me who runs {t} ?",
        "show me the analyst ratings on {t} .",
        "i would like to know the dividend yield on {t} .",
        "{t} market capitalisation ?",
        "give me the price target on {t} .",
        "anything on when {t} reports next ?",
    ),
}

# Three shape families deliberately left out of the pool above, so that widening it does not leave the
# project without an unseen shape to measure on: a leading subordinate clause, a two-clause aside that
# states a situation before it asks, and an opening admission of not knowing. Held out here means the
# router never matches against them, the same contract `_HELD_OUT_FRAMES` holds for training.
_HELD_OUT_PARAPHRASES = {
    "performance": (
        "given how the market has been , where has {t} ended up ?",
        "i am looking at {t} and i cannot tell how it has done .",
        "not sure what {t} has been doing , can you fill me in ?",
    ),
    "overbought": (
        "given the run it has had , is {t} overbought ?",
        "i keep hearing {t} is overbought and i want to check .",
        "not sure if {t} is overbought , what does the rsi say ?",
    ),
    "volatility": (
        "given how the market has been , how volatile is {t} ?",
        "i am sizing a position in {t} and i need its volatility .",
        "not sure how much {t} swings around , can you check ?",
    ),
    "drawdown": (
        "given the peak it made , how far has {t} fallen ?",
        "i am holding {t} and i want to see the worst of the fall .",
        "not sure how deep the fall in {t} went , can you check ?",
    ),
    "risk": (
        "given how it has traded , how risky is {t} next week ?",
        "i am deciding whether to hold {t} through next week and i need the risk .",
        "not sure what the week ahead looks like for {t} , can you read it ?",
    ),
    "buy": (
        "given where it is trading , should i buy {t} ?",
        "i have cash sitting idle and i am looking at {t} .",
        "not sure whether to buy {t} , what do you think ?",
    ),
    "unsupported": (
        "given the results season , when does {t} report ?",
        "i am building a spreadsheet and i need the book value of {t} .",
        "not sure what sector {t} is in , can you check ?",
    ),
}

# Every frame above is a clean, punctuated, single clause, and nobody types that way. The corpus is
# already half conversational English and the encoder places an unseen word 15/15, so the register is
# the gap the task data leaves open rather than anything pretraining is missing.
_FILLERS = ("hey", "so", "ok", "hi", "quick q")

_SHORTHAND = (
    ("what is", "whats"), ("how is", "hows"), ("i am", "im"), ("it is", "its"),
    ("does not", "doesnt"), ("right now", "rn"), ("please", "pls"), ("you", "u"),
)


def _misspell(word, rng):
    """A dropped letter, two swapped, or one doubled — the three mistakes a keyboard actually makes."""
    at = int(rng.integers(1, len(word) - 1))
    mistake = int(rng.integers(3))
    if mistake == 0:
        return word[:at] + word[at + 1 :]
    if mistake == 1:
        return word[:at] + word[at + 1] + word[at] + word[at + 2 :]
    return word[:at] + word[at] + word[at:]


def casual(template, rng):
    """One frame rewritten the way somebody types it into a chat box.

    Takes the template, not the finished question, so the ticker is never corrupted: the answer quotes
    it back and a mistyped one would teach the model to copy the mistake. Nothing here adds a digit,
    which would be an unsupported figure in the question on sight.
    """
    words = template.split()
    if words[-1] in ".?" and rng.random() < 0.7:
        words.pop()
    text = " ".join(words)
    for spelled, short in _SHORTHAND:
        if spelled in text and rng.random() < 0.5:
            text = text.replace(spelled, short, 1)
    words = text.split()
    long_enough = [i for i, word in enumerate(words) if len(word) >= 5 and "{" not in word]
    if long_enough and rng.random() < 0.5:
        at = int(rng.choice(long_enough))
        words[at] = _misspell(words[at], rng)
    if rng.random() < 0.25:
        words.insert(0, _FILLERS[int(rng.integers(len(_FILLERS)))])
    return " ".join(words)


def _build(intent):
    """Every phrasing of one intent, as (text, novelty), frame by frame.

    The novelty names which axis training never saw, so a failure says whether the word, the shape or
    the pair of them is what the encoder could not place.
    """
    built = []
    for frame in _FRAMES[intent]:
        new_frame = frame == _HELD_OUT_FRAMES[intent]
        for word in _WORDS[intent] if "{w}" in frame else (None,):
            novel = [axis for axis, new in (("word", word in _HELD_OUT_WORDS[intent]),
                                            ("frame", new_frame)) if new]
            text = frame.replace("{w}", word) if word else frame
            built.append((text, " and ".join(novel) or TRAINED))
    return tuple(built)


_ASK = {intent: _build(intent) for intent in _FRAMES}


def _performance(ticker, facts, shown):
    trend = _trend_words(facts)
    return (
        f"{ticker} has returned {shown['return_20d']} over 20 days and "
        f"{shown['return_5d']} over the last 5 , and sits {trend} at {shown['close']} .",
        f"over 20 days {ticker} is {shown['return_20d']} , and {shown['return_5d']} over the last 5 . "
        f"it trades at {shown['close']} , {trend} .",
        f"the 20 day move is {shown['return_20d']} and the 5 day move {shown['return_5d']} . "
        f"{ticker} is {trend} , last at {shown['close']} .",
    )


def _overbought(ticker, facts, shown):
    words = _rsi_words(facts["rsi_14"])
    return (
        f"its 14 day rsi is {shown['rsi_14']} , which is {words} .",
        f"on a 14 day rsi of {shown['rsi_14']} , {ticker} is {words} .",
        f"{ticker} reads {shown['rsi_14']} on the 14 day rsi , so {words} .",
    )


def _volatility(ticker, facts, shown):
    return (
        f"annualised volatility over the last 20 days is {shown['volatility_20d']} , "
        f"and the average daily range is {shown['atr_14']} on a price of {shown['close']} .",
        f"{ticker} has run at {shown['volatility_20d']} annualised over 20 days , with an average "
        f"14 day range of {shown['atr_14']} against a price of {shown['close']} .",
        f"the last 20 days put it at {shown['volatility_20d']} annualised . its typical daily range "
        f"is {shown['atr_14']} , on {shown['close']} .",
    )


def _drawdown(ticker, facts, shown):
    return (
        f"the worst fall inside the last 60 days is {shown['drawdown_60d']} , "
        f"against a 50 day average of {shown['sma_50']} .",
        f"over 60 days {ticker} has taken {shown['drawdown_60d']} from its peak , with the 50 day "
        f"average at {shown['sma_50']} .",
        f"its deepest 60 day drop is {shown['drawdown_60d']} . the 50 day average sits at "
        f"{shown['sma_50']} .",
    )


def _risk(ticker, facts, shown):
    return (
        f"the model reads the week ahead as {shown['outlook']} , at {shown['outlook_confidence']} "
        f"confidence . over the last 20 days annualised volatility was {shown['volatility_20d']} "
        f"and the average daily range {shown['atr_14']} on a price of {shown['close']} .",
        f"i would call the coming week {shown['outlook']} , though only at "
        f"{shown['outlook_confidence']} confidence . trailing 20 day volatility is "
        f"{shown['volatility_20d']} , with a daily range around {shown['atr_14']} .",
        f"{shown['outlook']} is the call for the week ahead , at {shown['outlook_confidence']} . "
        f"for context {ticker} has run at {shown['volatility_20d']} over 20 days , trading at "
        f"{shown['close']} .",
    )


def _buy(ticker, facts, shown):
    """The buy question, answered with the abstention the measurements support.

    A week-ahead direction call scored 35.7% against a 36.0% baseline, so any confident wording here
    would be the model dressing a coin toss as a view. Every figure it does give is in the evidence.
    """
    return (
        f"i do not have a directional signal for {ticker} , and a one week call on it is little "
        f"better than a coin toss . what the evidence does say is that it is trading at "
        f"{shown['close']} , "
        f"{shown['return_20d']} over 20 days , with volatility {shown['volatility_20d']} . "
        f"this is not a recommendation to trade .",
        f"i cannot tell you which way {ticker} goes , and i will not pretend otherwise . what i can "
        f"say is that it is at {shown['close']} , {shown['return_20d']} over 20 days , running at "
        f"{shown['volatility_20d']} volatility . this is not a recommendation to trade .",
        f"no view on direction , because a one week call on {ticker} is a coin toss . the evidence "
        f"has it at {shown['close']} , {shown['return_20d']} over 20 days , volatility "
        f"{shown['volatility_20d']} . this is not a recommendation to trade .",
    )


def _unsupported(ticker, facts, shown):
    return (UNSUPPORTED,)


_ANSWER = {
    "performance": _performance,
    "overbought": _overbought,
    "volatility": _volatility,
    "drawdown": _drawdown,
    "risk": _risk,
    "buy": _buy,
    "unsupported": _unsupported,
}

INTENTS = tuple(_ASK)

EVIDENCE_KEYS = tuple(_FORMATS) + ("outlook", "outlook_confidence")


def _needs():
    """Which evidence facts each intent's answers quote, read off the templates themselves.

    Derived rather than listed, because a template that starts quoting another figure would leave a
    hand-written list silently wrong — and serving uses this to refuse an intent whose facts are
    missing instead of letting the decoder fill the gap. Substring matching over-declares rather than
    under-declares, which fails towards a refusal.
    """
    numeric = {name: 0.0 for name in _FORMATS}
    named = {name: name for name in EVIDENCE_KEYS}
    return {
        intent: tuple(name for name in EVIDENCE_KEYS
                      if any(name in answer for answer in _ANSWER[intent]("TICKER", numeric, named)))
        for intent in _ANSWER
    }


NEEDS = _needs()


def phrasings(intent):
    """How many ways this intent can be asked, held-out phrasing included."""
    return len(_ASK[intent])


def trained_phrasings(intent):
    """The phrasing indices training saw. A served question is rewritten into one of these, never
    answered in its own words, because the decoder scores 90.5% exact on wording it trained on and
    28.5% on wording it did not."""
    return tuple(p for p in range(phrasings(intent)) if not is_held_out(intent, p))


def canonical_phrasing(intent):
    """The phrasing a paraphrase is rewritten into: the intent's plainest trained wording.

    One wording rather than the nearest of several, because the decoder answers every trained phrasing
    at 90.5% and choosing between them would be a second ranking nobody has measured.
    """
    return trained_phrasings(intent)[0]


def paraphrases(intent):
    """Frames the router matches against and nothing trains on. `{t}` is the instrument, as above."""
    return _PARAPHRASES[intent]


def held_out_paraphrases(intent):
    """Frames in shape families the router's pool does not carry, to measure the widened pool on."""
    return _HELD_OUT_PARAPHRASES[intent]


def is_held_out(intent, phrasing):
    """True for the phrasings reserved to measure generalisation, which training must never see."""
    return novelty(intent, phrasing) != TRAINED


def novelty(intent, phrasing):
    """Which axis training never saw for this phrasing: the word, the frame, both, or neither."""
    return _ASK[intent][phrasing][1]


def ask(intent, ticker, phrasing, rng=None):
    """The question alone, for anything measuring the wording without needing a snapshot.

    With an rng the wording is roughed up into chat register; the intent and the facts asked for are
    untouched, so the same answer is still the right one.
    """
    template = _ASK[intent][phrasing][0]
    return (casual(template, rng) if rng is not None else template).format(t=ticker)


def row(intent, ticker, facts, shown, phrasing, rng=None):
    """(question, answer) for one intent asked the `phrasing` way.

    The answer form rotates independently of the question phrasing, so the model cannot learn that a
    given wording of the question implies a given wording of the answer.
    """
    answers = _ANSWER[intent](ticker, facts, shown)
    return ask(intent, ticker, phrasing, rng), answers[phrasing % len(answers)]
