"""A facts answer written by our code from the share's facts lines, so every figure and date in it is a line's own,
under a lead-in that answers the question asked.

The question's area picks the lines that answer it ("what does it do" the profile, "is it expensive" the valuation), or
a metric it names picks that metric's line: from the passage the facts picker chose, or from the share's other lines when
the passage lacks them. What no line gives is said plainly, then the closest facts. Each lead-in is picked by a hash of
the question, so the wording varies from question to question and never from run to run.
"""
import re
from collections import namedtuple

from backend.advisory.domains import FACTS_DOMAIN_WORDS
from backend.models.agent.finance import FACTS_ASKED
from backend.models.data.stock_qa import (DEFINITIONS, FACT_SEPARATOR, FUNDAMENTAL_LABELS, METRICS, STOCK_LINE, lean,
                                          own_names, stable_rank, without_names)
from backend.models.guardrails.guardrails import metrics_named

# What a question can ask of a share's facts: its words, the words of the lines that answer it, what a lead-in calls it
# and what a reply says is missing. The last two, a price move and the profile, yield to any other area named with them:
# "did its profit fall" is profit, and "what does its debt look like" is debt.
Area = namedtuple("Area", "asks says noun lacks")
Reply = namedtuple("Reply", "text evidence")
DRAWDOWN = ("below its peak", "below its highest close")
PRICE = Area(("technical", "trend", "average", " ema", " sma", "rsi", "relative strength", "macd", "momentum", "support",
              "resistance", "chart", "price"),
             ("[technical]", "[pattern]", "[candlestick]", "price change", *DRAWDOWN), "price", "readings of its price")
# Words from the facts-labelled questions of price_short_templates.json that named no area: 44 of the 122. Their lines
# are the returns, the averages, a breakout, the run of highs and lows, and how far the price is below its peak
MOVED = PRICE._replace(asks=(" rise", " rising", " rose", " fall", " fell", " move", " broke", " range", "volume",
                             " trading", "strength", "stronger", "weaker", " higher", " lower", " going ", " recover",
                             " bounce", "upside", "doing better", "best period", "worst period", "buyers", "sellers",
                             " pressure", " level"),
                       says=("price change", "-day return", "-day average", "-day high", "swing structure", *DRAWDOWN))
AREAS = (
    Area(("debt", "borrow", "leverag", "loan", " owe", "balance sheet", "solven", "liquid"),
         ("debt/equity", "current ratio", "covers interest", "shareholders' equity"),
         "debt", "figures on its debt"),
    Area(("dividend", "yield", "payout"), ("dividend yield", "free cash flow yield"), "dividend", "its dividend yield"),
    Area(("cash",), ("free cash flow", "cash and equivalents"), "cash", "figures on its cash"),
    Area(("expensive", "cheap", "valued", "valuation", "p/e", " pe ", "peg", "ebitda", "price to earnings", "book",
          "priced", "worth"), ("p/e", "p/b", "ev/", "peg", "market value"), "valuation", "figures on its valuation"),
    Area(("grow", "revenue", "sales", "topline", "top line"),
         ("revenue", "eps changed", "net income", "growth per year"), "growth", "figures on its growth"),
    Area(("profit", "margin", "return on", "roe", "roic", "roa", "making money", "earn", "loss"),
         ("return on", "margin", "net income"), "profit", "figures on its profit"),
    Area(("risk", "safe", "downside", "volatil", "wrong", "threat", "problem", "beta", "drawdown"),
         ("[risk]", "[regime]", "volatil", "beta"), "risk", "figures on its risk"),
    PRICE,
    Area(("competit", "rival", "peer"), ("peer ",), "competitors", "a list of its competitors"),
    MOVED,
    Area(FACTS_DOMAIN_WORDS["profile"], ("[profile]", "(yahoo profile)"), "business", "a description of its business"),
)
LEADS = ("Here is what our facts say about the {noun} of {name}.", "For {name}, our facts on its {noun} are these.",
         "Our facts give this on the {noun} of {name}.")
ANY_LEADS = ("Here is what our facts say about {name}.", "For {name}, our facts are these.",
             "Our facts give this on {name}.")
GAPS = ("For {name}, our facts here do not give {what}.", "The facts we hold for {name} do not include {what}.",
        "Nothing in our facts for {name} gives {what}.")
CLOSEST = ("The closest facts we hold are these.", "Here is what we do hold.", "What our facts do give is this.")

# How questions spell out what METRICS abbreviates: 35 of the 178 held facts questions asked for the "price to earnings
# ratio"
SPELT_OUT = ((r"price[ -]to[ -]earnings", "P/E"), (r"debt[ -]to[ -]equity", "Debt/equity"),
             (r"earnings per share", "EPS"))
# "what does" opens price questions too ("what does its volume say"), so a price move's broad words yield to the profile
# only when another of its words names it or the question asks what the company does: "what does ITC do abroad"
_ASKS_WHAT_IT_DOES = re.compile(r"\bwhat does (\S+ ){0,3}?(do|make|sell)s?\b", re.I)
_TAG = re.compile(r"\[([^\]]+)\] ")
_SECTOR = re.compile(r"Sector (.+), industry (.+) \(yahoo profile\)\.")
# the lines `stock_qa.facts` writes as a label and its figure, which read as a sentence with an "is" between them
_LABELS = (*FUNDAMENTAL_LABELS.values(), "Market value", "Cash and equivalents")


def reply(question, passage, all_lines):
    """The answer to `question` and the evidence it rests on: `passage`, which `stock_qa.facts_passage` laid out with the
    share's line first, then any of the share's `all_lines` it states that the passage lacks."""
    first, *rest = passage.split(FACT_SEPARATOR)
    names, name = own_names(first), "{} ({})".format(*STOCK_LINE.match(first).groups())
    facts = [line for line in rest if line != DEFINITIONS]
    # the picker fits the passage to the generator's window, so the line asked for is often one of the share's others
    others = [line for line in map(lean, all_lines[1:]) if line != DEFINITIONS and line not in facts]
    # read without the share's names, as the picker reads them: "Beta Drugs" asks for no beta
    asking = without_names(question, names)
    for spelt, metric in SPELT_OUT:
        asking = re.sub(spelt, metric, asking, flags=re.I)
    asked = metrics_named(asking, METRICS)
    named = [area for area in AREAS if any(word in f" {asking.lower()} " for word in area.asks)]
    if _ASKS_WHAT_IT_DOES.search(asking) or any(word in f" {asking.lower()} " for word in FACTS_ASKED["profile"]):
        named = [area for area in named if area is not MOVED]
    # a question naming several, as a scorecard does, is answered from every fact the picker chose for it
    area = named[0] if named and set(named[1:]) <= set(AREAS[-2:]) else None
    held = {line: metrics_named(without_names(line, names), METRICS) for line in facts + others}
    stating = [line for line in facts + others if set(held[line]) & set(asked)]
    missing = [metric for metric in asked if not any(metric in held[line] for line in stating)]
    of_area = _of(area, facts, names) or ([] if stating else _of(area, others, names))
    answering = stating if asked else of_area if area else facts
    lacks = (" or ".join(f"its {_spoken(metric[0])}" for metric in missing) if asked
             else area.lacks if area else "any facts")
    gap = _pick(GAPS, question).format(name=name, what=lacks)
    evidence = FACT_SEPARATOR.join([passage, *(line for line in stating + of_area if line in others)])
    if answering:
        lead = _pick(LEADS, question).format(noun=area.noun, name=name) if area else _pick(ANY_LEADS, question).format(
            name=name)
        return Reply(" ".join([lead, *map(stated, answering), *([gap] if missing else [])]), evidence)
    closest = of_area or facts
    return Reply(" ".join([gap, *([_pick(CLOSEST, question), *map(stated, closest)] if closest else [])]), evidence)


def _of(area, lines, names):
    """The lines giving `area`, read without the share's names; a line saying a figure is "Not available" answers only a
    question asking for that figure."""
    return [line for line in lines if area and any(word in without_names(line, names).lower() for word in area.says)
            and not stated(line).startswith("Not available")]


def stated(line):
    """One facts line as a sentence, every figure, unit and date in it as the passage gives them."""
    tag = _TAG.match(line)
    text = line[tag.end():] if tag else line
    sector = _SECTOR.fullmatch(text)
    if sector:
        return f"Yahoo lists its sector as {sector.group(1)} and its industry as {sector.group(2)}."
    label = next((label for label in _LABELS if re.match(rf"{re.escape(label)} [-+\d]", text)), None)
    if label:
        text = f"{label} is {text[len(label) + 1:]}"
    head, cut, day = text.rpartition(". (")
    if cut and day.endswith(")."):
        text = f"{head} ({day}"
    return f"Our model's reading: {text}" if tag and "model output" in tag.group(1) else text


def _spoken(term):
    """"Return on equity" as a sentence says it, "return on equity"; "P/E" and "EV/sales" as they are."""
    return term[0].lower() + term[1:] if term[1:2].islower() else term


def _pick(frames, question):
    return frames[int(stable_rank(question, frames[0]), 16) % len(frames)]
