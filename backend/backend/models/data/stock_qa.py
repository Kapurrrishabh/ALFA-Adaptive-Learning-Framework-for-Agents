"""Stock questions answered from a dated snapshot: which stocks, the facts a row may use, the split it
belongs to, and the check an answer must pass before anything trains on it.

Two template sets feed it: the ten topic questions of stock_qa_1000.csv and the 480 templates of the
question library. Neither is trained on as written. Each placeholder is filled with a real stock and the
answer is written from that stock's snapshot, so every figure in an answer can be found again in the
facts its row carries -- the property `advisory` relies on, at a wider range of questions.

Pure functions; `training/prepare_stock_qa.py` fetches and writes.
"""

import hashlib
import re
from typing import Dict, Iterable, List, Mapping, Sequence, Set

from backend.advisory.domains import FACTS_DOMAIN_WORDS, named_domains
from backend.advisory.fundamental.fundamentals import TRAILING_PE
from backend.advisory.report import UNAVAILABLE
from backend.advisory.service import company_terms
from backend.config import resolve_symbol
from backend.models.guardrails import guardrails

SLOT = re.compile(r"\{([a-z_0-9]+)\}")
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# Every row carries this line, so an answer may use the indicator settings and thresholds it explains:
# they are then figures its evidence states, under the same rule the server screens answers with.
DEFINITIONS = ("Definitions: RSI(14) runs from 0 to 100, and readings above 70 are often called overbought and below 30 "
               "oversold. MACD uses 12, 26 and 9 days. The 20-, 50- and 200-day averages are simple averages of closing "
               "prices. A beta of 1 means the stock tends to move with the market.")
ADVICE = [r"\byou should (buy|sell)\b", r"\bbuy now\b", r"\bsell now\b", r"\bguarantee", r"\bwill (definitely|surely)\b",
          r"\bsure to\b", r"\bcan(no|')t lose\b", r"\bstrong buy\b", r"\btrim it\b", r"\bbefore (you buy|buying|trading|acting)\b"]
# "fell -23.9%" reads to a beginner as a rise; the direction word already carries the sign
DOUBLE_NEGATIVE = re.compile(r"\b(fell|falls?|falling|down|dropped|drops?|declined?|declining|lower|lost|loses|decreased|shrank|"
                             r"slipped|slid|below|under)\s+(by\s+)?[-−]\d"
                             r"|\b(fall|drop|loss|losses|drawdown|decline)\b[^.;:\n]{0,60}?\b(is|was|of)\s+[-−]\d"
                             r"|\b(rose|grew|gained|climbed|increased|up)\s+(by\s+)?\+\d", re.I)
# A term, and the plain words that show an answer has said what it means. A heuristic for the revision pass,
# not a rule: an answer may explain a term in words this list does not hold.
MEANING_CUES = {
    r"\bP/E\b": r"times (its |a year's |the |yearly |annual )?(profit|earnings)|profit per share|earnings per share|price (against|compared with|relative to|divided by)|for every (rupee|dollar|₹|\$)|years of (profit|earnings)",
    r"\bROE\b|return on equity": r"owners'|shareholders'|owners’|shareholders’|money (the )?owners",
    r"\bROA\b|return on assets": r"everything (it|the company|the business) owns|all (it|the company) owns|its assets|total assets",
    r"\bROIC\b|return on invested capital": r"owners and lenders|lenders and owners|all (the )?money put in",
    r"EV/EBITDA": r"whole business|debt included|including (its )?debt|enterprise value",
    r"\bPEG\b": r"divided by (its )?(profit|earnings) growth|adjust\w* (the )?P/E for growth|growth-adjusted",
    r"\bP/B\b|price to book": r"net assets|owners'? money per share|owners’ money per share|worth on (its|the) books|what (it|the company) owns minus what it owes",
    r"forward P/E": r"expected (profit|earnings)|next year's (profit|earnings)|profit (it is )?expected|earnings (are )?expected",
    r"MACD": r"momentum",
    r"\bEPS\b": r"profit per share|earnings per share",
    r"interest cover": r"times (its |the )?interest|covers (its |the )?interest|interest bill",
    r"net margin": r"kept as profit|of every (rupee|dollar|₹|\$) of (sales|revenue)|share of (revenue|sales) (it )?(keeps|kept)|left as profit",
    r"operating margin": r"running costs|day-to-day costs|after (its )?costs of running",
    r"\bpercentile\b": r"higher than|out of 100|of (its )?past (days|readings)|than on \d+% of",
    r"risk-off": r"cautious|nervous|avoid(ing)? risk|defensive|selling riskier",
    r"\bregime\b": r"(rising|falling) 200-day|long-term (trend|direction)|broad (trend|state)|kind of market",
    r"\bdoji\b|shooting star|hanging man|dark cloud|engulfing|hammer": r"candle|a (single )?day's (bar|price)|opened|closed (near|close to|far)",
    r"(rising|falling|price|sideways) channel|\bwedge\b|\btriangle\b|double top|double bottom|triple top|head and shoulders|broadening (formation|pattern)": r"lines? (drawn|joining|that join)|shape|price (has )?(bounced|moved) between|highs and lows",
    r"FCF yield|free cash flow yield": r"share of (its |the company's )?market value|as a share of|per (rupee|dollar|₹|\$) of market value",
    r"\bRSI\b": r"0 to 100|0-100|zero to 100|gains against|recent gains",
    r"\bATRs?\b": r"daily (price )?range|daily move|average day's (move|range)",
    r"\bbeta\b": r"moves? (with|when|as) the (market|index)|market moves|sensitiv",
    r"debt/equity|debt-to-equity": r"debt (against|for every|compared with)|owners' money|owners’ money|borrow",
    r"\bEBIT\b": r"before interest and tax",
    r"current ratio": r"short-term (assets|debts|bills)",
    r"\bCVaR\b": r"worst 5%|average loss",
    r"Brier": r"better than (the )?base rate|worse than (the )?base rate|above (0|zero) (means|is)|below (0|zero) (means|is)",
    r"\bAUC\b": r"coin|0\.5 means|chance|separat|tells? (ups|rises) from",
    r"\bNIM\b|net interest margin": r"earns on (its )?loans|pays on deposits|lending (minus|less)",
    r"GNPA|NNPA": r"not being repaid|bad loans|unpaid|default",
    r"\bCASA\b": r"current and savings|low-cost deposits",
    r"\bCET1\b|capital adequacy": r"own capital|capital (behind|cushion|buffer)",
}
FILLER = re.compile(r"(our data|the facts|we) (lacks?|has no|have no|does not (have|include)|do not have)[^.]{0,40}\b(ADX|ATR|Bollinger|P/S|30-day volatility)", re.I)
ANSWER_WORDS = (60, 300)
SHORT_WORDS = (6, 50)


def stable_rank(key: str, salt: str) -> str:
    return hashlib.sha1(f"{salt}|{key}".encode()).hexdigest()


def held_out(keys: Iterable[str], share: float, group: Mapping[str, str], salt: str) -> Set[str]:
    """`share` of the keys in each group, the same ones on every run."""
    by_group: Dict[str, List[str]] = {}
    for k in keys:
        by_group.setdefault(group.get(k, ""), []).append(k)
    out: Set[str] = set()
    for members in by_group.values():
        members.sort(key=lambda k: stable_rank(k, salt))
        out.update(members[:max(1, round(share * len(members)))])
    return out


def split_for(stocks: Sequence[str], template: str, test_stocks: Set[str], validation_stocks: Set[str],
              test_templates: Set[str]) -> str:
    """A row is test if any stock it names, or its template, is held out; validation if a stock is."""
    if template in test_templates or test_stocks & set(stocks):
        return "test"
    return "validation" if validation_stocks & set(stocks) else "train"


def _pct(v) -> str:
    return f"{v:+.2f}%"


def money(value: float, currency: str) -> str:
    """Large amounts in the units people read: crore for rupees, millions or billions for dollars."""
    if currency == "INR":
        return f"{value / 1e7:,.2f} crore INR"
    if abs(value) >= 1e9:
        return f"{value / 1e9:,.2f} billion {currency}"
    return f"{value / 1e6:,.2f} million {currency}"


# Two sentences never fit the facts window. The first, cut at 20 words, is 35 tokens or fewer in 160 of the 200 snapshots.
PROFILE_WORDS = 20
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
# The period in "Apple Inc.", "Dr. Reddy's" or "ASML Holding N.V." ends no sentence. Cutting at it left 21 of the 185
# profiles round 7 fetched holding only the company's name.
NOT_AN_END = re.compile(r"(?:\b(?:Inc|Co|Corp|Ltd|Pvt|Bros|Dr|St|Mr|Mrs|Ms|Jr|Sr)|(?<![A-Za-z])[A-Za-z])\.$")
FUNDAMENTAL_LABELS = {
    "gross_margin": "Gross margin", "roa": "Return on assets", "revenue_cagr": "Revenue growth per year (CAGR)",
    "latest_quarter_revenue_yoy": "Latest quarter revenue versus a year earlier",
    "latest_quarter_net_income_yoy": "Latest quarter net income versus a year earlier",
    "forward_pe": "Forward P/E", "price_to_book": "Price to book (P/B)", "ev_to_ebitda": "EV/EBITDA",
    "ev_to_sales": "EV/sales", "dividend_yield": "Dividend yield",
}
# How the analysis dates a valuation figure: by the day it was fetched, which was one day in every training row, and
# ALFA learned to write it. The figure rests on the last close, so its line carries that day instead.
RETRIEVED = re.compile(r"snapshot retrieved \d{4}-\d{2}-\d{2}")


def _dated(period, closed):
    return closed if RETRIEVED.fullmatch(str(period)) else period


def first_sentence(text):
    ends = (end.start() for end in SENTENCE_END.finditer(text) if not NOT_AN_END.search(text[:end.start()]))
    return text[:next(ends, len(text))]


def facts(snap: Mapping) -> List[str]:
    """Every line a row's answer may draw on, each with its source and the date or period it describes."""
    res, sym, ccy = snap["results"], snap["symbol"], snap["currency"]
    tech, risk, fund = res.get("technical", {}), res.get("risk", {}), res.get("fundamental", {})
    td, rd, fd = tech.get("details") or {}, risk.get("details") or {}, fund.get("details") or {}
    as_of = td.get("as_of") or tech.get("as_of")
    closed = snap["quality"]["last_date"]
    close = f"; last close {td['price']:,.2f} on {as_of}" if td.get("price") is not None and as_of else ""
    out = [f"Stock: {snap['name']} ({resolve_symbol(sym, snap['market'])}), {snap['market']}, {ccy}{close}.", DEFINITIONS]
    prof = fd.get("profile") or {}
    if prof.get("summary") and prof["summary"] != UNAVAILABLE:
        words = first_sentence(prof["summary"]).split()
        cut = " ".join(words[:PROFILE_WORDS]).rstrip(",;:") + "..."
        out.append("[profile] " + (" ".join(words) if len(words) <= PROFILE_WORDS else cut))
    if fd.get("sector") and fd["sector"] != UNAVAILABLE:
        out.append(f"Sector {fd['sector']}, industry {fd.get('industry')} (yahoo profile).")
    if prof.get("market_cap"):
        out.append(f"Market value {money(prof['market_cap'], ccy)} (yahoo, as of {closed}).")
    rets = rd.get("returns") or {}
    names = (("1d_pct", "1 day"), ("1w_pct", "1 week"), ("1m_pct", "1 month"), ("3m_pct", "3 months"),
             ("6m_pct", "6 months"), ("1y_pct", "1 year"))
    parts = [f"{label} {_pct(rets[k])}" for k, label in names if rets.get(k) is not None]
    if parts and as_of:
        out.append(f"Price change to {as_of}: {', '.join(parts)} (computed from yahoo daily bars).")
    for domain in ("technical", "risk", "regime", "pattern", "candlestick", "fundamental", "forecast",
                   "historical", "news_sentiment"):
        r = res.get(domain) or {}
        for e in r.get("evidence", []):
            p = e.get("provenance") or {}
            when = _dated(p.get("period") or p.get("as_of"), closed)
            kind = ", model output" if e.get("is_model_output") else ""
            claim = e["claim"].replace("RSI is ", "RSI(14) is ")      # an answer may then quote the 14 its RSI is read over
            if str(p.get("source")).endswith(TRAILING_PE):
                claim += " It is over the last twelve months, not the financial year the yearly figures are for."
            out.append(f"[{domain}{kind}] {claim} (source {p.get('source')}; {when}).")
        if r.get("state") == "unavailable":
            out.append(f"[{domain}] Not available: {r.get('error') or 'no data'}.")
    if td.get("volume_vs_20d_avg") is not None and as_of:
        out.append(f"[technical] Volume on {as_of} was {td['volume_vs_20d_avg']:.2f} times its 20-day average.")
    for key, label in (("support_levels", "Support levels"), ("resistance_levels", "Resistance levels")):
        if td.get(key):
            out.append(f"[technical] {label} found in the price history: "
                       f"{', '.join(f'{v:,.2f}' for v in td[key])} {ccy} (as of {as_of}).")
    if rd.get("current_drawdown_pct") is not None:
        out.append(f"[risk] The price is {abs(rd['current_drawdown_pct']):.2f}% below its highest close in the loaded history (as of {as_of}).")
    metrics = fd.get("metrics") or {}
    for key, label in FUNDAMENTAL_LABELS.items():
        m = metrics.get(key)
        if m and m.get("value") is not None:
            unit = "%" if m.get("unit") == "%" else ("x" if m.get("unit") == "x" else "")
            out.append(f"[fundamental] {label} {m['value']:.2f}{unit} (source {m.get('source')}; {_dated(m.get('period'), closed)}).")
    if (metrics.get("cash") or {}).get("value") is not None:
        m = metrics["cash"]
        out.append(f"[fundamental] Cash and equivalents {money(m['value'], m.get('currency') or ccy)} "
                   f"(source {m.get('source')}; {m.get('period')}).")
    stated, priced = fd.get("statement_currency"), fd.get("price_currency")
    if stated and priced and stated != priced and UNAVAILABLE not in (stated, priced):
        out.append(f"[fundamental] Statements are in {stated}; the price is in {priced}.")
    if fd.get("unavailable"):
        out.append(f"[fundamental] Not available in our data: {', '.join(fd['unavailable'])}.")
    for ev in snap.get("upcoming_events") or []:
        out.append(f"Upcoming: {ev}.")
    dec = snap.get("decision") or {}
    if dec.get("overall_state"):
        out.append(f"[app's combined reading, model output] Overall evidence reads {dec['overall_state']} "
                   f"(confidence {dec.get('confidence', 0):.2f}); this is decision support, not advice.")
    return out


def key_facts(snap: Mapping) -> List[str]:
    """The few lines a comparison needs about a second stock, drawn from the same snapshot fields."""
    keep = ("Stock:", "Sector", "Price change", "P/E is", "Return on equity", "Debt/equity",
            "Revenue changed", "Annualized volatility", "Beta to the benchmark")
    return [f for f in facts(snap) if any(k in f for k in keep)]


def fill(template: str, slots: Mapping[str, str]) -> str:
    missing = sorted(set(SLOT.findall(template)) - set(slots))
    if missing:
        raise KeyError(f"template needs slots {missing}: {template[:80]!r}")
    return SLOT.sub(lambda m: str(slots[m.group(1)]), template)


def check(answer: str, short: str, question: str, row_facts: Sequence[str], own_names: Sequence[str],
          other_names: Mapping[str, Sequence[str]]) -> List[str]:
    """Every problem that stops an answer being trained on; empty means it passes.

    own_names: the symbol and names of the stock the row is about. other_names: every other stock of the
    universe, which an answer may name only when its question or facts already do.
    """
    source = " ".join(row_facts) + " " + question
    problems = _text_problems("answer", answer, source, "its facts or question", ANSWER_WORDS) \
        + _text_problems("short answer", short, source, "its facts or question", SHORT_WORDS)
    whole = f"{answer} {short}"
    if not any(_names_in(n, whole) for n in own_names):
        problems.append(f"the answer never names the stock ({', '.join(own_names)})")
    if guardrails.figures(answer) and not (set(ISO_DATE.findall(answer)) or "FY ending" in answer):
        problems.append("the answer states figures without the date or period they describe")
    for sym, names in other_names.items():
        for n in names:
            if _names_in(n, whole) and not _names_in(n, source) and not any(_names_in(n, o) for o in own_names):
                problems.append(f"the answer names another stock, {sym} ({n!r}), that its question and facts do not")
                break
    return problems


def _text_problems(label: str, text: str, source: str, said_by: str, words) -> List[str]:
    """What stops one written text training: its length, a figure or date `source` does not state, advice wording, a
    placeholder brace or a double negative."""
    problems, (lo, hi) = [], words
    n_words = len(text.split())
    if not lo <= n_words <= hi:
        problems.append(f"{label} has {n_words} words; it must have {lo} to {hi}")
    bad = guardrails.unsupported_figures(text, source)
    if bad:
        problems.append(f"{label} states figures not in {said_by}, as written there: {bad}")
    bad_dates = sorted(set(ISO_DATE.findall(text)) - set(ISO_DATE.findall(source)))
    if bad_dates:
        problems.append(f"{label} states dates not in {said_by}: {bad_dates}")
    for pat in ADVICE:
        if re.search(pat, text, re.I):
            problems.append(f"{label} uses advice wording matching {pat!r}")
    if "{" in text or "}" in text:
        problems.append(f"{label} still holds a placeholder brace")
    neg = DOUBLE_NEGATIVE.search(text)
    if neg:
        problems.append(f"{label} has a double negative, {neg.group(0)!r}: drop the minus sign, the word gives the direction")
    return problems


def check_passage_answer(answer: str, passage: str, company: str, names) -> List[str]:
    """Every problem that stops an answer written from one facts passage training; empty means it passes.

    The server screens a facts answer against its passage alone, by figures and by the wrong-share guard with the
    facts route's slots, so an answer passing here is one the server would serve. `names` are the server's.
    """
    problems = _text_problems("passage answer", answer, passage, "the passage", SHORT_WORDS)
    own = own_names(passage) + [company]
    if not any(_names_in(n, answer) for n in own):
        problems.append(f"the passage answer never names the stock ({', '.join(own)})")
    others = guardrails.other_shares(answer, passage, names, FACTS_SLOTS)
    if others:
        problems.append(f"the server's guard reads another share in it: {others}")
    mislabelled = guardrails.mislabelled_figures(answer, passage, METRICS)
    if mislabelled:
        problems.append(f"the server's guard reads a figure as another metric's: {mislabelled}")
    return problems


def _names_in(name: str, text: str) -> bool:
    return _name_pattern(name).search(text) is not None


def _name_pattern(name: str):
    """Symbols match in capitals only, so 'LT' is not found in 'salt'; names match in any case."""
    if name.isupper():
        return re.compile(rf"(?<![A-Za-z0-9&]){re.escape(name)}(?![A-Za-z0-9&])")
    return re.compile(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", re.I)


def style_flags(answer: str, question: str, watch_factors: str = "", row_facts: Sequence[str] = ()) -> List[str]:
    """What the revision pass should look at in one answer; unlike `check`, a flag does not stop training."""
    flags = []
    quarter = [f for f in row_facts if "Latest quarter" in f]
    if quarter and "FY ending" in answer and "quarter" not in answer.lower():
        flags.append("uses yearly figures but leaves out the latest quarter the facts give: " + quarter[0][:90])
    for term, cues in MEANING_CUES.items():
        if re.search(term, answer, re.I if term.islower() else 0) and not re.search(cues, answer, re.I) \
                and not re.search(term, question, re.I if term.islower() else 0):
            flags.append(f"uses {term.replace(chr(92) + 'b', '').split('|')[0]} with no plain meaning")
    filler = FILLER.search(answer)
    if filler and not re.search(filler.group(4), question, re.I):
        flags.append(f"lists a missing measure the question did not ask about: {filler.group(0)!r}")
    if len(answer.split()) > 270:
        flags.append(f"{len(answer.split())} words; aim for 150 to 250")
    if watch_factors:
        wf, low = watch_factors.rstrip("."), answer.lower()
        items_in_order = [i.strip().lower() for i in re.split(r",\s*|\s+and\s+", wf) if i.strip()]
        pasted = any(f"{a}, {b}" in low or f"{a} and {b}" in low for a, b in zip(items_in_order, items_in_order[1:]))
        if pasted or wf[:40].lower() in low:
            flags.append("pastes the watch factors as written; say them in plain words")
        items = [i.strip() for i in re.split(r",|\band\b", wf) if i.strip()]
        key = [max(re.findall(r"[A-Za-z]{4,}", i) or [i], key=len).lower() for i in items]
        found = sum(k in answer.lower() for k in key)
        if items and found < 0.4 * len(items):
            flags.append(f"covers {found} of the {len(items)} watch factors")
    return flags


SOURCE = re.compile(r"\s*\(source [^;]*;\s*(.*)\)\.?$")      # a source may hold brackets, "computed:rsi(14)"; it never holds ";"


def lean(line: str) -> str:
    """A fact without its source tag but with its date, which is what an answer has to quote."""
    return SOURCE.sub(lambda m: f" ({m.group(1).strip()}).", line)


FACT_SEPARATOR = " ; "
# The facts window: one passage of the generator's 128 tokens, 32 of them for the question and the rest for facts.
# 91% of the frozen set's questions fit in 32, and the 16 tokens given to facts ground 527 of its short answers to 48's 457.
QUESTION_TOKENS = 32
# The wrong-share guard's slots for a facts answer: none. Facts answers are free prose, and the advisory forms' slot
# words read "data" in "our data has no ..." as a share in over half the grounded facts targets.
FACTS_SLOTS = (frozenset(),) * 3
# The ratios a fact gives a figure, by its label here and the short forms an answer uses, for the server's label guard.
# Ratios only: adding the close, averages, revenue and market value refused 85 of 2,752 written passage answers, not 4.
METRICS = (("Return on invested capital", "ROIC"), ("Return on equity", "ROE"), ("Return on assets", "ROA"),
           ("Price to book", "P/B"), ("Free cash flow yield", "FCF yield"), ("Debt/equity", "debt-to-equity"),
           ("Current ratio",), ("Forward P/E",), ("P/E",), ("PEG",), ("Operating margin",), ("Net margin",),
           ("Gross margin",), ("Dividend yield",), ("EV/sales",), ("EV/EBITDA",), ("EPS",), ("RSI",), ("MACD",),
           ("Volatility",), ("Beta",))
STOCK_LINE = re.compile(r"Stock: (.+?) \(([^()]+)\), ")
AREA_TAG = re.compile(r"\[(\w+)[,\]]")
# Of the frozen set's 3,002 short answers, 0.10 lets 588 be grounded by their passage, 0.15 587, 0.20 585 and none 560
DOMAIN_PRIOR = 0.10
# Even asked what a company does, its profile scores about 0.17 below the figures: 0.30 puts it in 71 passages of the
# 224 what-it-does, competition, ESG and overview rows, against 15 at DOMAIN_PRIOR, for 10 fewer grounded short answers
PROFILE_PRIOR = 0.30


def own_names(stock_line: str) -> List[str]:
    """The ticker, symbol and names the first facts line gives its share, longest first."""
    found = STOCK_LINE.match(stock_line)
    if found is None:
        raise ValueError(f"the first facts line names no share as stock_qa.facts writes it: {stock_line[:80]!r}")
    name, ticker = found.groups()
    return sorted({ticker, name, *company_terms(ticker, name)[:2]}, key=len, reverse=True)


def without_names(text: str, names: Sequence[str]) -> str:
    for name in names:
        text = _name_pattern(name).sub("", text)
    return re.sub(r"\s+", " ", re.sub(r"\(\s*\)", "", text)).strip()


def facts_scorer(lines: Sequence[str], similarity):
    """`score(question)`, one number per facts line, which `pick_facts` orders the lines by.

    `similarity(texts)` is the encoder's, read with the share's own names taken out of the question and the lines:
    otherwise headlines, which repeat the name, outrank the figures asked for. A line of an area the question names
    ("is it expensive" names fundamental) gets DOMAIN_PRIOR more, or PROFILE_PRIOR for the company's profile.
    """
    names = own_names(lines[0])
    cosine = similarity(tuple(without_names(line, names) for line in lines))
    areas = [tag.group(1) if tag else None for tag in map(AREA_TAG.search, lines)]

    def score(question):
        wanted = set(named_domains(question, FACTS_DOMAIN_WORDS))
        return [float(c) + (area in wanted) * (PROFILE_PRIOR if area == "profile" else DOMAIN_PRIOR)
                for c, area in zip(cosine(without_names(question, names)), areas)]
    return score


def pick_facts(question: str, lines: Sequence[str], score, count_tokens, budget: int) -> List[str]:
    """The facts a question needs, best first, that fit the generator's evidence window.

    `score(question)` gives one number per line, the router's own scorer over these lines, so training and
    serving choose evidence the same way. The stock's own line always leads: without it in the window
    the generator has no name to copy, and names another share.
    """
    if not lines:
        raise ValueError("no facts to pick from; build them with stock_qa.facts(snapshot)")
    first, rest = lean(lines[0]), [lean(line) for line in lines[1:]]
    scores = score(question)
    order = sorted(range(len(rest)), key=lambda i: -float(scores[i + 1]))
    chosen, used = [first], count_tokens(first)
    for i in order:
        # each line after the first costs its separator too, so the joined passage fits the budget
        cost = count_tokens(FACT_SEPARATOR + rest[i])
        if used + cost <= budget:
            chosen.append(rest[i])
            used += cost
    return chosen


def facts_passage(question: str, lines: Sequence[str], score, count_tokens, budget: int) -> str:
    """The one evidence passage the generator reads for a facts question, laid out the same in training and serving."""
    return FACT_SEPARATOR.join(pick_facts(question, lines, score, count_tokens, budget))
