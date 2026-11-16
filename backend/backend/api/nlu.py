"""Query understanding: natural language -> structured intent. Pure functions.

Rule-based on purpose: the intents are few and well defined, rules are
exact and testable, and a misparse is visible in `Intent`. The optional
Claude tool agent (llm.ToolAgent) handles phrasing these rules miss.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

from backend.advisory.domains import DAY_RANGE, named_domains
from backend.advisory.sentiment.feeds.feed import ENGLISH_WORDS, is_name

# Common names -> (symbol, market): indices, US stocks, names people write unlike NSE, and names of everyday words,
# which name_index leaves out (see it).
ALIASES = {
    "reliance": ("RELIANCE", "NSE"), "hdfc bank": ("HDFCBANK", "NSE"), "icici bank": ("ICICIBANK", "NSE"),
    "icici": ("ICICIBANK", "NSE"), "state bank": ("SBIN", "NSE"), "sbi": ("SBIN", "NSE"),
    "infosys": ("INFY", "NSE"), "tata consultancy": ("TCS", "NSE"), "airtel": ("BHARTIARTL", "NSE"),
    "bharti airtel": ("BHARTIARTL", "NSE"), "kotak": ("KOTAKBANK", "NSE"), "axis bank": ("AXISBANK", "NSE"),
    "bajaj finance": ("BAJFINANCE", "NSE"), "bajaj finserv": ("BAJAJFINSV", "NSE"),
    "larsen": ("LT", "NSE"), "l&t": ("LT", "NSE"), "maruti": ("MARUTI", "NSE"),
    "mahindra": ("M&M", "NSE"), "hindustan unilever": ("HINDUNILVR", "NSE"), "hul": ("HINDUNILVR", "NSE"),
    "asian paints": ("ASIANPAINT", "NSE"), "sun pharma": ("SUNPHARMA", "NSE"), "titan": ("TITAN", "NSE"),
    "wipro": ("WIPRO", "NSE"), "hcl tech": ("HCLTECH", "NSE"), "hcl": ("HCLTECH", "NSE"),
    "tech mahindra": ("TECHM", "NSE"), "nestle": ("NESTLEIND", "NSE"), "ultratech": ("ULTRACEMCO", "NSE"),
    "power grid": ("POWERGRID", "NSE"), "coal india": ("COALINDIA", "NSE"), "tata steel": ("TATASTEEL", "NSE"),
    "jsw steel": ("JSWSTEEL", "NSE"), "hindalco": ("HINDALCO", "NSE"), "cipla": ("CIPLA", "NSE"),
    "dr reddy": ("DRREDDY", "NSE"), "apollo hospitals": ("APOLLOHOSP", "NSE"), "zomato": ("ETERNAL", "NSE"),
    "eternal": ("ETERNAL", "NSE"), "trent": ("TRENT", "NSE"), "bharat electronics": ("BEL", "NSE"),
    "adani ports": ("ADANIPORTS", "NSE"), "eicher": ("EICHERMOT", "NSE"), "hero motocorp": ("HEROMOTOCO", "NSE"),
    "bajaj auto": ("BAJAJ-AUTO", "NSE"), "jio financial": ("JIOFIN", "NSE"), "shriram finance": ("SHRIRAMFIN", "NSE"),
    "sbi life": ("SBILIFE", "NSE"), "hdfc life": ("HDFCLIFE", "NSE"), "britannia": ("BRITANNIA", "NSE"),
    "tata consumer": ("TATACONSUM", "NSE"), "grasim": ("GRASIM", "NSE"), "ntpc": ("NTPC", "NSE"),
    "ongc": ("ONGC", "NSE"), "itc": ("ITC", "NSE"), "tcs": ("TCS", "NSE"), "lic": ("LICI", "NSE"),
    "nifty": ("^NSEI", "NSE"), "sensex": ("^BSESN", "BSE"), "bank nifty": ("^NSEBANK", "NSE"),
    "nvidia": ("NVDA", "US"), "apple": ("AAPL", "US"), "microsoft": ("MSFT", "US"), "amd": ("AMD", "US"),
    "tesla": ("TSLA", "US"), "amazon": ("AMZN", "US"), "alphabet": ("GOOGL", "US"), "google": ("GOOGL", "US"),
    "meta": ("META", "US"), "happiest minds": ("HAPPSTMNDS", "NSE"), "indian hotels": ("INDHOTEL", "NSE"),
}
US_TICKERS = {"NVDA", "AAPL", "MSFT", "AMD", "TSLA", "AMZN", "GOOGL", "GOOG", "META", "NFLX", "INTC"}
NOT_TICKERS = {"I", "A", "RSI", "PE", "EPS", "MACD", "ATR", "NSE", "BSE", "BUY", "SELL", "HOLD", "WATCH",
               "AVOID", "ETF", "IPO", "CEO", "CFO", "US", "USA", "INR", "USD", "FY", "EV", "EBITDA", "ROE",
               "ROA", "ROIC", "PEG", "FCF", "VAR", "CVAR", "SMA", "EMA", "OBV", "GDP", "RBI", "SEBI", "AI",
               "IT", "FMCG", "OK", "Q1", "Q2", "Q3", "Q4", "YOY", "QOQ", "TTM", "LTCG", "STCG", "STT",
               "HNI", "MF", "SIP", "PB", "DE", "NPA", "WHY", "WHAT", "HOW", "AND", "THE", "VS",
               "AVG", "INFO", "AGRI", "AGRO", "ECO", "IND"}

SECTOR_WORDS = {
    "Technology": ("it stocks", "it sector", "it compan", "it services", "tech", "software",
                   "technology"),
    "Financial Services": ("bank", "financial", "nbfc", "insurance", "finance"),
    "Healthcare": ("pharma", "healthcare", "hospital"),
    "Consumer Cyclical": ("auto", "automobile", "retail", "consumer cyclical"),
    "Consumer Defensive": ("fmcg", "consumer staples", "consumer defensive"),
    "Energy": ("energy", "oil", "gas"),
    "Basic Materials": ("metal", "steel", "cement", "chemical", "materials"),
    "Utilities": ("power", "utilit"),
    "Communication Services": ("telecom", "communication"),
    "Industrials": ("infra", "capital goods", "industrial", "defence", "defense"),
}


@dataclass
class Intent:
    name: str
    symbols: List[Tuple[str, str]] = field(default_factory=list)
    amount: Optional[float] = None
    horizon: Optional[str] = None
    risk_profile: Optional[str] = None
    include_sectors: List[str] = field(default_factory=list)
    exclude_sectors: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    days: Optional[int] = None
    raw: str = ""


AMOUNT = re.compile(
    r"(?P<cur>₹|rs\.?|inr|\$|usd)?\s*(?P<num>\d+(?:,\d{2,3})*(?:\.\d+)?)\s*"
    r"(?P<unit>k\b|thousand|lakhs?\b|lacs?\b|l\b|crores?\b|cr\b|million\b|mn\b|m\b)?", re.I)
UNIT = {"k": 1e3, "thousand": 1e3, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "l": 1e5,
        "crore": 1e7, "crores": 1e7, "cr": 1e7, "million": 1e6, "mn": 1e6, "m": 1e6}
MONEY_CONTEXT = re.compile(r"\b(have|invest|investing|capital|budget|add|adding|deploy|put|spend|with)\b", re.I)


def extract_amount(text: str) -> Optional[float]:
    for m in AMOUNT.finditer(text):
        cur, unit = m.group("cur"), (m.group("unit") or "").lower()
        num = float(m.group("num").replace(",", ""))
        tail = text[m.end():m.end() + 12].lower()
        if re.match(r"\s*(day|week|month|year|session|stock|share|compan|%)", tail) and not cur and not unit:
            continue
        if unit:
            return num * UNIT[unit]
        if cur:
            return num
        if num >= 1000 and MONEY_CONTEXT.search(text):
            return num
    return None


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9&]+", text.lower())


def proper_nouns() -> FrozenSet[str]:
    """The dictionary's capitalised words in lower case ("india", "delhi", "monte"), which english_words() leaves out."""
    return frozenset(w.lower() for w in ENGLISH_WORDS.read_text().split() if w.isalpha() and w[:1].isupper())


def name_index(names: Mapping[str, str], english: Set[str], proper: Set[str]) -> Dict[str, str]:
    """What people write for each listed company -> its symbol: the symbol, the name without "Limited", and the
    name's leading words when no other name starts with them ("happiest minds"). A lone English word ("idea", "beta") is
    left out, so IDEA is found only in capitals or by more of its name, and so are leading words that are all dictionary
    words, `proper` nouns too ("new delhi"). Each symbol is also its own key in capitals, which no lowercased phrase
    matches, so the values are every listed share, "Eternal Ltd." among them."""
    index: Dict[str, str] = {}
    starts: Dict[str, Set[str]] = {}
    for sym, name in names.items():
        words = _words(re.sub(r"\s+(ltd\.?|limited)$", "", name.strip(), flags=re.I))
        index[" ".join(words)] = sym
        lead = words[1:] if words[0] == "the" and len(words) > 1 else words
        for k in range(1, max(len(lead), 2)):
            starts.setdefault(" ".join(lead[:k]), set()).add(sym)
    index.update({" ".join(_words(sym)): sym for sym in names})
    everyday = set(english) | set(proper)
    # the dictionary has no plurals, and "indian metals" or "solar industries" in a question named a share
    is_everyday = lambda w: any(form in everyday for form in ((w, w[:-1], w[:-2], w[:-3] + "y") if w.endswith("s") else (w,)))
    index.update({w: next(iter(syms)) for w, syms in starts.items() if len(syms) == 1 and w not in index
                  and (" " not in w or not all(map(is_everyday, w.split())))})
    named = {p: sym for p, sym in index.items() if is_name(p, english) and p.upper() not in NOT_TICKERS}
    return {**{sym: sym for sym in names}, **named}


def extract_tickers(text: str, listings: Mapping[str, str]) -> List[Tuple[str, str]]:
    # not a symbol's own key: then every word of "SHOULD I TAKE PROFITS NOW?" that is a symbol counted as one (TAKE)
    known_set: Set[str] = {sym for phrase, sym in listings.items() if phrase != sym}
    found: List[Tuple[str, str]] = []
    rest = f" {text} "
    explicit = re.compile(r"(?<![\w&\-])([A-Za-z0-9&\-]{2,15})\.(ns|bo)\b", re.I)
    for base, suffix in explicit.findall(rest):
        sym = (base.upper(), "BSE" if suffix.lower() == "bo" else "NSE")
        if sym not in found:
            found.append(sym)
    rest = explicit.sub(" ", rest)
    # the longest name wins ("reliance power" over the alias "reliance"); on a tie the alias does ("amd" is the US one)
    phrases = {**{p: (sym, "NSE") for p, sym in listings.items()}, **{" ".join(_words(a)): v for a, v in ALIASES.items()}}
    tokens, longest = list(re.finditer(r"[a-z0-9&]+", rest.lower())), max(p.count(" ") + 1 for p in phrases)
    i = 0
    while i < len(tokens):
        n = next((n for n in range(min(longest, len(tokens) - i), 0, -1)
                  if " ".join(t.group() for t in tokens[i:i + n]) in phrases), 0)
        if n:
            sym = phrases[" ".join(t.group() for t in tokens[i:i + n])]
            if sym not in found:
                found.append(sym)
            start, end = tokens[i].start(), tokens[i + n - 1].end()
            rest = rest[:start] + " " * (end - start) + rest[end:]
        i += n or 1
    for tok in re.findall(r"\b[A-Z][A-Z0-9&\-]{1,14}\b", rest):
        base = tok
        if base in NOT_TICKERS:
            continue
        if base in US_TICKERS:
            sym = (base, "US")
        elif base in known_set:
            sym = (base, "NSE")
        elif len(base) >= 3 and text.strip().upper() != text.strip():
            # An unfamiliar all-caps word in mixed-case text is most likely a ticker.
            sym = (base, "NSE")
        else:
            continue
        if sym not in found:
            found.append(sym)
    return found


def extract_horizon(text: str) -> Optional[str]:
    """The horizon someone says they hold or trade for. "the 50 day average", "short-term trend", "daily swings" and "the
    intraday high" name a span a figure is measured over, and the horizon they set stayed for the rest of the chat."""
    t = text.lower()
    m = re.search(r"\b(?:for|within|next)\s+(?:the\s+)?(?:next\s+)?(\d+(?:\.\d+)?)\s*(year|yr|month|week|day)", t) \
        or re.search(r"\b(\d+(?:\.\d+)?)[- ]?(year|yr|month|week|day)s?\s+(?:horizon|hold)", t)
    if m:
        n, unit = float(m.group(1)), m.group(2)
        days = n * {"year": 365, "yr": 365, "month": 30, "week": 7, "day": 1}[unit]
        return "long" if days >= 365 else "medium" if days >= 45 else "short"
    # "a long term investment", "buy it long term?"; not "long term debt"
    term = r"[- ]term( (invest|hold|horizon|trad|buy|bet|view|play)|\s*([?.!,]|$))"
    if re.search(rf"\b(for|in) (the )?long (term|run)|long{term}|\bfor years\b|retire", t):
        return "long"
    if re.search(rf"\b(for|in) (the )?short (term|run)|short{term}|(swing|intraday) trad|\b(for|next) (a )?few weeks", t):
        return "short"
    if re.search(rf"\b(for|in) (the )?medium term|medium{term}|\b(for|next) (a )?few months", t):
        return "medium"
    return None


def extract_risk(text: str) -> Optional[str]:
    t = text.lower()
    if re.search(r"conservative|low[- ]risk|safe|cautious", t):
        return "conservative"
    if re.search(r"aggressive|high[- ]risk|risky", t):
        return "aggressive"
    if re.search(r"moderate|medium[- ]risk|balanced", t):
        return "moderate"
    return None


def extract_sectors(text: str) -> Tuple[List[str], List[str]]:
    t = f" {text.lower()} "
    include, exclude = [], []
    for sector, words in SECTOR_WORDS.items():
        for w in words:
            idx = t.find(w)
            if idx < 0 and sector == "Technology":
                # Uppercase IT is the sector; lowercase "it" is a pronoun.
                m = re.search(r"\bIT\b", text)
                idx = m.start() + 1 if m else -1
            if idx < 0:
                continue
            before = t[max(0, idx - 25):idx]
            (exclude if re.search(r"\b(no|not|avoid|exclude|excluding|except|without)\b", before)
             else include).append(sector)
            break
    return include, exclude


def detect_domain(text: str) -> Optional[str]:
    found = named_domains(text)
    return found[0] if found else None


def _window_days(text: str) -> Optional[int]:
    t = text.lower()
    if "today" in t:
        return 1
    if "week" in t:
        return 5
    if "month" in t:
        return 21
    if "quarter" in t or "3 months" in t:
        return 63
    if "year" in t:
        return 252
    return None


# what the LM router may choose from: every intent that answers without needing a stock or an amount
ROUTABLE = {
    "market_regime": "the state of the Indian market as a whole, its trend, volatility and how much to hold",
    "signals": "which stocks to buy now, the buy list, rebalancing",
    "stops": "whether any holding has hit its stop-loss",
    "alerts": "alerts that have triggered",
    "portfolio_summary": "how the user's own portfolio is doing",
    "portfolio_risk": "which holding carries most of the user's risk",
    "portfolio_exposure": "whether the user's portfolio leans too much on one sector",
    "knowledge": "what a finance word, ratio, indicator or method means",
}


def classify(text: str, listings: Mapping[str, str], short_forms: Set[str], has_session_symbol: bool,
             research_phrases: Sequence[str] = ()) -> Intent:
    t = text.lower().strip()
    symbols = extract_tickers(text, listings)
    amount = extract_amount(text)
    inc, exc = extract_sectors(text)
    base = dict(symbols=symbols, amount=amount, horizon=extract_horizon(text),
                risk_profile=extract_risk(text), include_sectors=inc, exclude_sectors=exc,
                domain=detect_domain(text), days=_window_days(text), raw=text)
    refers_back = bool(re.search(r"\b(it|this|that|the stock|this stock|this company)\b", t))

    def intent(name: str) -> Intent:
        return Intent(name=name, **base)

    # "what is a stop-loss?" asks what the words mean; "check my stops" asks about the user's own holdings
    asks_meaning = bool(re.match(r"(what is|what's|what are|what does|explain|define|meaning of|how does|how do|how is)", t)) \
        and not re.search(r"\b(current|today|now|latest|my)\b", t)
    if re.search(r"\balerts?\b", t) and not symbols and not asks_meaning:
        return intent("alerts")
    if re.search(r"\bstops?\b|stop[- ]loss", t) and not symbols and not asks_meaning:
        return intent("stops")
    if re.search(r"^(hi|hey|hello|yo|thanks|thank you|thx|ok|okay|bye|good (morning|afternoon|evening))\b[\s!.?]*$", t):
        return intent("greeting")
    index_only = bool(symbols) and all(sym in ("^NSEI", "^BSESN") for sym, _ in symbols)  # "is the nifty falling" is a market question
    if (not symbols or index_only) and re.search(r"market (trend|regime|condition|timing)|should i be (invested|in cash)|"
                                 r"is (the|this) (a )?(bull|bear)|downtrend|uptrend|reduce exposure|go to cash|"
                                 r"(market|nifty|sensex|index|indices)[^.?]*\b(down|up|fall|falling|fell|drop|dropping|"
                                 r"crash|crashing|correct|correcting|correction|rise|rising|rose|rally|rallying|"
                                 r"weak|volatile|red|green)\b", t):
        return intent("market_regime")
    if not symbols and re.search(r"what (should|to|can) i buy|which stocks? (should|to|can) i buy|buy list|"
                                 r"\bsignals?\b|\bpicks?\b|rebalanc|recommend|allocate|what to invest in|"
                                 r"momentum (stocks|portfolio)|low[- ]vol(atility)? (stocks|portfolio)", t):
        return intent("signals")
    if (symbols or has_session_symbol) and re.search(
            r"should i (buy|sell|invest|hold|exit|add)|(good|right) (time )?to (buy|sell|invest)|worth (buying|investing)|"
            r"buy or (not|sell)|is (it|this) a (good )?buy|can i buy|time to exit|(sell|hold) or (sell|hold)", t):
        # a holder's question: "should i sell" got "DON'T BUY", which says nothing about a share already owned
        holds = re.search(r"should i (sell|hold|exit)|(good|right) (time )?to sell|time to exit|(sell|hold) or (sell|hold)", t)
        return intent("should_sell" if holds else "should_buy")
    # a verdict only when the one stock named is the subject: "TCS fell, is gold a good buy?" is not about TCS
    subject = re.search(r"\bis (.+?) a (good )?buy\b", text, re.I)
    if subject and len(symbols) == 1 and not symbols[0][0].startswith("^") \
            and extract_tickers(subject.group(1), listings) == symbols:
        return intent("should_buy")
    if (symbols or has_session_symbol) and DAY_RANGE.search(t):
        return intent("day_range")
    if (symbols or has_session_symbol) and re.search(r"\bcorrect(ion|ing)\b|\bbear market\b", t):
        return intent("correction")
    if re.search(r"\breport\b", t) and (symbols or has_session_symbol):
        return intent("report")
    mine = re.search(r"\b(my|i hold|i own)\b", t) and re.search(
        r"portfolio|holding|stocks i|positions?|capital|cash|invest", t)
    if mine or re.search(r"\bportfolio\b", t):
        if amount and re.search(r"\badd|adding|invest|deploy|put\b", t):
            return intent("portfolio_what_if")
        if re.search(r"overexpos|exposure|sector", t):
            return intent("portfolio_exposure")
        if re.search(r"contribut|risk", t):
            return intent("portfolio_risk")
        if "momentum" in t:
            return intent("portfolio_momentum")
        if "fundamental" in t:
            return intent("portfolio_fundamentals")
        if re.search(r"volatil", t):
            return intent("portfolio_volatility")
        if re.search(r"cash|capital available|available capital|how much capital", t):
            return intent("portfolio_cash")
        if re.search(r"this week|changed|today|since", t):
            return intent("portfolio_changes")
        if re.search(r"sentiment", t):
            return intent("portfolio_sentiment")
        return intent("portfolio_summary")
    if re.search(r"\bcompare\b|\bvs\.?\b|\bversus\b", t):
        return intent("compare")
    if not symbols and any(p in t for p in research_phrases) and re.search(r"\b(find|show|which|list|screen)\b", t):
        return intent("research_query")
    if amount and not symbols and re.search(r"invest|research|stocks?|buy|find|show|which|what|suggest", t):
        return intent("screen")
    concept = re.match(r"(what is|what's|what are|what does|explain|define|meaning of|how does|how do|how is|"
                       r"how are\b.*\btaxed\b)", t)
    live = re.search(r"\b(current|today|now|latest|my)\b", t)
    # the ticker guess reads NAV in "what is NAV?" as a stock, but it is a word our lessons explain
    names_no_stock = all(sym in short_forms and sym not in listings.values() for sym, _ in symbols)
    asks_term = concept or symbols and re.search(r"\b(explain|describe|define|tell me about|understand|learn|meaning|"
                                                 r"meant|full form|stands? for)\b", t)
    if asks_term and names_no_stock and not live and not (refers_back and has_session_symbol):
        return Intent(name="knowledge", **{**base, "symbols": []})
    if re.search(r"\bwhy\b.*\b(fall|fell|falling|drop|dropped|down|decline|rise|rose|rising|up|rally|rallied|mov(e|ed|ing)|jump)", t) \
            and (symbols or has_session_symbol):
        return intent("explain_move")
    if re.search(r"what(?:'s| has| have)? changed|what is new|what's new", t) and (symbols or has_session_symbol):
        return intent("what_changed")
    if concept and has_session_symbol and refers_back:
        return intent("concept_with_data")
    if t.startswith("why") and has_session_symbol and not symbols:
        return intent("domain" if base["domain"] else "explain_decision")
    if amount and (symbols or has_session_symbol):
        return intent("budget_analysis")
    if base["domain"] and (symbols or has_session_symbol) and not re.search(r"\banaly[sz]e\b", t):
        return intent("domain")
    if symbols and re.search(r"\b(overbought|oversold)\b", t):
        return Intent(name="domain", **{**base, "domain": "technical"})
    if symbols:
        return intent("analyze")
    return intent("help")
