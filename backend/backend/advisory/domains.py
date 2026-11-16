"""The words that name each area of the app's analysis in a question. The chat's parser reads DOMAIN_WORDS to pick the
area it shows; the stock Q&A facts picker reads FACTS_DOMAIN_WORDS, those and a few more, to favour that area's facts."""

import re

DOMAIN_WORDS = {
    "candlestick": ("candle", "doji", "hammer", "engulfing"),
    "pattern": ("pattern", "triangle", "head and shoulders", "double top", "double bottom",
                "flag", "wedge", "channel", " forming"),
    "fundamental": ("fundamental", "valuation", "earnings", "revenue", "margin", "debt", "p/e",
                    " pe ", "roe", "balance sheet", "cash flow", "profit"),
    "news_sentiment": ("news", "sentiment", "headline", "announcement"),
    "forecast": ("forecast", "predict", "probability", "expected range", "outlook", "scenario"),
    "historical": ("similar setup", "analogue", "analog", "historically similar", "past setups"),
    "regime": ("regime", "market condition", "risk-on", "risk-off"),
    "risk": ("risk", "volatil", "drawdown", "beta", "sharpe", "var "),
    "technical": ("technical", "trend", "rsi", "macd", "moving average", "momentum", "support",
                  "resistance", "breakout", "chart"),
}
# Not the chat's: there "gold looks cheap" read as asking for the session stock's fundamentals
FACTS_WORDS = {
    "fundamental": ("valued", "expensive", "cheap", "dividend", "growth"),
    "news_sentiment": (" event", "catalyst"),
    "forecast": (" bull", " bear"),
    "risk": ("safe", "downside"),
    "profile": ("what does", "business", "make money", "makes money", "customer", "competit", "sector", "industry",
                "esg", "overview", "summar"),
}
FACTS_DOMAIN_WORDS = {domain: DOMAIN_WORDS.get(domain, ()) + FACTS_WORDS.get(domain, ())
                      for domain in {**DOMAIN_WORDS, **FACTS_WORDS}}

# The day's high or low, which the day's price bar holds: "what is the intraday high of hdfc bank today". Not "its 60 day
# high", a span's, nor the "low" of "why is infosys trading low today"
DAY_RANGE = re.compile(r"(?<!\d)(?<!\d[- ])\b(intraday|day'?s|today'?s|day)\s+(high|low|range)\b|"
                       r"\b(the|its)\s+(high|low)\s+(of\s+the\s+day|today)\b|\b(high|low)\s+of\s+the\s+day\b", re.I)


def named_domains(text, domain_words=DOMAIN_WORDS):
    """Every domain whose words `text` uses, in the mapping's order."""
    t = f" {text.lower()} "
    return [domain for domain, words in domain_words.items() if any(w in t for w in words)]
