"""What a reader sees: a turn laid out as a title, the answer, the figures behind it and where they came from.

Built by our code from the turn, after the agent has decided what to say, so the layout never depends on
the language model: it writes sentences only, and every figure on the "Key figures" line is read straight
out of the evidence. Markdown, which the chat renders.
"""
import json
import re
from datetime import date

TITLES = {"performance": "Recent performance", "overbought": "Momentum and RSI", "volatility": "Volatility",
          "drawdown": "Drawdown", "risk": "Week-ahead risk", "buy": "Should I buy?"}
# the snapshot's own names, in the order a reader wants them, with the units the advisory formats give them
FIGURES = (("close", "Last close", True), ("return_5d", "5 days", False), ("return_20d", "20 days", False),
           ("volatility_20d", "20-day volatility", False), ("rsi_14", "RSI (14)", False),
           ("drawdown_60d", "Off its 60-day high", False), ("outlook", "Week ahead", False))
SITES = (("stackexchange.com", "Stack Exchange"), ("wikipedia.org", "Wikipedia"), ("sec.gov", "SEC filing"),
         ("federalreserve.gov", "Federal Reserve"), ("rbi.org.in", "RBI"), ("sebi.gov.in", "SEBI"),
         ("economictimes.indiatimes.com", "Economic Times"), ("livemint.com", "Mint"),
         ("ndtvprofit.com", "NDTV Profit"), ("thehindubusinessline.com", "Hindu BusinessLine"))
LESSON = "lesson:"


def readable_day(day):
    """"2026-10-06" as "6 Oct 2026"; anything else unchanged."""
    try:
        return date.fromisoformat(str(day)[:10]).strftime("%-d %b %Y")
    except ValueError:
        return str(day)


def facts(evidence):
    """{name: value} from a snapshot's "name value ; name value" evidence line."""
    found = {}
    for part in evidence.split(" ; "):
        name, _, value = part.partition(" ")
        if name and value:
            found[name] = value
    return found


def figures_line(evidence, money):
    shown = facts(evidence)
    parts = [f"{label} {money}{_grouped(shown[name])}" if priced else f"{label} {shown[name]}"
             for name, label, priced in FIGURES if name in shown]
    return "**Key figures** · " + " · ".join(parts) if parts else ""


def _grouped(value):
    """"1217.00" as "1,217.00": display only, the evidence keeps the figure as the model saw it."""
    try:
        return f"{float(value):,.2f}"
    except ValueError:
        return value


def heading(ticker, names, title):
    called = f"{names[ticker]} ({ticker})" if ticker in names else ticker
    return (f"### {called}" + (f" · {title}" if title else "")) if ticker else f"### {title}"


def advisory(turn, names, money):
    """A price-snapshot answer: what it is about, the answer, then the figures it rests on."""
    lines = [heading(turn.ticker, names, TITLES.get(turn.intent, "")), "", _sentence_case(turn.served)]
    figures = figures_line(turn.evidence, money)
    return "\n".join(lines + (["", figures] if figures else []))


def sourced(turn, names, sources, money, title):
    """An answer read out of passages: the answer, the prices if they were part of it, then each source."""
    lines = [heading(turn.ticker, names, title), "", turn.served]
    prices = re.search(r"prices (.+?) ;; ", turn.evidence)
    if prices:
        lines += ["", figures_line(prices.group(1), money)]
    cited = [f"- {link}" for link in (_link(key, day, sources) for key, day in
                                       re.findall(r"document (.+?) ; published (\S+)", turn.evidence)) if link]
    return "\n".join(lines + (["", "**Sources**", *dict.fromkeys(cited)] if cited else []))


def present(turn, names, sources):
    """The markdown for a turn that spoke; a refusal is shown as the plain sentence it is."""
    if not turn.spoke:
        return turn.served
    money = "₹" if turn.ticker.endswith(".NS") else ("$" if turn.ticker else "")
    if turn.intent.startswith("reference"):
        lesson = re.search(rf"document {LESSON}(.+?) ; published", turn.evidence)
        title = ("What happened lately" if "recent" in turn.intent else lesson.group(1) if lesson
                 else "" if turn.ticker else "From the reference library")
        return sourced(turn, names, sources, money, title)
    return advisory(turn, names, money)


def _sentence_case(text):
    """ALFA's generator writes in lower case ("its 14 day rsi is 45 ."); a reader expects sentences."""
    text = re.sub(r"\s+([.,;:?])", r"\1", text)
    text = re.sub(r"(^|[.?!]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return re.sub(r"\brsi\b", "RSI", text)


def _link(key, day, sources):
    if key.startswith(LESSON):
        return f"StockIntel lessons · {key[len(LESSON):]}"
    url = key if key.startswith("http") else sources.get(key, "")
    if not url:
        return ""
    site = next((name for host, name in SITES if host in url), url.split("/")[2])
    return f"[{site}, {readable_day(day)}]({url})" if day not in ("", "undated") else f"[{site}]({url})"


def source_map(chunks, manifest):
    """{document key: source URL} for the documents an index holds, read once from the data manifest."""
    wanted = {chunk.document for chunk in chunks}
    found = {}
    with open(manifest, encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            key = entry["sha256"][:16]
            if key in wanted:
                found[key] = entry["url"]
    return found
