#!/usr/bin/env python3
"""Build the stock Q&A set: fetch a dated snapshot per stock, then lay out one row per question.

    python -m backend.models.training.prepare_stock_qa snapshots --qa ../stock_qa_1000.csv
    python -m backend.models.training.prepare_stock_qa rows --qa ../stock_qa_1000.csv --library <dir>
    python -m backend.models.training.prepare_stock_qa merge --answers <dir>
    python -m backend.models.training.prepare_stock_qa generator --frozen <dir> --artifacts <dir> [--passage-answers <dir>]
    python -m backend.models.training.prepare_stock_qa passages --frozen <dir>
    python -m backend.models.training.prepare_stock_qa check-passage --answers <file>
    python -m backend.models.training.prepare_stock_qa v3 --frozen <dir>
    python -m backend.models.training.prepare_stock_qa check-v3 --answers <file>

`snapshots` runs the app's own AnalysisService, so a row's facts are the figures the app would show.
`rows` writes each question with its facts and split, and no answer. `merge` joins written answers
to their rows, checks each one with `stock_qa.check`, and writes train, validation and test files with
only the rows that pass; every failure is listed in the report. `generator` encodes a frozen set for
train_generator, keeping only the rows whose target passes the server's guards against its facts passage: the
short answer, or with --passage-answers an answer written from that passage. `passages` writes every row's
passage, in batches for the writers, and `check-passage` checks what they wrote. `v3` fills each of the user's short
templates with real shares and gives each row the evidence the server reads for its label, and `check-v3` checks the
two answers written to each row.
"""
import argparse
import collections
import csv
import json
import logging
import random
import re
import sys
from pathlib import Path

import numpy as np

from backend.config import resolve_symbol
from backend.models.agent.context import evidence_room
from backend.models.core.config import ModelConfig
from backend.models.core.tokenizer.vocab import CLS_ID, PAD_ID, SEP_ID
from backend.models.data import advisory, slots, stock_qa
from backend.models.data.encode import build_sources
from backend.models.guardrails import guardrails
from backend.paths import ARTIFACTS, DATA

OUT = DATA / "stock_qa"
OUTSIDE_NIFTY500 = 80       # NSE shares ALFA has no price file for, so the set teaches it new names
OUTSIDE_EQUITY_L = 20       # smaller main-board shares outside the Nifty 500
MIN_ROWS = 250              # a year of bars: the risk figures need one
TEST_STOCK_SHARE, VALIDATION_STOCK_SHARE = 0.12, 0.08
PER_TEMPLATE = {"single_turn": 4, "multi_turn_followup": 5, "multimodal_or_tool_augmented": 7}

log = logging.getLogger("prepare_stock_qa")


def read_qa(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def held_symbols():
    return {p.stem.removesuffix(".NS") for p in (DATA / "prices").glob("*.csv")}


def universe(qa_rows):
    """(symbol, market, name, group) for every stock, in a fixed order: the CSV's 100, then NSE shares outside it."""
    from backend.database.sources.panel import _constituents_file, INDEX_LISTS, NSE_LISTS, plain_name
    seen, out = set(), []
    for r in qa_rows:
        market = "NSE" if r["exchange"] == "NSE" else "US"
        if r["ticker"] not in seen:
            seen.add(r["ticker"])
            out.append((r["ticker"], market, r["company"], "nifty50" if market == "NSE" else "us"))
    held = held_symbols()
    n500 = _constituents_file("nifty500", INDEX_LISTS)
    eq = _constituents_file("equity_l", NSE_LISTS)
    in500 = set(n500["Symbol"].str.strip())
    outside = sorted((s.strip(), plain_name(n)) for s, n in zip(n500["Symbol"], n500["Company Name"])
                     if s.strip() not in held | seen)
    small = sorted(((s.strip(), plain_name(n)) for s, n in zip(eq["SYMBOL"], eq["NAME OF COMPANY"])
                    if s.strip() not in in500 | held | seen and "dummy" not in n.lower()),
                   key=lambda t: stock_qa.stable_rank(t[0], "equity_l"))
    return out, [(s, "NSE", n, "nifty500_outside") for s, n in outside], \
        [(s, "NSE", n, "equity_l_outside") for s, n in small]


def cmd_snapshots(args):
    from backend.advisory.service import AnalysisService
    from backend.database.sources.yahoo import YahooProvider
    snap_dir = OUT / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    core, n500, small = universe(read_qa(args.qa))
    step = max(1, len(n500) // OUTSIDE_NIFTY500)
    plan = [(core, len(core)), (n500[::step] + n500[1::step], OUTSIDE_NIFTY500), (small, OUTSIDE_EQUITY_L)]
    services, failures, chosen = {}, {}, []
    provider = YahooProvider()
    for candidates, wanted in plan:
        got = 0
        for sym, market, name, group in candidates:
            if got >= wanted:
                break
            path = snap_dir / f"{sym}.json"
            if not path.exists():
                svc = services.setdefault(market, AnalysisService(provider, market))
                try:
                    snap = svc.analyze(sym).to_dict()
                except Exception as exc:          # one stock failing must not stop the other 199; each is recorded
                    failures[sym] = f"{type(exc).__name__}: {exc}"
                    continue
                if snap["quality"].get("rows", 0) < MIN_ROWS:
                    failures[sym] = f"only {snap['quality'].get('rows')} daily bars; need {MIN_ROWS}"
                    continue
                path.write_text(json.dumps(snap, default=str))
            got += 1
            chosen.append({"symbol": sym, "market": market, "name": name, "group": group})
            log.info("%d %s %s", len(chosen), sym, group)
        if got < wanted:
            raise SystemExit(f"only {got} of {wanted} stocks in one group have a usable snapshot; see failures")
    (OUT / "universe.json").write_text(json.dumps(chosen, indent=1))
    (OUT / "snapshot_failures.json").write_text(json.dumps(failures, indent=1))
    print(f"{len(chosen)} stocks, {len(failures)} skipped (snapshot_failures.json)")


BENCHMARK = {"NSE": "the Nifty 50", "US": "the S&P 500"}           # the index each market's beta is measured against
ALTERNATIVE_BENCHMARK = {"NSE": "the Nifty 500", "US": "the Nasdaq-100"}
NEWS_ITEMS = ["the company has won a contract worth about 5% of its yearly revenue", "the chief financial officer has resigned",
              "a regulator has opened an inquiry into the company's accounts", "the company plans to buy back its own shares"]
EVENTS = ["its latest quarterly results", "a large new order announcement", "a change of chief executive", "an analyst downgrade"]
ARGUMENTS = ["it has fallen a lot from its peak, so it must be cheap now", "everyone I know is buying it, so it will keep rising",
             "its P/E is low, so it is a bargain", "it beat earnings last quarter, so it will beat again"]
SCREENS = ["return on equity above 15%, debt/equity below 0.5 and positive revenue growth",
           "P/E below 25, revenue growth above 10% and a positive 1-year price change"]
AMBIGUOUS = {"NSE": ["Tata", "Bajaj", "Adani", "HDFC", "Mahindra", "Reliance"], "US": ["Alphabet"]}
CSV_TOPIC_QUESTIONS = "csv"


def metric(snap, key):
    m = ((snap["results"].get("fundamental") or {}).get("details") or {}).get("metrics", {}).get(key)
    return None if not m or m.get("value") is None else m["value"]


def tech(snap):
    return (snap["results"].get("technical") or {}).get("details") or {}


def kind_of(snap):
    """bank / nbfc / insurer / other, from the provider's industry name."""
    ind = (((snap["results"].get("fundamental") or {}).get("details") or {}).get("industry") or "").lower()
    if "bank" in ind:
        return "bank"
    if "insurance" in ind:
        return "insurer"
    if "credit" in ind or "mortgage" in ind or "financial conglomerate" in ind or "capital markets" in ind:
        return "nbfc"
    return "other"


def slot_values(tid, names, stock, snap, peers, rng, template_text=""):
    """Values for a template's slots, or None when this stock lacks a figure the question would state as fact.

    A value the question states about the stock comes from its snapshot. A value the asker supplies
    ("if it falls 20%") is drawn from a fixed list; the answer may use it because the question states it.
    """
    sym, market, name = stock["symbol"], stock["market"], stock["name"]
    ccy, t = snap["currency"], tech(snap)
    risk = (snap["results"].get("risk") or {}).get("details") or {}
    rets = risk.get("returns") or {}
    hist = t.get("macd_histogram")
    v = {"ticker": sym, "company": name, "company_name": name, "bank": name, "nbfc": name, "insurer": name,
         "benchmark": BENCHMARK[market], "alternative_benchmark": ALTERNATIVE_BENCHMARK[market],
         "index_name": "the Nifty 50" if market == "NSE" else "the Nasdaq-100",
         "market": "India's NSE" if market == "NSE" else "the US market",
         "horizon": rng.choice(["six months", "one year", "three years", "five years"]),
         "period": rng.choice(["the last year", "the last three years", "the last six months"]),
         "prior_period": "the same months a year ago", "lookback_period": rng.choice(["the last month", "the last three months"]),
         "timeframe": rng.choice(["daily", "weekly"]), "event": rng.choice(EVENTS),
         "rate_change": rng.choice(["50 basis point", "25 basis point"]), "rate_delta": "one percentage point",
         "growth_delta": "half a percentage point", "currency_pair": "USD/INR" if market == "NSE" else "EUR/USD",
         "portfolio_value": "Rs 10,00,000" if market == "NSE" else "$100,000",
         "position_pct": rng.choice(["8", "15", "25"]), "stress_pct": rng.choice(["10", "20", "30"]),
         "loss_pct": rng.choice(["15", "25", "35"]), "decline_pct": rng.choice(["10", "20", "30"]),
         "risk_profile": rng.choice(["low", "moderate", "high"]), "screen_criteria": rng.choice(SCREENS),
         "strategy_name": rng.choice(["50/200-day moving-average crossover", "RSI(14) mean-reversion", "12-month momentum"]),
         "investor_argument": rng.choice(ARGUMENTS), "news_item": rng.choice(NEWS_ITEMS), "new_evidence": rng.choice(NEWS_ITEMS),
         "earnings_summary": "revenue up 8% and profit down 3% year on year, both below what analysts expected",
         "audience": rng.choice(["a school student", "a retired person who is new to shares", "a professional fund manager"]),
         "factor": rng.choice(["the latest quarterly results", "the central bank's interest-rate decision", "oil prices"]),
         "prior_date": "a month ago", "adx_period": "14", "adx_value": rng.choice(["18", "27", "34"]),
         "new_horizon": "five-year", "old_horizon": "six-month", "source_1": "Yahoo Finance", "source_2": "the exchange's website",
         "growth_rate": rng.choice(["5%", "8%"]), "margin": rng.choice(["10%", "4%"]),
         "prior_conclusion": (snap.get("decision") or {}).get("overall_state") and
                             f"the overall evidence for {sym} reads {snap['decision']['overall_state']}",
         "profit_value": "120 crore INR" if market == "NSE" else "$2.1 billion",
         "cash_value": "45 crore INR" if market == "NSE" else "$0.8 billion",
         "missing_field": rng.choice(["analyst earnings estimates", "the latest earnings-call transcript", "options implied volatility"]),
         "move_pct": rng.choice(["6", "12"]) if tid == "STKQ-0184" else rng.choice(["6", "-8", "12"])}
    if peers:
        v.update(peer=peers[0]["name"], peer_group=", ".join([name] + [p["name"] for p in peers[:3]]),
                 tickers=", ".join([sym] + [p["symbol"] for p in peers[:3]]),
                 existing_holdings=", ".join(p["name"] for p in peers[:3]),
                 bank_1=name, bank_2=peers[0]["name"], **{f"peer_{i + 1}": q["symbol"] for i, q in enumerate(peers[:3])})
    as_of = t.get("as_of")
    v.update(as_of_date=as_of, price_timestamp=as_of)
    period = next((m.get("period") for m in (((snap["results"].get("fundamental") or {}).get("details") or {})
                                            .get("metrics") or {}).values() if m.get("period")), None)
    v["financial_period"] = period
    v["industry"] = ((snap["results"].get("fundamental") or {}).get("details") or {}).get("industry")
    pe, fpe, roe, roic = metric(snap, "pe"), metric(snap, "forward_pe"), metric(snap, "roe"), metric(snap, "roic")
    de, cr, dy = metric(snap, "debt_to_equity"), metric(snap, "current_ratio"), metric(snap, "dividend_yield")
    beta = (risk.get("benchmark") or {}).get("beta")
    fmt = lambda x, f: None if x is None else format(x, f)
    v.update(pe_ttm=fmt(pe, ".1f"), forward_pe=fmt(fpe, ".1f"), roe=fmt(roe, ".1f") and f"{roe:.1f}%",
             roic=fmt(roic, ".1f") and f"{roic:.1f}%", debt_equity=fmt(de, ".2f"), current_ratio=fmt(cr, ".2f"),
             dividend_yield=fmt(dy, ".2f"), beta=fmt(beta, ".2f"), rsi_14=fmt(t.get("rsi_14"), ".1f"),
             ps_ratio=rng.choice(["3.2", "8.5"]),
             implied_multiple=pe and f"{pe:.1f} times trailing earnings", peer_median=pe and f"a P/E of {pe * 0.8:.1f}")
    if hist is not None:
        v["macd_state"] = f"{'positive' if hist >= 0 else 'negative'} (histogram {hist:+.2f})"
        v["current_histogram"] = f"{hist:+.2f}"
        v["previous_histogram"] = f"{hist * 0.4 if abs(hist) > 0.05 else hist - 0.5:+.2f}"
    if rets.get("1m_pct") is not None:
        v["roc_value"] = f"{rets['1m_pct']:+.2f}%"
    if tid == "STKQ-0041" and v.get("rsi_14") and v.get("macd_state"):
        v.update(rsi_14=f"an RSI(14) of {v['rsi_14']}", macd_state=f"a MACD histogram that is {v['macd_state']}",
                 adx_value=f"an ADX(14) of {v['adx_value']}", roc_value=f"a 1-month change of {v.get('roc_value')}")
    levels_up, levels_down = t.get("resistance_levels") or [], t.get("support_levels") or []
    if levels_up or levels_down:
        v["resistance_level"] = f"{(levels_up or levels_down)[0]:,.2f}"
    if levels_down:
        v["support_level"] = f"{levels_down[0]:,.2f}"
    if t.get("price"):
        close = t["price"]
        v.update(entry_price=f"entry at {close:,.2f}", stop_price=f"stop at {close * 0.93:,.2f}",
                 target_price=f"target at {close * 1.12:,.2f}", price_or_pct=f"{close * 0.9:,.2f} {ccy}",
                 updated_data=f"the price is now {close * 0.95:,.2f} {ccy}" +
                              (f" and RSI(14) is {t['rsi_14'] - 6:.1f}" if t.get("rsi_14") else ""))
    up = [(k, label) for k, label in (("1m_pct", "the last month"), ("3m_pct", "the last three months"),
                                      ("6m_pct", "the last six months"), ("1y_pct", "the last year")) if (rets.get(k) or 0) > 0]
    if tid == "STKQ-0017":
        if not up:
            return None
        k, label = up[-1]
        v.update(return_pct=f"{rets[k]:.2f}", period=label)
    fcf, mcap = metric(snap, "free_cash_flow"), ((snap["results"].get("fundamental") or {}).get("details") or {}).get("profile", {}).get("market_cap")
    choices = [m for m, ok in (("return on equity", roe), ("P/E", pe), ("debt/equity", de), ("RSI(14)", t.get("rsi_14"))) if ok is not None]
    v["metric"] = rng.choice(choices) if choices else None
    if tid == "STKQ-0432":
        if fcf is None or not mcap:
            return None
        v.update(metric="free cash flow yield", inputs=f"free cash flow of {stock_qa.money(fcf, ccy)} and a market value of {stock_qa.money(mcap, ccy)}")
    if tid == "STKQ-0464":
        if roe is None:
            return None
        v.update(metric="return on equity", original_value=f"{roe:.1f}%", alternative_value=f"{roe * 1.15:.1f}%")
    if "the last {period}" in template_text or "the next {period}" in template_text:      # "over the last the last year"
        v["period"] = v["period"].removeprefix("the last ")
    out = {}
    for n in names:
        if v.get(n) is None:
            return None
        out[n] = v[n]
    return out


def library_templates(path):
    return [json.loads(line) for line in Path(path, "stock_question_templates.jsonl").read_text().splitlines() if line.strip()]


def needs(template):
    """The kind of stock a template must be filled with."""
    slots = set(stock_qa.SLOT.findall(template["question_template"]))
    for k in ("bank", "nbfc", "insurer"):
        if k in slots or f"{k}_1" in slots:
            return k
    return "financial" if template["category"] == "banks_nbfc_and_insurance_metrics" else "any"


def eligible(need, kind):
    return {"any": True, "financial": kind != "other"}.get(need, kind == need)


def peers_of(stock, stocks, snaps, kinds=None):
    """Up to three other stocks of the same market, preferring the same sector; only of `kinds` when given."""
    sector = lambda s: ((snaps[s["symbol"]]["results"].get("fundamental") or {}).get("details") or {}).get("sector")
    same = [s for s in stocks if s["market"] == stock["market"] and s["symbol"] != stock["symbol"]
            and (kinds is None or kinds[s["symbol"]] == kinds[stock["symbol"]])]
    same.sort(key=lambda s: (sector(s) != sector(stock), stock_qa.stable_rank(s["symbol"], stock["symbol"])))
    return same[:3]


ATTACHMENTS = {   # what the multimodal templates are given, since the set carries text, not files
    "STKQ-0471": "chart readings", "STKQ-0472": "financial table", "STKQ-0476": "news", "STKQ-0477": "holdings",
    "STKQ-0478": "dashboard", "STKQ-0480": "listings"}


def attachment(tid, stock, peers, snaps, listings):
    kind = ATTACHMENTS.get(tid)
    snap = snaps[stock["symbol"]]
    if kind is None:
        return ["Attached: nothing. The file the question mentions was not supplied, so only the snapshot below is available."]
    if kind == "holdings":
        weights = [40, 25, 20, 15]
        return ["Attached holdings (weights are the user's own figures): " +
                ", ".join(f"{s['name']} ({s['symbol']}) {w}%" for s, w in zip([stock] + peers, weights)) + "."]
    if kind == "listings":
        word = stock["_ambiguous"]
        found = [f"{n} ({s})" for s, n in listings if word.lower() in n.lower()][:12]
        return [f"Attached: NSE's list of main-board shares has {len(found)} names containing '{word}' among the first matches: "
                + "; ".join(found) + "."] if stock["market"] == "NSE" else \
            [f"Attached: Alphabet Inc. trades on Nasdaq as two share classes, GOOGL (Class A, with votes) and GOOG (Class C, no votes)."]
    keep = {"chart readings": ("[technical", "[pattern", "[candlestick", "Price change"),
            "financial table": ("[fundamental",), "news": ("[news_sentiment",), "dashboard": ("",)}[kind]
    return [f"Attached {kind}: {line}" for line in stock_qa.facts(snap) if any(line.startswith(k) or k in line[:20] for k in keep)]


def ambiguous_word(stock):
    """The word of a stock's name that other listed companies share ("Tata"), which STKQ-0480 asks about."""
    return next((w for w in AMBIGUOUS[stock["market"]] if w.lower() in stock["name"].lower()), None)


def row_facts(snaps, stock, peers, extra_facts):
    """A row's facts: its stock's own, then what the question attaches, then a few of each peer's."""
    fx = stock_qa.facts(snaps[stock["symbol"]]) + list(extra_facts)
    for p in peers:
        fx += [f"Peer {p['symbol']}: {line}" for line in stock_qa.key_facts(snaps[p["symbol"]])]
    return fx


def nse_listings():
    """(symbol, name) of every share on NSE's main board, which the listings attachment searches."""
    from backend.database.sources.panel import listed_shares
    shares = listed_shares()
    return list(zip(shares["symbol"], shares["name"]))


def snapshots_of(stocks):
    return {s["symbol"]: json.loads((OUT / "snapshots" / f"{s['symbol']}.json").read_text()) for s in stocks}


def refreshed_facts(row, by_sym, snaps, listings):
    """A frozen row's facts as `stock_qa.facts` writes them now, with the peers and attachment `cmd_rows` gave it."""
    stock, peers = dict(by_sym[row["symbol"]]), [by_sym[s] for s in row["stocks"][1:]]
    extra = []
    if row["interaction_type"] == "multimodal_or_tool_augmented":
        stock["_ambiguous"] = ambiguous_word(stock)
        extra = attachment(row["template_id"], stock, peers, snaps, listings)
    return row_facts(snaps, stock, peers, extra)


def cmd_rows(args):
    stocks = json.loads((OUT / "universe.json").read_text())
    snaps = snapshots_of(stocks)
    by_sym = {s["symbol"]: s for s in stocks}
    kinds = {s["symbol"]: kind_of(snaps[s["symbol"]]) for s in stocks}
    listings = nse_listings()
    templates = library_templates(args.library)
    group = {s["symbol"]: s["group"] for s in stocks}
    test_stocks = stock_qa.held_out(by_sym, TEST_STOCK_SHARE, group, "test")
    rest = [s for s in by_sym if s not in test_stocks]
    validation_stocks = stock_qa.held_out(rest, VALIDATION_STOCK_SHARE / (1 - TEST_STOCK_SHARE), group, "validation")
    tgroup = {t["template_id"]: t["category"] if t["interaction_type"] == "single_turn" else t["interaction_type"]
              for t in templates}
    test_templates = stock_qa.held_out(tgroup, 1 / 14, tgroup, "template")
    rows = []

    def row(rid, source, template_id, category, kind, question, stock, peers, extra_facts, **more):
        named = [stock["symbol"]] + [p["symbol"] for p in peers]
        rows.append({"id": rid, "source": source, "template_id": template_id, "category": category,
                     "interaction_type": kind, "symbol": stock["symbol"], "market": stock["market"],
                     "company": stock["name"], "stocks": named, "question": question,
                     "facts": row_facts(snaps, stock, peers, extra_facts),
                     "split": stock_qa.split_for(named, template_id, test_stocks, validation_stocks, test_templates),
                     **more})

    for r in read_qa(args.qa):
        stock = by_sym[r["ticker"]]
        row(r["id"], "stock_qa_1000", f"{CSV_TOPIC_QUESTIONS}:{r['topic']}", r["topic"], "single_turn", r["question"],
            stock, [], [], answer_outline=r["answer"], watch_factors=r["watch_factors"], company_type=r["company_type"])
    use = {s: 0 for s in by_sym}
    topic_questions = {}
    for r in read_qa(args.qa):
        topic_questions.setdefault(r["ticker"], []).append(r["question"])
    def make(t, candidates, pool, limit, rng, suffix="", same_kind=False):
        """Up to `limit` rows of one template, its stocks from `candidates` and its peers from `pool`."""
        tid, need = t["template_id"], needs(t)
        names = stock_qa.SLOT.findall(t["question_template"])
        order = sorted(candidates, key=lambda s: (use[s], stock_qa.stable_rank(s, tid)))
        made = 0
        for sym in order:
            if made >= limit:
                break
            stock = dict(by_sym[sym])
            if not eligible(need, kinds[sym]):
                continue
            if tid == "STKQ-0480":
                word = ambiguous_word(stock)
                if word is None:
                    continue
                stock["_ambiguous"] = word
            peers = peers_of(stock, pool, snaps, kinds if same_kind else None)
            if tid == "STKQ-0336":
                pick = {k: next((by_sym[s] for s in order if kinds[s] == k and by_sym[s]["market"] == "NSE"), None)
                        for k in ("bank", "nbfc", "insurer")}
                if kinds[sym] != "bank" or None in pick.values():
                    continue
                peers = [pick["nbfc"], pick["insurer"]]
            values = slot_values(tid, names, stock, snaps[sym], peers, rng, t["question_template"])
            if values is None:
                continue
            if tid == "STKQ-0336":
                values.update(nbfc=peers[0]["name"], insurer=peers[1]["name"])
            if tid == "STKQ-0480":
                values["company_name"] = stock["_ambiguous"]
            question = stock_qa.fill(t["question_template"], values)
            uses_peers = bool({"peer", "peer_1", "peer_group", "tickers", "existing_holdings", "bank_2", "nbfc"} & set(names)) \
                or t["category"] == "peer_comparison_and_relative_strength" or tid in ("STKQ-0477", "STKQ-0336")
            extra = attachment(tid, stock, peers, snaps, listings) if t["interaction_type"] == "multimodal_or_tool_augmented" else []
            more = {"answer_outline": t["answer_guidance"], "required_behaviours": t["required_behaviours"],
                    "difficulty": t["difficulty"]}
            if t["interaction_type"] == "multi_turn_followup":
                more["prior_question"] = rng.choice(topic_questions.get(sym) or
                                                    [f"Give me a simple fundamental and technical assessment of {stock['name']} ({sym})."])
            row(f"{tid}-{sym}{suffix}", "template_library", tid, t["category"], t["interaction_type"], question, stock,
                peers if uses_peers else [], extra, subject=f"{stock['name']} ({sym})", **more)
            for s in [sym] + ([p["symbol"] for p in peers] if uses_peers else []):
                use[s] += 1
            made += 1
        return made

    for t in templates:
        need, wanted = needs(t), PER_TEMPLATE[t["interaction_type"]]
        made = make(t, by_sym, stocks, wanted, random.Random(stock_qa.stable_rank(t["template_id"], "slots")))
        if made < wanted:
            log.warning("%s: only %d rows (need %s, %d eligible stocks)", t["template_id"], made, need,
                        sum(eligible(need, k) for k in kinds.values()))
    # Fixes run after every template so they cannot shift the stock rotation, and so the rows above stay
    # the ones already answered. Comparing two banks' bad loans needs a second bank, not a same-sector stock.
    pairs = [t for t in templates if "bank_2" in stock_qa.SLOT.findall(t["question_template"])]
    for t in pairs:
        wrong = [r for r in rows if r["template_id"] == t["template_id"] and kinds[r["stocks"][1]] != "bank"]
        rows[:] = [r for r in rows if r not in wrong]
        make(t, by_sym, stocks, len(wrong), random.Random(stock_qa.stable_rank(t["template_id"], "banks")), "-B", same_kind=True)
    # A template whose every row names a held-out stock, often as a peer, would be held out by accident. Two
    # more rows built only from training stocks keep it in training, so only test_templates are unseen.
    train_only = [s for s in by_sym if s not in test_stocks | validation_stocks]
    for t in templates:
        tid = t["template_id"]
        if tid not in test_templates and not any(r["template_id"] == tid and r["split"] == "train" for r in rows):
            added = make(t, train_only, [by_sym[s] for s in train_only], 2, random.Random(stock_qa.stable_rank(tid, "top-up")),
                         "-T", same_kind=t in pairs)
            log.info("%s: %d training rows added", tid, added)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "rows.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {sp: sum(r["split"] == sp for r in rows) for sp in ("train", "validation", "test")}
    (OUT / "split.json").write_text(json.dumps({"test_stocks": sorted(test_stocks), "validation_stocks": sorted(validation_stocks),
                                                "test_templates": sorted(test_templates), "rows": counts}, indent=1))
    print(f"{len(rows)} rows: {counts}")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def problems_of(row, ans, names):
    """Every check failure of one written answer (and its prior turn, for a follow-up)."""
    own = [row["symbol"], row["company"]]
    others = {s: n for s, n in names.items() if s not in row["stocks"]}
    found = []
    for key in ("answer", "short_answer"):
        if not isinstance(ans.get(key), str):
            found.append(f"missing {key}")
    if row["interaction_type"] == "multi_turn_followup" and not isinstance(ans.get("prior_answer"), str):
        found.append("missing prior_answer")
    if found:
        return found
    question = " ".join(filter(None, [row.get("prior_question"), row["question"], row.get("subject")]))
    found += stock_qa.check(ans["answer"], ans["short_answer"], question, row["facts"], own, others)
    if row["interaction_type"] == "multi_turn_followup":
        found += [f"prior answer: {p}" for p in stock_qa.check(
            ans["prior_answer"], ans["short_answer"], row["prior_question"], row["facts"], own, others)
            if "short answer" not in p]
    return found


def universe_names():
    return {s["symbol"]: [s["symbol"], s["name"]] for s in json.loads((OUT / "universe.json").read_text())}


def cmd_check(args):
    rows = {r["id"]: r for r in read_jsonl(OUT / "rows.jsonl")}
    names, bad = universe_names(), 0
    latest = {a.get("id"): a for a in read_jsonl(args.answers)}      # a fixed row is appended again; the last one counts
    for ans in latest.values():
        row = rows.get(ans.get("id"))
        found = ["unknown id"] if row is None else problems_of(row, ans, names)
        if found:
            bad += 1
            print(f"{ans.get('id')}:\n  - " + "\n  - ".join(found))
    print(f"{bad} failing")
    return 1 if bad else 0


def cmd_flags(args):
    """Style flags for every answer in one file, for the revision pass; a flag is a prompt to look, not a failure."""
    rows = {r["id"]: r for r in read_jsonl(OUT / "rows.jsonl")}
    latest = {a.get("id"): a for a in read_jsonl(args.answers)}
    flagged = 0
    for i, a in latest.items():
        r = rows[i]
        found = stock_qa.style_flags(a["answer"], r["question"], r.get("watch_factors", ""), r["facts"])
        if a.get("prior_answer"):
            found += [f"prior answer {f}" for f in stock_qa.style_flags(a["prior_answer"], r["prior_question"], "", r["facts"])
                      if "words; aim" not in f]
        if found:
            flagged += 1
            print(f"{i}:\n  - " + "\n  - ".join(found))
    print(f"{flagged} of {len(latest)} answers flagged")


def cmd_merge(args):
    rows = read_jsonl(OUT / "rows.jsonl")
    answers = {}
    for path in sorted(Path(args.answers).glob("*.jsonl")):
        for a in read_jsonl(path):
            answers[a["id"]] = a
    names, kept, failed, missing = universe_names(), {"train": [], "validation": [], "test": []}, {}, []
    for r in rows:
        a = answers.get(r["id"])
        if a is None:
            missing.append(r["id"])
            continue
        found = problems_of(r, a, names)
        if found:
            failed[r["id"]] = found
            continue
        kept[r["split"]].append({**r, "short_answer": a["short_answer"], "answer": a["answer"],
                                 **({"prior_answer": a["prior_answer"]} if "prior_answer" in a else {})})
    for split, items in kept.items():
        with (OUT / f"{split}.jsonl").open("w") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (OUT / "stock_qa.csv").open("w", newline="") as f:
        cols = ["id", "split", "source", "template_id", "category", "interaction_type", "market", "symbol", "company",
                "subject", "prior_question", "prior_answer", "question", "short_answer", "answer"]
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        for split in kept:
            w.writerows(kept[split])
    report = {"rows": len(rows), "kept": {k: len(v) for k, v in kept.items()}, "failed": len(failed),
              "missing": len(missing), "failures": failed, "missing_ids": missing}
    (OUT / "merge_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("failures", "missing_ids")}))


GENERATOR_DATASET = "stock_qa"
GENERATOR_SPLITS = {"train": "train", "validation": "validation", "test": "stockqa"}


def generator_split(row, held):
    """The generator split of one frozen row. Recomputed from split.json rather than read off the row, so a
    row naming a test stock or a test template cannot reach training."""
    split = stock_qa.split_for(row["stocks"], row["template_id"], set(held["test_stocks"]),
                               set(held["validation_stocks"]), set(held["test_templates"]))
    if split != row["split"]:
        raise ValueError(f"{row['id']} is marked {row['split']} but split.json puts it in {split}; re-freeze the set")
    return GENERATOR_SPLITS[split]


FOLLOW_UP = "multi_turn_followup"
PASSAGES = OUT / "passages_v1.jsonl"
SEC_NAMES = DATA / "raw/sec_edgar/company_tickers.json"
PASSAGE_BATCHES = OUT / "passage_batches"
BATCH_ROWS = 150


def passage_for(row, similarity, encode):
    """(the one facts passage a row's question is answered from, its ids), picked and sized as the server picks it."""
    room = evidence_room(ModelConfig.max_text_length, stock_qa.QUESTION_TOKENS)
    passage = stock_qa.facts_passage(row["question"], row["facts"], stock_qa.facts_scorer(row["facts"], similarity),
                                     lambda text: len(encode(text)), room)
    passage_ids = encode(passage)
    if len(passage_ids) > room:
        raise ValueError(f"{row['id']}: its facts passage is {len(passage_ids)} tokens, over the {room} the "
                         f"window leaves; build_sources would cut a figure the answer quotes")
    return passage, passage_ids


def drop_reason(target, row, passage, names):
    """Why a target cannot train the generator, or "" when it can: it must pass the guards the server screens a
    facts answer with, the figure check against its passage and the wrong-share guard with the facts slots."""
    question = row["question"]
    if guardrails.unsupported_figures(target, passage):
        if not guardrails.unsupported_figures(target, f"{passage} {question}"):
            return "a figure only the question states"
        if not guardrails.unsupported_figures(target, " ".join([passage, question, *row["facts"]])):
            return "a figure in a fact the window left out"
        return "a figure in no fact"
    if guardrails.other_shares(target, passage, names, stock_qa.FACTS_SLOTS):
        return "names a share the server's guard refuses"
    return ""


def generator_rows(rows, held, similarity, encode, names, written=None):
    """(source, keep, target) per kept row by generator split, and the rows dropped per (split, reason).

    The target is the short answer, or with `written` ({id: (passage, passage answer)}) an answer written from that
    passage, which must still be the one the row is given. The full answer outruns the 192-token target window.
    """
    length, answer_budget = ModelConfig.max_text_length, ModelConfig.max_answer_length
    kept, dropped = {name: [] for name in GENERATOR_SPLITS.values()}, collections.Counter()
    for row in rows:
        split = generator_split(row, held)
        if row["interaction_type"] == FOLLOW_UP:
            dropped[split, "a follow-up, whose earlier turn the source does not hold"] += 1
            continue
        passage, passage_ids = passage_for(row, similarity, encode)
        text = row["short_answer"]
        if written is not None:
            if row["id"] not in written:
                dropped[split, "no passage answer"] += 1
                continue
            answered_from, text = written[row["id"]]
            if answered_from != passage:
                raise ValueError(f"{row['id']}: its passage is no longer the one its answer was written from; rebuild "
                                 f"{PASSAGES.name} with `passages` and have its answers checked again")
        reason = drop_reason(text, row, passage, names)
        if not reason and written is not None and stock_qa.check_passage_answer(text, passage, row["company"], names):
            reason = "fails check-passage"
        if reason:
            dropped[split, reason] += 1
            continue
        source, keep = build_sources(encode(row["question"])[:stock_qa.QUESTION_TOKENS], [passage_ids], 1, length)
        target = np.full(answer_budget, PAD_ID, dtype=np.uint16)
        ids = [CLS_ID] + encode(text)[: answer_budget - 2] + [SEP_ID]
        target[: len(ids)] = ids
        kept[split].append((source, keep, target))
    return kept, dropped


def frozen_rows(frozen):
    """Every row of a frozen set with its facts re-derived from the snapshots by today's `stock_qa.facts`, which the
    server reads: the facts frozen with a row are laid out as they were the day it was written."""
    stocks = json.loads((frozen / "universe.json").read_text())
    by_sym, snaps, listings = {s["symbol"]: s for s in stocks}, snapshots_of(stocks), nse_listings()
    return [{**r, "facts": refreshed_facts(r, by_sym, snaps, listings)}
            for name in GENERATOR_SPLITS for r in read_jsonl(frozen / f"{name}.jsonl")]


def served_names():
    """The share names the live server's wrong-share guard reads an answer for, from the lists `ask.build` gives it."""
    from backend.models.agent import finance
    from backend.models.agent.run.ask import listed_names, nse_shares

    def no_fetch(ticker, path):
        raise RuntimeError(f"the data builder reads share names only; it does not fetch {ticker}")
    return finance.Market(DATA / "prices", None, SEC_NAMES, None, listed_names(),
                          nse_shares(), no_fetch).names


def encoder_tools(artifacts):
    """(encode, similarity) as the server has them; one similarity per distinct facts list, since peers and
    attachments add lines to a stock's own."""
    import functools
    from backend.models.core.tokenizer.wordpiece import WordPiece
    from backend.models.external import sentence
    return WordPiece.load(artifacts / "tokenizer.json").encode, functools.lru_cache(maxsize=None)(sentence.similarity)


def passage_answers(directory):
    """{id: (the passage it was written from, the passage answer)}; the last line of an id counts."""
    passages = {p["id"]: p["passage"] for p in read_jsonl(PASSAGES)}
    answers = {a["id"]: a["passage_answer"] for path in sorted(Path(directory).glob("*.jsonl")) for a in read_jsonl(path)}
    unknown = sorted(set(answers) - set(passages))
    if unknown:
        raise SystemExit(f"{len(unknown)} passage answers have no passage in {PASSAGES}, e.g. {unknown[:3]}")
    return {i: (passages[i], text) for i, text in answers.items()}


def cmd_generator(args):
    held = json.loads((args.frozen / "split.json").read_text())
    encode, similarity = encoder_tools(args.artifacts)
    written = passage_answers(args.passage_answers) if args.passage_answers else None
    kept, dropped = generator_rows(frozen_rows(args.frozen), held, similarity, encode, served_names(), written)
    for split, examples in kept.items():
        if not examples:
            raise SystemExit(f"no {split} row has a target that passes the server's guards; there is nothing to write")
        for part, values in zip(("source", "keep", "target"), zip(*examples)):
            np.save(args.artifacts / f"{GENERATOR_DATASET}_{split}_{part}.npy", np.stack(values))
        lost = {reason: n for (where, reason), n in dropped.items() if where == split}
        print(f"  {split}: {len(examples):,} kept, {sum(lost.values()):,} dropped {lost} -> {args.artifacts}/")
    (args.artifacts / f"{GENERATOR_DATASET}_MANIFEST.json").write_text(json.dumps({
        "targets": f"passage answers in {args.passage_answers}" if written is not None else "short answers",
        "kept": {split: len(examples) for split, examples in kept.items()},
        "dropped": {f"{split}: {reason}": n for (split, reason), n in sorted(dropped.items())}}, indent=1))


def batches(rows, size):
    """`rows` grouped by stock, whole stocks to a batch, each batch closed once it holds `size` rows."""
    by_stock, out, current = {}, [], []
    for row in rows:
        by_stock.setdefault(row["symbol"], []).append(row)
    for group in by_stock.values():
        current += group
        if len(current) >= size:
            out.append(current)
            current = []
    return out + ([current] if current else [])


def write_jsonl(path, rows):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_passages(args):
    held = json.loads((args.frozen / "split.json").read_text())
    encode, similarity = encoder_tools(args.artifacts)
    out, grounded = [], collections.Counter()
    for row in frozen_rows(args.frozen):
        split = generator_split(row, held)
        if row["interaction_type"] == FOLLOW_UP:
            continue
        passage, _ = passage_for(row, similarity, encode)
        out.append({**{k: row[k] for k in ("id", "split", "template_id", "interaction_type", "symbol", "company",
                                           "question")}, "passage": passage})
        grounded[split] += not guardrails.unsupported_figures(row["short_answer"], passage)
    write_jsonl(PASSAGES, out)
    PASSAGE_BATCHES.mkdir(exist_ok=True)
    for old in PASSAGE_BATCHES.glob("batch_*.jsonl"):
        old.unlink()
    made = batches(out, BATCH_ROWS)
    for n, batch in enumerate(made):
        write_jsonl(PASSAGE_BATCHES / f"batch_{n:02d}.jsonl", batch)
    print(f"{len(out):,} passages -> {PASSAGES}; {len(made)} batches -> {PASSAGE_BATCHES}/")
    print(f"short answers whose figures their passage states: {sum(grounded.values()):,} {dict(grounded)}")


def cmd_check_passage(args):
    passages = {p["id"]: p for p in read_jsonl(PASSAGES)}
    latest = {a.get("id"): a for a in read_jsonl(args.answers)}      # a fixed answer is appended again; the last counts
    names, bad = served_names(), 0
    for ans in latest.values():
        row = passages.get(ans.get("id"))
        found = (["unknown id"] if row is None else ["missing passage_answer"] if not isinstance(ans.get("passage_answer"), str)
                 else stock_qa.check_passage_answer(ans["passage_answer"], row["passage"], row["company"], names))
        if found:
            bad += 1
            print(f"{ans.get('id')}:\n  - " + "\n  - ".join(found))
    print(f"{bad} failing")
    return 1 if bad else 0


V3 = OUT / "v3"
# The v3 template answers by their path under DATA, the form finance.STOCK_QA names the frozen set by
V3_DATASET = V3.relative_to(DATA).as_posix()
V3_BATCH_ROWS = 100
# Each template is filled with two shares of each market, the first named by its name and the second by its symbol
V3_STOCKS = {"NSE": 2, "US": 2}
V3_WORDS = (25, 110)
# Provisional, the plan's bar: two answers sharing half their word trigrams teach one wording twice
V3_MOST_OVERLAP = 0.5
FACTS_EVIDENCE, PRICE_EVIDENCE = "facts", "advisory"
# The wrong-share guard's slots for each kind of evidence, as the server picks them
V3_SLOTS = {FACTS_EVIDENCE: stock_qa.FACTS_SLOTS, PRICE_EVIDENCE: advisory.SUBJECT_SLOTS}


def short_templates():
    """The user's short templates, train ("dev") and test, each with the label `finance` routes it by."""
    from backend.models.agent import finance
    held = json.loads(finance.SHORT_TEMPLATES.read_text(encoding="utf-8"))["templates"]
    return [{**t, "label": label} for split in ("dev", "test")
            for t, (label, _) in zip([t for t in held if t["split"] == split], finance.short_templates(split), strict=True)]


def price_evidence(stocks, encode):
    """{symbol: the evidence the server reads a price question from, at its last bar} for each stock it has bars of.

    A share outside the served price files is read from prices_wide, the Yahoo bars a live fetch gets; a US one the
    server cannot fetch, so it gets none.
    """
    from backend.models import serving
    from backend.models.agent import finance
    advisor = serving.load(ARTIFACTS / "price_head.npz")
    markets = [finance.Market(DATA / "prices", advisor), finance.Market(DATA / "prices_wide", advisor)]
    room, out = evidence_room(ModelConfig.max_text_length, advisory.QUESTION_TOKENS), {}
    for stock in stocks:
        ticker = resolve_symbol(stock["symbol"], stock["market"])
        market = next((m for m in markets if ticker in m.paths), None)
        if market is None:
            continue
        evidence, shown, _ = market.snapshot(ticker)
        missing = sorted(set(advisory.EVIDENCE_KEYS) - set(shown))
        if missing or len(encode(evidence)) > room:
            raise ValueError(f"{ticker}'s price evidence lacks {missing} or is over the {room} tokens the window leaves, "
                             f"so the server refuses or cannot read it: {evidence}")
        out[stock["symbol"]] = evidence
    return out


def v3_rows(templates, stocks, prices, facts_passage_of, held):
    """One row per template and stock, each template filled with V3_STOCKS shares of each market that have its evidence.

    A facts label reads `facts_passage_of(row)`; any other label reads the share's evidence in `prices`, an unsupported
    one too, as its answer says what that evidence lacks. A second share in a question is another of the same market,
    and `split_for` puts the row in test when its template or any share it names is held out.
    """
    from backend.models.agent import finance
    test_templates = {t["id"] for t in templates if t["split"] == "test"}
    test_stocks, validation_stocks = set(held["test_stocks"]), set(held["validation_stocks"])
    use, rows = collections.Counter(), []
    for t in templates:
        kind = FACTS_EVIDENCE if t["label"] == finance.FACTS else PRICE_EVIDENCE
        for market, count in V3_STOCKS.items():
            pool = [s for s in stocks if s["market"] == market and (kind == FACTS_EVIDENCE or s["symbol"] in prices)]
            if len(pool) < count:
                raise SystemExit(f"{t['id']} needs {count} {market} shares with {kind} evidence; there are {len(pool)}")
            pool.sort(key=lambda s: (use[s["symbol"]], stock_qa.stable_rank(s["symbol"], t["id"])))
            for at, stock in enumerate(pool[:count]):
                others = sorted((s for s in stocks if s["market"] == market and s is not stock),
                                key=lambda s: stock_qa.stable_rank(s["symbol"], f"{t['id']}|{stock['symbol']}"))
                named = [stock] + others[:max(0, t["question"].count("{t}") - 1)]
                fills = iter(s["name"] if at == 0 else s["symbol"] for s in named)
                row = {"id": f"{t['id']}-{stock['symbol']}",
                       "split": stock_qa.split_for([s["symbol"] for s in named], t["id"], test_stocks,
                                                   validation_stocks, test_templates),
                       "template_id": t["id"], "category": t["category"], "label": t["label"],
                       "symbol": stock["symbol"], "company": stock["name"],
                       "question": re.sub(r"\{t\}", lambda _: next(fills), t["question"])}
                evidence = facts_passage_of(row) if kind == FACTS_EVIDENCE else prices[stock["symbol"]]
                rows.append({**row, "evidence": evidence, "evidence_kind": kind})
                use[stock["symbol"]] += 1
    return rows


def cmd_v3(args):
    stocks = json.loads((OUT / "universe.json").read_text())
    held = json.loads((args.frozen / "split.json").read_text())
    encode, similarity = encoder_tools(args.artifacts)
    snaps, prices = snapshots_of(stocks), price_evidence(stocks, encode)
    rows = v3_rows(short_templates(), stocks, prices, lambda row: passage_for(
        {**row, "facts": stock_qa.facts(snaps[row["symbol"]])}, similarity, encode)[0], held)
    (V3 / "batches").mkdir(parents=True, exist_ok=True)
    write_jsonl(V3 / "rows.jsonl", rows)
    for old in (V3 / "batches").glob("batch_*.jsonl"):
        old.unlink()
    made = batches(rows, V3_BATCH_ROWS)
    for n, batch in enumerate(made):
        write_jsonl(V3 / "batches" / f"batch_{n:02d}.jsonl", batch)
    print(f"{len(rows):,} rows -> {V3}/rows.jsonl; {len(made)} batches -> {V3}/batches/; "
          f"{len(prices)} of {len(stocks)} stocks have price evidence")
    counts = {key: collections.Counter(f"{r['split']}, {r[key]}" for r in rows) for key in ("label", "category")}
    print(json.dumps({"split": collections.Counter(r["split"] for r in rows), **counts}, indent=1, sort_keys=True))


def v3_answer_problems(label, row, answer, names):
    """Every problem that stops one answer to a v3 row training; empty means it passes."""
    evidence, kind = row["evidence"], row["evidence_kind"]
    name, ticker = (stock_qa.STOCK_LINE.match(evidence).groups() if kind == FACTS_EVIDENCE
                    else (row["company"], guardrails.snapshot_values(evidence)["ticker"]))
    named = (name, ticker) if kind == FACTS_EVIDENCE else (ticker,)      # price evidence names no company
    found = stock_qa._text_problems(label, answer, evidence, "its evidence", V3_WORDS)
    if not any(stock_qa._names_in(n, answer) for n in named):
        found.append(f"{label} never names the stock as its evidence does: {' or '.join(named)}")
    others = guardrails.other_shares(answer, evidence, names, V3_SLOTS[kind])
    if others:
        _, before, after = V3_SLOTS[kind]
        found.append(f"the server's guard reads another share in {label}: {others}" + (
            f"; in a price answer a word after {sorted(before)} or before {sorted(after)} must be one the answer forms "
            "or the evidence use" if before else ""))
    if slots.to_slots(answer, evidence, name, ticker) is None:
        found.append(f"{label} writes a figure otherwise than its evidence does, so it has no slot form")
    # the word guards the server runs on this kind of answer, as core._misworded does
    if kind == FACTS_EVIDENCE:
        found += [f"the server's guard reads {figure} in {label} as {metric}"
                  for figure, metric in guardrails.mislabelled_figures(answer, evidence, stock_qa.METRICS)]
        found += [f"{label} gives {metric} a meaning of its own"
                  for metric in guardrails.own_meanings(answer, stock_qa.METRICS)]
    found += [f"{label} says the price is {said} its 20 and 50 day averages; its evidence says "
              f"{truth or 'no averages'}" for said, truth in guardrails.wrong_trend(answer, evidence)]
    return found


def trigram_overlap(first, second):
    """The Jaccard overlap of two texts' word trigrams; two texts with none are the same text."""
    grams = [set(zip(w, w[1:], w[2:])) for w in (re.findall(r"\w+", text.lower()) for text in (first, second))]
    union = grams[0] | grams[1]
    return len(grams[0] & grams[1]) / len(union) if union else 1.0


def v3_problems(row, entry, names):
    """Every problem with one line of v3 answers: each of its two answers, and how alike the two are."""
    answers = entry.get("answers")
    if not (isinstance(answers, list) and len(answers) == 2 and all(isinstance(a, str) for a in answers)):
        return [f"answers must be a list of two strings, got {answers!r:.80}"]
    found = [p for n, answer in enumerate(answers, 1) for p in v3_answer_problems(f"answer {n}", row, answer, names)]
    overlap = trigram_overlap(*answers)
    if overlap >= V3_MOST_OVERLAP:
        found.append(f"the two answers share {overlap:.0%} of their word trigrams; reword one until they share "
                     f"under {V3_MOST_OVERLAP:.0%}")
    return found


def cmd_check_v3(args):
    rows = {r["id"]: r for r in read_jsonl(V3 / "rows.jsonl")}
    latest = {a.get("id"): a for a in read_jsonl(args.answers)}      # a fixed pair is appended again; the last counts
    names, bad = served_names(), 0
    for entry in latest.values():
        row = rows.get(entry.get("id"))
        found = ["unknown id"] if row is None else v3_problems(row, entry, names)
        if found:
            bad += 1
            print(f"{entry.get('id')}:\n  - " + "\n  - ".join(found))
    print(f"{bad} failing")
    return 1 if bad else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshots")
    s.add_argument("--qa", required=True, type=Path)
    r = sub.add_parser("rows")
    r.add_argument("--qa", required=True, type=Path)
    r.add_argument("--library", required=True, type=Path)
    c = sub.add_parser("check", help="list every failing answer in one answers file")
    c.add_argument("--answers", required=True, type=Path)
    fl = sub.add_parser("flags", help="list the style flags of every answer in one answers file")
    fl.add_argument("--answers", required=True, type=Path)
    m = sub.add_parser("merge", help="join every answers file to its rows and write the split files")
    m.add_argument("--answers", required=True, type=Path)
    g = sub.add_parser("generator", help="encode a frozen set's grounded rows for train_generator")
    g.add_argument("--frozen", required=True, type=Path, help="a frozen set: split.json and the three split files")
    g.add_argument("--artifacts", required=True, type=Path, help="holds tokenizer.json; the arrays are written here")
    g.add_argument("--passage-answers", type=Path, help=f"train on the answers written from {PASSAGES.name}, not the short answers")
    ps = sub.add_parser("passages", help=f"write every frozen row's facts passage to {PASSAGES.name} and its batches")
    ps.add_argument("--frozen", required=True, type=Path)
    ps.add_argument("--artifacts", default=ARTIFACTS, type=Path, help="holds tokenizer.json")
    cp = sub.add_parser("check-passage", help="list every failing answer in one passage answers file")
    cp.add_argument("--answers", required=True, type=Path)
    v3 = sub.add_parser("v3", help=f"fill the user's short templates and write their rows and batches to {V3}")
    v3.add_argument("--frozen", required=True, type=Path, help="the frozen set whose split.json holds its test stocks")
    v3.add_argument("--artifacts", default=ARTIFACTS, type=Path, help="holds tokenizer.json")
    c3 = sub.add_parser("check-v3", help="list every failing line in one v3 answers file")
    c3.add_argument("--answers", required=True, type=Path)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return {"snapshots": cmd_snapshots, "rows": cmd_rows, "check": cmd_check, "flags": cmd_flags,
            "merge": cmd_merge, "generator": cmd_generator, "passages": cmd_passages,
            "check-passage": cmd_check_passage, "v3": cmd_v3, "check-v3": cmd_check_v3}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
