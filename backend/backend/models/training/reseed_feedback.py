#!/usr/bin/env python3
"""Append oracle-judged answers in the shape the agent serves to a feedback log, so its answer cut is fitted on them.

The agent is built as ask.py builds it and asked every intent in its trained phrasings, in turn, about every held
instrument at the last bar and at `--back` earlier dates a month apart. OracleTeacher judges each answer against
advisory.row for the phrasing the agent asked. Each row records the generator's digest, so a fit keeps only the rows
of the build that serves.

The log is append-only: this adds rows and never deletes one. `--rows` keeps what was appended, and `--from` appends
that file to another log without building the agent, so a second host gets the same rows.
"""
from backend.paths import ARTIFACTS, DATA

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from backend.models.agent.run.ask import ROUTERS, build, listed_names
from backend.models.agent.run.chat import generator_of
from backend.models.data import advisory, prices
from backend.models.learning import ORACLE, FeedbackLog
from backend.models.learning.teacher import OracleTeacher
from backend.models.serving import registry

# Trading days between the dates asked about, about a month.
STEP = 21


def phrasing_of(intent, ticker, asked):
    """Which phrasing of `intent` the agent asked, which is the one its gold answer is built for."""
    for phrasing in range(advisory.phrasings(intent)):
        if advisory.ask(intent, ticker, phrasing) == asked:
            return phrasing
    raise ValueError(f"the agent asked {asked!r}, which is no phrasing of {intent} about {ticker}")


def judged(agent, digest, back):
    """One row per turn the advisory generator answered, judged by the oracle, in FeedbackLog.append's terms."""
    oracle, tokenizer, rows = OracleTeacher(advisory.UNSUPPORTED), agent.tokenizer, []
    for index, ticker in enumerate(sorted(agent.market.paths)):
        dates, bars = prices.load_bars(agent.market.paths[ticker])
        for months in range(back + 1):
            end = len(dates) - 1 - STEP * months
            if end < advisory.BARS_NEEDED - 1:
                continue
            facts, as_of = advisory.snapshot(bars, end), str(dates[end])
            for intent in advisory.INTENTS:
                trained = advisory.trained_phrasings(intent)
                turn = agent.answer(advisory.ask(intent, ticker, trained[(index + months) % len(trained)]), as_of)
                if not turn.answer or turn.intent not in advisory.INTENTS:
                    continue
                _, shown, _ = agent.market.snapshot(ticker, as_of)
                gold = advisory.row(turn.intent, ticker, facts, shown, phrasing_of(turn.intent, ticker, turn.asked))[1]
                is_right, why = oracle.judge(turn.asked, turn.evidence, turn.answer,
                                             tokenizer.decode(tokenizer.encode(gold)))
                rows.append(dict(asked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 question=turn.asked, evidence=turn.evidence, answer=turn.answer,
                                 confidence=turn.confidence, is_right=bool(is_right), why=why,
                                 extra=dict(digest=digest, source="reseed", intent=turn.intent, as_of=as_of,
                                            spoke=turn.spoke)))
        if index % 10 == 9:
            print(f"  {index + 1} instruments, {len(rows)} judged turns", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="the feedback log the rows are appended to")
    parser.add_argument("--from", dest="source", default="",
                        help="a --rows file to append instead of asking the agent")
    parser.add_argument("--rows", default="", help="where to keep the asked rows, for --from on another host")
    parser.add_argument("--back", type=int, default=3, help="earlier dates asked about besides the last bar")
    parser.add_argument("--router", choices=sorted(ROUTERS), default="pretrained",
                        help="the router the deployment serves with (deploy/oracle/compose.yaml)")
    parser.add_argument("--checkpoint", default="",
                        help="a candidate to judge instead of the promoted generator, as ask.py's --checkpoint")
    args = parser.parse_args()

    if args.source:
        rows = json.loads(Path(args.source).read_text())
        digest = generator_of(ARTIFACTS / args.checkpoint if args.checkpoint
                              else registry.serving(registry.GENERATOR))["digest"]
        others = sorted({row["extra"]["digest"] for row in rows} - {digest})
        if others:
            raise ValueError(f"{args.source} holds rows of generator {others}, and {digest} is the one reseeded here; "
                             f"those rows would never be fitted on, so reseed from this host's generator instead")
    else:
        agent, _, _, served = build(ARTIFACTS, args.checkpoint, str(DATA / "prices"), "", args.log, 0.6, 10,
                                    str(ARTIFACTS / "price_head.npz"), None,
                                    str(DATA / "raw/sec_edgar/company_tickers.json"), str(ARTIFACTS / "returns.npz"),
                                    listed=listed_names(), router=args.router, labeller=ORACLE)
        digest = generator_of(served)["digest"]
        rows = judged(agent, digest, args.back)
        if args.rows:
            Path(args.rows).write_text(json.dumps(rows))
    with FeedbackLog(args.log) as feedback:
        before = len(feedback)
        feedback.extend(rows, ORACLE)
        print(f"{args.log}: {before} -> {len(feedback)} rows, +{len(rows)} judged by the oracle under {digest}")


if __name__ == "__main__":
    sys.exit(main())
