#!/usr/bin/env python3
"""Label generated answers with both judges and report how often they agree.

There is no financial analyst to ask, so the teacher is Claude. That only means anything if the two
judges are measured against each other instead of assumed equivalent, which is what this prints.

The agent is deliberately shown less than the oracle: question, evidence, answer, and no gold. Judging
with the gold in hand would reduce to string comparison and agreement would be 100% by construction.
Blind, a disagreement is informative — usually the model said the right thing in different words, and
the oracle's exact match is the one that is wrong.

Two passes, because a verdict has to be replayable:

  1. --ask   generates answers, labels them with the oracle into the feedback log, and writes the rows
             needing an agent verdict to artifacts/verdicts_wanted.jsonl.
  2. --tell  reads verdicts back from artifacts/verdicts_given.jsonl, appends them to the log next to
             the oracle's, and prints the agreement rate.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_answers import answer_rows, load_model  # noqa: E402
from selfagent.data import advisory  # noqa: E402
from selfagent.learn import AGENT, ORACLE, FeedbackLog  # noqa: E402
from selfagent.learn.teacher import AgentTeacher, OracleTeacher, verdict_key  # noqa: E402
from train_generator import load_split  # noqa: E402

WANTED = "verdicts_wanted.jsonl"
GIVEN = "verdicts_given.jsonl"
CACHE = "verdicts.jsonl"


def ask(artifacts, dataset, checkpoint, split, rows, feedback):
    """Generate, label with the oracle, and write down what the agent still has to look at."""
    tokenizer, model, config = load_model(artifacts, dataset, checkpoint)
    loaded = load_split(artifacts, dataset, split)
    rng = np.random.default_rng(config.seed)
    chosen = rng.permutation(len(loaded[0]))[:rows]
    scored = answer_rows(model, tokenizer, loaded, chosen, rng)

    oracle = OracleTeacher(advisory.UNSUPPORTED)
    agent = AgentTeacher(artifacts / CACHE)
    asked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    right = 0
    for question, evidence, gold, answer, sure in scored:
        is_right, why = oracle.judge(question, evidence, answer, gold)
        right += is_right
        feedback.append(asked_at, question, evidence, answer, sure, is_right, oracle.name, why,
                        split=split, gold=gold)
        agent.judge(question, evidence, answer)

    written = agent.write_requests(artifacts / WANTED)
    print(f"{len(scored)} rows labelled by the oracle ({right} right), {written} awaiting an agent "
          f"verdict in {artifacts / WANTED}")
    if written:
        print(f"Judge each one and write {{\"key\": ..., \"is_right\": true|false, \"why\": ...}} per "
              f"line to {artifacts / GIVEN}, then re-run with --tell")


def tell(artifacts, feedback):
    """Take the agent's verdicts, put them in the log beside the oracle's, and compare."""
    given = artifacts / GIVEN
    if not given.exists():
        raise SystemExit(f"no verdicts at {given}; run --ask first and judge the rows it writes")
    verdicts = {row["key"]: row for row in map(json.loads, given.read_text().splitlines()) if row}

    agent = AgentTeacher(artifacts / CACHE)
    unmatched, replayed = [], 0
    for row in feedback.rows(labeller=ORACLE):
        key = verdict_key(row["question"], row["answer"])
        verdict = verdicts.get(key)
        if verdict is None:
            unmatched.append(key)
            continue
        # Already in the cache means an earlier --tell logged it. Appending again would count one
        # verdict twice and quietly double its weight in every fit made from this log.
        if agent.judge(row["question"], row["evidence"], row["answer"])[0] is not None:
            replayed += 1
            continue
        agent.record(key, verdict["is_right"], verdict.get("why", ""))
        feedback.append(row["asked_at"], row["question"], row["evidence"], row["answer"],
                        row["confidence"], verdict["is_right"], AGENT, verdict.get("why", ""),
                        **row["extra"])

    agreed, both = feedback.agreement()
    if unmatched:
        print(f"{len(unmatched)} oracle rows had no agent verdict, e.g. {unmatched[:3]}")
    if replayed:
        print(f"{replayed} verdicts were already logged and were left alone")
    print(f"agreement {agreed}/{both} ({agreed / max(both, 1):.1%}) over answers both judges saw")
    for row in feedback.rows(labeller=AGENT):
        oracle_said = [r for r in feedback.rows(labeller=ORACLE)
                       if r["question"] == row["question"] and r["answer"] == row["answer"]]
        if oracle_said and oracle_said[0]["is_right"] != row["is_right"]:
            print(f"\n  disagreed: {row['question']}\n    answered: {row['answer']}"
                  f"\n    gold:     {oracle_said[0]['extra'].get('gold', '')}"
                  f"\n    oracle {oracle_said[0]['is_right']} ({oracle_said[0]['why']})"
                  f" / agent {row['is_right']} ({row['why']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--dataset", default="advisory")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--split", default="unseen")
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--log", default="artifacts/feedback.sqlite")
    parser.add_argument("--ask", action="store_true", help="generate and label with the oracle")
    parser.add_argument("--tell", action="store_true", help="read agent verdicts back and compare")
    args = parser.parse_args()
    if args.ask == args.tell:
        raise SystemExit("pass exactly one of --ask or --tell")

    artifacts = Path(args.artifacts)
    with FeedbackLog(args.log) as feedback:
        if args.ask:
            ask(artifacts, args.dataset, args.checkpoint, args.split, args.rows, feedback)
        else:
            tell(artifacts, feedback)


if __name__ == "__main__":
    main()
