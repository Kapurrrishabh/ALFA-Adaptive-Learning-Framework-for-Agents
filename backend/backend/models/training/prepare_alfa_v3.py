#!/usr/bin/env python3
"""Build alfa_v3, phase 2 of the v3 generator: every target in slot form, every source marked and laid out over its
passage windows by slots.windows, the function serving is to lay its sources out with too.

    python -m backend.models.training.prepare_alfa_v3 --served <dir> --wide <dir> --stale-passages <file> --out <dir>

run_alfa_v3.sh builds the two advisory input sets and then this. The rows:
- the advisory rows of the served 101 and of the wide shares, collected again from the prices each set's manifest
  records: marking needs the text, and the set's token ids have lost the ticker's case. The counts must match the set.
- the facts rows: the answers written from passages_v1, less the stale passages. A passage dated by the day it was
  fetched is dated by its last close instead, as stock_qa.facts writes those lines now.
- the v3 template rows, each of a row's two answers a row of its own.
A row whose slot form does not come back whole is dropped and counted by its cause. The facts and template rows are
repeated until they are FACTS_AND_TEMPLATES_SHARE of the scored target tokens.
"""
from backend.paths import ARTIFACTS, DATA

import argparse
import collections
import functools
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from backend.models.agent import finance
from backend.models.agent.context import evidence_room
from backend.models.agent.run.ask import listed_names
from backend.models.core.config import ModelConfig
from backend.models.core.tokenizer.vocab import CLS_ID, PAD_ID, SEP_ID
from backend.models.core.tokenizer.wordpiece import WordPiece
from backend.models.data import advisory, slots, stock_qa
from backend.models.data.encode import build_sources
from backend.models.guardrails import guardrails
from backend.models.training import prepare_advisory, prepare_stock_qa as qa, promote
from backend.models.training.prepare_alfa_v2 import SERVED, check_unseen, oversampling, scored_tokens

DATASET = "alfa_v3"
WIDE = "wide"
# Provisional, the share the plan set: what settles it is stockqa and template_test loss falling while unseen holds.
FACTS_AND_TEMPLATES_SHARE = 0.15
TEMPLATE_SPLITS = {"train": "train", "validation": "validation", "test": "template_test"}


def plain(text):
    return [figure.replace(",", "") for figure in guardrails.figures(text)]


class Layout:
    """Rows as the generator reads them. A snapshot gives several rows, so each source is marked and split once."""

    def __init__(self, tokenizer):
        self.tokenizer, self.length, self.budget = tokenizer, ModelConfig.max_text_length, ModelConfig.max_answer_length
        self.passages = functools.lru_cache(maxsize=None)(self._passages)

    def _passages(self, evidence, name, ticker, others, question_tokens):
        marked = slots.mark_source(evidence, name, ticker, *others)
        windows = slots.windows(marked, lambda text: len(self.tokenizer.encode(text)),
                                evidence_room(self.length, question_tokens))
        return marked, [self.tokenizer.encode(window) for window in windows]

    def row(self, question, evidence, answer, name, ticker, others, question_tokens):
        """(source, keep, target) of one row, or why its slot form does not come back whole."""
        slotted = slots.to_slots(answer, evidence, name, ticker, *others)
        if slotted is None:
            return "states a figure its evidence lacks or writes otherwise"
        ids = self.tokenizer.encode(slotted)
        if len(ids) > self.budget - 2:
            return f"its slot form is over the {self.budget}-token answer window"
        marked, passages = self.passages(evidence, name, ticker, others, question_tokens)
        if any(f"{slots.FIGURE} {place} " not in marked for place in re.findall(rf"{slots.FIGURE} (\d+)", slotted)):
            return "copies a figure marker its marked source never shows, a digit of a name the source marks"
        try:
            written = slots.from_slots(self.tokenizer.decode(ids), evidence, name, ticker)
        except ValueError:
            return "its slot form, as the tokenizer gives it back, names a figure its evidence lacks"
        answered = self.tokenizer.decode(self.tokenizer.encode(slots.from_slots(slotted, evidence, name, ticker)))
        if plain(written) != plain(answered):
            return "its slot form, as the tokenizer gives it back, fills to other figures than its answer"
        source, keep = build_sources(self.tokenizer.encode(question)[:question_tokens], passages, slots.WINDOWS,
                                     self.length)
        target = np.full(self.budget, PAD_ID, dtype=np.uint16)
        target[: len(ids) + 2] = [CLS_ID, *ids, SEP_ID]
        return source, keep, target


class Rows:
    """The kept rows of every source by (source, split), and the dropped ones by (source, split, cause)."""

    def __init__(self, layout):
        self.layout, self.kept, self.dropped = layout, collections.defaultdict(list), collections.Counter()
        self.held_out = collections.defaultdict(set)

    def add(self, source, split, laid):
        """Keep `laid`, a row as Layout.row lays it out, or count it dropped when it is the cause it was not."""
        if isinstance(laid, str):
            self.dropped[source, split, laid] += 1
        else:
            self.kept[source, split].append(laid)

    def hold_out(self, split, question, question_tokens):
        """Note a question a held-out template asks, as its row's first window holds it; no train row may ask it."""
        ids = self.layout.tokenizer.encode(question)[:question_tokens]
        self.held_out[split].add(np.array(ids, dtype=np.uint16).tobytes())

    def split(self, source, split):
        rows = self.kept[source, split]
        if not rows:
            raise SystemExit(f"no {source} {split} row came back whole; there is nothing to write")
        return tuple(np.stack(part) for part in zip(*rows))


def advisory_rows(rows, label, inputs, splits):
    """Lay out the rows of `splits` an advisory input set holds; refuses a set built before its trend words were checked
    or by other code than today's."""
    path = inputs / f"{SERVED}_MANIFEST.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    if manifest.get("trend_checked_by") != prepare_advisory.TREND_CHECKED_BY:
        raise SystemExit(f"{inputs} records no trend check in {path.name}, so it was built before the server's trend "
                         "rule; rebuild it with run_alfa_v3.sh inputs")
    prices = Path(manifest["prices"])
    collected = prepare_advisory.collect(prices, manifest["stride"], manifest["cutoff"],
                                         np.random.default_rng(ModelConfig.seed), manifest["limit"], False, True)
    collected = [row for row in collected if not prepare_advisory.drop_reason(*row)]
    counts = dict(collections.Counter(split for split, *_ in collected))
    if counts != manifest["rows"]:
        raise SystemExit(f"collected again, {inputs}'s prices give {counts} rows, not the {manifest['rows']} it holds: "
                         "the advisory code or the prices changed since; rebuild it with run_alfa_v3.sh inputs")
    # The name the server's Subject gives the share. Advisory text never names a company, so where the name turns up it
    # is inside another word ("rec" in "recent") that marking would turn into the name marker.
    market = finance.Market(prices, None, qa.SEC_NAMES, None, listed_names())
    for split, question, evidence, answer in collected:
        if split not in splits:
            continue
        ticker = guardrails.snapshot_values(evidence)["ticker"]
        name = market.subject(ticker).named
        if name in evidence or name in answer:
            rows.add(label, split, "the share's name is part of another word in its row")
            continue
        laid = rows.layout.row(question, evidence, answer, name, ticker, (), advisory.QUESTION_TOKENS)
        rows.add(label, split, laid)
    return manifest


def facts_rows(rows, stale):
    """Lay out the answers written from passages_v1 by generator split, less the `stale` passages."""
    frozen = DATA / finance.STOCK_QA
    held = json.loads((frozen / "split.json").read_text())
    frozen_rows = {row["id"]: row for row in qa.frozen_rows(frozen)}
    snaps = qa.snapshots_of(json.loads((frozen / "universe.json").read_text()))
    for key, (passage, answer) in qa.passage_answers(qa.OUT / "passage_answers").items():
        row = frozen_rows[key]
        split = qa.generator_split(row, held)
        if row["template_id"] in held["test_templates"]:
            rows.hold_out(split, row["question"], stock_qa.QUESTION_TOKENS)
        if key in stale:
            rows.add(finance.STOCK_QA, split, f"a stale passage: {stale[key]}")
            continue
        redated = stock_qa.RETRIEVED.sub(snaps[row["symbol"]]["quality"]["last_date"], passage)
        written_now = {stock_qa.lean(line) for line in row["facts"]}
        if any(line not in written_now for line in redated.split(stock_qa.FACT_SEPARATOR)):
            rows.add(finance.STOCK_QA, split, "holds a line today's facts no longer write")
            continue
        name, ticker = stock_qa.STOCK_LINE.match(redated).groups()
        laid = rows.layout.row(row["question"], redated, answer, name, ticker,
                               tuple({row["company"]} - {name, ticker}), stock_qa.QUESTION_TOKENS)
        if isinstance(laid, str) and redated != passage:
            fetched = re.findall(r"\d{4}-\d{2}-\d{2}", " ".join(stock_qa.RETRIEVED.findall(passage)))
            laid = ("quotes the day its passage was fetched, which re-dated by its last close it no longer says"
                    if any(day in answer for day in fetched) else f"re-dated by its last close, {laid}")
        rows.add(finance.STOCK_QA, split, laid)


def template_rows(rows):
    """Lay out both answers of each v3 template row; refuses a train or validation row of a held-out template."""
    test_templates = {t["id"] for t in qa.short_templates() if t["split"] == "test"}
    answers = {a["id"]: a["answers"] for path in sorted((qa.V3 / "answers").glob("batch_*.jsonl"))
               for a in qa.read_jsonl(path)}
    for row in qa.read_jsonl(qa.V3 / "rows.jsonl"):
        split = TEMPLATE_SPLITS[row["split"]]
        if row["template_id"] in test_templates and split != "template_test":
            raise SystemExit(f"{row['id']} is in {split} but its template {row['template_id']} is held out for test")
        evidence = row["evidence"]
        if row["evidence_kind"] == qa.FACTS_EVIDENCE:
            name, ticker = stock_qa.STOCK_LINE.match(evidence).groups()
            named = (name, ticker, tuple({row["company"]} - {name, ticker}), stock_qa.QUESTION_TOKENS)
        else:
            named = (row["company"], guardrails.snapshot_values(evidence)["ticker"], (), advisory.QUESTION_TOKENS)
        if row["template_id"] in test_templates:
            rows.hold_out(split, row["question"], named[-1])
        for answer in answers.get(row["id"], ()):
            rows.add(qa.V3_DATASET, split, rows.layout.row(row["question"], evidence, answer, *named))
        if row["id"] not in answers:
            rows.add(qa.V3_DATASET, split, "has no answers")


def questions(source):
    """Each row's question ids as bytes, read off its first window: they sit between its first two markers."""
    first = source[:, 0]
    ends = (first == SEP_ID).argmax(axis=1)
    return [first[row, 1:end].tobytes() for row, end in enumerate(ends)]


def check_held_out(train, held_out, test_names, tokenizer):
    """Stop the build when a frozen test stock or a question of a held-out template reaches train. A row held out for
    its stock may ask what train asks, as a template that names no share asks it of every share. The served rows are
    exempt from the stock check, as in alfa_v2: some test stocks are among the 101, kept so they do not regress."""
    check_unseen(train, test_names, tokenizer)
    asked = set(questions(np.concatenate([source for source, _, _ in train.values()])))
    for split, held in sorted(held_out.items()):
        leaked = len(held & asked)
        print(f"{split}: train asks {leaked or 'none'} of the {len(held)} questions its held-out templates ask")
        if leaked:
            raise SystemExit(f"train asks {leaked} questions of {split}'s held-out templates; rebuild without them")


def save(out, split, arrays):
    for part, array in zip(("source", "keep", "target"), arrays):
        np.save(out / f"{DATASET}_{split}_{part}.npy", array)
    print(f"  {split}: {len(arrays[0]):,} rows")
    return len(arrays[0])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--served", required=True, type=Path, help="prepare_advisory --casual's set of the served 101")
    parser.add_argument("--wide", required=True, type=Path, help="prepare_advisory --casual's set over prices_wide")
    parser.add_argument("--stale-passages", required=True, type=Path,
                        help="{passage id: why} for the passages_v1 passages whose lines today's facts write otherwise")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    tokenizer = WordPiece.load(ARTIFACTS / "tokenizer.json")
    rows = Rows(Layout(tokenizer))
    inputs = {SERVED: advisory_rows(rows, SERVED, args.served, ("train", "validation", "unseen", "casual")),
              WIDE: advisory_rows(rows, WIDE, args.wide, ("train", "unseen"))}
    facts_rows(rows, json.loads(args.stale_passages.read_text()))
    template_rows(rows)
    for (source, split, cause), count in sorted(rows.dropped.items()):
        print(f"dropped {count:,} {source} {split}: {cause}")

    train = {name: rows.split(name, "train") for name in (SERVED, WIDE, finance.STOCK_QA, qa.V3_DATASET)}
    held = json.loads((DATA / finance.STOCK_QA / "split.json").read_text())
    universe = json.loads((DATA / finance.STOCK_QA / "universe.json").read_text())
    test_names = held["test_stocks"] + [s["name"] for s in universe if s["symbol"] in held["test_stocks"]]
    check_held_out(train, rows.held_out, test_names, tokenizer)

    tokens = {name: scored_tokens(target) for name, (_, _, target) in train.items()}
    written = tokens[finance.STOCK_QA] + tokens[qa.V3_DATASET]
    copies = oversampling(written, tokens[SERVED] + tokens[WIDE], FACTS_AND_TEMPLATES_SHARE)
    repeats = {SERVED: 1, WIDE: 1, finance.STOCK_QA: copies, qa.V3_DATASET: copies}
    mixed = [np.concatenate([np.repeat(part, repeats[name], axis=0) for name, part in zip(train, parts)])
             for parts in zip(*train.values())]
    shares = {name: {"rows": repeats[name] * len(train[name][0]) / len(mixed[0]),
                     "scored_tokens": repeats[name] * tokens[name] / sum(repeats[n] * tokens[n] for n in train)}
              for name in train}
    print(f"facts and template train rows repeated {copies}x: "
          f"{sum(s['scored_tokens'] for n, s in shares.items() if repeats[n] > 1):.1%} of the scored target tokens")

    args.out.mkdir(parents=True, exist_ok=True)
    order = np.random.default_rng(ModelConfig.seed).permutation(len(mixed[0]))
    counts = {"train": save(args.out, "train", [part[order] for part in mixed])}
    # train_generator scores a split's first rows only, so the facts and template rows go first
    validation = [rows.split(name, "validation") for name in (finance.STOCK_QA, qa.V3_DATASET, SERVED)]
    counts["validation"] = save(args.out, "validation", [np.concatenate(p) for p in zip(*validation)])
    for split, arrays in (("stockqa", rows.split(finance.STOCK_QA, "stockqa")),
                          ("template_test", rows.split(qa.V3_DATASET, "template_test")),
                          ("unseen", rows.split(SERVED, "unseen")), ("casual", rows.split(SERVED, "casual"))):
        counts[split] = save(args.out, split, arrays)
    # the wide rows come a ticker at a time, so its first rows would be a few tickers
    wide = rows.split(WIDE, "unseen")
    counts["wide_unseen"] = save(args.out, "wide_unseen",
                                 [p[np.random.default_rng(ModelConfig.seed).permutation(len(p))] for p in wide])

    shutil.copy2(ARTIFACTS / "tokenizer.json", args.out / "tokenizer.json")
    trained_on = list(train)
    if set(trained_on) - set(promote.DATASETS):
        raise SystemExit(f"{trained_on} names datasets promote.DATASETS {promote.DATASETS} does not know")
    (args.out / "MANIFEST.json").write_text(json.dumps({
        "train": {name: len(part[0]) for name, part in train.items()}, "copies": repeats,
        "scored_tokens": tokens, "shares": shares,
        "validation_starts_with": {name: len(part[0])
                                   for name, part in zip((finance.STOCK_QA, qa.V3_DATASET), validation)},
        "layout": {"windows": slots.WINDOWS, "window_tokens": ModelConfig.max_text_length,
                   "question_tokens": {"advisory": advisory.QUESTION_TOKENS, "facts": stock_qa.QUESTION_TOKENS}},
        "inputs": {SERVED: {"path": str(args.served), **inputs[SERVED]},
                   WIDE: {"path": str(args.wide), **inputs[WIDE]}},
        "stale_passages": {"path": str(args.stale_passages),
                           "sha1": hashlib.sha1(args.stale_passages.read_bytes()).hexdigest()},
        "dropped": {f"{source} {split}: {cause}": n for (source, split, cause), n in sorted(rows.dropped.items())},
        # what promote.py records the checkpoint trained on
        "trained_on": trained_on, "rows": counts, "seed": ModelConfig.seed}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
