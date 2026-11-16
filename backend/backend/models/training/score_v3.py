#!/usr/bin/env python3
"""Answer alfa_v3's held-out rows and judge each answer as the server would: v3's slot checkpoints, and a plain one such
as v2 step36000 on the same rows.

    python -m backend.models.training.score_v3 --data <alfa_v3_data> --out <report.md> [--rows N] \
        --slots <v3 checkpoint>... [--plain <v2 checkpoint>...]

Every split's rows are collected again as text by prepare_alfa_v3's own functions and laid out by its Layout, and each
must be one of the rows it stored for the split. A slot answer is filled by slots.from_slots; a marker naming a figure
its source lacks is a slot-fill failure, which the server refuses. A plain checkpoint reads the row's question and
unmarked evidence as context.assemble lays out a served v2 turn, in one passage. Each answer then meets the server's
guards, and the terms it defines are judged against TERM_MEANINGS. <report>.json beside the report keeps every row.
"""
from backend.paths import ARTIFACTS

import argparse
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from backend.advisory.report import TERM_MEANINGS
from backend.knowledge_base.lessons import STOP
from backend.models.agent import finance
from backend.models.agent.context import assemble
from backend.models.core.config import ModelConfig
from backend.models.core.tokenizer.vocab import CLS_ID, PAD_ID, SEP_ID
from backend.models.core.tokenizer.wordpiece import WordPiece, pretokenize
from backend.models.data import slots, stock_qa
from backend.models.guardrails import guardrails
from backend.models.networks.generator import ANSWER_TEMPERATURE, ANSWER_TOP_P
from backend.models.training import prepare_alfa_v3 as v3, prepare_stock_qa as qa
from backend.models.training.check_answers import load_model, repeated_fraction
from backend.models.training.train_generator import load_split

# alfa_v3's held-out splits, each by the (source, split) prepare_alfa_v3 collects its rows under
SPLITS = {"stockqa": (finance.STOCK_QA, "stockqa"), "template_test": (qa.V3_DATASET, "template_test"),
          "unseen": (v3.SERVED, "unseen"), "casual": (v3.SERVED, "casual"), "wide_unseen": (v3.WIDE, "unseen")}
SAMPLES = 10
BATCH = 8
# What stops an answer before a guard reads it: a slot marker naming no figure, or a plain row its window cannot hold
SLOT_FILL, OVER_WINDOW = "slot fill", "over one window"
GUARDS = ("unsupported figure", "another share", "wrong trend", "mislabelled figure", "own meaning")
COLUMNS = ("exact", "refused", SLOT_FILL, OVER_WINDOW, *GUARDS, "figures not in evidence", "defines a term",
           "contradicts a meaning", "repeated 4-grams", "distinct words", "mean words")
RATES = {"exact", "refused", SLOT_FILL, OVER_WINDOW, *GUARDS, "defines a term", "contradicts a meaning",
         "repeated 4-grams"}
# The other spellings answers give TERM_MEANINGS' terms, beside the label guard's. Its other metrics are read as terms
# of no meaning, so the equity in "debt / equity" is not taken for one.
SPELLINGS = {"Max drawdown": ("maximum drawdown",), "EPS": ("earnings per share",),
             "ATRs": ("ATR", "average true range"), "P/E": ("price-to-earnings", "price to earnings"),
             "RSI": ("relative strength index",)}
OPPOSITES = {"rising": "falling", "above": "below", "more": "less", "positive": "negative", "before": "after",
             "biggest": "smallest", "short": "long"}
OPPOSITES.update({second: first for first, second in list(OPPOSITES.items())})


def _read(text):
    return " ".join(pretokenize(text))


def _content(text):
    return {w[:-1] if w.endswith("s") and not w.endswith("ss") else w for w in text.split()
            if w.isalpha() and len(w) > 1 and w not in STOP}


def _spellings():
    """{a spelling as pretokenize reads it: its TERM_MEANINGS term}, the label guard's other metrics as None."""
    spelt = {_read(spelling): None for group in stock_qa.METRICS for spelling in group}
    for term in TERM_MEANINGS:
        group = next((group for group in stock_qa.METRICS if term.lower() in map(str.lower, group)), ())
        spelt.update(dict.fromkeys(map(_read, (term, *group, *SPELLINGS.get(term, ()))), term))
    return spelt


_SPELT = _spellings()
_TERMS = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, sorted(_SPELT, key=len, reverse=True))) + r")(?!\w)")
_MEANT = {term: _content(_read(meaning)) for term, meaning in TERM_MEANINGS.items()}
# A meaning right after a term or the figure given it: ", a ...", "( ... )", "means ...", "is a ..."
_DEFINITION = re.compile(r"(?: (?:reads|is|was|of|at) \S+)?(?: , (?:which (?:is|means|measures) |that is |meaning )?"
                         r"(?=(?:a|an|the|how|what) )| \( | (?:means|measures|compares|weighs|tracks) | is "
                         r"(?=(?:a|an|the|how) ))(?P<meaning>.+?)(?= [,;.()]|$)")


def defined_terms(answer):
    """(term, the meaning `answer` gives it, whether that contradicts TERM_MEANINGS) for each term it defines.

    A meaning contradicts when it shares at least two words with another term's entry and more than with its own, that
    term being no part of it ("the biggest fall from a peak" is max drawdown's, which holds drawdown), or when it holds
    the opposite of one of its own entry's words and not the word. One stating a figure is not read, as own_meanings.
    """
    said, found = _read(answer), []
    for named in _TERMS.finditer(said):
        term, meant = _SPELT[named.group(0)], _DEFINITION.match(said, named.end())
        if term is None or meant is None or guardrails.figures(meant.group("meaning")):
            continue
        meaning = meant.group("meaning")
        shared = {other: len(_content(meaning) & words) for other, words in _MEANT.items()}
        closest = max(shared, key=shared.get)
        apart = not (closest.lower() in term.lower() or term.lower() in closest.lower()
                     or closest.lower() in TERM_MEANINGS[term].lower())
        words, own = set(meaning.split()), set(_read(TERM_MEANINGS[term]).split())
        flipped = any(OPPOSITES[word] in words and word not in words for word in own & OPPOSITES.keys())
        found.append((term, meaning, flipped or (apart and shared[closest] >= 2 and shared[closest] > shared[term])))
    return found


def refusals(answer, evidence, names):
    """{guard: what it found} for each server guard that stops `answer`, given what core._written gives them."""
    facts = stock_qa.STOCK_LINE.match(evidence) is not None
    found = {"unsupported figure": guardrails.unsupported_figures(answer, evidence),
             "another share": guardrails.other_shares(answer, evidence, names,
                                                      finance.FACTS_SLOTS if facts else finance.SUBJECT_SLOTS),
             "wrong trend": guardrails.wrong_trend(answer, evidence),
             "mislabelled figure": facts and guardrails.mislabelled_figures(answer, evidence, finance.FACTS_METRICS),
             "own meaning": facts and guardrails.own_meanings(answer, finance.FACTS_METRICS)}
    return {guard: hit for guard, hit in found.items() if hit}


class Recorded(v3.Layout):
    """Layout.row, each laid-out row kept with the text it was laid out from, which filling and guarding need."""

    def row(self, *text):
        laid = super().row(*text)
        return laid if isinstance(laid, str) else (*laid, text)


def checked(name, kept, stored):
    """`kept` when it is the split as stored: as many rows, each one of the stored ones."""
    source, _, target = stored
    held = {s.tobytes() + t.tobytes() for s, t in zip(source, target)}
    strays = sum(s.tobytes() + t.tobytes() not in held for s, _, t, _ in kept)
    if strays or len(kept) != len(source):
        raise SystemExit(f"{name}: collected again it is {len(kept):,} rows, {strays:,} of them not stored, against "
                         f"the {len(source):,} stored; score the data alfa_v3 was built from, or rebuild it")
    return kept


def held_out(data, tokenizer):
    """{split: its rows as Recorded lays them out}, from the inputs and stale passages alfa_v3's manifest records."""
    manifest = json.loads((data / "MANIFEST.json").read_text())
    rows = v3.Rows(Recorded(tokenizer))
    for label, splits in ((v3.SERVED, ("unseen", "casual")), (v3.WIDE, ("unseen",))):
        v3.advisory_rows(rows, label, Path(manifest["inputs"][label]["path"]), splits)
    v3.facts_rows(rows, json.loads(Path(manifest["stale_passages"]["path"]).read_text()))
    v3.template_rows(rows)
    return {name: checked(name, rows.kept[key], load_split(data, v3.DATASET, name)) for name, key in SPLITS.items()}


def plain_layout(tokenizer, config, text):
    """(source, keep) of one passage, as context.assemble lays out a served v2 turn; None when it will not."""
    question, evidence, *_, question_tokens = text
    try:
        context = assemble(tokenizer, question, [evidence], config.max_text_length, question_tokens)
    except ValueError:
        return None
    return context.source[0], context.keep[0]


def generated(checkpoint, plain, chosen):
    """{split: each chosen row's answer as the model wrote it, None where a plain layout will not hold the row}, sampled
    as served from one seed a split, so every checkpoint draws alike."""
    if (checkpoint.parent / "tokenizer.json").read_bytes() != (ARTIFACTS / "tokenizer.json").read_bytes():
        raise SystemExit(f"{checkpoint} reads another tokenizer than the one alfa_v3's rows are laid out with")
    tokenizer, model, config = load_model(checkpoint.parent, "", checkpoint.name)
    answers = {}
    for name, rows in chosen.items():
        started, rng = time.time(), np.random.default_rng(ModelConfig.seed)
        laid = [plain_layout(tokenizer, config, text) if plain else (source, keep) for source, keep, _, text in rows]
        ready = [at for at, row in enumerate(laid) if row is not None]
        answers[name] = [None] * len(rows)
        for start in range(0, len(ready), BATCH):
            batch = ready[start:start + BATCH]
            produced = model.generate(np.stack([laid[at][0] for at in batch]), np.stack([laid[at][1] for at in batch]),
                                      ANSWER_TEMPERATURE, ANSWER_TOP_P, rng)
            for at, ids in zip(batch, produced):
                answers[name][at] = tokenizer.decode(list(ids))
        print(f"{checkpoint.name} on {name}: {len(rows)} rows in {time.time() - started:.0f}s", flush=True)
    return answers


def judged(written, text, wanted, slot_form, tokenizer, names):
    """One row's record: the answer `written`, filled when in `slot_form`, through every guard. Exact is `written` being
    the `wanted` target word for word, as check_answers compares them: filled, the tokenizer reads "INR." otherwise."""
    question, evidence, gold, name, ticker, _, _ = text
    record = {"question": question, "evidence": evidence, "gold": gold, "written": written, "answer": None,
              "exact": written is not None and written.split() == wanted.split()}
    if written is None:
        return {**record, "refused": {OVER_WINDOW: True}}
    answer = written
    if slot_form:
        try:
            answer = slots.from_slots(written, evidence, name, ticker)
        except ValueError as error:
            return {**record, "refused": {SLOT_FILL: str(error)}}
    return {**record, "answer": answer, "said": tokenizer.decode(tokenizer.encode(answer)),
            "refused": refusals(answer, evidence, names), "figures": len(guardrails.figures(answer)),
            "unsupported": guardrails.unsupported_figures(answer, evidence), "meanings": defined_terms(answer)}


def summary(records):
    """The report's numbers for one checkpoint on one split. Length, words and repeats are of the answers it has."""
    answered = [r for r in records if r["answer"] is not None]
    rows = len(records)
    mean = lambda values: float(np.mean(values)) if values else float("nan")
    return {"rows": rows, "exact": sum(r["exact"] for r in records) / rows,
            "refused": sum(bool(r["refused"]) for r in records) / rows,
            **{guard: sum(guard in r["refused"] for r in records) / rows
               for guard in (SLOT_FILL, OVER_WINDOW, *GUARDS)},
            "figures not in evidence": f"{sum(len(r['unsupported']) for r in answered)}/"
                                       f"{sum(r['figures'] for r in answered)}",
            "defines a term": sum(bool(r["meanings"]) for r in answered) / rows,
            "contradicts a meaning": sum(any(wrong for *_, wrong in r["meanings"]) for r in answered) / rows,
            "repeated 4-grams": mean([repeated_fraction(r["said"]) for r in answered]),
            "distinct words": len({word for r in answered for word in r["said"].split()}),
            "mean words": mean([len(r["said"].split()) for r in answered])}


def _cell(column, value):
    return f"{value:.1%}" if column in RATES else f"{value:.1f}" if isinstance(value, float) else str(value)


def report(scored, stored, args, minutes):
    """The markdown: a table a split, then SAMPLES rows of it with each checkpoint's answer and what stopped it."""
    lines = ["# alfa_v3 held-out scores", "",
             f"{args.rows} rows a split, chosen with seed {ModelConfig.seed}, sampled at temperature "
             f"{ANSWER_TEMPERATURE} and top-p {ANSWER_TOP_P} as served, from {args.data}. The pass took {minutes:.1f} "
             "minutes.", "",
             "`exact` is the answer as written being its training target word for word: the slot form for a slot "
             "checkpoint, the plain answer for a plain one. A slot answer is then filled by slots.from_slots, and "
             f"`{SLOT_FILL}` is a marker naming a figure its source lacks. `{OVER_WINDOW}` is a plain row "
             "context.assemble will not lay out. `refused` is any of those or a server guard. `gold` is each row's "
             "slot target as the tokenizer gives it back, filled and checked the same way. `contradicts a meaning` is "
             "defined_terms' judgement against TERM_MEANINGS.", ""]
    for name in SPLITS:
        lines += [f"## {name}: {min(args.rows, stored[name])} of {stored[name]:,} rows", "",
                  "| checkpoint | " + " | ".join(COLUMNS) + " |", "|---" * (len(COLUMNS) + 1) + "|"]
        lines += [f"| {label} | " + " | ".join(_cell(c, split[name]["summary"][c]) for c in COLUMNS) + " |"
                  for label, split in scored.items()]
        lines.append("")
    for name in SPLITS:
        lines += [f"## {name} samples", ""]
        for at, row in enumerate(scored["gold"][name]["records"][:SAMPLES]):
            lines += [f"**Q** {row['question']}", "", f"**E** {row['evidence']}", ""]
            for label, split in scored.items():
                record = split[name]["records"][at]
                stopped = "; ".join(f"{guard}: {hit}" for guard, hit in record["refused"].items())
                lines.append(f"- **{label}**: {record['answer'] or record['written']}"
                             + (f" — *refused, {stopped}*" if stopped else ""))
            lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path, help="alfa_v3's data: its MANIFEST.json and split arrays")
    parser.add_argument("--out", required=True, type=Path, help="the markdown report; its .json beside it")
    parser.add_argument("--rows", type=int, default=200, help="rows a split, chosen as check_answers chooses them")
    parser.add_argument("--slots", nargs="*", default=[], type=Path, help="checkpoints trained on alfa_v3's slot rows")
    parser.add_argument("--plain", nargs="*", default=[], type=Path, help="checkpoints that read evidence unmarked")
    args = parser.parse_args(argv)
    models = [(path, False) for path in args.slots] + [(path, True) for path in args.plain]
    if not models:
        parser.error("give a checkpoint to score with --slots or --plain")

    started = time.time()
    tokenizer = WordPiece.load(ARTIFACTS / "tokenizer.json")
    held = held_out(args.data, tokenizer)
    chosen = {name: [rows[at] for at in np.random.default_rng(ModelConfig.seed).permutation(len(rows))[:args.rows]]
              for name, rows in held.items()}
    # one process a checkpoint: generation re-encodes its whole prefix each token, and the machine has the cores
    with ProcessPoolExecutor(len(models)) as pool:
        answers = list(pool.map(generated, [path for path, _ in models], [plain for _, plain in models],
                                [chosen] * len(models)))
    names = qa.served_names()

    # each row's target as each kind of checkpoint is trained on it: the slot form stored, or its answer as it is
    targets = {True: {name: [tokenizer.decode([i for i in target if i not in (PAD_ID, CLS_ID, SEP_ID)])
                             for _, _, target, _ in rows] for name, rows in chosen.items()},
               False: {name: [tokenizer.decode(tokenizer.encode(text[2])) for *_, text in rows]
                       for name, rows in chosen.items()}}

    def split_scores(answers_to, slot_form):
        records = {name: [judged(answer, text, wanted, slot_form, tokenizer, names) for answer, (*_, text), wanted
                          in zip(answers_to[name], chosen[name], targets[slot_form][name])] for name in SPLITS}
        return {name: {"records": rows, "summary": summary(rows)} for name, rows in records.items()}
    scored = {f"{path.parent.name}/{path.name}": split_scores(answers_to, not plain)
              for (path, plain), answers_to in zip(models, answers)}
    # what a slot checkpoint writing its target exactly would be scored as
    scored["gold"] = split_scores(targets[True], True)
    minutes = (time.time() - started) / 60
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report(scored, {name: len(rows) for name, rows in held.items()}, args, minutes))
    args.out.with_suffix(".json").write_text(json.dumps(scored))
    print(f"wrote {args.out} and its .json in {minutes:.1f} minutes")


if __name__ == "__main__":
    sys.exit(main())
