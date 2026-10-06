"""The collected Q&A records as retrievable documents: one per thread, holding the answer alone.

Why this exists next to `corpus.documents`. The same threads sit on disk twice. `data/text/stackexchange`
holds each one flattened -- title, then the asker's question, then every answer appended -- which is what
pretraining reads. `data/qa` holds the same posts as records, keeping what flattening throws away: which
answer the asker accepted, and how the site voted.

Indexing the flattened form means indexing question text, and a question retrieved for a question is not
an answer. Measured, that is not a small effect: asking the served index "what is a stop loss order ?"
quoted a stranger's wash-sale question back, while the passages ranked second to fourth were all about
stop losses. Indexing the records instead puts only endorsed answers in the corpus, so the worst a quote
can be is the wrong answer rather than somebody else's question.

**The document key is the one the manifest already files this thread under**, not a new id. An answer and
the thread it came from are one source, and `data/manifest.jsonl` is where its URL and CC BY-SA licence
are recorded -- a citation that resolves nowhere is not attribution. A thread the manifest does not hold
is skipped for that reason rather than indexed unattributable.
"""

import json
import re
from pathlib import Path

from backend.models.data.qa_pairs import best_answer, load_threads

SOURCE = "stackexchange"

# The URL the collector recorded per question, which is what the manifest keys on.
_THREAD = re.compile(r"https://([^/]+)/questions/(\d+)")


class Answer:
    """One endorsed answer, shaped like what `chunks` reads off a document.

    Not a `Document`: that one reads its text off disk on access, because the extracted corpus is larger
    than memory, and these arrive as records from a file the trainer already loads whole.
    """

    __slots__ = ("key", "source", "day", "text")

    def __init__(self, key, source, text):
        self.key = key
        self.source = source
        self.text = text
        # The records carry no creation date, so these answer only questions that name no date. The
        # index is what enforces that; `--undated` at the build is what lets them in at all.
        self.day = ""

    def __repr__(self):
        return f"Answer({self.key!r}, {self.source!r}, {len(self.text)} chars)"


def keys_by_thread(manifest):
    """{(site, question id): document key} for the threads the manifest holds."""
    keys = {}
    with open(manifest, encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            if entry["source"] != SOURCE:
                continue
            found = _THREAD.match(entry["url"])
            if found:
                keys[found.group(1), found.group(2)] = entry["sha256"][:16]
    return keys


def answers(manifest, qa_root):
    """One `Answer` per thread whose asker accepted an answer or whose voters endorsed one.

    Sourced per site -- `qa_money`, `qa_quant`, `qa_economics` -- because the three are different
    corpora wearing one name: money is people asking about their own money, and the other two are
    coursework and derivations. Naming them separately is what lets the budget weight them differently
    and what makes a bad retrieval traceable to the site it came from.
    """
    keys = keys_by_thread(manifest)
    questions, held = load_threads(Path(qa_root))
    built = []
    for question in questions:
        chosen = best_answer(question, held.get((question["site"], question["id"]), ()))
        if chosen is None:
            continue
        key = keys.get((question["site"], question["id"]))
        if key is not None:
            built.append(Answer(key, f"qa_{question['site'].split('.')[0]}", chosen["text"]))
    if not built:
        raise ValueError(
            f"no thread under {qa_root} is recorded in {manifest}; the two were collected from "
            f"different runs, and an answer whose URL and licence are unrecorded cannot be cited")
    return built
