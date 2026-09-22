"""Turning collected Stack Exchange threads into things the generator can train on.

Two kinds of pair come out of the same records. A supervised pair is a question and the answer its
asker accepted, which is what teaches the decoder to write. A preference pair is two answers to one
question where the site voted one above the other, which is the signal a reward model needs — and it
is free, because the votes were already cast. Nothing here asks a human for anything.

Reading the files is the only I/O, and it happens at the top so the rest stays pure and testable.
"""

import json
import re
from collections import defaultdict

# Targets are cut at a sentence boundary, never mid-clause. A target that stops mid-clause teaches
# the model to stop mid-clause, because it never sees the end marker follow a finished thought.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# A fallback answer needs at least one upvote. Without that the target is an answer nobody endorsed,
# which is a worse lesson than no lesson: 51% of our questions have no accepted answer.
MINIMUM_FALLBACK_SCORE = 1


def load_threads(qa_root):
    """(questions, answers_by_key) over every data/qa/<site> directory.

    Keyed by (site, id) because post ids restart per site, so an id alone collides across the three.
    """
    questions, answers = [], defaultdict(list)
    for site in sorted(path for path in qa_root.iterdir() if path.is_dir()):
        with (site / "questions.jsonl").open(encoding="utf-8") as handle:
            questions.extend(json.loads(line) for line in handle)
        with (site / "answers.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                answers[(record["site"], record["question_id"])].append(record)
    return questions, answers


def question_text(question):
    """The title carries the actual question and the body only elaborates, so the title leads."""
    title = (question.get("title") or "").strip()
    body = (question.get("text") or "").strip()
    return f"{title}\n\n{body}".strip()


def truncate_to_sentences(text, encode, budget):
    """The longest run of whole sentences that fits, or "" if even the first will not.

    Budget is in tokens and leaves room for the two markers the generator brackets an answer with.
    """
    remaining = budget - 2
    kept, used = [], 0
    for sentence in _SENTENCE_END.split(text.strip()):
        length = len(encode(sentence))
        if used + length > remaining:
            break
        kept.append(sentence)
        used += length
    return " ".join(kept)


def best_answer(question, candidates):
    """The answer to learn from, or None if this thread has nothing worth learning.

    An accepted answer wins outright: the person with the problem said it solved it, which is a
    stronger signal than a vote count from people who did not have it.
    """
    accepted = question.get("accepted_answer_id")
    if accepted is not None:
        for candidate in candidates:
            if str(candidate["id"]) == str(accepted):
                return candidate
    scored = [candidate for candidate in candidates if (candidate["score"] or 0) >= MINIMUM_FALLBACK_SCORE]
    return max(scored, key=lambda candidate: candidate["score"]) if scored else None


def supervised_pairs(questions, answers, encode, budget):
    """(key, question, answer) per thread, answers already truncated to the budget.

    The key travels with the pair so the retriever can exclude this thread when it grounds this
    example. Without that it retrieves the answer it is being asked to produce, the model learns to
    copy one passage verbatim, and faithfulness looks perfect while nothing has been learned.
    """
    for question in questions:
        key = (question["site"], question["id"])
        chosen = best_answer(question, answers.get(key, ()))
        if chosen is None:
            continue
        target = truncate_to_sentences(chosen["text"], encode, budget)
        if target:
            yield key, question_text(question), target


def preference_pairs(questions, answers):
    """(key, question, chosen, rejected) wherever the site ranked one answer above another.

    Only the widest gap per thread is used. Adjacent scores are noise — vote counts on a ten-year-old
    thread reflect who saw it, not only how good it was — so a clear gap is the only honest pair.
    """
    for question in questions:
        key = (question["site"], question["id"])
        ranked = sorted(
            (candidate for candidate in answers.get(key, ()) if candidate["score"] is not None),
            key=lambda candidate: candidate["score"],
        )
        if len(ranked) < 2 or ranked[-1]["score"] <= ranked[0]["score"]:
            continue
        yield key, question_text(question), ranked[-1]["text"], ranked[0]["text"]
