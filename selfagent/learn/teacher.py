"""Who decides whether an answer was good, when there is no financial analyst to ask.

Two judges behind one call. The oracle judges rows that came with a gold answer, which is cheap and
exact and only exists on the built dataset. The agent judges rows that did not — a real question typed
by a user — and the agent here is Claude, working in session.

The agent's verdicts are cached to a file and replayed, never re-asked. That is what makes a learning
curve reproducible: the second run of an experiment has to see the same feedback as the first. A cache
miss is reported as unjudged rather than guessed, because a fabricated label is indistinguishable from
a real one once it is in the log, and it would flatter every number downstream.
"""

import hashlib
import json
from pathlib import Path

from ..agent import guardrails
from .store import AGENT, ORACLE


def verdict_key(question, answer):
    """What a cached verdict is filed under: the pair that was actually judged."""
    return hashlib.sha256(f"{question}\x00{answer}".encode()).hexdigest()[:16]


def pair_key(question, chosen, rejected):
    """What a pairwise verdict is filed under. Sorted, so the same two answers file under one key however
    the corpus happened to rank them -- otherwise re-running with the order flipped would ask twice."""
    return verdict_key(question, "\x00".join(sorted((chosen, rejected))))


def shows_rejected_first(key):
    """Whether the judge sees the lower-ranked answer first. Derived from the key rather than stored, so it
    is the same on every run and `--tell` can undo it without a file to lose.

    It has to vary: put the preferred answer first every time and a judge reads the position instead of the
    answer, and the agreement number then measures the layout."""
    return int(key, 16) % 2 == 1


class OracleTeacher:
    """Judges against the gold answer the dataset carries. Exact, and only available on built rows."""

    name = ORACLE

    def __init__(self, refusal):
        self.refusal = refusal

    def judge(self, question, evidence, answer, gold=None):
        """(is_right, why). Requires a gold answer — without one there is nothing exact to compare to.

        Word-split rather than string equality, matching how check_answers.py scores exact match: a
        labeller that counted a row right where the scorer counted it wrong would put the feedback log
        and the reported accuracy permanently out of step. An invented figure is named ahead of a
        wording difference because it is the failure the whole design exists to catch.
        """
        if gold is None:
            raise ValueError(
                f"the oracle needs a gold answer and got none for {question!r}; "
                f"route rows without one to the agent instead"
            )
        unsupported = guardrails.unsupported_figures(answer, evidence)
        if unsupported:
            return False, f"states figures the evidence does not: {unsupported}"
        refused = guardrails.is_refusal(answer, self.refusal)
        if refused != guardrails.is_refusal(gold, self.refusal):
            return False, "refused an answerable question" if refused else "answered without evidence"
        if answer.split() != gold.split():
            return False, "differs from the gold answer"
        return True, "matches the gold answer"


class AgentTeacher:
    """Judges by looking up a verdict Claude wrote, and records a request when there is none.

    The loop is: run, collect the pending file, have the agent fill it in, run again. Slower than an
    API call and it costs nothing, needs no key, and leaves an auditable record of every judgement that
    shaped the model — which for this project is the point rather than a consolation.
    """

    name = AGENT

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._verdicts = self._read()
        self.asked = []

    def _read(self):
        if not self.path.exists():
            return {}
        verdicts = {}
        for line in self.path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                verdicts[row["key"]] = (row["is_right"], row.get("why", ""))
        return verdicts

    def judge(self, question, evidence, answer, gold=None):
        """(is_right, why), or (None, reason) when the agent has not seen this pair yet."""
        key = verdict_key(question, answer)
        if key in self._verdicts:
            return self._verdicts[key]
        self.asked.append({"key": key, "question": question, "evidence": evidence, "answer": answer})
        return None, "no verdict yet"

    def write_requests(self, path):
        """The rows needing a verdict, as one JSON object per line for the agent to read and answer."""
        path = Path(path)
        path.write_text("".join(json.dumps(row) + "\n" for row in self.asked))
        return len(self.asked)

    def record(self, key, is_right, why=""):
        """One verdict, appended to the cache so the next run replays it instead of asking again."""
        with self.path.open("a") as handle:
            handle.write(json.dumps({"key": key, "is_right": bool(is_right), "why": why}) + "\n")
        self._verdicts[key] = (bool(is_right), why)


class PairJudge:
    """Judges which of two answers to the same question is the better one, again by reading Claude's
    verdicts back from a file.

    Not a subclass of AgentTeacher, and not sharing its verdict file: the question being answered is a
    different one -- "which of these two" rather than "was this right" -- and a caller that swapped one for
    the other would get a boolean whose meaning had silently changed. The mechanics are duplicated, which
    is the cheaper of the two mistakes available here.

    What the boolean means: the corpus ranked these two answers, and True says a reader agrees with that
    ranking. So the agreement rate this produces is a measurement of the vote labels themselves, and the
    pairs it disagrees on are the ones worth looking at before trusting 22,379 of them.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._verdicts = self._read()
        self.asked = []

    def _read(self):
        if not self.path.exists():
            return {}
        verdicts = {}
        for line in self.path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                verdicts[row["key"]] = (row["agrees"], row.get("why", ""))
        return verdicts

    def judge(self, question, chosen, rejected):
        """(agrees_with_the_corpus, why), or (None, reason) when this pair has no verdict yet."""
        key = pair_key(question, chosen, rejected)
        if key in self._verdicts:
            return self._verdicts[key]
        first, second = (rejected, chosen) if shows_rejected_first(key) else (chosen, rejected)
        self.asked.append({"key": key, "question": question, "a": first, "b": second})
        return None, "no verdict yet"

    def write_requests(self, path):
        """The pairs needing a verdict, one JSON object per line, with no sign of which way they were
        ranked. That omission is the whole value of the number they produce."""
        path = Path(path)
        path.write_text("".join(json.dumps(row) + "\n" for row in self.asked))
        return len(self.asked)

    def record(self, key, better, why=""):
        """One verdict. `better` is "a" or "b" as the request file laid them out, not as the corpus did."""
        if better not in ("a", "b"):
            raise ValueError(f"a pairwise verdict is 'a' or 'b', got {better!r} for pair {key}")
        agrees = (better == "b") if shows_rejected_first(key) else (better == "a")
        with self.path.open("a") as handle:
            handle.write(json.dumps({"key": key, "agrees": agrees, "why": why}) + "\n")
        self._verdicts[key] = (agrees, why)
