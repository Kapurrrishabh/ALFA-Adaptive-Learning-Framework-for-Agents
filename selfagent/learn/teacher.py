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
