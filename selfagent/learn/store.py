"""The feedback log: every answer the agent gave, and what turned out to be true about it.

This is the one durable thing in the learning loop. Everything else — a calibrator, a threshold, a
ranking — is derived from it and can be thrown away and refitted. So the log is append-only and records
who judged each row, because a result that only holds under one judge is not a result.

SQLite on purpose: the loop has to run on a laptop with no service to start, and a single file is also
the easiest thing to copy next to a set of numbers when reporting them.
"""

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    asked_at    TEXT    NOT NULL,
    question    TEXT    NOT NULL,
    evidence    TEXT    NOT NULL,
    answer      TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    is_right    INTEGER,
    labeller    TEXT    NOT NULL,
    why         TEXT,
    extra       TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS feedback_by_labeller ON feedback (labeller);
"""

# Which judge decided a row. The oracle only speaks where ground truth exists; the agent speaks where
# judgement is needed, and its verdicts are stored rather than re-asked so a run can be replayed.
ORACLE = "oracle"
AGENT = "agent"
HUMAN = "human"


class FeedbackLog:
    """Append-only feedback, and the training set that replays out of it.

    Rows are never updated or deleted. A label that turns out wrong is superseded by a later row for the
    same question, not edited: the history of what the agent believed is the data the thesis is about.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Served from an event loop, the thread using this is not the thread that opened it. Callers
        # still use it from one thread at a time.
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def close(self):
        self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def append(self, asked_at, question, evidence, answer, confidence, is_right, labeller, why="",
               **extra):
        """One judged turn. `is_right` may be None for a turn nobody has judged yet.

        `labeller` is required rather than defaulted: a row whose judge is unknown cannot be excluded
        from a measurement later, and that is exactly when it matters.
        """
        if labeller not in (ORACLE, AGENT, HUMAN):
            raise ValueError(f"unknown labeller {labeller!r}; expected one of {ORACLE}, {AGENT}, {HUMAN}")
        cursor = self._connection.execute(
            "INSERT INTO feedback (asked_at, question, evidence, answer, confidence, is_right,"
            " labeller, why, extra) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (asked_at, question, evidence, answer, float(confidence),
             None if is_right is None else int(bool(is_right)), labeller, why, json.dumps(extra)),
        )
        self._connection.commit()
        return cursor.lastrowid

    def __len__(self):
        return self._connection.execute("SELECT count(*) FROM feedback").fetchone()[0]

    def rows(self, labeller=None, judged_only=True, limit=0):
        """Every row, oldest first, so a replay sees the feedback in the order it arrived.

        Order matters more than it looks: a learning curve that shuffled its history would be measuring
        a different experiment, one where the agent had tomorrow's feedback today.
        """
        where = ["1=1"]
        parameters = []
        if labeller is not None:
            where.append("labeller = ?")
            parameters.append(labeller)
        if judged_only:
            where.append("is_right IS NOT NULL")
        query = f"SELECT * FROM feedback WHERE {' AND '.join(where)} ORDER BY id"
        if limit:
            query += f" LIMIT {int(limit)}"
        return [self._as_dict(row) for row in self._connection.execute(query, parameters)]

    def agreement(self):
        """Where the oracle and the agent judged the same answer, how often they agreed.

        The number that says whether agent-as-teacher is usable at all. Matched on (question, answer)
        because that pair is what was judged — the same question answered differently is a different row.
        """
        seen = {}
        for row in self.rows(judged_only=True):
            seen.setdefault((row["question"], row["answer"]), {})[row["labeller"]] = row["is_right"]
        both = [v for v in seen.values() if ORACLE in v and AGENT in v]
        agreed = sum(v[ORACLE] == v[AGENT] for v in both)
        return agreed, len(both)

    @staticmethod
    def _as_dict(row):
        as_dict = dict(row)
        as_dict["extra"] = json.loads(as_dict["extra"])
        as_dict["is_right"] = None if as_dict["is_right"] is None else bool(as_dict["is_right"])
        return as_dict
