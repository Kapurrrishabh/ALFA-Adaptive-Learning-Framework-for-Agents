"""Who the user is, what they asked, and what they hold. One SQLite file, nothing to start.

Two rules shape the schema.

**A position is not stored.** Trades are the log; a holding is a fold of it. A `holdings` table next to
a `trades` table is two places that have to agree about the same fact, and they drift the first time a
write half-fails. The reference UI asks for symbol, quantity and average price, so that is what the fold
returns, and nothing can report a position the trades do not support.

**A user id is never an argument.** Every read and write takes a session token and resolves it here, so
"see only your own" is the only way the methods can be called rather than a check each one remembers to
do. Passing a user id in would make the wrong-user read expressible, and then a missing `WHERE` clause
in one method is a data leak.

Feedback stays in `selfagent.learn.store.FeedbackLog`, which owns that table and is what the learning
loop refits from. This store opens the *same file*, so a message can hold a real foreign key to the
feedback row that judged it, and the loop keeps its one durable log rather than gaining a rival.

The reference repo has no accounts to copy: its portfolio is a process-global list, lost on restart and
shared by everyone, and its login page talks to Firebase, which needs a hosted key this project does not
use. So identity is local and the password hashing is stdlib.
"""

import hashlib
import secrets
import sqlite3
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path

from selfagent.learn import FeedbackLog

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT    NOT NULL UNIQUE,
    salt          TEXT    NOT NULL,
    password_hash TEXT    NOT NULL,
    joined_at     TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token      TEXT    PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users (id),
    started_at TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users (id),
    started_at TEXT    NOT NULL,
    subject    TEXT    NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations (id),
    at              TEXT    NOT NULL,
    question        TEXT    NOT NULL,
    served          TEXT    NOT NULL,
    intent          TEXT    NOT NULL,
    ticker          TEXT    NOT NULL,
    evidence        TEXT    NOT NULL,
    confidence      REAL,
    stated          REAL,
    spoke           INTEGER NOT NULL,
    because         TEXT    NOT NULL DEFAULT '',
    feedback_id     INTEGER REFERENCES feedback (id)
);
CREATE TABLE IF NOT EXISTS trades (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id  INTEGER NOT NULL REFERENCES users (id),
    at       TEXT    NOT NULL,
    symbol   TEXT    NOT NULL,
    side     TEXT    NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity REAL    NOT NULL CHECK (quantity > 0),
    price    REAL    NOT NULL CHECK (price > 0)
);
CREATE INDEX IF NOT EXISTS conversations_by_user ON conversations (user_id);
CREATE INDEX IF NOT EXISTS messages_by_conversation ON messages (conversation_id);
CREATE INDEX IF NOT EXISTS trades_by_user ON trades (user_id, symbol);
"""

BUY = "buy"
SELL = "sell"

# Deliberately slow: the cost of one login is the cost of one guess. 200,000 rounds measures at about
# 0.09s here, which nobody notices once and an attacker pays on every attempt.
ROUNDS = 200_000

# How many past turns a caller gets back for the open conversation. It is what the UI shows, not
# something the decoder reads -- it answers one question at a time -- so no measurement rides on it.
BUFFER_TURNS = 10

Position = namedtuple("Position", "symbol quantity average_price")
Message = namedtuple("Message", "at question served intent ticker evidence confidence stated spoke "
                                "because feedback_id")


class NotYours(LookupError):
    """Raised when a token is valid but the row it asked for belongs to someone else."""


def fold_positions(trades):
    """(symbol, quantity, average price) per symbol, from trades in the order they happened.

    Average cost, so a sell takes stock out at the average and leaves the average where it was. Kept
    free of the database because it is the one piece of arithmetic here worth testing directly.
    """
    held = {}
    for symbol, side, quantity, price in trades:
        have, average = held.get(symbol, (0.0, 0.0))
        if side == BUY:
            average = (have * average + quantity * price) / (have + quantity)
            have += quantity
        else:
            if quantity > have:
                raise ValueError(
                    f"cannot sell {quantity} of {symbol} holding {have}; the trade log would not "
                    f"support the position it implies")
            have -= quantity
        held[symbol] = (have, average if have else 0.0)
    return [Position(symbol, have, average) for symbol, (have, average) in sorted(held.items())
            if have]


class Store:
    """Accounts, conversations and simulated trades for one deployment, in one file."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript(SCHEMA)
        self._connection.commit()
        # Same file, its own connection: the feedback table belongs to the learning loop, and this is
        # how a message can point at a judged row without either side owning the other's schema.
        self.feedback = FeedbackLog(self.path)

    def close(self):
        self.feedback.close()
        self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ---- accounts

    def register(self, email, password):
        """A new account. Returns its id, and raises if the email is taken or the password is empty."""
        if not password:
            raise ValueError(f"no password given for {email}; an account needs one to be logged into")
        salt = secrets.token_hex(16)
        try:
            cursor = self._connection.execute(
                "INSERT INTO users (email, salt, password_hash, joined_at) VALUES (?, ?, ?, ?)",
                (email, salt, _hash(password, salt), _now()))
        except sqlite3.IntegrityError as error:
            raise ValueError(f"{email} already has an account") from error
        self._connection.commit()
        return cursor.lastrowid

    def log_in(self, email, password):
        """A session token, or ValueError. One message for both failures, so it names neither."""
        row = self._connection.execute(
            "SELECT id, salt, password_hash FROM users WHERE email = ?", (email,)).fetchone()
        if row is None or not secrets.compare_digest(row["password_hash"],
                                                     _hash(password, row["salt"])):
            raise ValueError(f"no account matches that email and password ({email})")
        token = secrets.token_urlsafe(32)
        self._connection.execute(
            "INSERT INTO sessions (token, user_id, started_at) VALUES (?, ?, ?)",
            (token, row["id"], _now()))
        self._connection.commit()
        return token

    def log_out(self, token):
        self._connection.execute("DELETE FROM sessions WHERE token = ?", (token,))
        self._connection.commit()

    def user_of(self, token):
        """The account a token belongs to. The only route from a request to a user id."""
        row = self._connection.execute(
            "SELECT user_id FROM sessions WHERE token = ?", (token,)).fetchone()
        if row is None:
            raise NotYours("that session is not signed in; log in again")
        return row["user_id"]

    # ---- conversations

    def start_conversation(self, token, subject=""):
        cursor = self._connection.execute(
            "INSERT INTO conversations (user_id, started_at, subject) VALUES (?, ?, ?)",
            (self.user_of(token), _now(), subject))
        self._connection.commit()
        return cursor.lastrowid

    def conversations(self, token):
        """(id, started_at, subject, how many messages) for this user's conversations, newest first."""
        return [tuple(row) for row in self._connection.execute(
            "SELECT c.id, c.started_at, c.subject, COUNT(m.id) FROM conversations c "
            "LEFT JOIN messages m ON m.conversation_id = c.id "
            "WHERE c.user_id = ? GROUP BY c.id ORDER BY c.id DESC", (self.user_of(token),))]

    def remember(self, token, conversation_id, turn, feedback_id=None):
        """Stores one `Turn`, including a refused one: what the agent would not say is history too."""
        self._must_own(token, conversation_id)
        cursor = self._connection.execute(
            "INSERT INTO messages (conversation_id, at, question, served, intent, ticker, evidence, "
            "confidence, stated, spoke, because, feedback_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (conversation_id, _now(), turn.question, turn.served, turn.intent, turn.ticker or "",
             turn.evidence, _number(turn.confidence), _number(turn.stated), int(turn.spoke),
             turn.because or "", feedback_id))
        self._connection.commit()
        return cursor.lastrowid

    def recent(self, token, conversation_id, limit=BUFFER_TURNS):
        """The session buffer: the last few messages of one conversation, oldest first."""
        self._must_own(token, conversation_id)
        rows = self._connection.execute(
            "SELECT at, question, served, intent, ticker, evidence, confidence, stated, spoke, "
            "because, feedback_id FROM messages WHERE conversation_id = ? "
            "ORDER BY id DESC LIMIT ?", (conversation_id, limit)).fetchall()
        return [Message(*row) for row in reversed(rows)]

    # ---- portfolio

    def record_trade(self, token, symbol, side, quantity, price):
        """A simulated trade. Refused if the log would then imply a position the user never had."""
        if side not in (BUY, SELL):
            raise ValueError(f"a trade is '{BUY}' or '{SELL}', not {side!r}")
        user = self.user_of(token)
        # Folded before the insert, so an over-sell is refused instead of stored and skipped later.
        fold_positions(self._trades_of(user) + [(symbol, side, quantity, price)])
        cursor = self._connection.execute(
            "INSERT INTO trades (user_id, at, symbol, side, quantity, price) VALUES (?, ?, ?, ?, ?, ?)",
            (user, _now(), symbol, side, quantity, price))
        self._connection.commit()
        return cursor.lastrowid

    def holdings(self, token):
        return fold_positions(self._trades_of(self.user_of(token)))

    def trades(self, token):
        """Every trade this user made, oldest first."""
        return [tuple(row) for row in self._connection.execute(
            "SELECT at, symbol, side, quantity, price FROM trades WHERE user_id = ? "
            "ORDER BY id", (self.user_of(token),))]

    def _trades_of(self, user_id):
        return [tuple(row) for row in self._connection.execute(
            "SELECT symbol, side, quantity, price FROM trades WHERE user_id = ? ORDER BY id",
            (user_id,))]

    def _must_own(self, token, conversation_id):
        owned = self._connection.execute(
            "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, self.user_of(token))).fetchone()
        if owned is None:
            raise NotYours(f"conversation {conversation_id} does not belong to this session")


def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), ROUNDS).hex()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _number(value):
    """None for a value that is not a number, so a refused turn stores a null and not a NaN."""
    return None if value is None or value != value else float(value)
