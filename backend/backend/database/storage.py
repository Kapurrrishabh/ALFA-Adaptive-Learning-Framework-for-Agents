"""SQLite persistence. One file, WAL mode, every row timestamped with its
source. Holdings are never stored: they are derived from transactions, so
positions and trade history cannot disagree. The full logical data model
(docs/DESIGN.md §12) maps onto these tables as the system grows.
"""
from __future__ import annotations

import hashlib
import json
import functools
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.advisory.fusion.evidence import utcnow_iso
from backend.config import DEFAULT_PORTFOLIO

SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol TEXT NOT NULL, date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    source TEXT NOT NULL, retrieved_at TEXT NOT NULL,
    PRIMARY KEY (symbol, date)
);
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL, analysis_ts TEXT NOT NULL, data_ts TEXT,
    label TEXT NOT NULL, score REAL, confidence REAL, payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_analyses_symbol_ts ON analyses(symbol, analysis_ts);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL, model_version TEXT NOT NULL, symbol TEXT NOT NULL,
    horizon INTEGER NOT NULL, as_of TEXT NOT NULL, input_hash TEXT NOT NULL,
    output REAL NOT NULL, created_at TEXT NOT NULL,
    realized REAL, realized_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_predictions_symbol ON predictions(symbol, as_of);
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL, created_at TEXT NOT NULL, body TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL, created_at TEXT NOT NULL, kind TEXT NOT NULL,
    severity TEXT NOT NULL, message TEXT NOT NULL, reasons TEXT NOT NULL,
    acknowledged INTEGER NOT NULL DEFAULT 0   -- no longer read, see alert_acks; dropping a column cannot be undone
);
CREATE TABLE IF NOT EXISTS portfolios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE, cash REAL NOT NULL DEFAULT 0,
    risk_profile TEXT NOT NULL DEFAULT 'moderate', horizon TEXT NOT NULL DEFAULT 'medium',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL, side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity REAL NOT NULL CHECK (quantity > 0), price REAL NOT NULL CHECK (price > 0),
    fees REAL NOT NULL DEFAULT 0, trade_date TEXT NOT NULL, sector TEXT
);
CREATE INDEX IF NOT EXISTS ix_tx_portfolio ON transactions(portfolio_id, trade_date);
-- funds stay out of transactions: the stock report prices each symbol there on the exchange
CREATE TABLE IF NOT EXISTS fund_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    scheme_code INTEGER NOT NULL, side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    units REAL NOT NULL CHECK (units > 0), nav REAL NOT NULL CHECK (nav > 0),
    fees REAL NOT NULL DEFAULT 0 CHECK (fees >= 0), trade_date TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_fund_tx_portfolio ON fund_transactions(portfolio_id, trade_date);
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL, value REAL NOT NULL, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pattern_validation (
    pattern TEXT NOT NULL, family TEXT NOT NULL, horizon INTEGER NOT NULL,
    result TEXT NOT NULL, universe TEXT NOT NULL, computed_at TEXT NOT NULL,
    PRIMARY KEY (pattern, family)
);
CREATE TABLE IF NOT EXISTS signal_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL, as_of TEXT NOT NULL, strategy TEXT NOT NULL, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL, version TEXT NOT NULL, status TEXT NOT NULL
        CHECK (status IN ('candidate', 'approved', 'deployed', 'retired', 'rejected')),
    training_period TEXT, features TEXT, hyperparameters TEXT, metrics TEXT,
    backtest TEXT, limitations TEXT, created_at TEXT NOT NULL, approved_by TEXT,
    UNIQUE (name, version)
);
CREATE TABLE IF NOT EXISTS alert_acks (
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    alert_id INTEGER NOT NULL REFERENCES alerts(id) ON DELETE CASCADE,
    PRIMARY KEY (portfolio_id, alert_id)
);
CREATE TABLE IF NOT EXISTS usage (
    who TEXT NOT NULL, day TEXT NOT NULL, kind TEXT NOT NULL, count INTEGER NOT NULL,
    PRIMARY KEY (who, day, kind)
);
"""
EVERYONE = "*"
_OWN_REPORTS = ("SELECT r.id, r.symbol, r.created_at, r.body FROM reports r"
                " JOIN portfolios p ON p.id = r.portfolio_id WHERE p.name = ?")
# Provisional: a report is about 6 KB (render_report on synthetic data), so a full portfolio keeps about 0.3 MB.
# Raise it if users ask for reports older than the newest 50.
REPORTS_KEPT = 50
# Provisional: only the stop check reads a plan, and only the newest; older ones explain an old stop level. A plan is
# about 5.5 KB (12 saved in two local databases), so a portfolio keeps about 0.3 MB.
PLANS_KEPT = 50


def _refuse_everyone(who: str) -> None:
    if who == EVERYONE:
        raise ValueError(f"User id {who!r} is the all-users usage row; charge or refund a real user id.")


def _serialized(cls):
    """One SQLite connection is shared by the web server's threads; every public
    method runs under one re-entrant lock so calls never interleave on it."""
    for name, fn in list(vars(cls).items()):
        if callable(fn) and not name.startswith("_") and not isinstance(fn, staticmethod):
            def wrap(f):
                @functools.wraps(f)
                def locked(self, *a, **k):
                    with self._lock:
                        return f(self, *a, **k)
                return locked
            setattr(cls, name, wrap(fn))
    return cls


@_serialized
class Store:
    def __init__(self, path: str):
        if sqlite3.sqlite_version_info < (3, 35):
            raise RuntimeError(f"SQLite {sqlite3.sqlite_version} is too old: the usage counters need RETURNING, "
                               "added in SQLite 3.35. Run on a Python or Docker image with a newer SQLite.")
        self.path = path
        self._lock = threading.RLock()
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        if path != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)
        self._add_owner_columns()

    def _add_owner_columns(self) -> None:
        """Older files have reports and plans with no owner. The shared key's one user wrote them, so the default
        portfolio owns them; with no default portfolio they stay ownerless, and nobody can read them."""
        with self.conn:
            self.conn.execute("BEGIN")      # else sqlite3 runs the ALTER on its own, apart from the backfill
            for table in ("reports", "signal_plans"):
                if "portfolio_id" not in {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN portfolio_id INTEGER"
                                      " REFERENCES portfolios(id) ON DELETE CASCADE")
                    self.conn.execute(f"UPDATE {table} SET portfolio_id = (SELECT id FROM portfolios WHERE name = ?)",
                                      (DEFAULT_PORTFOLIO,))

    def _portfolio_id(self, name: str) -> Optional[int]:
        row = self.conn.execute("SELECT id FROM portfolios WHERE name = ?", (name,)).fetchone()
        return None if row is None else int(row[0])

    # --- market data cache -------------------------------------------------
    def save_ohlcv(self, symbol: str, df: pd.DataFrame, source: str) -> int:
        now = utcnow_iso()
        rows = [(symbol, str(ix.date()), float(r.open), float(r.high), float(r.low),
                 float(r.close), float(r.volume), source, now) for ix, r in df.iterrows()]
        with self.conn:
            self.conn.executemany("INSERT OR REPLACE INTO ohlcv VALUES (?,?,?,?,?,?,?,?,?)", rows)
        return len(rows)

    def load_ohlcv(self, symbol: str) -> pd.DataFrame:
        df = pd.read_sql_query("SELECT date, open, high, low, close, volume FROM ohlcv "
                               "WHERE symbol = ? ORDER BY date", self.conn, params=(symbol,),
                               parse_dates=["date"], index_col="date")
        return df

    # --- analyses / reports ------------------------------------------------
    def save_analysis(self, symbol: str, payload: Dict[str, Any]) -> int:
        d = payload["decision"]
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO analyses (symbol, analysis_ts, data_ts, label, score, confidence, payload)"
                " VALUES (?,?,?,?,?,?,?)",
                (symbol, d["analysis_timestamp"], d["data_timestamp"], d["decision_support_label"],
                 d["score"], d["confidence"], json.dumps(payload, default=str)))
        return int(cur.lastrowid)

    def analyses(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT payload FROM analyses WHERE symbol = ? "
                                 "ORDER BY analysis_ts DESC, id DESC LIMIT ?", (symbol, limit))
        return [json.loads(r["payload"]) for r in rows]

    def save_report(self, portfolio: str, symbol: str, body: str) -> int:
        pid = self._portfolio_id(portfolio)
        if pid is None:
            raise ValueError("Set up your portfolio first: each report is kept with it. "
                             "Add your cash or a holding, then ask for the report again.")
        with self.conn:
            cur = self.conn.execute("INSERT INTO reports (portfolio_id, symbol, created_at, body) VALUES (?,?,?,?)",
                                    (pid, symbol, utcnow_iso(), body))
            self.conn.execute("DELETE FROM reports WHERE portfolio_id = ? AND id NOT IN (SELECT id FROM reports"
                              " WHERE portfolio_id = ? ORDER BY id DESC LIMIT ?)", (pid, pid, REPORTS_KEPT))
        return int(cur.lastrowid)

    def reports(self, portfolio: str, symbol: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        q = _OWN_REPORTS
        args: tuple = (portfolio,)
        if symbol:
            q += " AND r.symbol = ?"
            args += (symbol,)
        q += " ORDER BY r.id DESC LIMIT ?"
        return [dict(r) for r in self.conn.execute(q, args + (limit,))]

    def report(self, portfolio: str, report_id: int) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(_OWN_REPORTS + " AND r.id = ?", (portfolio, report_id)).fetchone()
        return None if row is None else dict(row)

    # --- predictions (reproducibility) --------------------------------------
    @staticmethod
    def input_hash(obj: Any) -> str:
        return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def log_prediction(self, model: str, version: str, symbol: str, horizon: int, as_of: str,
                       inputs: Any, output: float) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO predictions (model, model_version, symbol, horizon, as_of, input_hash,"
                " output, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (model, version, symbol, horizon, as_of, self.input_hash(inputs), output, utcnow_iso()))
        return int(cur.lastrowid)

    def predictions(self, symbol: Optional[str] = None, unresolved: bool = False) -> List[Dict[str, Any]]:
        q, args = "SELECT * FROM predictions WHERE 1=1", []
        if symbol:
            q, args = q + " AND symbol = ?", [symbol]
        if unresolved:
            q += " AND realized IS NULL"
        return [dict(r) for r in self.conn.execute(q + " ORDER BY as_of", args)]

    def resolve_prediction(self, pred_id: int, realized: float) -> None:
        with self.conn:
            self.conn.execute("UPDATE predictions SET realized = ?, realized_at = ? WHERE id = ?",
                              (realized, utcnow_iso(), pred_id))

    # --- signal plans --------------------------------------------------------------
    def save_plan(self, portfolio: str, payload: Dict[str, Any]) -> Optional[int]:
        """Not kept for a portfolio that is not set up: only the stop check reads plans, and it needs holdings."""
        pid = self._portfolio_id(portfolio)
        if pid is None:
            return None
        with self.conn:
            cur = self.conn.execute("INSERT INTO signal_plans (portfolio_id, created_at, as_of, strategy, payload)"
                                    " VALUES (?,?,?,?,?)", (pid, utcnow_iso(), payload["as_of"], payload["strategy"],
                                                            json.dumps(payload, default=str)))
            self.conn.execute("DELETE FROM signal_plans WHERE portfolio_id = ? AND id NOT IN (SELECT id FROM signal_plans"
                              " WHERE portfolio_id = ? ORDER BY id DESC LIMIT ?)", (pid, pid, PLANS_KEPT))
        return int(cur.lastrowid)

    def latest_plan(self, portfolio: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT s.payload FROM signal_plans s JOIN portfolios p ON p.id = s.portfolio_id"
                                " WHERE p.name = ? ORDER BY s.id DESC LIMIT 1", (portfolio,)).fetchone()
        return json.loads(row["payload"]) if row else None

    # --- pooled pattern validation --------------------------------------------
    def save_pattern_validation(self, family: str, pattern: str, horizon: int,
                                result: Dict[str, Any], universe: List[str]) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO pattern_validation VALUES (?,?,?,?,?,?)",
                              (pattern, family, horizon, json.dumps(result), ",".join(universe),
                               utcnow_iso()))

    def pattern_priors(self, family: str) -> Dict[str, Dict[str, Any]]:
        rows = self.conn.execute("SELECT pattern, result FROM pattern_validation WHERE family = ?",
                                 (family,))
        return {r["pattern"]: json.loads(r["result"]) for r in rows}

    # --- alerts ------------------------------------------------------------
    def save_alert(self, symbol: str, kind: str, severity: str, message: str,
                   reasons: List[str]) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO alerts (symbol, created_at, kind, severity, message, reasons)"
                " VALUES (?,?,?,?,?,?)",
                (symbol, utcnow_iso(), kind, severity, message, json.dumps(reasons)))
        return int(cur.lastrowid)

    def alerts(self, portfolio: str, symbols: List[str], include_acknowledged: bool = False,
               limit: int = 50) -> List[Dict[str, Any]]:
        """Alerts on `symbols`, each marked acknowledged or not by this portfolio."""
        marks = ",".join("?" * len(symbols))
        # filtered outside: in the inner query `acknowledged` would mean the old column of that name
        q = ("SELECT * FROM (SELECT a.id, a.symbol, a.created_at, a.kind, a.severity, a.message, a.reasons,"
             " EXISTS (SELECT 1 FROM alert_acks k JOIN portfolios p ON p.id = k.portfolio_id"
             f" WHERE k.alert_id = a.id AND p.name = ?) AS acknowledged FROM alerts a WHERE a.symbol IN ({marks}))")
        q += "" if include_acknowledged else " WHERE NOT acknowledged"
        rows = self.conn.execute(q + " ORDER BY id DESC LIMIT ?", (portfolio, *symbols, limit))
        return [{**dict(r), "reasons": json.loads(r["reasons"])} for r in rows]

    def acknowledge_alert(self, portfolio: str, alert_id: int) -> None:
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO alert_acks (portfolio_id, alert_id) SELECT p.id, a.id"
                              " FROM portfolios p JOIN alerts a ON a.id = ? WHERE p.name = ?", (alert_id, portfolio))

    # --- portfolios ----------------------------------------------------------
    def save_portfolio(self, name: str, cash: float, risk_profile: str, horizon: str,
                       transactions: List[Dict[str, Any]]) -> None:
        """Replace a portfolio's settings and transactions in one database transaction."""
        with self.conn:
            self.conn.execute(
                "INSERT INTO portfolios (name, cash, risk_profile, horizon, created_at) VALUES (?,?,?,?,?)"
                " ON CONFLICT(name) DO UPDATE SET cash = excluded.cash,"
                " risk_profile = excluded.risk_profile, horizon = excluded.horizon",
                (name, cash, risk_profile, horizon, utcnow_iso()))
            pid = self.conn.execute("SELECT id FROM portfolios WHERE name = ?", (name,)).fetchone()[0]
            self.conn.execute("DELETE FROM transactions WHERE portfolio_id = ?", (pid,))
            self.conn.executemany(
                "INSERT INTO transactions (portfolio_id, symbol, side, quantity, price, fees,"
                " trade_date, sector) VALUES (?,?,?,?,?,?,?,?)",
                [(pid, t["symbol"], t["side"], t["quantity"], t["price"], t.get("fees", 0.0),
                  t["trade_date"], t.get("sector")) for t in transactions])

    def portfolio(self, name: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM portfolios WHERE name = ?", (name,)).fetchone()
        if row is None:
            return None
        tx = [dict(r) for r in self.conn.execute(
            "SELECT symbol, side, quantity, price, fees, trade_date, sector FROM transactions"
            " WHERE portfolio_id = ? ORDER BY trade_date, id", (row["id"],))]
        return {**dict(row), "transactions": tx}

    def save_fund_transactions(self, name: str, transactions: List[Dict[str, Any]]) -> None:
        """Replace a portfolio's mutual fund transactions in one database transaction."""
        pid = self._portfolio_id(name)
        if pid is None:
            raise ValueError(f"There is no portfolio {name!r} to keep fund transactions in; save the portfolio first.")
        with self.conn:
            self.conn.execute("DELETE FROM fund_transactions WHERE portfolio_id = ?", (pid,))
            self.conn.executemany(
                "INSERT INTO fund_transactions (portfolio_id, scheme_code, side, units, nav, fees, trade_date)"
                " VALUES (?,?,?,?,?,?,?)", [(pid, t["scheme_code"], t["side"], t["units"], t["nav"], t.get("fees", 0.0),
                                             t["trade_date"]) for t in transactions])

    def fund_transactions(self, name: str) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT f.id, f.scheme_code, f.side, f.units, f.nav, f.fees, f.trade_date FROM fund_transactions f"
            " JOIN portfolios p ON p.id = f.portfolio_id WHERE p.name = ? ORDER BY f.trade_date, f.id", (name,))]

    def delete_portfolio(self, name: str) -> None:
        """Its transactions, fund transactions, snapshots, reports, plans and alert acknowledgements go with it (ON DELETE CASCADE)."""
        with self.conn:
            self.conn.execute("DELETE FROM portfolios WHERE name = ?", (name,))

    def rename_portfolio(self, old: str, new: str) -> None:
        if self._portfolio_id(new) is not None:
            raise ValueError(f"Portfolio {new!r} already exists, so {old!r} cannot take its name.")
        with self.conn:
            if self.conn.execute("UPDATE portfolios SET name = ? WHERE name = ?", (new, old)).rowcount == 0:
                raise ValueError(f"There is no portfolio {old!r} to rename.")

    def save_snapshot(self, portfolio: str, value: float, payload: Dict[str, Any]) -> None:
        pid = self._portfolio_id(portfolio)
        with self.conn:
            self.conn.execute("INSERT INTO portfolio_snapshots (portfolio_id, created_at, value, payload)"
                              " VALUES (?,?,?,?)", (pid, utcnow_iso(), value,
                                                    json.dumps(payload, default=str)))

    def snapshots(self, portfolio: str, limit: int = 10) -> List[Dict[str, Any]]:
        pid = self._portfolio_id(portfolio)
        if pid is None:
            return []
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in self.conn.execute(
            "SELECT created_at, value, payload FROM portfolio_snapshots WHERE portfolio_id = ?"
            " ORDER BY id DESC LIMIT ?", (pid, limit))]

    # --- usage counters ------------------------------------------------------
    def spend(self, who: str, kind: str, day: str, limit: int, everyone_limit: int) -> Optional[str]:
        """None when counted, else the cap that stopped it ('you' or 'everyone'), with both counts unchanged."""
        _refuse_everyone(who)
        with self.conn:
            if not self._count_one(who, kind, day, limit):
                return "you"
            if not self._count_one(EVERYONE, kind, day, everyone_limit):
                self.conn.rollback()
                return "everyone"
        return None

    def _count_one(self, who: str, kind: str, day: str, limit: int) -> bool:
        return self.conn.execute(
            "INSERT INTO usage (who, day, kind, count) SELECT ?, ?, ?, 1 WHERE ? > 0"
            " ON CONFLICT (who, day, kind) DO UPDATE SET count = count + 1 WHERE count < ?"
            " RETURNING count", (who, day, kind, limit, limit)).fetchone() is not None

    def refund(self, who: str, kind: str, day: str) -> None:
        _refuse_everyone(who)
        with self.conn:
            if self._lower_one(who, kind, day):
                self._lower_one(EVERYONE, kind, day)

    def _lower_one(self, who: str, kind: str, day: str) -> bool:
        return self.conn.execute("UPDATE usage SET count = count - 1 WHERE who = ? AND day = ? AND kind = ?"
                                 " AND count > 0", (who, day, kind)).rowcount > 0

    def delete_usage(self, who: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM usage WHERE who = ?", (who,))

    def usage(self, who: str, day: str) -> Dict[str, int]:
        return {r["kind"]: r["count"] for r in self.conn.execute(
            "SELECT kind, count FROM usage WHERE who = ? AND day = ?", (who, day))}
