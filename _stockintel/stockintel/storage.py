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

from .evidence import utcnow_iso

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
    acknowledged INTEGER NOT NULL DEFAULT 0
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
"""


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

    def save_report(self, symbol: str, body: str) -> int:
        with self.conn:
            cur = self.conn.execute("INSERT INTO reports (symbol, created_at, body) VALUES (?,?,?)",
                                    (symbol, utcnow_iso(), body))
        return int(cur.lastrowid)

    def reports(self, symbol: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        q = "SELECT id, symbol, created_at, body FROM reports"
        args: tuple = ()
        if symbol:
            q += " WHERE symbol = ?"
            args = (symbol,)
        q += " ORDER BY id DESC LIMIT ?"
        return [dict(r) for r in self.conn.execute(q, args + (limit,))]

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
    def save_plan(self, payload: Dict[str, Any]) -> int:
        with self.conn:
            cur = self.conn.execute("INSERT INTO signal_plans (created_at, as_of, strategy, payload)"
                                    " VALUES (?,?,?,?)", (utcnow_iso(), payload["as_of"], payload["strategy"],
                                                          json.dumps(payload, default=str)))
        return int(cur.lastrowid)

    def latest_plan(self) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT payload FROM signal_plans ORDER BY id DESC LIMIT 1").fetchone()
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

    def alerts(self, include_acknowledged: bool = False, limit: int = 50) -> List[Dict[str, Any]]:
        q = "SELECT * FROM alerts" + ("" if include_acknowledged else " WHERE acknowledged = 0")
        rows = self.conn.execute(q + " ORDER BY id DESC LIMIT ?", (limit,))
        return [{**dict(r), "reasons": json.loads(r["reasons"])} for r in rows]

    def acknowledge_alert(self, alert_id: int) -> None:
        with self.conn:
            self.conn.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))

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

    def save_snapshot(self, portfolio: str, value: float, payload: Dict[str, Any]) -> None:
        row = self.conn.execute("SELECT id FROM portfolios WHERE name = ?", (portfolio,)).fetchone()
        with self.conn:
            self.conn.execute("INSERT INTO portfolio_snapshots (portfolio_id, created_at, value, payload)"
                              " VALUES (?,?,?,?)", (row[0], utcnow_iso(), value,
                                                    json.dumps(payload, default=str)))

    def snapshots(self, portfolio: str, limit: int = 10) -> List[Dict[str, Any]]:
        row = self.conn.execute("SELECT id FROM portfolios WHERE name = ?", (portfolio,)).fetchone()
        if row is None:
            return []
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in self.conn.execute(
            "SELECT created_at, value, payload FROM portfolio_snapshots WHERE portfolio_id = ?"
            " ORDER BY id DESC LIMIT ?", (row[0], limit))]
