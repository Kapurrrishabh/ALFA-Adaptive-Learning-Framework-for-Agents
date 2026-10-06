"""Model registry with a controlled promotion lifecycle.

    candidate -> (beats deployed out of sample) -> approved (named human) -> deployed

A model can only be deployed after it was approved, and only approved after
its out-of-sample metric beat the currently deployed version. There is no
automatic path: this is the hook a future self-learning agent would use, and
it keeps that agent from changing production behaviour on its own.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from backend.advisory.fusion.evidence import utcnow_iso
from backend.database.storage import Store


class PromotionError(Exception):
    pass


class ModelRegistry:
    def __init__(self, store: Store):
        self.conn = store.conn

    def register(self, name: str, version: str, *, training_period: str, features: list,
                 hyperparameters: Dict[str, Any], metrics: Dict[str, Any],
                 backtest: Optional[Dict[str, Any]], limitations: str) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO model_versions (name, version, status, training_period, features,"
                " hyperparameters, metrics, backtest, limitations, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (name, version, "candidate", training_period, json.dumps(features),
                 json.dumps(hyperparameters), json.dumps(metrics), json.dumps(backtest),
                 limitations, utcnow_iso()))
        return int(cur.lastrowid)

    def get(self, name: str, version: str) -> Dict[str, Any]:
        row = self.conn.execute("SELECT * FROM model_versions WHERE name = ? AND version = ?",
                                (name, version)).fetchone()
        if row is None:
            raise KeyError(f"model {name} version {version} is not registered")
        out = dict(row)
        for k in ("features", "hyperparameters", "metrics", "backtest"):
            out[k] = json.loads(out[k]) if out[k] else None
        return out

    def deployed(self, name: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT version FROM model_versions WHERE name = ? AND status = "
                                "'deployed'", (name,)).fetchone()
        return self.get(name, row[0]) if row else None

    def check_promotion(self, name: str, version: str, metric: str = "brier_skill",
                        higher_is_better: bool = True, floor: Optional[float] = 0.0) -> Tuple[bool, str]:
        """`floor` is the baseline the metric must beat regardless of the incumbent
        (Brier skill 0 = no better than the historical base rate)."""
        cand = self.get(name, version)
        value = (cand["metrics"] or {}).get(metric)
        if value is None:
            return False, f"candidate has no out-of-sample '{metric}' metric"
        if floor is not None and (value <= floor if higher_is_better else value >= floor):
            return False, f"candidate {metric}={value} does not beat the baseline ({floor})"
        current = self.deployed(name)
        if current is None:
            return True, f"no deployed version; candidate {metric}={value}"
        incumbent = (current["metrics"] or {}).get(metric)
        if incumbent is None:
            return True, f"deployed version lacks '{metric}'; candidate {metric}={value}"
        better = value > incumbent if higher_is_better else value < incumbent
        verdict = "beats" if better else "does not beat"
        return better, (f"candidate {metric}={value} {verdict} deployed "
                        f"{current['version']} ({incumbent})")

    def approve(self, name: str, version: str, approved_by: str, **check_kwargs) -> str:
        if not approved_by.strip():
            raise PromotionError("approval needs a named human approver")
        ok, reason = self.check_promotion(name, version, **check_kwargs)
        if not ok:
            raise PromotionError(f"cannot approve {name} {version}: {reason}")
        self._set_status(name, version, "approved", approved_by=approved_by)
        return reason

    def deploy(self, name: str, version: str) -> None:
        if self.get(name, version)["status"] != "approved":
            raise PromotionError(f"{name} {version} must be approved before deployment")
        with self.conn:
            self.conn.execute("UPDATE model_versions SET status = 'retired' WHERE name = ? AND "
                              "status = 'deployed'", (name,))
            self.conn.execute("UPDATE model_versions SET status = 'deployed' WHERE name = ? AND "
                              "version = ?", (name, version))

    def reject(self, name: str, version: str) -> None:
        self._set_status(name, version, "rejected")

    def _set_status(self, name: str, version: str, status: str,
                    approved_by: Optional[str] = None) -> None:
        with self.conn:
            self.conn.execute("UPDATE model_versions SET status = ?, approved_by = COALESCE(?, "
                              "approved_by) WHERE name = ? AND version = ?",
                              (status, approved_by, name, version))
