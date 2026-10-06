"""Probabilistic forecasting with walk-forward validation.

Targets are P(return over h days > 0) and an h-day range; never a point price.
Walk-forward: expanding training window, retrained every STEP bars, and each
training row must have its label fully realized before the test block starts
(t + h < first test bar) — the purge that stops overlapping labels leaking.
Features at t use only bars <= t. Every candidate must beat the climatology
base rate out of sample; if none does, the engine says so and fusion gives
the forecast zero weight.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from backend.advisory.fusion.evidence import DomainResult, Evidence, Provenance, unavailable
from backend.advisory.statistics import indicators as ind
from backend.advisory.statistics.stats import auc, brier, calibration_table, fit_logistic, log_loss, norm_ppf, predict_logistic

DOMAIN = "forecast"
MIN_TRAIN = 250
STEP = 21
EWMA_LAMBDA = 0.94          # RiskMetrics daily decay; standard, not tuned here
INTERVAL = 0.80
# Skill gate. PROVISIONAL: a Brier skill of +0.5% and AUC 0.52 over >= 200
# out-of-sample predictions is a low bar on purpose; raise it once pooled
# multi-stock evaluation is in place.
MIN_BSS, MIN_AUC, MIN_OOS = 0.005, 0.52, 200


def features(df: pd.DataFrame, bench: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    c = df["close"]
    f = pd.DataFrame(index=df.index)
    f["r1"] = c.pct_change()
    f["r5"] = c.pct_change(5)
    f["r20"] = c.pct_change(20)
    f["r60"] = c.pct_change(60)
    f["rsi"] = ind.rsi(c) / 100.0 - 0.5
    f["dist50"] = c / ind.sma(c, 50) - 1.0
    f["vol20"] = ind.historical_volatility(c, 20)
    f["vol_ratio"] = f["vol20"] / ind.historical_volatility(c, 60)
    f["volume"] = np.log(ind.volume_ratio(df["volume"]).clip(lower=0.05))
    f["macd"] = ind.macd(c)[2] / c
    if bench is not None and len(bench):
        f["bench_r20"] = bench["close"].pct_change(20).reindex(df.index).ffill()
    return f.replace([np.inf, -np.inf], np.nan)


def labels(close: pd.Series, h: int) -> pd.Series:
    fwd = close.shift(-h) / close - 1.0
    return (fwd > 0).astype(float).where(fwd.notna())


class Climatology:
    name = "climatology"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Climatology":
        self.p = float(y.mean())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.p)


class Logistic:
    name = "logistic"

    def __init__(self, l2: float = 5.0):
        self.l2 = l2

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Logistic":
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-12
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        self.w = fit_logistic(Z, y, l2=self.l2)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        return predict_logistic(Z, self.w)


class GradientBoosting:
    """Shallow, heavily regularized trees; the non-linear candidate."""
    name = "gbm"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        from threadpoolctl import threadpool_limits
        if len(np.unique(y)) < 2:
            self.const = float(y.mean())
            return self
        self.const = None
        # On ~1k rows, OpenMP thread start-up costs more than the fit (measured
        # ~4x CPU for no speed-up); one thread is faster.
        with threadpool_limits(limits=1):
            self.model = HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=40,
                l2_regularization=1.0, random_state=0).fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.const is not None:
            return np.full(len(X), self.const)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            return self.model.predict_proba(X)[:, 1]


MODELS = {"climatology": Climatology, "logistic": Logistic, "gbm": GradientBoosting}


def ewma_sigma(close: pd.Series) -> pd.Series:
    lr = np.log(close).diff()
    return np.sqrt((lr ** 2).ewm(alpha=1 - EWMA_LAMBDA, adjust=False).mean())


def walk_forward(df: pd.DataFrame, h: int, models: Sequence[str] = ("climatology", "logistic"),
                 bench: Optional[pd.DataFrame] = None, min_train: int = MIN_TRAIN,
                 step: int = STEP) -> Dict[str, object]:
    F = features(df, bench)
    y = labels(df["close"], h)
    valid = F.notna().all(axis=1).to_numpy()
    X_all, y_all = F.to_numpy(), y.to_numpy()
    n = len(df)
    valid_pos = np.where(valid)[0]
    if len(valid_pos) < min_train + 50:
        return {"error": f"only {len(valid_pos)} usable rows; need {min_train + 50}"}
    preds: Dict[str, List[float]] = {m: [] for m in models}
    test_rows: List[int] = []
    audit: List[Dict[str, int]] = []
    last_labelled = n - 1 - h
    for s in range(valid_pos[0] + min_train, last_labelled + 1, step):
        test = [p for p in range(s, min(s + step, last_labelled + 1)) if valid[p]]
        train = valid_pos[valid_pos <= s - h - 1]
        if not test or len(train) < int(min_train * 0.8):
            continue
        audit.append({"train_last": int(train[-1]), "label_end": int(train[-1] + h),
                      "test_first": int(test[0])})
        for m in models:
            model = MODELS[m]().fit(X_all[train], y_all[train])
            preds[m].extend(model.predict(X_all[test]).tolist())
        test_rows.extend(test)
    if not test_rows:
        return {"error": "walk-forward produced no test predictions"}
    yt = y_all[test_rows]
    metrics: Dict[str, Dict[str, object]] = {}
    base_brier = brier(yt, preds["climatology"]) if "climatology" in preds else None
    vol = X_all[test_rows, list(F.columns).index("vol20")]
    hi_vol = vol > np.median(vol)
    for m, p in preds.items():
        p = np.asarray(p)
        b = brier(yt, p)
        metrics[m] = {"n": len(yt), "brier": round(b, 4), "log_loss": round(log_loss(yt, p), 4),
                      "auc": round(auc(yt, p), 3), "accuracy": round(float(((p > 0.5) == yt).mean()), 3),
                      "brier_skill": None if not base_brier else round(1 - b / base_brier, 4),
                      "calibration": calibration_table(yt, p),
                      "by_vol_regime": {
                          "high_vol": {"n": int(hi_vol.sum()), "auc": round(auc(yt[hi_vol], p[hi_vol]), 3)},
                          "low_vol": {"n": int((~hi_vol).sum()), "auc": round(auc(yt[~hi_vol], p[~hi_vol]), 3)}}}
    # Range calibration: does the EWMA interval cover INTERVAL of outcomes?
    sig = ewma_sigma(df["close"]).to_numpy()
    z = norm_ppf(0.5 + INTERVAL / 2)
    fwd_log = np.log(df["close"].shift(-h) / df["close"]).to_numpy()
    band = z * sig[test_rows] * math.sqrt(h)
    coverage = float((np.abs(fwd_log[test_rows]) <= band).mean())
    return {"horizon": h, "metrics": metrics, "interval_coverage": round(coverage, 3),
            "interval_target": INTERVAL, "test_start": str(df.index[test_rows[0]].date()),
            "test_end": str(df.index[test_rows[-1]].date()), "audit": audit,
            "features": list(F.columns),
            "predictions": {m: pd.Series(p, index=df.index[test_rows]) for m, p in preds.items()}}


def _usable(m: Dict[str, object]) -> bool:
    return (m.get("brier_skill") is not None and m["brier_skill"] > MIN_BSS
            and m["auc"] == m["auc"] and m["auc"] > MIN_AUC and m["n"] >= MIN_OOS)


def forecast_now(df: pd.DataFrame, h: int, model_name: str,
                 bench: Optional[pd.DataFrame] = None) -> float:
    F = features(df, bench)
    y = labels(df["close"], h)
    known = F.notna().all(axis=1) & y.notna()
    if not F.iloc[-1].notna().all():
        raise ValueError("latest bar has incomplete features; cannot forecast")
    model = MODELS[model_name]().fit(F[known].to_numpy(), y[known].to_numpy())
    return float(model.predict(F.iloc[[-1]].to_numpy())[0])


def analyze(df: pd.DataFrame, bench: Optional[pd.DataFrame] = None,
            horizons: Sequence[int] = (5, 20),
            models: Sequence[str] = ("climatology", "logistic")) -> DomainResult:
    if df is None or len(df) < MIN_TRAIN + 150:
        return unavailable(DOMAIN, f"need at least {MIN_TRAIN + 150} bars for a validated forecast")
    as_of = str(df.index[-1].date())
    price = float(df["close"].iloc[-1])
    sigma = float(ewma_sigma(df["close"]).iloc[-1])
    z = norm_ppf(0.5 + INTERVAL / 2)
    per_h: Dict[str, object] = {}
    ev: List[Evidence] = []
    contributions, confidences = [], []
    for h in horizons:
        wf = walk_forward(df, h, models, bench)
        if "error" in wf:
            per_h[f"{h}d"] = {"error": wf["error"]}
            continue
        candidates = {m: v for m, v in wf["metrics"].items() if m != "climatology"}
        best = max(candidates, key=lambda m: candidates[m]["brier_skill"] or -1) if candidates else None
        base_rate = float(labels(df["close"], h).dropna().mean())
        usable = best is not None and _usable(candidates[best])
        band = z * sigma * math.sqrt(h)
        entry = {"base_rate_up": round(base_rate, 3), "oos": wf["metrics"],
                 "oos_period": f"{wf['test_start']} to {wf['test_end']}",
                 "expected_range": {"low": round(price * math.exp(-band), 2),
                                    "high": round(price * math.exp(band), 2),
                                    "probability": INTERVAL,
                                    "oos_coverage": wf["interval_coverage"]},
                 "model": best if usable else "climatology", "skill_demonstrated": usable}
        prov = Provenance(source=f"model:forecast:{entry['model']}:h{h}", as_of=as_of)
        if usable:
            p = forecast_now(df, h, best, bench)
            entry["probability_up"] = round(p, 3)
            skill = candidates[best]["brier_skill"]
            weight = min(1.0, skill / 0.02)
            edge = p - base_rate
            strength = min(0.7, abs(edge) * 3) * weight
            contributions.append(np.sign(edge) * strength)
            confidences.append(weight)
            ev.append(Evidence(domain=DOMAIN, direction=int(np.sign(edge)), strength=round(strength, 3),
                               claim=f"{best} model: P(up over {h}d) = {p:.2f} vs base rate "
                                     f"{base_rate:.2f}; out-of-sample Brier skill "
                                     f"{skill:+.3f}, AUC {candidates[best]['auc']:.2f}.",
                               value=p, is_model_output=True, provenance=prov))
        else:
            entry["probability_up"] = round(base_rate, 3)
            why = ("no candidate model" if best is None else
                   f"best candidate ({best}) Brier skill {candidates[best]['brier_skill']:+.3f}, "
                   f"AUC {candidates[best]['auc']:.2f}")
            ev.append(Evidence(domain=DOMAIN, direction=0, strength=0.0,
                               claim=f"No model beat the base rate out of sample for {h}d ({why}); "
                                     f"reporting the historical base rate {base_rate:.2f} instead.",
                               value=base_rate, is_model_output=True, provenance=prov))
        ev.append(Evidence(domain=DOMAIN, direction=0, strength=0.0,
                           claim=f"{int(INTERVAL * 100)}% expected {h}d range "
                                 f"{entry['expected_range']['low']:,.2f}–{entry['expected_range']['high']:,.2f} "
                                 f"(EWMA volatility; historical coverage {wf['interval_coverage']:.0%}).",
                           value=band, is_model_output=True, provenance=prov))
        per_h[f"{h}d"] = entry
    if not per_h or all("error" in v for v in per_h.values()):
        return unavailable(DOMAIN, "walk-forward evaluation failed for every horizon")
    score = float(np.mean(contributions)) if contributions else 0.0
    confidence = float(np.mean(confidences)) if confidences else 0.0
    state = ("no demonstrated edge" if not contributions else
             "mildly positive" if 0 < score < 0.3 else "positive" if score >= 0.3 else
             "mildly negative" if -0.3 < score < 0 else "negative" if score <= -0.3 else "neutral")
    return DomainResult(domain=DOMAIN, state=state, score=score, evidence=ev,
                        details={"horizons": per_h, "ewma_daily_vol_pct": round(sigma * 100, 3)},
                        as_of=as_of, confidence=round(confidence, 2))
