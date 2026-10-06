"""Small numpy-only statistics toolkit (no scipy dependency)."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, |err| < 1.2e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf needs 0 < p < 1, got {p}")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    lo, hi = 0.02425, 1 - 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > hi:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def two_sided_p(z: float) -> float:
    """Normal-approximation p-value; adequate for the n >= 20 samples we require."""
    return 2.0 * (1.0 - norm_cdf(abs(z)))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney rank statistic (ties get average rank)."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 5) -> list:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.sum():
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(mask.sum()),
                         "mean_predicted": round(float(p[mask].mean()), 3),
                         "observed_rate": round(float(y[mask].mean()), 3)})
    return rows


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0,
                 iters: int = 50) -> np.ndarray:
    """L2-regularized logistic regression by Newton-Raphson (IRLS).

    X must already include a bias column; the bias is not penalized.
    """
    n, k = X.shape
    w = np.zeros(k)
    penalty = np.full(k, l2)
    penalty[0] = 0.0
    for _ in range(iters):
        z = np.clip(X @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = X.T @ (p - y) + penalty * w
        hess = (X * (p * (1 - p))[:, None]).T @ X + np.diag(penalty) + 1e-9 * np.eye(k)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def predict_logistic(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))


def variance_ratio(log_ret: np.ndarray, q: int) -> Tuple[float, float]:
    """Lo-MacKinlay variance ratio VR(q) with heteroskedasticity-robust z.

    VR > 1: returns trend (positive autocorrelation); VR < 1: mean reversion.
    """
    r = np.asarray(log_ret, float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < q * 4:
        return float("nan"), float("nan")
    mu = r.mean()
    dev = r - mu
    var1 = np.sum(dev ** 2) / (n - 1)
    rq = np.convolve(r, np.ones(q), mode="valid")
    m = q * (n - q + 1) * (1 - q / n)
    varq = np.sum((rq - q * mu) ** 2) / m
    vr = varq / var1
    # Heteroskedasticity-consistent asymptotic variance (Lo & MacKinlay 1988).
    denom = np.sum(dev ** 2) ** 2
    theta = 0.0
    for j in range(1, q):
        delta = np.sum(dev[j:] ** 2 * dev[:-j] ** 2) / denom
        theta += (2 * (q - j) / q) ** 2 * delta
    z = (vr - 1) / math.sqrt(theta) if theta > 0 else float("nan")
    return float(vr), float(z)


def summarize(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
            "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "hit_rate": float((x > 0).mean()),
            "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90))}
