"""Numpy uplift ranking metrics (higher cate_pred = more responsive)."""

from typing import Callable

import numpy as np


def _sorted(cate_pred, outcome, treatment):
    cate_pred = np.asarray(cate_pred, dtype=np.float64).ravel()
    outcome = np.asarray(outcome, dtype=np.float64).ravel()
    treatment = np.asarray(treatment, dtype=np.float64).ravel()
    order = np.argsort(-cate_pred, kind="mergesort")
    return outcome[order], treatment[order]


def qini_curve(cate_pred, outcome, treatment):
    """Return (x, q): targeted fraction vs cumulative incremental responders."""
    y, t = _sorted(cate_pred, outcome, treatment)
    n = len(y)
    yt, yc = np.cumsum(y * t), np.cumsum(y * (1 - t))
    nt, nc = np.cumsum(t), np.cumsum(1 - t)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(nc > 0, nt / nc, 0.0)
    q = yt - yc * ratio
    x = np.arange(1, n + 1) / n
    return np.concatenate([[0.0], x]), np.concatenate([[0.0], q])


def qini_coefficient(cate_pred, outcome, treatment, normalize=True) -> float:
    x, q = qini_curve(cate_pred, outcome, treatment)
    coef = float(np.trapz(q, x) - 0.5 * q[-1])
    return coef / len(outcome) if normalize else coef


def auuc(cate_pred, outcome, treatment, normalize=True) -> float:
    y, t = _sorted(cate_pred, outcome, treatment)
    n = len(y)
    yt, yc = np.cumsum(y * t), np.cumsum(y * (1 - t))
    nt, nc = np.cumsum(t), np.cumsum(1 - t)
    with np.errstate(divide="ignore", invalid="ignore"):
        rt = np.where(nt > 0, yt / nt, 0.0)
        rc = np.where(nc > 0, yc / nc, 0.0)
    idx = np.arange(1, n + 1)
    area = float(np.trapz((rt - rc) * idx, idx / n))
    return area / n if normalize else area


def uplift_at_k(cate_pred, outcome, treatment, k=0.3) -> float:
    y, t = _sorted(cate_pred, outcome, treatment)
    k_idx = max(1, int(round(len(y) * k)))
    yk, tk = y[:k_idx], t[:k_idx]
    nt, nc = tk.sum(), (1 - tk).sum()
    rt = (yk * tk).sum() / nt if nt > 0 else 0.0
    rc = (yk * (1 - tk)).sum() / nc if nc > 0 else 0.0
    return float(rt - rc)


def pehe(cate_pred, cate_true) -> float:
    cate_pred = np.asarray(cate_pred, dtype=np.float64).ravel()
    cate_true = np.asarray(cate_true, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((cate_pred - cate_true) ** 2)))


def bootstrap_metric(metric_fn: Callable, cate_pred, outcome, treatment,
                     n_boot=200, seed=0, ci=0.95, **metric_kwargs) -> dict:
    cate_pred = np.asarray(cate_pred).ravel()
    outcome = np.asarray(outcome).ravel()
    treatment = np.asarray(treatment).ravel()
    n = len(cate_pred)
    rng = np.random.default_rng(seed)
    vals = np.array([
        metric_fn(cate_pred[i], outcome[i], treatment[i], **metric_kwargs)
        for i in (rng.integers(0, n, size=n) for _ in range(n_boot))
    ])
    return {"mean": float(vals.mean()), "std": float(vals.std()),
            "lo": float(np.quantile(vals, (1 - ci) / 2)),
            "hi": float(np.quantile(vals, 1 - (1 - ci) / 2))}
