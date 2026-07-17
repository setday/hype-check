"""Point metrics and paired bootstrap CIs for lockbox evaluation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np

from src.metrics import utils


METRIC_FNS = {
    "qini": lambda c, y, t: utils.compute_qini_coefficient(c, y, t, normalized=True),
    "auuc": lambda c, y, t: utils.compute_auuc(c, y, t, normalized=True),
    "uplift@10": lambda c, y, t: utils.compute_uplift_at_k(c, y, t, k=0.1),
    "uplift@30": lambda c, y, t: utils.compute_uplift_at_k(c, y, t, k=0.3),
}


def compute_point_metrics(cate_pred, outcome, treatment) -> Dict[str, float]:
    cate_pred = np.asarray(cate_pred, dtype=np.float64).ravel()
    outcome = np.asarray(outcome, dtype=np.float64).ravel()
    treatment = np.asarray(treatment, dtype=np.float64).ravel()
    return {name: float(fn(cate_pred, outcome, treatment)) for name, fn in METRIC_FNS.items()}


def _paired_bootstrap_indices(outcome, treatment, n_boot: int, seed: int) -> List[np.ndarray]:
    """Resample within treatment and control separately; same indices for all models."""
    outcome = np.asarray(outcome).ravel()
    treatment = np.asarray(treatment).astype(int).ravel()
    rng = np.random.default_rng(seed)
    idx_t = np.flatnonzero(treatment == 1)
    idx_c = np.flatnonzero(treatment == 0)
    if len(idx_t) == 0 or len(idx_c) == 0:
        raise ValueError("Paired bootstrap requires both treated and control units.")

    samples = []
    for _ in range(n_boot):
        bt = rng.choice(idx_t, size=len(idx_t), replace=True)
        bc = rng.choice(idx_c, size=len(idx_c), replace=True)
        samples.append(np.concatenate([bt, bc]))
    return samples


def paired_bootstrap_metrics(
    predictions: Dict[str, np.ndarray],
    outcome,
    treatment,
    *,
    n_boot: int = 200,
    seed: int = 42,
    quantiles=(0.05, 0.95),
    reference_model: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute paired bootstrap CIs for each model and optional deltas vs reference."""
    outcome = np.asarray(outcome, dtype=np.float64).ravel()
    treatment = np.asarray(treatment, dtype=np.float64).ravel()
    boot_idx = _paired_bootstrap_indices(outcome, treatment, n_boot, seed)

    results: Dict[str, Dict[str, float]] = {}
    boot_values: Dict[str, Dict[str, List[float]]] = {m: {k: [] for k in METRIC_FNS} for m in predictions}

    for b_idx in boot_idx:
        y_b = outcome[b_idx]
        t_b = treatment[b_idx]
        for model_name, cate in predictions.items():
            c_b = np.asarray(cate, dtype=np.float64).ravel()[b_idx]
            for metric_name, fn in METRIC_FNS.items():
                boot_values[model_name][metric_name].append(float(fn(c_b, y_b, t_b)))

    q_lo, q_hi = quantiles
    for model_name, metric_dict in boot_values.items():
        row: Dict[str, float] = {}
        for metric_name, values in metric_dict.items():
            arr = np.asarray(values, dtype=np.float64)
            row[metric_name] = float(np.mean(arr))
            row[f"{metric_name}_std"] = float(np.std(arr))
            row[f"{metric_name}_q{int(q_lo*100)}"] = float(np.quantile(arr, q_lo))
            row[f"{metric_name}_q{int(q_hi*100)}"] = float(np.quantile(arr, q_hi))
        results[model_name] = row

    if reference_model is not None:
        if reference_model not in predictions:
            raise KeyError(f"Reference model '{reference_model}' not in predictions.")
        for model_name in predictions:
            if model_name == reference_model:
                continue
            for metric_name in METRIC_FNS:
                delta = np.asarray(boot_values[model_name][metric_name]) - np.asarray(
                    boot_values[reference_model][metric_name]
                )
                results[model_name][f"{metric_name}_delta_vs_{reference_model}"] = float(delta.mean())
                results[model_name][f"{metric_name}_delta_vs_{reference_model}_q{int(q_lo*100)}"] = float(
                    np.quantile(delta, q_lo)
                )
                results[model_name][f"{metric_name}_delta_vs_{reference_model}_q{int(q_hi*100)}"] = float(
                    np.quantile(delta, q_hi)
                )
    return results
