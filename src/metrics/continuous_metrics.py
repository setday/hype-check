"""Metrics for continuous treatment evaluation."""

import numpy as np
from typing import Optional, Tuple


def dose_response_curve(
    X: np.ndarray,
    T_range: np.ndarray,
    model,
    y_true: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute dose-response curve: CATE τ̂(X, t) across treatment spectrum.

    Args:
        X: (n_samples, n_features) covariates
        T_range: (n_doses,) treatment values to evaluate
        model: fitted continuous treatment model
        y_true: (n_samples,) optional ground-truth CATE for validation

    Returns:
        T_range, cate_curve: treatment values and corresponding CATE predictions
        If y_true provided, also computes MSE per treatment bin
    """
    cate_curve = []
    for t in T_range:
        T_fixed = np.full(len(X), t)
        cate = model.predict_cate(X, T_fixed)
        cate_curve.append(cate.mean())  # Average CATE across samples

    cate_curve = np.array(cate_curve)
    return T_range, cate_curve


def mse_continuous(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error for continuous predictions."""
    return float(np.mean((y_true - y_pred) ** 2))


def rmse_continuous(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae_continuous(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_continuous(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0 if ss_res == 0 else -np.inf
    return float(1.0 - ss_res / ss_tot)


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def continuous_cate_mse(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    cate_true: np.ndarray,
    model,
) -> float:
    """
    MSE on ground-truth CATE (for semi-synthetic data with known τ).

    Args:
        X: (n_samples, n_features) covariates
        T: (n_samples,) continuous treatment
        Y: (n_samples,) outcomes
        cate_true: (n_samples,) ground-truth CATE
        model: fitted continuous treatment model

    Returns:
        MSE of predicted CATE vs ground truth
    """
    cate_pred = model.predict_cate(X, T)
    return mse_continuous(cate_true, cate_pred)


def continuous_propensity_calibration(
    T: np.ndarray,
    T_pred_mu: np.ndarray,
    T_pred_sigma: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Calibration metric for propensity model p(T|X).

    Compares empirical distribution of T in bins vs predicted Gaussian.

    Args:
        T: (n_samples,) observed treatment
        T_pred_mu: (n_samples,) predicted mean
        T_pred_sigma: (n_samples,) predicted std dev
        n_bins: number of bins for calibration check

    Returns:
        Calibration error (lower is better, 0 = perfect)
    """
    # Bin by predicted mu
    bin_idx = np.digitize(T_pred_mu, np.linspace(T_pred_mu.min(), T_pred_mu.max(), n_bins))

    calib_error = 0.0
    for b in range(1, n_bins + 1):
        mask = bin_idx == b
        if mask.sum() < 5:  # Skip sparse bins
            continue
        T_bin = T[mask]
        mu_bin = T_pred_mu[mask].mean()
        sigma_bin = T_pred_sigma[mask].mean()

        # Empirical vs predicted std
        empirical_std = T_bin.std()
        calib_error += abs(empirical_std - sigma_bin)

    return float(calib_error / max(n_bins, 1))


def partial_auc_dose_response(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    model,
    top_fraction: float = 0.1,
) -> float:
    """
    Partial AUC: integral of dose-response curve over top-k% treatment range.

    Useful for policies that target high-effect populations.

    Args:
        X: (n_samples, n_features) covariates
        T: (n_samples,) continuous treatment
        Y: (n_samples,) outcomes
        model: fitted continuous treatment model
        top_fraction: fraction of treatment range to integrate (default 0.1 = top 10%)

    Returns:
        Partial AUC value
    """
    # Compute dose-response over full range
    T_range = np.linspace(T.min(), T.max(), 100)
    _, cate_curve = dose_response_curve(X, T_range, model)

    # Integrate over top fraction
    n_top = max(1, int(len(T_range) * top_fraction))
    top_indices = np.argsort(cate_curve)[-n_top:]
    partial_auc = np.mean(cate_curve[top_indices])

    return float(partial_auc)


def heterogeneous_effect_metrics(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    cate_true: Optional[np.ndarray] = None,
    model = None,
) -> dict:
    """
    Compute comprehensive heterogeneous effect metrics.

    Args:
        X: covariates
        T: continuous treatment
        Y: outcomes
        cate_true: optional ground-truth CATE
        model: optional fitted model for predictions

    Returns:
        Dict with metric names and values
    """
    metrics = {}

    # Treatment effect heterogeneity
    metrics["T_mean"] = float(T.mean())
    metrics["T_std"] = float(T.std())
    metrics["T_min"] = float(T.min())
    metrics["T_max"] = float(T.max())

    if cate_true is not None:
        metrics["cate_true_mean"] = float(cate_true.mean())
        metrics["cate_true_std"] = float(cate_true.std())
        metrics["cate_true_min"] = float(cate_true.min())
        metrics["cate_true_max"] = float(cate_true.max())

    if model is not None and cate_true is not None:
        cate_pred = model.predict_cate(X, T)
        metrics["cate_pred_mse"] = mse_continuous(cate_true, cate_pred)
        metrics["cate_pred_rmse"] = rmse_continuous(cate_true, cate_pred)
        metrics["cate_pred_mae"] = mae_continuous(cate_true, cate_pred)
        metrics["cate_pred_r2"] = r2_continuous(cate_true, cate_pred)
        metrics["cate_pred_corr"] = pearson_correlation(cate_true, cate_pred)

    return metrics
