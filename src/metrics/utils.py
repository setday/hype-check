"""Numpy uplift ranking metrics (higher cate_pred = more responsive)."""

from typing import Callable

import numpy as np


def sort_by_cate_pred(cate_pred: np.ndarray, outcome: np.ndarray, treatment: np.ndarray):
    sort_idx = np.argsort(cate_pred, kind="mergesort")[::-1]  # Descending order of CATE

    sorted_outcome = outcome[sort_idx]
    sorted_treatment = treatment[sort_idx]

    return sorted_outcome, sorted_treatment


def compute_qini_curve(cate_pred: np.ndarray, outcome: np.ndarray, treatment: np.ndarray, normalized: bool = True):
    """Return (x, q): targeted fraction vs cumulative incremental responders."""
    sorted_outcome, sorted_treatment = sort_by_cate_pred(cate_pred, outcome, treatment)
    
    # Cumulative response for treatment and control groups
    cumsum_response_t = np.cumsum(sorted_outcome * sorted_treatment)
    cumsum_response_c = np.cumsum(sorted_outcome * (1 - sorted_treatment))
    cumsum_count_t = np.cumsum(sorted_treatment)
    cumsum_count_c = np.cumsum(1 - sorted_treatment)

    # Masking extreme values
    mask_cumsum_count_c = cumsum_count_c == 0
    cumsum_count_c[mask_cumsum_count_c] = 1

    # Qini curve = difference in response rates
    ratios = cumsum_count_t / cumsum_count_c
    incremental_uplift = cumsum_response_t - cumsum_response_c * ratios

    if normalized:
        qini_curve = incremental_uplift / cumsum_count_t[-1]
    else:
        qini_curve = incremental_uplift

    qini_curve[mask_cumsum_count_c] = cumsum_response_t[mask_cumsum_count_c]
    qini_curve = np.concatenate([[0.0], qini_curve])

    return qini_curve


def compute_qini_coefficient(cate_pred: np.ndarray, outcome: np.ndarray, treatment: np.ndarray, normalized: bool = True) -> float:
    qini_curve = compute_qini_curve(cate_pred, outcome, treatment, normalized=normalized)

    # Percentage of population (x-axis)
    percent_pop = np.linspace(0.0, 1.0, len(qini_curve))

    # Qini = area under Qini curve (trapezoid rule)
    qini = float(np.trapezoid(qini_curve, percent_pop) - 0.5 * qini_curve[-1])

    return qini


def compute_auuc(cate_pred: np.ndarray, outcome: np.ndarray, treatment: np.ndarray, normalized: bool = True) -> float:
    sorted_outcome, sorted_treatment = sort_by_cate_pred(cate_pred, outcome, treatment)
    
    # Cumulative response for treatment and control groups
    cumsum_response_t = np.cumsum(sorted_outcome * sorted_treatment)
    cumsum_response_c = np.cumsum(sorted_outcome * (1 - sorted_treatment))
    cumsum_count_t = np.cumsum(sorted_treatment)
    cumsum_count_c = np.cumsum(1 - sorted_treatment)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        response_t = np.where(cumsum_count_t > 0, cumsum_response_t / cumsum_count_t, 0.0)
        response_c = np.where(cumsum_count_c > 0, cumsum_response_c / cumsum_count_c, 0.0)
    
    n = len(cumsum_response_c)
    idx = np.arange(1, n + 1)
    area = float(np.trapezoid((response_t - response_c) * idx, idx / n))
    return area / n if normalized else area


def compute_uplift_at_k(cate_pred: np.ndarray, outcome: np.ndarray, treatment: np.ndarray, k: float = 0.3) -> float:
    sorted_outcome, sorted_treatment = sort_by_cate_pred(cate_pred, outcome, treatment)
    
    n = len(sorted_outcome)
    k_idx = max(1, int(round(n * k)))

    sorted_outcome = sorted_outcome[:k_idx]
    sorted_treatment = sorted_treatment[:k_idx]
    
    # Cumulative response for treatment and control groups
    sum_response_t = np.sum(sorted_outcome * sorted_treatment)
    sum_response_c = np.sum(sorted_outcome * (1 - sorted_treatment))
    sum_count_t = np.sum(sorted_treatment)
    sum_count_c = np.sum(1 - sorted_treatment)

    response_t_k = sum_response_t / sum_count_t if sum_count_t > 0 else 0.0
    response_c_k = sum_response_c / sum_count_c if sum_count_c > 0 else 0.0

    return float(response_t_k - response_c_k)


def compute_pehe(cate_pred: np.ndarray, cate_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((cate_pred - cate_true) ** 2)))
