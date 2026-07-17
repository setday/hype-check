"""Numpy uplift ranking metrics (higher cate_pred = more responsive)."""

import numpy as np


def sort_by_cate_pred(cate_pred: np.ndarray, outcome: np.ndarray, treatment: np.ndarray):
    sort_idx = np.argsort(cate_pred, kind="mergesort")[::-1]  # Descending order of CATE

    sorted_outcome = outcome[sort_idx]
    sorted_treatment = treatment[sort_idx]

    return sorted_outcome, sorted_treatment


def compute_qini_curve(cate_pred: np.ndarray, outcome: np.ndarray, treatment: np.ndarray):
    """Return (x, q): targeted fraction vs cumulative incremental responders."""
    sorted_outcome, sorted_treatment = sort_by_cate_pred(cate_pred, outcome, treatment)
    
    # Cumulative response for treatment and control groups
    cumsum_response_t = np.cumsum(sorted_outcome * sorted_treatment)
    cumsum_response_c = np.cumsum(sorted_outcome * (1 - sorted_treatment))
    cumsum_count_t = np.cumsum(sorted_treatment)
    cumsum_count_c = np.cumsum(1 - sorted_treatment).astype(np.float32)

    # Masking extreme values
    cumsum_count_c[cumsum_count_c == 0] = float('inf')

    # Qini curve = difference in response rates
    ratios = cumsum_count_t / cumsum_count_c
    incremental_uplift = cumsum_response_t - cumsum_response_c * ratios

    qini_curve = np.concatenate([[0.0], incremental_uplift])

    return qini_curve


def compute_uplift_curve(cate_pred: np.ndarray, outcome: np.ndarray, treatment: np.ndarray):
    """Return (x, q): targeted fraction vs cumulative incremental responders."""
    sorted_outcome, sorted_treatment = sort_by_cate_pred(cate_pred, outcome, treatment)
    
    # Cumulative response for treatment and control groups
    cumsum_response_t = np.cumsum(sorted_outcome * sorted_treatment)
    cumsum_response_c = np.cumsum(sorted_outcome * (1 - sorted_treatment))
    cumsum_count_t = np.cumsum(sorted_treatment).astype(np.float32)
    cumsum_count_c = np.cumsum(1 - sorted_treatment).astype(np.float32)

    # Masking extreme values
    cumsum_count_c[cumsum_count_c == 0] = float('inf')
    cumsum_count_t[cumsum_count_t == 0] = float('inf')

    # Uplift curve = difference in response rates
    incremental_uplift = (cumsum_response_t / cumsum_count_t - cumsum_response_c / cumsum_count_c) * np.arange(1, len(sorted_outcome) + 1)

    uplift_curve = np.concatenate([[0.0], incremental_uplift])

    return uplift_curve


def compute_qini_coefficient(cate_pred: np.ndarray, outcome: np.ndarray, treatment: np.ndarray, normalized: bool = True) -> float:
    qini_curve = compute_qini_curve(cate_pred, outcome, treatment)

    # Percentage of population (x-axis)
    percent_pop = np.linspace(0.0, 1.0, len(qini_curve))

    # Qini = area under Qini curve (trapezoid rule)
    qini = float(np.trapezoid(qini_curve, percent_pop) - 0.5 * qini_curve[-1])

    if normalized:
        ideal_qini_curve = compute_qini_curve(outcome * treatment - outcome * (1 - treatment), outcome, treatment)
        qini = qini / float(np.trapezoid(ideal_qini_curve, percent_pop) - 0.5 * qini_curve[-1])

    return qini


def compute_auuc(cate_pred: np.ndarray, outcome: np.ndarray, treatment: np.ndarray, normalized: bool = True) -> float:
    uplift_curve = compute_uplift_curve(cate_pred, outcome, treatment)

    # Percentage of population (x-axis)
    percent_pop = np.linspace(0.0, 1.0, len(uplift_curve))

    # AUUC = area under Uplift curve (trapezoid rule)
    auuc = float(np.trapezoid(uplift_curve, percent_pop) - 0.5 * uplift_curve[-1])

    if normalized:
        cr_num = np.sum((outcome == 1) & (treatment == 0))
        tn_num = np.sum((outcome == 0) & (treatment == 1))

        summand = outcome if cr_num > tn_num else treatment
        perfect_uplift = 2 * (outcome == treatment) + summand
        
        ideal_uplift_curve = compute_uplift_curve(perfect_uplift, outcome, treatment)
        auuc = auuc / float(np.trapezoid(ideal_uplift_curve, percent_pop) - 0.5 * uplift_curve[-1])

    return auuc


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
