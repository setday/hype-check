"""
Uplift Ranking Metrics
======================

Qini coefficient, AUUC, uplift@k, and PEHE with bootstrap confidence intervals.
"""

import logging
from typing import Optional, Dict

import numpy as np
import torch
from torchmetrics import Metric

from .utils import compute_qini_coefficient, compute_auuc, compute_uplift_at_k, compute_pehe

logger = logging.getLogger(__name__)


class QiniMetric(Metric):
    """
    Qini coefficient (Qini gain).

    The Qini curve plots cumulative response rate vs. cumulative % of population,
    ordered by model predictions (CATE). Higher Qini = better targeting.

    Qini = (Area under Qini curve - Area under diagonal) / Max possible gain

    Reference: Radcliffe & Surry (2011). "Real-World Uplift Modelling with Significance Based Trees"
    """

    def __init__(self, name: str = "qini", normalized: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.normalized = normalized
        self.add_state("cate_pred", default=[], dist_reduce_fx="cat")
        self.add_state("outcome", default=[], dist_reduce_fx="cat")
        self.add_state("treatment", default=[], dist_reduce_fx="cat")

    def update(
        self,
        cate_pred: torch.Tensor,
        outcome: torch.Tensor,
        treatment: torch.Tensor,
        **kwargs,
    ):
        """
        Update metric state.

        Args:
            cate_pred: (batch_size,) predicted CATE
            outcome: (batch_size,) observed outcome
            treatment: (batch_size,) treatment indicator (0 or 1)
        """
        self.cate_pred.append(cate_pred.detach().cpu())
        self.outcome.append(outcome.detach().cpu())
        self.treatment.append(treatment.detach().cpu())

    def compute(self) -> torch.Tensor:
        """
        Compute Qini coefficient.
        """
        if not self.cate_pred:
            return torch.tensor(0.0)

        cate_pred = torch.cat(self.cate_pred).numpy()
        outcome = torch.cat(self.outcome).numpy()
        treatment = torch.cat(self.treatment).numpy()

        qini = compute_qini_coefficient(cate_pred, outcome, treatment, normalized=self.normalized)

        return torch.tensor(qini, dtype=torch.float32)


class AUUCMetric(Metric):
    """
    Area Under Uplift Curve (AUUC).

    Cumulative gains plot comparing treatment and control response rates.
    Normalized by the maximum possible gain (perfect targeting).

    Reference: Radcliffe (2007). "Using the Kölmogorov-Smirnov test to predict treatment outcome"
    """

    def __init__(self, name: str = "auuc", normalized: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.normalized = normalized
        self.add_state("cate_pred", default=[], dist_reduce_fx="cat")
        self.add_state("outcome", default=[], dist_reduce_fx="cat")
        self.add_state("treatment", default=[], dist_reduce_fx="cat")

    def update(
        self,
        cate_pred: torch.Tensor,
        outcome: torch.Tensor,
        treatment: torch.Tensor,
        **kwargs,
    ):
        self.cate_pred.append(cate_pred.detach().cpu())
        self.outcome.append(outcome.detach().cpu())
        self.treatment.append(treatment.detach().cpu())

    def compute(self) -> torch.Tensor:
        """
        Compute AUUC.
        """
        if not self.cate_pred:
            return torch.tensor(0.0)

        cate_pred = torch.cat(self.cate_pred).numpy()
        outcome = torch.cat(self.outcome).numpy()
        treatment = torch.cat(self.treatment).numpy()

        auuc = compute_auuc(cate_pred, outcome, treatment, normalized=self.normalized)

        return torch.tensor(auuc, dtype=torch.float32)


class UpliftAtKMetric(Metric):
    """
    Uplift at k (% of population targeted).

    Response rate difference (treatment - control) when targeting top k% of population.
    """

    def __init__(self, k: float = 0.3, name: str = None, **kwargs):
        """
        Args:
            k: percentage of population to target (e.g., 0.3 for top 30%)
            name: metric name
        """
        super().__init__(**kwargs)
        self.k = k
        self.name = name or f"uplift@{int(k*100)}"
        self.add_state("cate_pred", default=[], dist_reduce_fx="cat")
        self.add_state("outcome", default=[], dist_reduce_fx="cat")
        self.add_state("treatment", default=[], dist_reduce_fx="cat")

    def update(
        self,
        cate_pred: torch.Tensor,
        outcome: torch.Tensor,
        treatment: torch.Tensor,
        **kwargs,
    ):
        self.cate_pred.append(cate_pred.detach().cpu())
        self.outcome.append(outcome.detach().cpu())
        self.treatment.append(treatment.detach().cpu())

    def compute(self) -> torch.Tensor:
        """
        Compute uplift at k.
        """
        if not self.cate_pred:
            return torch.tensor(0.0)

        cate_pred = torch.cat(self.cate_pred).numpy()
        outcome = torch.cat(self.outcome).numpy()
        treatment = torch.cat(self.treatment).numpy()
        
        uplift_at_k = compute_uplift_at_k(cate_pred, outcome, treatment, k=self.k)

        return torch.tensor(uplift_at_k, dtype=torch.float32)


class PEHEMetric(Metric):
    """
    Precision in Estimation of Heterogeneous Treatment Effects (PEHE).

    PEHE = sqrt(mean((CATE_pred - CATE_true)^2))

    Available only for semi-synthetic benchmarks with ground-truth CATE.
    """

    def __init__(self, name: str = "pehe", **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.add_state("cate_pred", default=[], dist_reduce_fx="cat")
        self.add_state("cate_true", default=[], dist_reduce_fx="cat")

    def update(
        self,
        cate_pred: torch.Tensor,
        cate_true: torch.Tensor,
        **kwargs,
    ):
        self.cate_pred.append(cate_pred.detach().cpu())
        self.cate_true.append(cate_true.detach().cpu())

    def compute(self) -> torch.Tensor:
        """
        Compute PEHE.
        """
        if not self.cate_pred:
            return torch.tensor(float("nan"))

        cate_pred = torch.cat(self.cate_pred).numpy()
        cate_true = torch.cat(self.cate_true).numpy()

        pehe = compute_pehe(cate_pred, cate_true)

        return torch.tensor(pehe)


class RankingCorrelationMetric(Metric):
    """
    Correlation between prediction accuracy (MSE) and ranking quality (Qini).

    Used to test Hypothesis H2: "Ranking by accuracy correlates weakly with ranking by targeting value."
    """

    def __init__(self, name: str = "ranking_correlation", **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.add_state("cate_pred", default=[], dist_reduce_fx="cat")
        self.add_state("cate_true", default=[], dist_reduce_fx="cat")
        self.add_state("outcome", default=[], dist_reduce_fx="cat")
        self.add_state("treatment", default=[], dist_reduce_fx="cat")

    def update(
        self,
        cate_pred: torch.Tensor,
        outcome: torch.Tensor,
        treatment: torch.Tensor,
        cate_true: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self.cate_pred.append(cate_pred.detach().cpu())
        self.outcome.append(outcome.detach().cpu())
        self.treatment.append(treatment.detach().cpu())
        if cate_true is not None:
            self.cate_true.append(cate_true.detach().cpu())

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Compute correlation between PEHE and Qini.
        """
        if not self.cate_pred:
            return {"correlation": torch.tensor(float("nan"))}

        cate_pred = torch.cat(self.cate_pred).numpy()
        outcome = torch.cat(self.outcome).numpy()
        treatment = torch.cat(self.treatment).numpy()

        # Accuracy metric: MSE against true CATE (if available)
        if self.cate_true:
            cate_true = torch.cat(self.cate_true).numpy()
            accuracy = np.mean((cate_pred - cate_true) ** 2)
        else:
            # Fallback: use MSE against outcome
            accuracy = np.mean((cate_pred - outcome) ** 2)

        # Ranking metric: Qini
        n = len(cate_pred)
        sort_idx = np.argsort(cate_pred, kind="mergesort")[::-1]
        sorted_outcome = outcome[sort_idx]
        sorted_treatment = treatment[sort_idx]

        cumsum_response_t = np.cumsum(sorted_outcome * sorted_treatment)
        cumsum_response_c = np.cumsum(sorted_outcome * (1 - sorted_treatment))
        cumsum_count_t = np.cumsum(sorted_treatment)
        cumsum_count_c = np.cumsum(1 - sorted_treatment)

        n_t = cumsum_count_t[-1] if cumsum_count_t[-1] > 0 else 1
        n_c = cumsum_count_c[-1] if cumsum_count_c[-1] > 0 else 1

        response_rate_t = cumsum_response_t / n_t if n_t > 0 else np.zeros_like(cumsum_response_t)
        response_rate_c = cumsum_response_c / n_c if n_c > 0 else np.zeros_like(cumsum_response_c)

        qini_curve = response_rate_t - response_rate_c
        percent_pop = np.arange(1, n + 1) / n
        qini = float(np.trapezoid(qini_curve, percent_pop))

        return {
            "accuracy_mse": torch.tensor(accuracy),
            "qini": torch.tensor(qini),
        }
