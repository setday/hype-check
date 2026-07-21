"""CausalPFN wrapper for continuous treatment.

Uses bin-based discretization strategy:
- Continuous T is discretized into K bins (deciles by default)
- CausalPFN is trained on discretized T
- Predictions are interpolated between bin centers for continuous doses
"""

import logging
import os
from typing import Dict, Optional

import numpy as np
import torch

from src.models.uplift_model import FrozenFoundationModel

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

logger = logging.getLogger(__name__)


def _select_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.mps.is_available():
        return "mps"
    return "cpu"


class CausalPFNContinuous(FrozenFoundationModel):
    """CausalPFN wrapper supporting continuous treatments via discretization."""

    display_name = "CausalPFNContinuous"

    def __init__(self, config: dict):
        super().__init__(config)

        self.device = config.get("device", _select_device())
        self.max_context = config.get("max_context", 5000)
        self.verbose = bool(config.get("verbose", False))
        self.n_bins = int(config.get("n_bins", 10))  # Discretization bins
        self._est = None
        self._is_fitted = False
        self._bin_edges = None
        self._bin_centers = None

    def _make_estimator(self):
        from causalpfn import CATEEstimator
        if self.device == "cpu":
            torch.set_num_threads(1)
        return CATEEstimator(device=self.device, verbose=self.verbose)

    @staticmethod
    def _np(a):
        return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)

    def fit(self, X, T, Y):
        """
        Fit CausalPFN on discretized continuous treatment.

        Args:
            X: (n_samples, n_features) covariates
            T: (n_samples,) continuous treatment
            Y: (n_samples,) outcomes
        """
        X = self._np(X).astype(np.float32)
        T = self._np(T).astype(np.float32).ravel()
        Y = self._np(Y).astype(np.float32).ravel()

        # Compute bin edges and centers
        self._bin_edges = np.linspace(T.min(), T.max(), self.n_bins + 1)
        self._bin_centers = (self._bin_edges[:-1] + self._bin_edges[1:]) / 2

        # Discretize treatment
        T_binned = np.digitize(T, self._bin_edges) - 1
        T_binned = np.clip(T_binned, 0, self.n_bins - 1).astype(np.float32)

        # Subsample if needed
        if self.max_context is not None and len(X) > self.max_context:
            idx = np.random.default_rng(0).choice(len(X), size=self.max_context, replace=False)
            X, T_binned, Y = X[idx], T_binned[idx], Y[idx]

        # Fit on discretized treatment
        self._est = self._make_estimator()
        self._est.fit(X, T_binned, Y)
        self._is_fitted = True
        logger.info("CausalPFNContinuous fitted (n=%d, n_bins=%d, T range=[%.2f, %.2f]).",
                    len(X), self.n_bins, T.min(), T.max())
        return self

    def predict_cate(self, X, T: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict continuous treatment effect.

        Args:
            X: (n_samples, n_features) covariates
            T: (n_samples,) optional continuous treatment. If None, use bin centers.

        Returns:
            (n_samples,) CATE predictions
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_cate().")

        X = self._np(X).astype(np.float32)

        if T is None:
            # Default: predict at median bin (0.5 normalized)
            T_norm = np.full(len(X), 0.5, dtype=np.float32)
        else:
            # Normalize T to [0, 1] range
            T = self._np(T).astype(np.float32).ravel()
            T_norm = (T - self._bin_edges[0]) / (self._bin_edges[-1] - self._bin_edges[0])
            T_norm = np.clip(T_norm, 0.0, 1.0)

        # Map normalized T to bin indices
        bin_idx = (T_norm * (self.n_bins - 1)).astype(int)
        bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)

        # Get predictions from CausalPFN for each bin index
        T_binned = bin_idx.astype(np.float32)
        cate_binned = np.asarray(self._est.estimate_cate(X, T_binned)).ravel()

        return cate_binned

    def forward(self, features, treatment=None, **batch) -> Dict[str, torch.Tensor]:
        cate = self.predict_cate(features, treatment)
        device = features.device if isinstance(features, torch.Tensor) else "cpu"
        return {"cate_pred": torch.as_tensor(cate, dtype=torch.float32, device=device)}
