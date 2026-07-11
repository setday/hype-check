"""Untuned CausalPFN wrapper (training-free in-context CATE).

fit(X, T, Y) sets the in-context sample; predict_cate(X) returns CATE.
Needs `pip install causalpfn`. Paper: https://arxiv.org/abs/2506.07918
"""

import logging
import os

# faiss + torch can load OpenMP twice on macOS/arm64 and segfault; set before import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from typing import Dict

import numpy as np
import torch

from src.models.uplift_model import FrozenFoundationModel

logger = logging.getLogger(__name__)


def causalpfn_available() -> bool:
    try:
        import causalpfn  # noqa: F401
        return True
    except Exception:
        return False


class CausalPFNModel(FrozenFoundationModel):
    display_name = "CausalPFN"

    def __init__(self, config: dict):
        super().__init__(config)
        self.device = config.get("device", None)
        self.max_context = config.get("max_context", 5000)
        self.verbose = bool(config.get("verbose", False))
        self._est = None
        self._is_fitted = False

    def _make_estimator(self):
        from causalpfn import CATEEstimator
        device = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(device) == "cpu":
            torch.set_num_threads(1)
        return CATEEstimator(device=device, verbose=self.verbose)

    @staticmethod
    def _np(a):
        return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)

    def fit(self, X, T, Y):
        X = self._np(X).astype(np.float32)
        T = self._np(T).astype(np.float32).ravel()
        Y = self._np(Y).astype(np.float32).ravel()

        if self.max_context is not None and len(X) > self.max_context:
            idx = np.random.default_rng(0).choice(len(X), size=self.max_context, replace=False)
            X, T, Y = X[idx], T[idx], Y[idx]

        self._est = self._make_estimator()
        self._est.fit(X, T, Y)
        self._is_fitted = True
        logger.info("CausalPFN context set (n=%d).", len(X))
        return self

    def predict_cate(self, X) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_cate().")
        return np.asarray(self._est.estimate_cate(self._np(X).astype(np.float32))).ravel()

    def forward(self, features, treatment=None, **batch) -> Dict[str, torch.Tensor]:
        cate = self.predict_cate(features)
        device = features.device if isinstance(features, torch.Tensor) else "cpu"
        return {"cate_pred": torch.as_tensor(cate, dtype=torch.float32, device=device)}
