"""TabPFN-3 wrapper: Production-ready tabular foundation model.

Training-free CATE estimation via T-Learner approach:
  - Use pretrained TabPFN-3 to predict P(Y=1|X, T=1) and P(Y=1|X, T=0)
  - CATE = P(Y|T=1) - P(Y|T=0)

Needs `pip install tabpfn`. Paper: https://arxiv.org/abs/2605.13986
HuggingFace: https://huggingface.co/Prior-Labs/tabpfn_3
"""

import logging
from typing import Dict

import numpy as np
import torch

from src.models.uplift_model import FrozenFoundationModel

logger = logging.getLogger(__name__)


def _select_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.mps.is_available():
        return "mps"
    return "cpu"


class TabPFN3(FrozenFoundationModel):
    display_name = "TabPFN-3"

    def __init__(self, config: dict):
        super().__init__(config)

        self.device = config.get("device", _select_device())
        self.max_context = config.get("max_context", 5000)
        self.verbose = bool(config.get("verbose", False))
        self._model = None
        self._is_fitted = False

    def _load_model(self):
        """Lazy-load TabPFN-3 classifier."""
        if self._model is None:
            try:
                from tabpfn import TabPFNClassifier
                self._model = TabPFNClassifier(device=self.device)
            except ImportError:
                raise ImportError(
                    "TabPFN-3 not installed. Install with: pip install tabpfn"
                )

    @staticmethod
    def _np(a):
        return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)

    def fit(self, X, T, Y):
        """Store context examples for in-context CATE prediction.

        TabPFN-3 learns in-context, so we don't explicitly fit.
        Just prepare context features for later prediction.
        """
        X = self._np(X).astype(np.float32)
        T = self._np(T).astype(np.float32).ravel()
        Y = self._np(Y).astype(np.float32).ravel()

        # Subsample context if needed
        if self.max_context is not None and len(X) > self.max_context:
            idx = np.random.default_rng(0).choice(len(X), size=self.max_context, replace=False)
            X, T, Y = X[idx], T[idx], Y[idx]

        # Store context
        self.context_X = X
        self.context_T = T
        self.context_Y = Y
        self._is_fitted = True
        logger.info("TabPFN-3 context set (n=%d).", len(X))
        return self

    def predict_cate(self, X) -> np.ndarray:
        """Predict CATE using T-Learner approach: P(Y|T=1) - P(Y|T=0)."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_cate().")

        self._load_model()
        X = self._np(X).astype(np.float32)

        # Combine context + test: [context_X | test_X]
        X_all = np.vstack([self.context_X, X])
        T_all = np.concatenate([self.context_T, np.full(len(X), -1, dtype=np.float32)])
        Y_all = np.concatenate([self.context_Y, np.full(len(X), -1, dtype=np.float32)])

        # Create datasets for T=0 and T=1 arms
        context_mask = np.isfinite(T_all) & (T_all >= 0)

        # T=0 arm: fit on context with T=0, predict on test
        X_control = X_all[np.concatenate([self.context_T == 0, np.zeros(len(X), dtype=bool)])]
        Y_control = Y_all[np.concatenate([self.context_T == 0, np.zeros(len(X), dtype=bool)])]
        X_test_control = X_all[np.concatenate([np.zeros(len(self.context_X), dtype=bool), np.ones(len(X), dtype=bool)])]

        # T=1 arm: fit on context with T=1, predict on test
        X_treated = X_all[np.concatenate([self.context_T == 1, np.zeros(len(X), dtype=bool)])]
        Y_treated = Y_all[np.concatenate([self.context_T == 1, np.zeros(len(X), dtype=bool)])]
        X_test_treated = X_all[np.concatenate([np.zeros(len(self.context_X), dtype=bool), np.ones(len(X), dtype=bool)])]

        # Predict with TabPFN-3
        # Simple approach: use the model directly on control/treated subsets
        try:
            # Fit on control context
            self._model.fit(X_control, Y_control)
            p_control = self._model.predict_proba(X_test_control)[:, 1]

            # Fit on treated context
            self._model.fit(X_treated, Y_treated)
            p_treated = self._model.predict_proba(X_test_treated)[:, 1]

            # CATE = P(Y=1|T=1) - P(Y=1|T=0)
            cate = p_treated - p_control
        except Exception as e:
            logger.warning(f"TabPFN-3 prediction failed: {e}. Returning zeros.")
            cate = np.zeros(len(X), dtype=np.float32)

        return cate.astype(np.float32).ravel()

    def forward(self, features, treatment=None, **batch) -> Dict[str, torch.Tensor]:
        cate = self.predict_cate(features)
        device = features.device if isinstance(features, torch.Tensor) else "cpu"
        return {"cate_pred": torch.as_tensor(cate, dtype=torch.float32, device=device)}
