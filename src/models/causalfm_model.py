"""CausalFM wrapper: Foundation Models for Causal Inference.

Prior function network (PFN) approach for heterogeneous treatment effect (HTE)
estimation. Learns CATE in-context without fine-tuning.

Official repo: https://github.com/yccm/CausalFM-toolkit (polished variant)
Paper: "Causal Foundation Models for Heterogeneous Treatment Effect Estimation"
"""

import logging
import os
import sys
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


class CausalFM(FrozenFoundationModel):
    display_name = "CausalFM"

    def __init__(self, config: dict):
        super().__init__(config)

        self.device = config.get("device", _select_device())
        self.max_context = config.get("max_context", 5000)
        self.treatment_balanced = bool(config.get("treatment_balanced", False))
        self.verbose = bool(config.get("verbose", False))
        self._model = None
        self._is_fitted = False

        # Add external/CausalFM to path for imports
        causalfm_path = os.path.join(os.path.dirname(__file__), "../../external/CausalFM")
        if os.path.exists(causalfm_path) and causalfm_path not in sys.path:
            sys.path.insert(0, causalfm_path)

    def _load_model(self):
        """Lazy-load CausalFM model."""
        if self._model is None:
            try:
                # Import from CausalFM-toolkit submodule
                from causalfm import CausalFMEstimator
                self._model = CausalFMEstimator(device=self.device)
                if self.verbose:
                    logger.info("CausalFM model loaded successfully.")
            except ImportError as e:
                raise ImportError(
                    f"CausalFM not properly installed. Ensure external/CausalFM is initialized. Error: {e}"
                )

    @staticmethod
    def _np(a):
        return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)

    def fit(self, X, T, Y):
        """Store context examples for in-context CATE learning.

        Optionally stratifies on treatment to ensure balanced context.
        """
        X = self._np(X).astype(np.float32)
        T = self._np(T).astype(np.float32).ravel()
        Y = self._np(Y).astype(np.float32).ravel()

        # Treatment-balanced context sampling
        if self.treatment_balanced and self.max_context is not None:
            n_per_arm = self.max_context // 2
            idx_control = np.where(T == 0)[0]
            idx_treated = np.where(T == 1)[0]

            # Stratified sampling
            rng = np.random.default_rng(0)
            idx_control_sample = rng.choice(
                idx_control, size=min(n_per_arm, len(idx_control)), replace=False
            )
            idx_treated_sample = rng.choice(
                idx_treated, size=min(n_per_arm, len(idx_treated)), replace=False
            )
            idx = np.concatenate([idx_control_sample, idx_treated_sample])
            X, T, Y = X[idx], T[idx], Y[idx]
        elif self.max_context is not None and len(X) > self.max_context:
            # Simple random subsampling
            idx = np.random.default_rng(0).choice(len(X), size=self.max_context, replace=False)
            X, T, Y = X[idx], T[idx], Y[idx]

        # Store context
        self.context_X = X
        self.context_T = T
        self.context_Y = Y
        self._is_fitted = True
        logger.info("CausalFM context set (n=%d, treatment_balanced=%s).", len(X), self.treatment_balanced)
        return self

    def predict_cate(self, X) -> np.ndarray:
        """Predict CATE using CausalFM's in-context learning.

        Passes context examples + test features to CausalFM,
        which learns point estimates of heterogeneous treatment effects.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_cate().")

        self._load_model()
        X = self._np(X).astype(np.float32)

        try:
            # Prepare context data
            context_data = {
                "X": self.context_X,
                "T": self.context_T,
                "Y": self.context_Y,
            }

            # Forward pass through CausalFM
            # Returns CATE predictions for test set
            with torch.no_grad():
                cate = self._model.estimate_cate(
                    X=X,
                    context_data=context_data,
                    device=self.device,
                )
            cate = self._np(cate).astype(np.float32).ravel()
        except Exception as e:
            logger.warning(f"CausalFM prediction failed: {e}. Returning zeros.")
            cate = np.zeros(len(X), dtype=np.float32)

        return cate

    def forward(self, features, treatment=None, **batch) -> Dict[str, torch.Tensor]:
        cate = self.predict_cate(features)
        device = features.device if isinstance(features, torch.Tensor) else "cpu"
        return {"cate_pred": torch.as_tensor(cate, dtype=torch.float32, device=device)}
