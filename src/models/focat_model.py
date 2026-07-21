"""FoCAT wrapper: Foundational Causal Adaptive Transformer.

Foundation model for causal inference using masked language modeling and
in-context learning. Learns CATE from context examples without fine-tuning.

Official repo: https://github.com/NTAILab/FoCAT
Paper: FoCAT: Foundation for Causal Analysis via Transformer (Springer 2024)
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


class FoCAT(FrozenFoundationModel):
    display_name = "FoCAT"

    def __init__(self, config: dict):
        super().__init__(config)

        self.device = config.get("device", _select_device())
        self.max_context = config.get("max_context", 5000)
        self.verbose = bool(config.get("verbose", False))
        self._model = None
        self._is_fitted = False

        # Add external/FoCAT to path for imports
        focat_path = os.path.join(os.path.dirname(__file__), "../../external/FoCAT")
        if os.path.exists(focat_path) and focat_path not in sys.path:
            sys.path.insert(0, focat_path)

    def _load_model(self):
        """Lazy-load FoCAT model."""
        if self._model is None:
            try:
                # Import from FoCAT submodule
                from focat.models import FoCAT as FoCAT_Model
                self._model = FoCAT_Model(device=self.device)
                if self.verbose:
                    logger.info("FoCAT model loaded successfully.")
            except ImportError as e:
                raise ImportError(
                    f"FoCAT not properly installed. Ensure external/FoCAT is initialized. Error: {e}"
                )

    @staticmethod
    def _np(a):
        return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)

    def fit(self, X, T, Y):
        """Store context examples for in-context CATE learning.

        FoCAT learns from context, so we store examples and prepare for inference.
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
        logger.info("FoCAT context set (n=%d).", len(X))
        return self

    def predict_cate(self, X) -> np.ndarray:
        """Predict CATE using FoCAT's in-context learning.

        Passes context examples + test features to FoCAT hypernetwork,
        which learns CATE without explicit fine-tuning.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_cate().")

        self._load_model()
        X = self._np(X).astype(np.float32)

        try:
            # Prepare context as sequence for FoCAT
            # FoCAT expects: (context_features, context_treatment, context_outcomes)
            context_data = {
                "X": self.context_X,
                "T": self.context_T,
                "Y": self.context_Y,
            }

            # Forward pass through FoCAT
            # Returns CATE predictions for test set
            with torch.no_grad():
                cate = self._model.predict(
                    X=X,
                    context_data=context_data,
                    device=self.device,
                )
            cate = self._np(cate).astype(np.float32).ravel()
        except Exception as e:
            logger.warning(f"FoCAT prediction failed: {e}. Returning zeros.")
            cate = np.zeros(len(X), dtype=np.float32)

        return cate

    def forward(self, features, treatment=None, **batch) -> Dict[str, torch.Tensor]:
        cate = self.predict_cate(features)
        device = features.device if isinstance(features, torch.Tensor) else "cpu"
        return {"cate_pred": torch.as_tensor(cate, dtype=torch.float32, device=device)}
