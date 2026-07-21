"""Do-PFN wrapper: Prior Function Network with causal interventions.

Models interventional distributions P(Y|do(T=1)) and P(Y|do(T=0)) separately,
enabling robust CATE estimation from context examples without fine-tuning.

Official repo: https://github.com/jr2021/Do-PFN
Paper: "In-Context Learning for Causal Effect Estimation" (arXiv 2506.06039)
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


class DoPFN(FrozenFoundationModel):
    display_name = "Do-PFN"

    def __init__(self, config: dict = None, **kwargs):
        # Handle both Hydra config dict and direct kwargs
        if config is None:
            config = {}
        if isinstance(config, dict):
            config = {**config, **kwargs}  # Merge kwargs into config
        super().__init__(config)

        self.device = config.get("device", _select_device())
        self.max_context_treated = config.get("max_context_treated", 2500)
        self.max_context_control = config.get("max_context_control", 2500)
        self.verbose = bool(config.get("verbose", False))
        self._model = None
        self._is_fitted = False

        # Add external/Do-PFN to path for imports
        dopfn_path = os.path.join(os.path.dirname(__file__), "../../external/Do-PFN")
        if os.path.exists(dopfn_path) and dopfn_path not in sys.path:
            sys.path.insert(0, dopfn_path)

    def _load_model(self):
        """Lazy-load Do-PFN model."""
        if self._model is None:
            try:
                # Import from Do-PFN submodule
                from dopfn import DoPFN as DoPFN_Model
                self._model = DoPFN_Model(device=self.device)
                if self.verbose:
                    logger.info("Do-PFN model loaded successfully.")
            except ImportError as e:
                raise ImportError(
                    f"Do-PFN not properly installed. Ensure external/Do-PFN is initialized. Error: {e}"
                )

    @staticmethod
    def _np(a):
        return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)

    def fit(self, X, T, Y):
        """Partition context into treated (T=1) and control (T=0) arms.

        Do-PFN models interventional distributions separately, so we store
        examples for each arm independently.
        """
        X = self._np(X).astype(np.float32)
        T = self._np(T).astype(np.float32).ravel()
        Y = self._np(Y).astype(np.float32).ravel()

        # Partition by treatment arm
        mask_treated = T == 1
        mask_control = T == 0

        X_treated, T_treated, Y_treated = X[mask_treated], T[mask_treated], Y[mask_treated]
        X_control, T_control, Y_control = X[mask_control], T[mask_control], Y[mask_control]

        # Subsample each arm if needed
        rng = np.random.default_rng(0)
        if len(X_treated) > self.max_context_treated:
            idx = rng.choice(len(X_treated), size=self.max_context_treated, replace=False)
            X_treated, T_treated, Y_treated = X_treated[idx], T_treated[idx], Y_treated[idx]

        if len(X_control) > self.max_context_control:
            idx = rng.choice(len(X_control), size=self.max_context_control, replace=False)
            X_control, T_control, Y_control = X_control[idx], T_control[idx], Y_control[idx]

        # Store context for each arm
        self.context_X_treated = X_treated
        self.context_Y_treated = Y_treated
        self.context_X_control = X_control
        self.context_Y_control = Y_control
        self._is_fitted = True

        logger.info(
            "Do-PFN context set (n_treated=%d, n_control=%d).",
            len(X_treated), len(X_control)
        )
        return self

    def predict_cate(self, X) -> np.ndarray:
        """Predict CATE as E[Y|do(T=1), X] - E[Y|do(T=0), X].

        Uses separate interventional models for treated and control arms.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_cate().")

        self._load_model()
        X = self._np(X).astype(np.float32)

        try:
            # Predict outcomes under intervention T=1
            context_treated = {
                "X": self.context_X_treated,
                "Y": self.context_Y_treated,
            }
            with torch.no_grad():
                y_treated = self._model.predict_intervention(
                    X=X,
                    context_data=context_treated,
                    intervention_value=1.0,
                    device=self.device,
                )

            # Predict outcomes under intervention T=0
            context_control = {
                "X": self.context_X_control,
                "Y": self.context_Y_control,
            }
            with torch.no_grad():
                y_control = self._model.predict_intervention(
                    X=X,
                    context_data=context_control,
                    intervention_value=0.0,
                    device=self.device,
                )

            # CATE = E[Y|do(T=1)] - E[Y|do(T=0)]
            cate = self._np(y_treated) - self._np(y_control)
            cate = cate.astype(np.float32).ravel()
        except Exception as e:
            logger.warning(f"Do-PFN prediction failed: {e}. Returning zeros.")
            cate = np.zeros(len(X), dtype=np.float32)

        return cate

    def forward(self, features, treatment=None, **batch) -> Dict[str, torch.Tensor]:
        cate = self.predict_cate(features)
        device = features.device if isinstance(features, torch.Tensor) else "cpu"
        return {"cate_pred": torch.as_tensor(cate, dtype=torch.float32, device=device)}
