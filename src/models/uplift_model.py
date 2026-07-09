"""
Uplift Model Base Classes
==========================

UpliftModel: extends AbstractModel for treatment effect estimation.
FrozenFoundationModel: wraps inference-only models like CausalPFN.
"""

import logging
from typing import Dict, Any

import torch
import torch.nn as nn

from src.models.abstract_model import AbstractModel

logger = logging.getLogger(__name__)


class UpliftModel(AbstractModel):
    """
    Base class for uplift/CATE estimation models.

    Extends AbstractModel to handle treatment as an input feature.
    Subclasses implement calculate_loss() for training (if applicable)
    and forward() to return CATE predictions.
    """

    def __init__(self, config: dict):
        super().__init__(config)

    def forward(self, features: torch.Tensor, treatment: torch.Tensor = None, **batch) -> Dict[str, torch.Tensor]:
        """
        Forward pass for CATE prediction.

        Args:
            features: (batch_size, n_features) input features
            treatment: (batch_size,) treatment indicator (0 or 1), optional for inference
            **batch: other fields (outcome, cate_true, etc.)

        Returns:
            dict with key "cate_pred": (batch_size,) predicted CATE
        """
        raise NotImplementedError("forward must return {'cate_pred': tensor}")

    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate training loss.

        For supervised methods, typically MSE(cate_pred, outcome_diff).
        Override in subclasses; default raises NotImplementedError.

        Args:
            batch: dict with features, treatment, outcome, ...

        Returns:
            scalar loss tensor
        """
        raise NotImplementedError("calculate_loss must be implemented in subclass.")


class FrozenFoundationModel(UpliftModel):
    """
    Wrapper for inference-only foundation models (e.g., CausalPFN).

    These models are pretrained and not fine-tuned. They require
    no training_step; only validation/inference.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._frozen = True

    def forward(self, features: torch.Tensor, treatment: torch.Tensor = None, **batch) -> Dict[str, torch.Tensor]:
        """
        Override in subclass to call foundation model inference.
        """
        raise NotImplementedError("Subclass must implement foundation model call.")

    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Frozen models do not support training.
        """
        raise RuntimeError(f"{self.__class__.__name__} is a frozen model and does not support training.")

    @property
    def is_frozen(self) -> bool:
        return self._frozen


class SLearnerWrapper(UpliftModel):
    """
    S-Learner (Single model): trains one model on all data with treatment as a feature.

    CATE = f(X, T=1) - f(X, T=0)

    Uses a wrapped sklearn regressor (e.g., XGBRegressor).
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.base_model = None
        self._is_fitted = False

    def fit(self, X: torch.Tensor, T: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Fit the S-Learner on (X, T, Y) data.

        Args:
            X: (n, n_features) feature matrix
            T: (n,) treatment indicator
            Y: (n,) outcome
        """
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler

        X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        T_np = T.cpu().numpy() if isinstance(T, torch.Tensor) else T
        Y_np = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else Y

        # Stack features with treatment
        X_with_t = np.hstack([X_np, T_np.reshape(-1, 1)])

        # Fit regressor
        self.base_model = GradientBoostingRegressor(
            **self.config.get("params", {})
        )
        self.base_model.fit(X_with_t, Y_np)
        self._is_fitted = True
        logger.info("S-Learner fitted on training data.")

    def forward(self, features: torch.Tensor, treatment: torch.Tensor = None, **batch) -> Dict[str, torch.Tensor]:
        """
        Predict CATE: f(X, T=1) - f(X, T=0).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

        X_np = features.cpu().numpy() if isinstance(features, torch.Tensor) else features

        # Predict for T=1
        X_t1 = np.hstack([X_np, np.ones((len(X_np), 1))])
        pred_t1 = self.base_model.predict(X_t1)

        # Predict for T=0
        X_t0 = np.hstack([X_np, np.zeros((len(X_np), 1))])
        pred_t0 = self.base_model.predict(X_t0)

        # CATE = pred_t1 - pred_t0
        cate_pred = torch.from_numpy(pred_t1 - pred_t0).float()

        return {"cate_pred": cate_pred}

    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        S-Learner training happens offline. Return dummy loss if called.
        """
        raise NotImplementedError("S-Learner is trained offline. Use fit() method.")


class TLearnerWrapper(UpliftModel):
    """
    T-Learner (Two models): train separate models on treatment and control groups.

    CATE = f_1(X) - f_0(X)

    Each model sees only its group's data.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.model_1 = None
        self.model_0 = None
        self._is_fitted = False

    def fit(self, X: torch.Tensor, T: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Fit T-Learner on (X, T, Y) data.
        """
        from sklearn.ensemble import GradientBoostingRegressor

        X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        T_np = T.cpu().numpy() if isinstance(T, torch.Tensor) else T
        Y_np = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else Y

        # Fit model on treatment group
        mask_1 = T_np == 1
        if mask_1.sum() > 0:
            self.model_1 = GradientBoostingRegressor(
                **self.config.get("params", {})
            )
            self.model_1.fit(X_np[mask_1], Y_np[mask_1])

        # Fit model on control group
        mask_0 = T_np == 0
        if mask_0.sum() > 0:
            self.model_0 = GradientBoostingRegressor(
                **self.config.get("params", {})
            )
            self.model_0.fit(X_np[mask_0], Y_np[mask_0])

        self._is_fitted = True
        logger.info("T-Learner fitted on treatment and control groups.")

    def forward(self, features: torch.Tensor, treatment: torch.Tensor = None, **batch) -> Dict[str, torch.Tensor]:
        """
        Predict CATE: f_1(X) - f_0(X).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        X_np = features.cpu().numpy() if isinstance(features, torch.Tensor) else features

        pred_1 = self.model_1.predict(X_np) if self.model_1 else np.zeros(len(X_np))
        pred_0 = self.model_0.predict(X_np) if self.model_0 else np.zeros(len(X_np))

        cate_pred = torch.from_numpy(pred_1 - pred_0).float()

        return {"cate_pred": cate_pred}

    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("T-Learner is trained offline. Use fit() method.")


# Import numpy for sklearn wrappers
import numpy as np
