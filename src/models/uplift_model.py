"""EconML meta-learner baselines (S/T/X/DR on LightGBM) + frozen-model base.

Common interface: fit(X, T, Y) -> predict_cate(X); forward() for Lightning.
Outcomes are binary, so base outcome models are regressors on the 0/1 label and
propensity models are classifiers.
"""

import logging
from typing import Dict

import numpy as np
import torch

from src.models.abstract_model import AbstractModel

logger = logging.getLogger(__name__)


def _lgbm_regressor(**params):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(**{"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.05,
                            "n_jobs": -1, "verbose": -1, **params})


def _lgbm_classifier(**params):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**{"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.05,
                             "n_jobs": -1, "verbose": -1, **params})


class UpliftModel(AbstractModel):
    def __init__(self, config: dict):
        super().__init__(config)

    def forward(self, features, treatment=None, **batch) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def calculate_loss(self, batch) -> torch.Tensor:
        raise NotImplementedError("Meta-learners are trained offline; use fit().")


class FrozenFoundationModel(UpliftModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self._frozen = True

    def calculate_loss(self, batch) -> torch.Tensor:
        raise RuntimeError(f"{self.__class__.__name__} is frozen and does not support training.")

    @property
    def is_frozen(self) -> bool:
        return self._frozen


class _EconMLWrapper(UpliftModel):
    display_name = "EconML"

    def __init__(self, config: dict):
        super().__init__(config)
        self.params = dict(config.get("params", {}))
        _rs = self.params.pop("random_state", None)
        self.seed = int(config.get("seed", _rs if _rs is not None else 42))
        self.est = None
        self._is_fitted = False

    def _build(self):
        raise NotImplementedError

    @staticmethod
    def _np(a):
        return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)

    def fit(self, X, T, Y):
        X, T, Y = self._np(X), self._np(T).ravel(), self._np(Y).ravel()
        self.est = self._build()
        self.est.fit(Y, T, X=X)
        self._is_fitted = True
        logger.info("%s fitted (n=%d, d=%d).", self.display_name, X.shape[0], X.shape[1])
        return self

    def predict_cate(self, X) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(f"{self.display_name} must be fitted before prediction.")
        return np.asarray(self.est.effect(self._np(X))).ravel()

    def forward(self, features, treatment=None, **batch) -> Dict[str, torch.Tensor]:
        cate = self.predict_cate(features)
        device = features.device if isinstance(features, torch.Tensor) else "cpu"
        return {"cate_pred": torch.as_tensor(cate, dtype=torch.float32, device=device)}


class SLearnerWrapper(_EconMLWrapper):
    display_name = "S-Learner"

    def _build(self):
        from econml.metalearners import SLearner
        return SLearner(overall_model=_lgbm_regressor(random_state=self.seed, **self.params))


class TLearnerWrapper(_EconMLWrapper):
    display_name = "T-Learner"

    def _build(self):
        from econml.metalearners import TLearner
        return TLearner(models=_lgbm_regressor(random_state=self.seed, **self.params))


class XLearnerWrapper(_EconMLWrapper):
    display_name = "X-Learner"

    def _build(self):
        from econml.metalearners import XLearner
        return XLearner(
            models=_lgbm_regressor(random_state=self.seed, **self.params),
            propensity_model=_lgbm_classifier(random_state=self.seed),
            cate_models=_lgbm_regressor(random_state=self.seed, **self.params),
        )


class DRLearnerWrapper(_EconMLWrapper):
    display_name = "DR-Learner"

    def _build(self):
        from econml.dr import DRLearner
        return DRLearner(
            model_propensity=_lgbm_classifier(random_state=self.seed),
            model_regression=_lgbm_regressor(random_state=self.seed, **self.params),
            model_final=_lgbm_regressor(random_state=self.seed, **self.params),
            cv=int(self.config.get("cv", 3)),
            random_state=self.seed,
        )
