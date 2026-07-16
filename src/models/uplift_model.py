"""Self-implemented meta-learners (S/T/X/DR on LightGBM) + frozen-model base.

No EconML dependency — meta-learners are implemented directly with LightGBM.
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


class _BaseLearner(UpliftModel):
    """Shared plumbing for self-implemented meta-learners.

    Handles seed extraction, tensor→numpy conversion, and the Lightning
    forward() wrapper so each subclass only needs fit() + predict_cate().
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.params = dict(config.get("params", {}))
        _rs = self.params.pop("random_state", None)
        self.seed = int(config.get("seed", _rs if _rs is not None else 42))

    @staticmethod
    def _np(a):
        return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)

    def fit(self, X, T, Y):
        raise NotImplementedError

    def predict_cate(self, X) -> np.ndarray:
        raise NotImplementedError

    def forward(self, features, treatment=None, **batch) -> Dict[str, torch.Tensor]:
        cate = self.predict_cate(features)
        device = features.device if isinstance(features, torch.Tensor) else "cpu"
        return {"cate_pred": torch.as_tensor(cate, dtype=torch.float32, device=device)}


class SLearner(_BaseLearner):
    """S-Learner: one model on [X, T]; CATE = predict(T=1) - predict(T=0)."""
    display_name = "S-Learner"

    def __init__(self, config: dict):
        super().__init__(config)
        self.model = None

    def fit(self, X, T, Y):
        X, T, Y = self._np(X), self._np(T).ravel(), self._np(Y).ravel()
        X_with_T = np.column_stack([X, T])
        self.model = _lgbm_regressor(random_state=self.seed, **self.params)
        self.model.fit(X_with_T, Y)
        return self

    def predict_cate(self, X) -> np.ndarray:
        X = self._np(X)
        X1 = np.column_stack([X, np.ones(len(X))])
        X0 = np.column_stack([X, np.zeros(len(X))])
        return self.model.predict(X1) - self.model.predict(X0)


class TLearner(_BaseLearner):
    """T-Learner: separate models for treated and control; CATE = f₁(X) − f₀(X)."""
    display_name = "T-Learner"

    def __init__(self, config: dict):
        super().__init__(config)
        self.model1 = None
        self.model0 = None

    def fit(self, X, T, Y):
        X, T, Y = self._np(X), self._np(T).ravel(), self._np(Y).ravel()
        mask1 = T == 1
        mask0 = ~mask1

        self.model1 = _lgbm_regressor(random_state=self.seed, **self.params)
        self.model1.fit(X[mask1], Y[mask1])
        self.model0 = _lgbm_regressor(random_state=self.seed, **self.params)
        self.model0.fit(X[mask0], Y[mask0])

        return self

    def predict_cate(self, X) -> np.ndarray:
        X = self._np(X)
        return self.model1.predict(X) - self.model0.predict(X)


class XLearner(_BaseLearner):
    """X-Learner: impute ITEs, model them, propensity-weighted ensemble.

    1. Train outcome models μ₁, μ₀ on each arm.
    2. Impute ITEs on the opposite arm: τ̃₁ = Y − μ₀(X) for treated,
       τ̃₀ = μ₁(X) − Y for control.
    3. Model τ̃₁ ~ X, τ̃₀ ~ X.
    4. Ensemble via propensity weight: τ̂ = g(X)·τ̃₀ + (1−g(X))·τ̃₁.
    """
    display_name = "X-Learner"

    def __init__(self, config: dict):
        super().__init__(config)
        self.model1 = None
        self.model0 = None
        self.cate_model1 = None
        self.cate_model0 = None
        self.propensity_model = None

    def fit(self, X, T, Y):
        X, T, Y = self._np(X), self._np(T).ravel(), self._np(Y).ravel()
        mask1 = T == 1
        mask0 = ~mask1

        # Step 1 — outcome models per arm
        self.model1 = _lgbm_regressor(random_state=self.seed, **self.params)
        self.model1.fit(X[mask1], Y[mask1])
        self.model0 = _lgbm_regressor(random_state=self.seed, **self.params)
        self.model0.fit(X[mask0], Y[mask0])

        # Step 2 — cross-imputed ITEs
        ite1 = Y[mask1] - self.model0.predict(X[mask1])
        ite0 = self.model1.predict(X[mask0]) - Y[mask0]

        # Step 3 — model ITEs
        self.cate_model1 = _lgbm_regressor(random_state=self.seed, **self.params)
        self.cate_model1.fit(X[mask1], ite1)
        self.cate_model0 = _lgbm_regressor(random_state=self.seed, **self.params)
        self.cate_model0.fit(X[mask0], ite0)

        # Step 4 — propensity for the ensemble weight
        self.propensity_model = _lgbm_classifier(random_state=self.seed)
        self.propensity_model.fit(X, T)

        return self

    def predict_cate(self, X) -> np.ndarray:
        X = self._np(X)
        tau1 = self.cate_model1.predict(X)
        tau0 = self.cate_model0.predict(X)
        g = self.propensity_model.predict_proba(X)[:, 1]
        return g * tau0 + (1 - g) * tau1


class DRLearner(_BaseLearner):
    """DR-Learner: doubly robust with cross-fitting, no EconML dependency.

    For each of K folds:
      1. Train propensity ê(X) and arm-specific regressions μ̂₁, μ̂₀ on the
         out-of-fold partition.
      2. Compute the AIPW pseudo-outcome Γ on the held-in fold:

         Γ = μ̂₁ − μ̂₀ + (T − ê) / (ê(1−ê)) · (Y − μ̂_T)

    A final model is then fit on (X, Γ) across all folds.
    """
    display_name = "DR-Learner"

    def __init__(self, config: dict):
        super().__init__(config)
        self.cv = int(config.get("cv", 3))
        self._final_model = None

    def fit(self, X, T, Y):
        from sklearn.model_selection import KFold

        X, T, Y = self._np(X), self._np(T).ravel(), self._np(Y).ravel()
        n = len(X)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.seed)
        gamma = np.empty(n, dtype=np.float64)

        fold = 0
        for train_idx, score_idx in kf.split(X):
            fold += 1
            X_tr, T_tr, Y_tr = X[train_idx], T[train_idx], Y[train_idx]
            X_sc, T_sc, Y_sc = X[score_idx], T[score_idx], Y[score_idx]

            # Propensity
            prop = _lgbm_classifier(random_state=self.seed)
            prop.fit(X_tr, T_tr)
            e_hat = prop.predict_proba(X_sc)[:, 1]

            # Regression per arm
            mask1 = T_tr == 1
            reg1 = _lgbm_regressor(random_state=self.seed, **self.params)
            reg1.fit(X_tr[mask1], Y_tr[mask1])
            reg0 = _lgbm_regressor(random_state=self.seed, **self.params)
            reg0.fit(X_tr[~mask1], Y_tr[~mask1])

            mu1 = reg1.predict(X_sc)
            mu0 = reg0.predict(X_sc)
            mu_t = mu1 * T_sc + mu0 * (1 - T_sc)

            # AIPW pseudo-outcome with clipped propensity
            eps = 1e-6
            e_clip = np.clip(e_hat, eps, 1.0 - eps)
            gamma[score_idx] = (
                mu1 - mu0
                + (T_sc - e_clip) / (e_clip * (1.0 - e_clip)) * (Y_sc - mu_t)
            )

        # Final model on the pseudo-outcomes
        self._final_model = _lgbm_regressor(random_state=self.seed, **self.params)
        self._final_model.fit(X, gamma)

        logger.info("%s cross-fit over %d folds done (n=%d).", self.display_name, self.cv, n)
        return self

    def predict_cate(self, X) -> np.ndarray:
        return self._final_model.predict(self._np(X))
