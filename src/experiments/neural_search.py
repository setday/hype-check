"""Optuna search spaces and CV objective for neural uplift models."""

from __future__ import annotations

from typing import Any, Callable, Dict, Type

import numpy as np
import optuna

from src.experiments.metrics_eval import compute_point_metrics
from src.experiments.splits import SplitBundle, iter_cv_train_val
from src.models.neural.cfrnet import CFRNet
from src.models.neural.descn import DESCN
from src.models.neural.dragonnet import DragonNet
from src.models.neural.efin import EFIN
from src.models.neural.tarnet import TARNet

NEURAL_MODELS: Dict[str, Type] = {
    "dragonnet": DragonNet,
    "tarnet": TARNet,
    "cfrnet": CFRNet,
    "efin": EFIN,
    "descn": DESCN,
}


def _suggest_common(trial: optuna.Trial, base: dict) -> dict:
    cfg = dict(base)
    cfg.update({
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 200, 256]),
        "n_layers": trial.suggest_int("n_layers", 2, 4),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024, 2048]),
        "max_epochs": trial.suggest_int("max_epochs", 20, 80),
        "patience": trial.suggest_categorical("patience", [3, 5, 8]),
    })
    return cfg


def suggest_params(trial: optuna.Trial, model_key: str, base: dict) -> dict:
    cfg = _suggest_common(trial, base)
    if model_key == "dragonnet":
        cfg["alpha"] = trial.suggest_float("alpha", 0.1, 5.0, log=True)
        cfg["beta"] = trial.suggest_float("beta", 0.1, 5.0, log=True)
    elif model_key == "cfrnet":
        cfg["mmd_weight"] = trial.suggest_float("mmd_weight", 0.01, 10.0, log=True)
    elif model_key == "descn":
        cfg["alpha"] = trial.suggest_float("alpha", 0.1, 2.0, log=True)
    return cfg


def cv_qini_objective(
    trial: optuna.Trial,
    *,
    model_key: str,
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    split: SplitBundle,
    base_config: dict,
    Y_eval: np.ndarray | None = None,
) -> float:
    """Optuna objective: mean validation Qini over shared 3-fold CV."""
    if model_key not in NEURAL_MODELS:
        raise KeyError(f"Unknown neural model '{model_key}'")

    y_metric = Y if Y_eval is None else Y_eval
    params = suggest_params(trial, model_key, base_config)
    fold_qinis = []

    for fold_train, fold_val in iter_cv_train_val(split.train_idx, split.folds):
        model = NEURAL_MODELS[model_key](params)
        model.fit(
            X[fold_train], T[fold_train], Y[fold_train],
            X_val=X[fold_val], T_val=T[fold_val], Y_val=Y[fold_val],
        )
        cate = model.predict_cate(X[fold_val])
        qini = compute_point_metrics(cate, y_metric[fold_val], T[fold_val])["qini"]
        fold_qinis.append(qini)
        trial.set_user_attr(f"fold_qini_{len(fold_qinis)-1}", qini)

    mean_qini = float(np.mean(fold_qinis))
    trial.set_user_attr("mean_cv_qini", mean_qini)
    return mean_qini


def build_model(model_key: str, params: dict):
    if model_key not in NEURAL_MODELS:
        raise KeyError(f"Unknown neural model '{model_key}'")
    return NEURAL_MODELS[model_key](params)
