"""Optuna tuning + lockbox evaluation for neural uplift models."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import numpy as np
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from optuna.samplers import TPESampler

from src.models.neural.base import configure_torch_gpu
from src.experiments.data import load_protocol_arrays
from src.experiments.metrics_eval import compute_point_metrics, paired_bootstrap_metrics
from src.experiments.neural_search import NEURAL_MODELS, build_model, cv_qini_objective
from src.experiments.splits import build_split_bundle
from src.utils.init_utils import ROOT_PATH

logger = logging.getLogger(__name__)


def _maybe_init_wandb(cfg: DictConfig):
    wb = cfg.get("wandb", {})
    if not wb.get("enabled", False):
        return None
    import os
    import wandb

    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        try:
            wandb.login(key=api_key)
        except (ValueError, wandb.errors.UsageError):
            pass  # already logged in or newer key format handled by wandb.init

    entity = wb.get("entity") or os.environ.get("WANDB_ENTITY")
    run_name = wb.get("run_name") or os.environ.get("WANDB_RUN_NAME")

    run = wandb.init(
        project=wb.get("project", "hype-check"),
        entity=entity,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
    )

    import torch
    run.config.update({
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })
    if torch.cuda.is_available():
        logger.info("W&B run on GPU: %s", torch.cuda.get_device_name(0))
    return run


def _optuna_wandb_callback(wandb_run, dataset_name: str, model_key: str, log_trials: bool):
    if wandb_run is None or not log_trials:
        return None
    import wandb

    def _callback(study, trial):
        payload = {
            f"optuna/{dataset_name}/{model_key}/trial": trial.number,
            f"optuna/{dataset_name}/{model_key}/cv_qini": trial.value,
            f"optuna/{dataset_name}/{model_key}/best_so_far": study.best_value,
        }
        for param, value in trial.params.items():
            payload[f"optuna/{dataset_name}/{model_key}/hp/{param}"] = value
        wandb.log(payload, step=trial.number)

    return _callback


def _save_predictions(path: Path, cate: np.ndarray, model: str, dataset: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"dataset": dataset, "model": model, "cate_pred": cate})
    if path.exists():
        prev = pd.read_parquet(path)
        prev = prev[~((prev["dataset"] == dataset) & (prev["model"] == model))]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_parquet(path, index=False)


def _gpu_training_config(cfg: DictConfig) -> dict:
    gpu = cfg.get("gpu") or {}
    use_cuda = str(cfg.get("device") or "").startswith("cuda")
    return {
        "use_amp": bool(gpu.get("use_amp", use_cuda)),
        "num_workers": int(gpu.get("num_workers", 2 if use_cuda else 0)),
        "pin_memory": bool(gpu.get("pin_memory", use_cuda)),
    }


def tune_one_model(
    model_key: str,
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    Y_eval: np.ndarray,
    split,
    cfg: DictConfig,
    outdir: Path,
    dataset_name: str,
    wandb_run=None,
    *,
    outcome_transform: str = "identity",
) -> dict:
    opt_cfg = cfg.optuna
    base = {
        "seed": int(cfg.global_settings.seed),
        "device": cfg.get("device"),
        "val_fraction": 0.0,  # CV supplies validation; no internal split
        **_gpu_training_config(cfg),
    }

    study_name = f"{dataset_name}_{model_key}"
    storage = opt_cfg.get("storage")
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=TPESampler(seed=int(opt_cfg.sampler_seed)),
        storage=storage,
        load_if_exists=bool(storage),
    )

    def objective(trial):
        return cv_qini_objective(
            trial,
            model_key=model_key,
            X=X, T=T, Y=Y,
            split=split,
            base_config=base,
            Y_eval=Y_eval,
        )

    logger.info("Starting Optuna for %s (%d trials)", model_key, opt_cfg.n_trials)
    wb_cfg = cfg.get("wandb", {})
    callbacks = []
    cb = _optuna_wandb_callback(
        wandb_run, dataset_name, model_key, bool(wb_cfg.get("log_optuna_trials", False)),
    )
    if cb is not None:
        callbacks.append(cb)

    study.optimize(
        objective,
        n_trials=int(opt_cfg.n_trials),
        n_jobs=int(opt_cfg.n_jobs),
        show_progress_bar=True,
        catch=(RuntimeError, ValueError),
        callbacks=callbacks or None,
    )

    trials_df = study.trials_dataframe()
    trials_df.to_csv(outdir / f"optuna_trials_{dataset_name}_{model_key}.csv", index=False)

    best_params = study.best_params.copy()
    best_params.update({"seed": base["seed"], "device": base["device"], "val_fraction": 0.0})
    with open(outdir / f"best_params_{dataset_name}_{model_key}.json", "w") as f:
        json.dump(best_params, f, indent=2)

    # Refit on full train (80%) with best params
    train_idx = split.train_idx
    test_idx = split.test_idx

    model = build_model(model_key, best_params)
    t0 = time.perf_counter()
    model.fit(X[train_idx], T[train_idx], Y[train_idx])
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    cate_test = model.predict_cate(X[test_idx])
    infer_time = time.perf_counter() - t1

    pred_path = outdir / "predictions_lockbox.parquet"
    _save_predictions(pred_path, cate_test, NEURAL_MODELS[model_key].display_name, dataset_name)

    point = compute_point_metrics(cate_test, Y_eval[test_idx], T[test_idx])
    row = {
        "dataset": dataset_name,
        "model": NEURAL_MODELS[model_key].display_name,
        "model_key": model_key,
        "outcome_transform": outcome_transform,
        "optuna_best_value": study.best_value,
        "optuna_n_trials": len(study.trials),
        "train_time_s": round(train_time, 2),
        "inference_time_s": round(infer_time, 2),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        **{f"point_{k}": v for k, v in point.items()},
        "best_params_json": json.dumps(best_params),
    }

    if wandb_run is not None:
        import wandb
        wandb.log({f"{dataset_name}/{model_key}/{k}": v for k, v in row.items() if isinstance(v, (int, float, str))})
        wandb.log({f"{dataset_name}/{model_key}/trials": wandb.Table(dataframe=trials_df)})

    return row, cate_test


@hydra.main(version_base=None, config_path="config", config_name="neural_tune_pilot")
def main(cfg: DictConfig):
    configure_torch_gpu()
    outdir = Path(cfg.get("outdir", ROOT_PATH / "results" / "neural_tune"))
    outdir.mkdir(parents=True, exist_ok=True)
    wandb_run = _maybe_init_wandb(cfg)
    tf_cfg = cfg.get("transforms", {})
    outcome_transform = str(tf_cfg.get("name", "identity"))
    outcome_transform_eps = float(tf_cfg.get("eps", 1e-3))

    all_rows: List[dict] = []
    lockbox_predictions: Dict[str, np.ndarray] = {}
    split_bundles: Dict[str, object] = {}
    eval_outcomes: Dict[str, np.ndarray] = {}
    treatments: Dict[str, np.ndarray] = {}

    for dataset_name in cfg.datasets:
        ds_cfg = cfg.dataset_configs[dataset_name]
        X, T, Y_fit, meta = load_protocol_arrays(
            dataset_name,
            cohort_limit=ds_cfg.get("cohort_limit"),
            cohort_seed=int(ds_cfg.get("cohort_seed", 20260715)),
            outcome_transform=outcome_transform,
            outcome_transform_eps=outcome_transform_eps,
        )
        Y_eval = meta.get("y_eval", Y_fit)
        X = np.ascontiguousarray(X, dtype=np.float32)
        Y_fit = np.ascontiguousarray(Y_fit, dtype=np.float32)
        Y_eval = np.ascontiguousarray(Y_eval, dtype=np.float32)
        split = build_split_bundle(
            T, Y_eval,
            test_size=float(cfg.split.test_size),
            split_seed=int(cfg.split.split_seed),
            n_splits=int(cfg.split.n_splits),
            cv_seed=int(cfg.split.cv_seed),
        )
        split_bundles[dataset_name] = split
        eval_outcomes[dataset_name] = Y_eval
        treatments[dataset_name] = T

        meta_for_save = {k: v for k, v in meta.items() if k != "y_eval"}
        split_meta = {
            "dataset": dataset_name,
            "meta": meta_for_save,
            "train_n": len(split.train_idx),
            "test_n": len(split.test_idx),
        }
        with open(outdir / f"split_{dataset_name}.json", "w") as f:
            json.dump(split_meta, f, indent=2, default=str)

        for model_key in cfg.models:
            row, cate = tune_one_model(
                model_key, X, T, Y_fit, Y_eval, split, cfg, outdir, dataset_name, wandb_run,
                outcome_transform=outcome_transform,
            )
            all_rows.append(row)
            lockbox_predictions[f"{dataset_name}::{model_key}"] = cate

    # Paired bootstrap on lockbox — same indices for all models per dataset
    boot_rows = []
    for dataset_name in cfg.datasets:
        split = split_bundles[dataset_name]
        Y_eval = eval_outcomes[dataset_name]
        T = treatments[dataset_name]
        test_idx = split.test_idx
        preds = {
            k.split("::", 1)[1]: v
            for k, v in lockbox_predictions.items()
            if k.startswith(f"{dataset_name}::")
        }
        display_preds = {NEURAL_MODELS[k].display_name: v for k, v in preds.items()}
        boot = paired_bootstrap_metrics(
            display_preds,
            Y_eval[test_idx],
            T[test_idx],
            n_boot=int(cfg.bootstrap.n_replicates),
            seed=int(cfg.bootstrap.seed),
            quantiles=tuple(cfg.bootstrap.quantiles),
        )
        for model_name, metrics in boot.items():
            boot_rows.append({"dataset": dataset_name, "model": model_name, **metrics})

    metrics_df = pd.DataFrame(all_rows)
    boot_df = pd.DataFrame(boot_rows)
    metrics_df.to_csv(outdir / "lockbox_metrics.csv", index=False)
    boot_df.to_csv(outdir / "lockbox_bootstrap.csv", index=False)

    print("\n=== LOCKBOX POINT METRICS ===")
    print(metrics_df.round(5).to_string(index=False))
    print("\n=== LOCKBOX BOOTSTRAP ===")
    print(boot_df.round(5).to_string(index=False))

    if wandb_run is not None:
        import wandb
        wandb.log({"lockbox_metrics": wandb.Table(dataframe=metrics_df)})
        wandb.log({"lockbox_bootstrap": wandb.Table(dataframe=boot_df)})
        wandb.finish()


if __name__ == "__main__":
    main()
