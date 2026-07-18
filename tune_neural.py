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

    return wandb.init(
        project=wb.get("project", "hype-check"),
        entity=wb.get("entity", "hype-check"),
        name=wb.get("run_name"),
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
    )


def _save_predictions(path: Path, cate: np.ndarray, model: str, dataset: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"dataset": dataset, "model": model, "cate_pred": cate})
    if path.exists():
        prev = pd.read_parquet(path)
        prev = prev[~((prev["dataset"] == dataset) & (prev["model"] == model))]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_parquet(path, index=False)


def tune_one_model(
    model_key: str,
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    split,
    cfg: DictConfig,
    outdir: Path,
    dataset_name: str,
    wandb_run=None,
) -> dict:
    opt_cfg = cfg.optuna
    base = {
        "seed": int(cfg.global_settings.seed),
        "device": cfg.get("device"),
        "val_fraction": 0.0,  # CV supplies validation; no internal split
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
        )

    logger.info("Starting Optuna for %s (%d trials)", model_key, opt_cfg.n_trials)
    study.optimize(
        objective,
        n_trials=int(opt_cfg.n_trials),
        n_jobs=int(opt_cfg.n_jobs),
        show_progress_bar=True,
        catch=(RuntimeError, ValueError),
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

    point = compute_point_metrics(cate_test, Y[test_idx], T[test_idx])
    row = {
        "dataset": dataset_name,
        "model": NEURAL_MODELS[model_key].display_name,
        "model_key": model_key,
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
    outdir = Path(cfg.get("outdir", ROOT_PATH / "results" / "neural_tune"))
    outdir.mkdir(parents=True, exist_ok=True)
    wandb_run = _maybe_init_wandb(cfg)

    all_rows: List[dict] = []
    lockbox_predictions: Dict[str, np.ndarray] = {}

    for dataset_name in cfg.datasets:
        ds_cfg = cfg.dataset_configs[dataset_name]
        X, T, Y, meta = load_protocol_arrays(
            dataset_name,
            cohort_limit=ds_cfg.get("cohort_limit"),
            cohort_seed=int(ds_cfg.get("cohort_seed", 20260715)),
        )
        split = build_split_bundle(
            T, Y,
            test_size=float(cfg.split.test_size),
            split_seed=int(cfg.split.split_seed),
            n_splits=int(cfg.split.n_splits),
            cv_seed=int(cfg.split.cv_seed),
        )

        split_meta = {
            "dataset": dataset_name,
            "meta": meta,
            "train_n": len(split.train_idx),
            "test_n": len(split.test_idx),
        }
        with open(outdir / f"split_{dataset_name}.json", "w") as f:
            json.dump(split_meta, f, indent=2)

        for model_key in cfg.models:
            row, cate = tune_one_model(
                model_key, X, T, Y, split, cfg, outdir, dataset_name, wandb_run,
            )
            all_rows.append(row)
            lockbox_predictions[f"{dataset_name}::{model_key}"] = cate

    # Paired bootstrap on lockbox — same indices for all models per dataset
    boot_rows = []
    for dataset_name in cfg.datasets:
        ds_cfg = cfg.dataset_configs[dataset_name]
        X, T, Y, _ = load_protocol_arrays(
            dataset_name,
            cohort_limit=ds_cfg.get("cohort_limit"),
            cohort_seed=int(ds_cfg.get("cohort_seed", 20260715)),
        )
        split = build_split_bundle(
            T, Y,
            test_size=float(cfg.split.test_size),
            split_seed=int(cfg.split.split_seed),
            n_splits=int(cfg.split.n_splits),
            cv_seed=int(cfg.split.cv_seed),
        )
        test_idx = split.test_idx
        preds = {
            k.split("::", 1)[1]: v
            for k, v in lockbox_predictions.items()
            if k.startswith(f"{dataset_name}::")
        }
        display_preds = {NEURAL_MODELS[k].display_name: v for k, v in preds.items()}
        boot = paired_bootstrap_metrics(
            display_preds,
            Y[test_idx],
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
