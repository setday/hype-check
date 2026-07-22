"""Refit from saved best_params and recompute lockbox + bootstrap metrics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.data import load_protocol_arrays
from src.experiments.metrics_eval import compute_point_metrics, paired_bootstrap_metrics
from src.experiments.neural_search import NEURAL_MODELS, build_model
from src.experiments.splits import build_split_bundle

DATASET_CONFIGS = {
    "hillstrom_mens": {"cohort_limit": None, "cohort_seed": 20260715},
    "hillstrom_womens": {"cohort_limit": None, "cohort_seed": 20260715},
    "x5": {"cohort_limit": None, "cohort_seed": 20260715},
    "criteo": {"cohort_limit": 300000, "cohort_seed": 20260715},
    "lzd": {"cohort_limit": None, "cohort_seed": 20260715},
}


def _save_predictions(path: Path, cate: np.ndarray, model: str, dataset: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"dataset": dataset, "model": model, "cate_pred": cate})
    if path.exists():
        prev = pd.read_parquet(path)
        prev = prev[~((prev["dataset"] == dataset) & (prev["model"] == model))]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_parquet(path, index=False)


def _optuna_meta(outdir: Path, dataset: str, model_key: str) -> tuple[float, int]:
    trials_path = outdir / f"optuna_trials_{dataset}_{model_key}.csv"
    if not trials_path.exists():
        return float("nan"), 0
    trials = pd.read_csv(trials_path)
    val_col = "value" if "value" in trials.columns else "mean_validation_qini"
    complete = trials[trials["state"] == "COMPLETE"] if "state" in trials.columns else trials
    return float(complete[val_col].max()), len(complete)


def recompute_run(
    run_dir: Path,
    *,
    model_key: str = "dragonnet",
    bootstrap_n: int,
    bootstrap_seed: int,
    quantiles: tuple[float, float],
):
    run_dir.mkdir(parents=True, exist_ok=True)

    param_files = sorted(run_dir.glob(f"best_params_*_{model_key}.json"))
    if not param_files:
        raise FileNotFoundError(f"No best_params_*_{model_key}.json in {run_dir}")

    all_rows = []
    lockbox_predictions: dict[str, np.ndarray] = {}

    for params_path in param_files:
        stem = params_path.stem  # best_params_{dataset}_{model_key}
        dataset_name = stem.replace("best_params_", "").replace(f"_{model_key}", "")

        with open(params_path) as f:
            best_params = json.load(f)

        ds_cfg = DATASET_CONFIGS[dataset_name]
        X, T, Y, _ = load_protocol_arrays(
            dataset_name,
            cohort_limit=ds_cfg["cohort_limit"],
            cohort_seed=int(ds_cfg["cohort_seed"]),
        )
        split = build_split_bundle(
            T, Y,
            test_size=0.2,
            split_seed=20260716,
            n_splits=3,
            cv_seed=42,
        )
        train_idx = split.train_idx
        test_idx = split.test_idx

        model = build_model(model_key, best_params)
        t0 = time.perf_counter()
        model.fit(X[train_idx], T[train_idx], Y[train_idx])
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        cate_test = model.predict_cate(X[test_idx])
        infer_time = time.perf_counter() - t1

        _save_predictions(
            run_dir / "predictions_lockbox.parquet",
            cate_test,
            NEURAL_MODELS[model_key].display_name,
            dataset_name,
        )

        point = compute_point_metrics(cate_test, Y[test_idx], T[test_idx])
        optuna_best, optuna_n = _optuna_meta(run_dir, dataset_name, model_key)
        row = {
            "dataset": dataset_name,
            "model": NEURAL_MODELS[model_key].display_name,
            "model_key": model_key,
            "optuna_best_value": optuna_best,
            "optuna_n_trials": optuna_n,
            "train_time_s": round(train_time, 2),
            "inference_time_s": round(infer_time, 2),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            **{f"point_{k}": v for k, v in point.items()},
            "best_params_json": json.dumps(best_params),
        }
        all_rows.append(row)
        lockbox_predictions[f"{dataset_name}::{model_key}"] = cate_test
        print(f"[{dataset_name}] qini={point['qini']:.5f} auuc={point['auuc']:.5f}")

    boot_rows = []
    for params_path in param_files:
        dataset_name = params_path.stem.replace("best_params_", "").replace(f"_{model_key}", "")
        ds_cfg = DATASET_CONFIGS[dataset_name]
        _, T, Y, _ = load_protocol_arrays(
            dataset_name,
            cohort_limit=ds_cfg["cohort_limit"],
            cohort_seed=int(ds_cfg["cohort_seed"]),
        )
        split = build_split_bundle(T, Y, test_size=0.2, split_seed=20260716, n_splits=3, cv_seed=42)
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
            n_boot=bootstrap_n,
            seed=bootstrap_seed,
            quantiles=quantiles,
        )
        for model_name, metrics in boot.items():
            boot_rows.append({"dataset": dataset_name, "model": model_name, **metrics})

    metrics_df = pd.DataFrame(all_rows)
    boot_df = pd.DataFrame(boot_rows)
    metrics_df.to_csv(run_dir / "lockbox_metrics.csv", index=False)
    boot_df.to_csv(run_dir / "lockbox_bootstrap.csv", index=False)

    print(f"\nWrote {run_dir / 'lockbox_metrics.csv'} ({len(metrics_df)} rows)")
    print(f"Wrote {run_dir / 'lockbox_bootstrap.csv'} ({len(boot_df)} rows)")
    print("\n=== LOCKBOX POINT METRICS ===")
    print(metrics_df.round(5).to_string(index=False))
    print("\n=== LOCKBOX BOOTSTRAP ===")
    print(boot_df.round(5).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--model-key", default="dragonnet")
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--quantiles", nargs=2, type=float, default=(0.025, 0.975))
    args = p.parse_args()
    recompute_run(
        Path(args.run_dir),
        model_key=args.model_key,
        bootstrap_n=args.bootstrap_n,
        bootstrap_seed=args.bootstrap_seed,
        quantiles=tuple(args.quantiles),
    )


if __name__ == "__main__":
    main()
