"""Upload DragonNet Optuna + lockbox results to Weights & Biases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXPERIMENT_NOTES = """
# DragonNet W1 — uplift targeting on marketing RCTs

## Model
**DragonNet** (Shi et al. 2019): shared MLP encoder + heads Y0, Y1, propensity;
targeted regularization. Predicts CATE = Y1 − Y0 for uplift ranking.

## Protocol (Salavat W1)
- **Split:** 80% train / 20% lockbox, stratified by (T, Y), split_seed=20260716
- **Tuning:** Optuna TPE, 40 trials, sampler_seed=42
- **CV:** 3-fold on train, objective = mean validation Qini
- **Refit:** best hyperparams on full 80% train
- **Eval:** single lockbox pass — Qini, AUUC, uplift@10/30
- **Bootstrap:** paired B=1000, quantiles 2.5%/97.5%, same indices across models

## Datasets
| key | description |
|-----|-------------|
| hillstrom_mens | Hillstrom, mens arm only (exclude womens-only) |
| hillstrom_womens | Hillstrom, womens arm only |
| x5 | RetailHero / X5, full 200k rows |
| criteo | Criteo visit, cohort 300k (cohort_seed=20260715) |
| lzd | LZD cleaned RCT |

## Artifacts uploaded
- `optuna_trials_*` — all 40 trials per dataset (distribution of CV Qini)
- `best_params_*` — winning LightGBM-style NN hyperparams per dataset
- `lockbox_metrics` — point estimates on lockbox
- `lockbox_bootstrap` — paired bootstrap CIs
- `lockbox_predictions` — CATE scores on lockbox (parquet artifact)
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="results/neural_tune_full")
    p.add_argument("--project", default="hype-check")
    p.add_argument("--entity", default="hype-check")
    p.add_argument("--run-name", default="dragonnet-w1-full")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    import os
    import wandb

    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        try:
            wandb.login(key=api_key)
        except (ValueError, wandb.errors.UsageError):
            pass

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        job_type="dragonnet-w1-eval",
        notes=EXPERIMENT_NOTES,
        config={
            "model": "DragonNet",
            "optuna_trials": 40,
            "cv_folds": 3,
            "lockbox_fraction": 0.2,
            "split_seed": 20260716,
            "cohort_seed_criteo": 20260715,
            "bootstrap_n": 1000,
        },
        reinit=True,
    )

    # Lockbox summary tables
    for name in ["lockbox_metrics.csv", "lockbox_bootstrap.csv"]:
        path = run_dir / name
        if path.exists():
            df = pd.read_csv(path)
            wandb.log({name.replace(".csv", ""): wandb.Table(dataframe=df)})
            print(f"Logged {path} ({len(df)} rows)")

    # Per-dataset Optuna studies
    trial_files = sorted(run_dir.glob("optuna_trials_*_dragonnet.csv"))
    if not trial_files:
        trial_files = sorted(run_dir.glob("optuna_trials_*.csv"))
    for trials_path in trial_files:
        df = pd.read_csv(trials_path)
        key = trials_path.stem
        wandb.log({key: wandb.Table(dataframe=df)})
        if "value" in df.columns and df["value"].notna().any():
            wandb.log({f"{key}/best_cv_qini": float(df["value"].max())})
        print(f"Logged {trials_path} ({len(df)} trials)")

    # Best hyperparameters per dataset
    for params_path in sorted(run_dir.glob("best_params_*_dragonnet.json")):
        with open(params_path) as f:
            params = json.load(f)
        key = params_path.stem
        wandb.config.update({key: params})
        print(f"Logged config {key}")

    # Predictions artifact
    pred_path = run_dir / "predictions_lockbox.parquet"
    if pred_path.exists():
        art = wandb.Artifact("lockbox_predictions_dragonnet", type="predictions")
        art.add_file(str(pred_path))
        run.log_artifact(art)
        print(f"Logged artifact {pred_path}")

    # Split metadata
    for split_path in sorted(run_dir.glob("split_*.json")):
        with open(split_path) as f:
            meta = json.load(f)
        wandb.config.update({split_path.stem: meta})

    url = run.url
    run.finish()
    print(f"\nDone: {url}")


if __name__ == "__main__":
    main()
