"""Upload neural Optuna + lockbox results to Weights & Biases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from src.experiments.neural_search import NEURAL_MODELS

PROTOCOL_NOTES = """
## Protocol (Salavat W1)
- **Split:** 80% train / 20% lockbox, stratified by (T, Y), split_seed=20260716
- **Tuning:** Optuna TPE, sampler_seed=42
- **CV:** 3-fold on train, objective = mean validation Qini
- **Refit:** best hyperparams on full 80% train
- **Eval:** single lockbox pass — Qini, AUUC, uplift@10/30
- **Bootstrap:** paired bootstrap, same indices across models
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--model-key", required=True, choices=list(NEURAL_MODELS.keys()))
    p.add_argument("--project", default="hype-check")
    p.add_argument("--entity", default="hype-check")
    p.add_argument("--run-name", required=True)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    model_key = args.model_key
    display_name = NEURAL_MODELS[model_key].display_name

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
        job_type=f"{model_key}-w1-eval",
        notes=f"# {display_name} W1 — uplift targeting on marketing RCTs\n\n{PROTOCOL_NOTES}",
        config={
            "model": display_name,
            "model_key": model_key,
            "cv_folds": 3,
            "lockbox_fraction": 0.2,
            "split_seed": 20260716,
            "cohort_seed_criteo": 20260715,
        },
        reinit=True,
    )

    for name in ["lockbox_metrics.csv", "lockbox_bootstrap.csv"]:
        path = run_dir / name
        if path.exists():
            df = pd.read_csv(path)
            wandb.log({name.replace(".csv", ""): wandb.Table(dataframe=df)})
            print(f"Logged {path} ({len(df)} rows)")

    trial_files = sorted(run_dir.glob(f"optuna_trials_*_{model_key}.csv"))
    for trials_path in trial_files:
        df = pd.read_csv(trials_path)
        key = trials_path.stem
        wandb.log({key: wandb.Table(dataframe=df)})
        if "value" in df.columns and df["value"].notna().any():
            wandb.log({f"{key}/best_cv_qini": float(df["value"].max())})
        print(f"Logged {trials_path} ({len(df)} trials)")

    for params_path in sorted(run_dir.glob(f"best_params_*_{model_key}.json")):
        with open(params_path) as f:
            params = json.load(f)
        wandb.config.update({params_path.stem: params})
        print(f"Logged config {params_path.stem}")

    pred_path = run_dir / "predictions_lockbox.parquet"
    if pred_path.exists():
        art = wandb.Artifact(f"lockbox_predictions_{model_key}", type="predictions")
        art.add_file(str(pred_path))
        run.log_artifact(art)
        print(f"Logged artifact {pred_path}")

    for split_path in sorted(run_dir.glob("split_*.json")):
        with open(split_path) as f:
            meta = json.load(f)
        wandb.config.update({split_path.stem: meta})

    url = run.url
    run.finish()
    print(f"\nDone: {url}")


if __name__ == "__main__":
    main()
