"""Upload lockbox metrics / Optuna trials to Weights & Biases.

Usage:
  export WANDB_API_KEY=...   # from https://wandb.ai/authorize
  python scripts/upload_to_wandb.py --run-dir results/neural_tune_pilot --run-name dragonnet-hillstrom-mens-pilot
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Directory with lockbox_metrics.csv etc.")
    p.add_argument("--project", default="hype-check")
    p.add_argument("--entity", default="hype-check")
    p.add_argument("--run-name", default="neural-upload")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    import os
    import wandb

    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)

    run = wandb.init(project=args.project, entity=args.entity, name=args.run_name, reinit=True)

    for name in ["lockbox_metrics.csv", "lockbox_bootstrap.csv"]:
        path = run_dir / name
        if path.exists():
            df = pd.read_csv(path)
            wandb.log({name.replace(".csv", ""): wandb.Table(dataframe=df)})
            print(f"Logged {path}")

    for trials_path in sorted(run_dir.glob("optuna_trials_*.csv")):
        df = pd.read_csv(trials_path)
        wandb.log({trials_path.stem: wandb.Table(dataframe=df)})
        print(f"Logged {trials_path}")

    for params_path in sorted(run_dir.glob("best_params_*.json")):
        with open(params_path) as f:
            params = json.load(f)
        wandb.config.update({params_path.stem: params})
        print(f"Logged {params_path}")

    pred_path = run_dir / "predictions_lockbox.parquet"
    if pred_path.exists():
        artifact = wandb.Artifact("lockbox_predictions", type="predictions")
        artifact.add_file(str(pred_path))
        run.log_artifact(artifact)
        print(f"Logged artifact {pred_path}")

    run.finish()
    print(f"Done: https://wandb.ai/{args.entity}/{args.project}/runs/{run.id}")


if __name__ == "__main__":
    main()
