"""Smoke test for DataSphere Jobs (2 Optuna trials, hillstrom + dragonnet)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    data_root = os.environ.get("HYPECHECK_DATA_ROOT", "/home/jupyter/datasets/data_A_cleaned")
    if not os.environ.get("WANDB_API_KEY"):
        print("ERROR: set WANDB_API_KEY", file=sys.stderr)
        return 1

    os.chdir(REPO_ROOT)
    os.environ["PYTHONPATH"] = str(REPO_ROOT)
    os.environ["HYPECHECK_DATA_ROOT"] = data_root
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")

    import torch
    print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    cmd = [
        sys.executable, str(REPO_ROOT / "tune_neural.py"),
        "--config-name", "neural_tune_gpu_full",
        "datasets=[hillstrom_mens]",
        "models=[dragonnet]",
        "optuna.n_trials=2",
        "bootstrap.n_replicates=10",
        f"wandb.run_name={os.environ.get('WANDB_RUN_NAME', 'smoke-gpu')}",
        f"outdir={REPO_ROOT / 'results' / 'neural_tune_gpu_smoke'}",
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
