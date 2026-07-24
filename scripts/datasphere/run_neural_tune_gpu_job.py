"""Entry point for DataSphere Jobs / VS Code DataSphere Jobs Toolkit.

Runs the same W1 neural Optuna pipeline as scripts/run_neural_tune_yandex_gpu.sh.
Data on cloud project disk (default): /home/jupyter/datasets/data_A_cleaned
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    data_root = os.environ.get(
        "HYPECHECK_DATA_ROOT",
        "/home/jupyter/datasets/data_A_cleaned",
    )
    if not Path(data_root).exists():
        print(f"ERROR: HYPECHECK_DATA_ROOT not found: {data_root}", file=sys.stderr)
        return 1

    if not os.environ.get("WANDB_API_KEY"):
        print("ERROR: set WANDB_API_KEY in job env vars or DataSphere secrets", file=sys.stderr)
        return 1

    os.chdir(REPO_ROOT)
    os.environ["PYTHONPATH"] = str(REPO_ROOT)
    os.environ.setdefault("HYPECHECK_DATA_ROOT", data_root)
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "8")

    outdir = REPO_ROOT / "results" / "neural_tune_gpu_full"
    outdir.mkdir(parents=True, exist_ok=True)
    run_name = os.environ.get("WANDB_RUN_NAME", "neural-w1-gpu-150t")

    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")
    print("HYPECHECK_DATA_ROOT:", data_root)
    print("Outdir:", outdir)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "tune_neural.py"),
        "--config-name", "neural_tune_gpu_full",
        f"wandb.run_name={run_name}",
        f"outdir={outdir}",
        f"optuna.storage=sqlite:///{outdir / 'optuna.db'}",
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
