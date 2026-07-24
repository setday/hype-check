#!/usr/bin/env bash
# Full W1 neural Optuna on Yandex Cloud GPU (gt4i.1, T4).
#
# Prerequisites on the VM / DataSphere node:
#   - NVIDIA driver + CUDA (check: nvidia-smi)
#   - Python 3.10+
#   - Repo cloned, data parquets at HYPECHECK_DATA_ROOT
#   - export WANDB_API_KEY=...   (https://wandb.ai/authorize)
#   - optional: export WANDB_ENTITY=your-team
#   - optional: export WANDB_RUN_NAME=neural-w1-gpu-150t
#
# Do NOT commit cloud passwords or API keys to git.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${HYPECHECK_DATA_ROOT:?Set HYPECHECK_DATA_ROOT to cleaned parquet root (features/ + outcomes/ per dataset)}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY from https://wandb.ai/authorize}"

export PYTHONPATH="${REPO_ROOT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TRANSFORMERS_NO_TF=1
export USE_TF=0

# Prefer CUDA wheels if the default torch build is CPU-only.
if ! python - <<'PY' 2>/dev/null
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  echo "CUDA not visible — install GPU PyTorch, e.g.:"
  echo "  pip install torch --index-url https://download.pytorch.org/whl/cu124"
  exit 1
fi

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY

RUN_NAME="${WANDB_RUN_NAME:-neural-w1-gpu-150t-$(date +%Y%m%d-%H%M)}"
OUTDIR="${REPO_ROOT}/results/neural_tune_gpu_full"
mkdir -p "$OUTDIR"

echo "=== Starting neural_tune_gpu_full (150 Optuna trials × 5 models × 5 datasets) ==="
echo "Outdir: $OUTDIR"
echo "W&B run: $RUN_NAME"

python tune_neural.py \
  --config-name neural_tune_gpu_full \
  "wandb.run_name=${RUN_NAME}" \
  "outdir=${OUTDIR}" \
  "optuna.storage=sqlite:///${OUTDIR}/optuna.db" \
  2>&1 | tee "${OUTDIR}/run.log"

echo "=== Done. Results in ${OUTDIR} ==="
echo "W&B: https://wandb.ai/${WANDB_ENTITY:-hype-check}/hype-check"
