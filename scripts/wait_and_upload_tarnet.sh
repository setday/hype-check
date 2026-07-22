#!/bin/bash
# Wait for tune_neural TARNet full run, then upload to W&B.
set -e
REPO="/Users/kseniashk/uplift_папка/hype-check-repo"
LOG="$REPO/results/wandb_upload_tarnet.log"
PID="${1:-}"

cd "$REPO"
export PYTHONPATH="."

if [[ -n "$PID" ]]; then
  echo "$(date): waiting for PID $PID..." | tee -a "$LOG"
  while kill -0 "$PID" 2>/dev/null; do sleep 120; done
  echo "$(date): process finished." | tee -a "$LOG"
else
  echo "$(date): waiting for tune_neural tarnet full..." | tee -a "$LOG"
  while pgrep -f "neural_tune_tarnet_full" >/dev/null; do sleep 120; done
  echo "$(date): no tarnet full running." | tee -a "$LOG"
fi

echo "$(date): recompute lockbox for all datasets..." | tee -a "$LOG"
export HYPECHECK_DATA_ROOT="${HYPECHECK_DATA_ROOT:-/Users/kseniashk/uplift_папка/Данные uplift/data_A_cleaned}"
export OMP_NUM_THREADS=1
"$HOME/anaconda3/bin/python" scripts/recompute_lockbox.py \
  --run-dir results/neural_tune_tarnet_full \
  --model-key tarnet \
  --bootstrap-n 1000 \
  --quantiles 0.025 0.975 \
  2>&1 | tee -a "$LOG"

echo "$(date): uploading TARNet full to W&B..." | tee -a "$LOG"
if [[ -n "$WANDB_API_KEY" ]]; then
  "$HOME/anaconda3/bin/python" scripts/upload_neural_wandb.py \
    --run-dir results/neural_tune_tarnet_full \
    --model-key tarnet \
    --run-name tarnet-w1-full \
    2>&1 | tee -a "$LOG"
else
  echo "$(date): WANDB_API_KEY not set, skip upload." | tee -a "$LOG"
fi
echo "$(date): done." | tee -a "$LOG"
