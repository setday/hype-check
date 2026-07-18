#!/bin/bash
# Wait for tune_neural.py to finish, then upload DragonNet results to W&B.
set -e
REPO="/Users/kseniashk/uplift_папка/hype-check-repo"
LOG="$REPO/results/wandb_upload.log"
PID="${1:-}"

cd "$REPO"

if [[ -n "$PID" ]]; then
  echo "$(date): waiting for PID $PID..." | tee -a "$LOG"
  while kill -0 "$PID" 2>/dev/null; do sleep 60; done
  echo "$(date): process finished." | tee -a "$LOG"
else
  echo "$(date): waiting for any tune_neural.py..." | tee -a "$LOG"
  while pgrep -f "tune_neural.py" >/dev/null; do sleep 60; done
  echo "$(date): no tune_neural running." | tee -a "$LOG"
fi

echo "$(date): uploading to W&B..." | tee -a "$LOG"
"$HOME/anaconda3/bin/python" scripts/upload_dragonnet_wandb.py \
  --run-dir results/neural_tune_full \
  --run-name dragonnet-w1-full \
  2>&1 | tee -a "$LOG"

echo "$(date): upload complete." | tee -a "$LOG"
