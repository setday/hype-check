#!/usr/bin/env bash
# One-time local setup for DataSphere Jobs from VS Code/Cursor.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
YC_BIN="${HOME}/yandex-cloud/bin/yc"
FEDERATION_ID="${FEDERATION_ID:-}"

echo "== Hype Check · DataSphere local setup =="
echo "Repo: ${REPO_ROOT}"

if [[ ! -x "${YC_BIN}" ]]; then
  echo "ERROR: yc not found at ${YC_BIN}"
  echo "Run: curl -sSL https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash"
  exit 1
fi

echo "OK: yc $( "${YC_BIN}" --version )"

if [[ -z "${FEDERATION_ID}" ]]; then
  echo ""
  echo "FEDERATION_ID is required for the hackathon account."
  echo "Get it from Nikita or from the browser URL when you open https://center.yandex.cloud/"
  echo "Look for federation-id / yc_federation_hint in the address bar."
  echo ""
  echo "Then run:"
  echo "  export FEDERATION_ID=aje********"
  echo "  bash scripts/datasphere/setup_local.sh"
  exit 1
fi

echo ""
echo "Starting federated login (NOT personal Yandex ID):"
echo "  Online_project6_2@smiles2026.idp.yandexcloud.net"
echo ""

"${YC_BIN}" init --federation-id="${FEDERATION_ID}"

echo ""
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WARN: WANDB_API_KEY is not set in this shell."
else
  echo "OK: WANDB_API_KEY is set"
fi

echo ""
echo "Project ID: bt1rnun041kpnonujvlj"
echo "Next: Run and Debug → 'DataSphere: neural GPU smoke (2 trials)'"
