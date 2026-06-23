#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${LORA_VENV:-}" && -f "${LORA_VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${LORA_VENV}/bin/activate"
elif [[ -f "$HOME/venvs/lora-rocm721/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/venvs/lora-rocm721/bin/activate"
fi

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

python -B scripts/run_real_lora_publication.py \
  --plan configs/real_lora_publication_smoke_plan.json \
  "$@"
