#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH=.
python3 -m pytest -q
python3 scripts/run_transformer_publication.py \
  --plan configs/transformer_publication_smoke_plan.yaml \
  "$@"
