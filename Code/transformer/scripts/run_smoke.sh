#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=.
python3 scripts/run_synthetic_transformer.py --config configs/smoke_modular.yaml
latest=$(ls -td runs/*_smoke_modular | head -1)
python3 scripts/summarize_run.py "$latest"
tar -czf synthetic_transformer_rank_smoke_run.tar.gz "$latest"
echo "wrote synthetic_transformer_rank_smoke_run.tar.gz"
