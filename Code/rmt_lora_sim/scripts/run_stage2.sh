#!/usr/bin/env bash
set -euo pipefail

# Stage 2 has two jobs:
# 1. rank_alpha_grid: separate nominal rank from LoRA scaling alpha/rank.
# 2. layerwise_rank_prediction: test whether early-gradient detectable rank predicts layerwise rank demand.

PYTHONPATH=. python3 scripts/run_lora_rank.py --config configs/rank_alpha_grid.yaml --plot
latest_grid=$(ls -td runs/*rank_alpha_grid | head -1)
PYTHONPATH=. python3 scripts/robust_analysis.py "$latest_grid/metrics.csv"

PYTHONPATH=. python3 scripts/run_layerwise_rank_prediction.py --config configs/layerwise_rank_prediction.yaml --plot
latest_layer=$(ls -td runs/*layerwise_rank_prediction | head -1)
PYTHONPATH=. python3 scripts/robust_analysis.py "$latest_layer/metrics.csv"

tar -czf rmt_lora_stage2_runs.tar.gz "$latest_grid" "$latest_layer"
echo "wrote rmt_lora_stage2_runs.tar.gz"
