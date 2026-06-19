#!/usr/bin/env bash
set -euo pipefail

# Stage 3 fixes the stage2 layerwise target.
# Stage2 used unconstrained best rank, which became the largest allowed rank.
# Stage3 evaluates rank/quality tradeoff targets and hard-knee finite-sample spectra.

PYTHONPATH=. python3 scripts/analyze_layerwise_targets.py "$(ls -td runs/*layerwise_rank_prediction | head -1)" || true

PYTHONPATH=. python3 scripts/run_layerwise_rank_prediction_v2.py --config configs/layerwise_hard_knee.yaml --plot
latest_hard=$(ls -td runs/*layerwise_hard_knee | head -1)
PYTHONPATH=. python3 scripts/analyze_layerwise_targets.py "$latest_hard"

# This second run is more expensive.  Keep it because it tests the sample-limited/overfitting regime.
PYTHONPATH=. python3 scripts/run_layerwise_rank_prediction_v2.py --config configs/layerwise_sample_limited.yaml --plot
latest_sample=$(ls -td runs/*layerwise_sample_limited | head -1)
PYTHONPATH=. python3 scripts/analyze_layerwise_targets.py "$latest_sample"

tar -czf rmt_lora_stage3_runs.tar.gz "$latest_hard" "$latest_sample"
echo "wrote rmt_lora_stage3_runs.tar.gz"
