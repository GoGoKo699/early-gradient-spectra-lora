#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=. python3 scripts/run_lora_rank.py --config configs/lora_rank_clean.yaml --plot
PYTHONPATH=. python3 scripts/run_alpha_sweep.py --config configs/alpha_noise_intruder.yaml --plot
PYTHONPATH=. python3 scripts/run_merge_conflict.py --config configs/merge_conflict.yaml --plot

latest_rank=$(ls -td runs/*lora_rank_clean | head -1)
latest_alpha=$(ls -td runs/*alpha_noise_intruder | head -1)
latest_merge=$(ls -td runs/*merge_conflict | head -1)

PYTHONPATH=. python3 scripts/robust_analysis.py "$latest_rank/metrics.csv"
PYTHONPATH=. python3 scripts/robust_analysis.py "$latest_alpha/metrics.csv"
PYTHONPATH=. python3 scripts/robust_analysis.py "$latest_merge/metrics.csv"

tar -czf rmt_lora_stage1b_runs.tar.gz "$latest_rank" "$latest_alpha" "$latest_merge"
echo "wrote rmt_lora_stage1b_runs.tar.gz"
