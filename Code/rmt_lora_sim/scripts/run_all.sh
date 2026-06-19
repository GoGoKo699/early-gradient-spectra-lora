#!/usr/bin/env bash
set -euo pipefail
python scripts/run_bbp.py --config configs/bbp.yaml --plot
python scripts/run_lora_rank.py --config configs/lora_rank.yaml --plot
python scripts/run_alpha_sweep.py --config configs/alpha_sweep.yaml --plot
python scripts/run_merging.py --config configs/merging.yaml --plot
