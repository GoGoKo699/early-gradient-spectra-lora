#!/usr/bin/env bash
set -euo pipefail

# Fast check: 3 seeds x 32 synthetic layers for each condition.
PYTHONPATH=. python3 scripts/run_stage4_replicates.py --seeds 101,103,107 --n-layers 32 --modes hard,sample
