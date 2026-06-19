#!/usr/bin/env bash
set -euo pipefail

# Paper-grade Stage4 replication used for the reported aggregate table:
# 5 seeds x 48 synthetic layers for each condition.
PYTHONPATH=. python3 scripts/run_stage4_replicates.py --seeds 101,103,107,109,113 --n-layers 48 --modes hard,sample
