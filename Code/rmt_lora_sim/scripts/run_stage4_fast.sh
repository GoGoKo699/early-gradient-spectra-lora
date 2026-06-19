#!/usr/bin/env bash
set -euo pipefail

# Development profile: 3 seeds x 32 synthetic layers x 2 conditions.
PYTHONPATH=. python3 scripts/run_stage4_replicates.py \
  --profile fast \
  --tag stage4_fast \
  --no-plot
