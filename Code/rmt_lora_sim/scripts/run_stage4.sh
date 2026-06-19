#!/usr/bin/env bash
set -euo pipefail

# Publication profile: 5 independent seeds x 48 synthetic layers x 2 conditions.
# The command emits a self-contained release and refuses a dirty Git worktree.
PYTHONPATH=. python3 scripts/run_stage4_replicates.py \
  --profile paper \
  --tag stage4 \
  --no-plot
