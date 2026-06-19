#!/usr/bin/env bash
set -euo pipefail

# One seed x six reduced-size layers x two conditions. This exercises the full
# observed-best target, source-provenance, aggregate, validation, and archive path.
PYTHONPATH=. python3 scripts/run_stage4_replicates.py \
  --profile smoke \
  --tag stage4_smoke \
  --no-plot
