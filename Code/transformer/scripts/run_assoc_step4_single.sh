#!/usr/bin/env bash
set -euo pipefail

CFG=${CFG:-configs/assoc_small_step4_s101.yaml}
LOG=${LOG:-assoc_small_step4_s101.log}

if [[ ! -f "$CFG" ]]; then
  echo "Missing $CFG. Apply synthetic_transformer_step4_patch.zip from the project root." >&2
  exit 1
fi

PYTHONPATH=. python3 scripts/run_synthetic_transformer.py --config "$CFG" 2>&1 | tee "$LOG"
latest=$(ls -td runs/*_assoc_small_step4_s101 | head -1)
PYTHONPATH=. python3 scripts/summarize_run.py "$latest" || true

tar -czf synthetic_transformer_rank_assoc_step4_single.tar.gz "$latest" "$LOG"
echo "wrote synthetic_transformer_rank_assoc_step4_single.tar.gz"
