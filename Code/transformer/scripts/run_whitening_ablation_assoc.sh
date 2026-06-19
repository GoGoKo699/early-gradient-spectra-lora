#!/usr/bin/env bash
set -euo pipefail
BASE_RUN=${BASE_RUN:-$(ls -td runs/*assoc_small_step4_s101* runs/*assoc_small_step4_s202* runs/*assoc_small_step4_s303* 2>/dev/null | head -1)}
THREADS=${THREADS:-12}
WHITENINGS=${WHITENINGS:-none,diag,full}
SPECTRAL_RULES=${SPECTRAL_RULES:-effective_rank,soft_dimension,marginal_gain_soft}
if [[ -z "$BASE_RUN" ]]; then
  echo "Could not find an associative base run. Set BASE_RUN=/path/to/run." >&2
  exit 1
fi
PYTHONPATH=. python3 scripts/run_whitening_ablation_from_run.py \
  --base-run "$BASE_RUN" \
  --name "whitening_ablation_assoc_from_$(basename "$BASE_RUN")" \
  --threads "$THREADS" \
  --whitenings "$WHITENINGS" \
  --spectral-rules "$SPECTRAL_RULES" \
  2>&1 | tee "whitening_ablation_assoc_$(basename "$BASE_RUN").log"
latest=$(ls -td runs/*whitening_ablation_assoc_from_* | head -1)
tar -czf synthetic_transformer_rank_whitening_ablation_assoc.tar.gz "$latest" "whitening_ablation_assoc_$(basename "$BASE_RUN").log"
echo "wrote synthetic_transformer_rank_whitening_ablation_assoc.tar.gz"
