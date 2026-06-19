#!/usr/bin/env bash
set -euo pipefail
BASE_RUN=${BASE_RUN:-$(ls -td runs/*modular_small_step2* runs/*modular_small_step3* 2>/dev/null | head -1)}
THREADS=${THREADS:-12}
WHITENINGS=${WHITENINGS:-none,diag,full}
SPECTRAL_RULES=${SPECTRAL_RULES:-effective_rank,soft_dimension,marginal_gain_soft}
if [[ -z "$BASE_RUN" ]]; then
  echo "Could not find a modular base run. Set BASE_RUN=/path/to/run." >&2
  exit 1
fi
PYTHONPATH=. python3 scripts/run_whitening_ablation_from_run.py \
  --base-run "$BASE_RUN" \
  --name "whitening_ablation_modular_from_$(basename "$BASE_RUN")" \
  --threads "$THREADS" \
  --whitenings "$WHITENINGS" \
  --spectral-rules "$SPECTRAL_RULES" \
  2>&1 | tee "whitening_ablation_modular_$(basename "$BASE_RUN").log"
latest=$(ls -td runs/*whitening_ablation_modular_from_* | head -1)
tar -czf synthetic_transformer_rank_whitening_ablation_modular.tar.gz "$latest" "whitening_ablation_modular_$(basename "$BASE_RUN").log"
echo "wrote synthetic_transformer_rank_whitening_ablation_modular.tar.gz"
