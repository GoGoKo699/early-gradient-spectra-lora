#!/usr/bin/env bash
set -euo pipefail

# Adds replicate modular-small runs using the Step 2 fair allocation rules.
# Default only adds two new seeds, assuming you already ran modular_small_step2 once.
# Override with: SEEDS="202 303 404" bash scripts/run_modular_step3_replicates.sh

SEEDS=${SEEDS:-"202 303"}
THREADS=${THREADS:-12}
BASE_CONFIG=${BASE_CONFIG:-"configs/modular_small_step2.yaml"}

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "Missing $BASE_CONFIG. Apply synthetic_transformer_step2_patch.zip first." >&2
  exit 1
fi

mkdir -p configs/_step3_generated

for seed in $SEEDS; do
  cfg="configs/_step3_generated/modular_small_step3_s${seed}.yaml"
  python3 - "$BASE_CONFIG" "$cfg" "$seed" "$THREADS" <<'PY'
import sys, yaml
base_path, out_path, seed_s, threads_s = sys.argv[1:]
with open(base_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
seed = int(seed_s)
threads = int(threads_s)
cfg.setdefault('run', {})['seed'] = seed
cfg['run']['name'] = f'modular_small_step3_s{seed}'
cfg['run']['torch_threads'] = threads
cfg['run']['device'] = 'cpu'
with open(out_path, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
print(out_path)
PY
  log="modular_small_step3_s${seed}.log"
  echo "=== running seed ${seed}; log=${log} ==="
  PYTHONPATH=. python3 scripts/run_synthetic_transformer.py --config "$cfg" 2>&1 | tee "$log"
  latest=$(ls -td runs/*_modular_small_step3_s${seed} | head -1)
  PYTHONPATH=. python3 scripts/summarize_run.py "$latest" || true
done

PYTHONPATH=. python3 scripts/aggregate_step3_replicates.py --runs-root runs --out step3_aggregate

tar -czf synthetic_transformer_rank_modular_step3_replicates.tar.gz \
  runs/*_modular_small_step2 \
  runs/*_modular_small_step3_s* \
  step3_aggregate \
  modular_small_step3_s*.log 2>/dev/null || \
  tar -czf synthetic_transformer_rank_modular_step3_replicates.tar.gz \
    runs/*_modular_small_step3_s* step3_aggregate modular_small_step3_s*.log

echo "wrote synthetic_transformer_rank_modular_step3_replicates.tar.gz"
