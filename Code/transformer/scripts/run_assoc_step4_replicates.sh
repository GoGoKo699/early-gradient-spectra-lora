#!/usr/bin/env bash
set -euo pipefail

# Replicate the associative-recall Step 4 config. Run the single seed first; only
# run this script if the base model learns the base task.
SEEDS=${SEEDS:-"202 303"}
THREADS=${THREADS:-12}
BASE_CONFIG=${BASE_CONFIG:-"configs/assoc_small_step4_s101.yaml"}

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "Missing $BASE_CONFIG. Apply synthetic_transformer_step4_patch.zip first." >&2
  exit 1
fi

mkdir -p configs/_step4_generated

for seed in $SEEDS; do
  cfg="configs/_step4_generated/assoc_small_step4_s${seed}.yaml"
  python3 - "$BASE_CONFIG" "$cfg" "$seed" "$THREADS" <<'PY'
import sys, yaml
base_path, out_path, seed_s, threads_s = sys.argv[1:]
with open(base_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
seed = int(seed_s)
threads = int(threads_s)
cfg.setdefault('run', {})['seed'] = seed
cfg['run']['name'] = f'assoc_small_step4_s{seed}'
cfg['run']['torch_threads'] = threads
cfg['run']['device'] = 'cpu'
with open(out_path, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
print(out_path)
PY
  log="assoc_small_step4_s${seed}.log"
  echo "=== running associative seed ${seed}; log=${log} ==="
  PYTHONPATH=. python3 scripts/run_synthetic_transformer.py --config "$cfg" 2>&1 | tee "$log"
  latest=$(ls -td runs/*_assoc_small_step4_s${seed} | head -1)
  PYTHONPATH=. python3 scripts/summarize_run.py "$latest" || true
done

PYTHONPATH=. python3 scripts/aggregate_step4_tasks.py --runs-root runs --out step4_assoc_aggregate --patterns '*_assoc_small_step4_s*'

tar -czf synthetic_transformer_rank_assoc_step4_replicates.tar.gz \
  runs/*_assoc_small_step4_s* \
  step4_assoc_aggregate \
  assoc_small_step4_s*.log 2>/dev/null || \
  tar -czf synthetic_transformer_rank_assoc_step4_replicates.tar.gz \
    runs/*_assoc_small_step4_s* step4_assoc_aggregate

echo "wrote synthetic_transformer_rank_assoc_step4_replicates.tar.gz"
