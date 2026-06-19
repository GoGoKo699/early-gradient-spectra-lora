#!/usr/bin/env bash
set -euo pipefail

if [ -n "${LORA_VENV:-}" ] && [ -f "${LORA_VENV}/bin/activate" ]; then source "${LORA_VENV}/bin/activate"; fi
if [ -n "${LORA_PROXY_ENV:-}" ] && [ -f "${LORA_PROXY_ENV}" ]; then source "${LORA_PROXY_ENV}"; fi

unset ALL_PROXY
unset all_proxy

export HF_HUB_DISABLE_XET=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

cd "$(dirname "$0")/.."

OUT="real_lora_runs/gpt2_attnproj_3seed_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

for seed in 101 103 107; do
  echo
  echo "================================================================================"
  echo "Running attnproj seed ${seed}"
  echo "================================================================================"

  python run_real_lora_validation.py \
    --model models/gpt2_local \
    --dataset_mode local \
    --train_text_file data/wikitext2_local/train.txt \
    --val_text_file data/wikitext2_local/validation.txt \
    --target_suffixes c_attn,attn.c_proj,c_fc \
    --block_size 128 \
    --batch_size 4 \
    --calib_batches 8 \
    --eval_batches 32 \
    --steps 200 \
    --uniform_rank 4 \
    --min_rank 1 \
    --max_rank 16 \
    --max_train_texts 5000 \
    --max_val_texts 1000 \
    --max_train_blocks 1024 \
    --max_val_blocks 256 \
    --log_every 20 \
    --dtype float32 \
    --seed "$seed" \
    --out_dir "$OUT"
done

python scripts/aggregate_real_replicates.py "$OUT"

echo
echo "replicate_dir: $OUT"
