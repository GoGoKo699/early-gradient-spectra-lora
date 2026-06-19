#!/usr/bin/env bash
set -euo pipefail
if [ -n "${LORA_VENV:-}" ] && [ -f "${LORA_VENV}/bin/activate" ]; then source "${LORA_VENV}/bin/activate"; fi
cd "$(dirname "$0")/.."
python run_real_lora_validation.py \
  --model gpt2 \
  --dataset_mode hf \
  --dataset_name Salesforce/wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --train_split train \
  --val_split validation \
  --block_size 128 \
  --batch_size 4 \
  --calib_batches 4 \
  --eval_batches 8 \
  --steps 1 \
  --uniform_rank 4 \
  --min_rank 1 \
  --max_rank 12 \
  --max_train_texts 2000 \
  --max_val_texts 500 \
  --max_train_blocks 512 \
  --max_val_blocks 128 \
  --log_every 1 \
  --dtype float32 \
  --calibrate_only
