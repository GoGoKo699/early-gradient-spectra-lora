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
  --dtype float32
