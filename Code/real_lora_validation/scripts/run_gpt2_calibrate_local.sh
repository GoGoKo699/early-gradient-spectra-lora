#!/usr/bin/env bash
set -euo pipefail
if [ -n "${LORA_VENV:-}" ] && [ -f "${LORA_VENV}/bin/activate" ]; then source "${LORA_VENV}/bin/activate"; fi
if [ -n "${LORA_PROXY_ENV:-}" ] && [ -f "${LORA_PROXY_ENV}" ]; then source "${LORA_PROXY_ENV}"; fi
export HF_HUB_DISABLE_XET=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
cd "$(dirname "$0")/.."

python run_real_lora_validation.py \
  --model models/gpt2_local \
  --dataset_mode local \
  --train_text_file data/wikitext2_local/train.txt \
  --val_text_file data/wikitext2_local/validation.txt \
  --target_suffixes c_attn,c_fc \
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
