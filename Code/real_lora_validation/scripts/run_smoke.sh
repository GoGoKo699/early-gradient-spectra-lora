#!/usr/bin/env bash
set -euo pipefail
if [ -n "${LORA_VENV:-}" ] && [ -f "${LORA_VENV}/bin/activate" ]; then source "${LORA_VENV}/bin/activate"; fi
cd "$(dirname "$0")/.."
python hf_gpu_smoke.py
python run_real_lora_validation.py \
  --model sshleifer/tiny-gpt2 \
  --dataset_mode builtin \
  --block_size 64 \
  --batch_size 4 \
  --calib_batches 2 \
  --eval_batches 4 \
  --steps 10 \
  --uniform_rank 2 \
  --min_rank 1 \
  --max_rank 4 \
  --max_train_blocks 64 \
  --max_val_blocks 16 \
  --log_every 2 \
  --dtype float32
