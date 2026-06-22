#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${LORA_VENV:-}" && -f "${LORA_VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${LORA_VENV}/bin/activate"
elif [[ -f "$HOME/venvs/lora-rocm721/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/venvs/lora-rocm721/bin/activate"
fi

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

RID=${REAL_LORA_RUN_ID:-"real_lora_protocol_smoke_$(date -u +%Y%m%dT%H%M%SZ)"}
MODEL=${REAL_LORA_SMOKE_MODEL:-sshleifer/tiny-gpt2}
python hf_gpu_smoke.py --model "$MODEL"

RUN_ROOT=${REAL_LORA_RUN_ROOT:-"$ROOT/runs/real_lora_source_runs"}
RELEASE_ROOT=${REAL_LORA_RELEASE_ROOT:-"$ROOT/runs/real_lora_releases"}

python run_real_lora_validation.py \
  --run-id "$RID" \
  --run-kind smoke \
  --out-dir "$RUN_ROOT" \
  --model "$MODEL" \
  --dataset_mode builtin \
  --block_size 32 \
  --batch_size 2 \
  --calib_batches 2 \
  --eval_batches 2 \
  --steps 4 \
  --uniform_rank 2 \
  --min_rank 1 \
  --max_rank 3 \
  --fisher_reference_rank 2 \
  --target_suffixes c_attn,c_proj,c_fc \
  --max_targets 4 \
  --max_train_texts 32 \
  --max_val_texts 16 \
  --max_train_blocks 16 \
  --max_val_blocks 8 \
  --log_every 1 \
  --dtype float32 \
  --strategies uniform,gradient_norm,spectral_effective,eva_activation,fim_gradient_variance,gora_sensitivity \
  --include_identity_control \
  --identity_tolerance 1e-7 \
  --deterministic_algorithms

RUN_DIR="$RUN_ROOT/$RID"
python scripts/validate_real_lora_run.py "$RUN_DIR" --expected-kind smoke
python scripts/build_real_lora_release.py \
  "$RUN_DIR" \
  --release-root "$RELEASE_ROOT" \
  --release-id "$RID" \
  --kind smoke
python scripts/validate_real_lora_release.py \
  "$RELEASE_ROOT/$RID" \
  --expected-kind smoke

(
  cd "$RELEASE_ROOT"
  sha256sum -c "${RID}.tar.gz.sha256"
)

printf 'Real LoRA protocol smoke release: PASS\n'
printf 'archive: %s/%s.tar.gz\n' "$RELEASE_ROOT" "$RID"
