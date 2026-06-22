#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH=.

RID="transformer_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="runs/transformer_source_runs/${RID}_modular_seed321"
LOG_DIR="runs/transformer_logs"
RELEASE_ROOT="runs/transformer_releases"
LOG="$LOG_DIR/${RID}.log"
mkdir -p "$LOG_DIR"

{
  python3 scripts/run_synthetic_transformer.py \
    --config configs/smoke_rank_scaling.yaml \
    --run-dir "$RUN_DIR"
  python3 scripts/validate_rank_scaling_run.py "$RUN_DIR"
} 2>&1 | tee "$LOG"

python3 scripts/build_transformer_release.py \
  --run "$RUN_DIR" \
  --release-root "$RELEASE_ROOT" \
  --release-id "$RID" \
  --kind smoke \
  --command-log "$LOG"

python3 scripts/validate_transformer_release.py \
  "$RELEASE_ROOT/$RID" \
  --expected-kind smoke

(
  cd "$RELEASE_ROOT"
  sha256sum -c "${RID}.tar.gz.sha256"
)

echo "Transformer rank-scaling smoke release: PASS"
echo "RELEASE_DIR=$ROOT/$RELEASE_ROOT/$RID"
echo "ARCHIVE=$ROOT/$RELEASE_ROOT/${RID}.tar.gz"
echo "SHA256_SIDECAR=$ROOT/$RELEASE_ROOT/${RID}.tar.gz.sha256"
