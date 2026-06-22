# Synthetic-transformer release protocol

## Smoke gate

From `Code/transformer`:

```bash
PYTHONPATH=. python3 -m pytest -q
bash scripts/run_rank_scaling_smoke.sh
```

The wrapper creates one immutable source run under
`synthetic_transformer_publication_protocol_v4`, validates it, copies the exact
experiment code and environment metadata into a self-contained release,
generates recursive SHA-256 checksums, validates the copied release, and writes
a compressed archive plus a sidecar checksum.

The release is written under:

```text
runs/transformer_releases/<release_id>/
  release_manifest.json
  SHA256SUMS.txt
  source_runs/<source_run_id>/
  code_snapshot/
  provenance/
```

The archive and sidecar are:

```text
runs/transformer_releases/<release_id>.tar.gz
runs/transformer_releases/<release_id>.tar.gz.sha256
```

The current identifier is stored in `runs/transformer_releases/LATEST`.

## Verification

```bash
RID=$(cat runs/transformer_releases/LATEST)
python3 scripts/validate_transformer_release.py \
  "runs/transformer_releases/$RID" \
  --expected-kind smoke

(
  cd runs/transformer_releases
  sha256sum -c "${RID}.tar.gz.sha256"
)
```

The validator rejects missing or unlisted files, altered checksums, symbolic
links, invalid source runs, mismatched task/seed or protocol metadata, missing
code snapshots, and incomplete provenance. The release manifest records the
pre-specified scaling, allocation, reference, metric, and independent-unit roles
validated from every source run.

## Publication releases

`build_transformer_release.py` accepts repeated `--run` arguments. A publication
release built by this script must contain every pre-specified corrected primary
task/seed source run. Each source run retains its immutable config, base model,
raw calibration arrays, raw sweep and allocation rows, figures, exact stochastic
seeds, and success certificate. The publication gate requires an explicitly
supplied aggregate directory and checks its independent-run count and declared
run-cluster inference unit. Whitening evidence still requires its own validated
package before manuscript import; it is not silently appended to the primary
release.

The strict publication gate requires at least two task families, five independent
task/base-model seeds per family, all three scaling modes, exhaustive modular
evaluation, 4,096 permutation maxima per module, 2,000 edge-uncertainty
resamples, and the pre-specified 0.995 null quantile.

The independent unit is the task/base-model seed run. Adaptation replicates and
budgets are nested observations and must be averaged or clustered within that
unit before confirmatory inference. No manuscript import is permitted directly
from a smoke release or from compact legacy summaries.

Root project checksums remain deferred until the final manuscript and all
validated releases are frozen.
