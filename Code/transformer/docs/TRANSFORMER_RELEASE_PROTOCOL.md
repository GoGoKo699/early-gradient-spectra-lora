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

## Pre-specified publication driver

The canonical full experiment is
`configs/transformer_publication_plan.yaml`. The plan is copied into the
aggregate and cryptographically bound to both the aggregate manifest and the
top-level release manifest. Generated per-task/per-seed configs also record the
plan version, plan SHA-256, and intended release identifier.

The full plan currently requires:

- modular arithmetic and associative recall as distinct task families;
- seeds `101, 103, 107, 109, 113` for every task;
- `fixed_update_scale` as the primary scaling condition, with `standard` and
  `rslora` retained as separate sensitivity conditions;
- `soft_dimension` versus its linked `uniform_exact_cost` comparator;
- final validation loss as the primary metric;
- three adaptation replicates nested within each task/base-seed run;
- equal weighting of the pre-specified budgets within each independent run;
- equal weighting of the modular and associative-recall task-family means;
- a pre-specified two-sided alpha of 0.05;
- 10,000 within-task run-cluster bootstrap resamples and a two-sided exact
  task-stratified sign-flip test across all ten task/seed runs, enumerating all
  1,024 sign patterns (nominal minimum two-sided p-value 0.001953125);
- exhaustive modular evaluation; and
- 4,096 permutation maxima plus 2,000 edge-uncertainty resamples per module.

Run a plan-only gate before execution:

```bash
PYTHONPATH=. python3 scripts/run_transformer_publication.py \
  --plan configs/transformer_publication_plan.yaml \
  --validate-plan-only
```

The driver requires a clean committed Git worktree, enumerates exact source-run
paths, validates each run before it can enter the aggregate, rejects repeated
task/seed units, and validates the finished release and archive sidecar. It
supports `--resume`, but reuses an existing source run only after the complete
source validator passes and only when its generated config is byte-identical to
the pre-specified one. All child processes use the noninteractive Matplotlib
`Agg` backend and an ignored package-local config cache to make headless runs
deterministic with respect to display and font-cache state.

The multi-task smoke plan exercises the same orchestration and aggregate
binding with deliberately reduced computation. Its numerical values are never
publication evidence:

```bash
bash scripts/run_transformer_publication_smoke.sh
```

The full study is launched only after the smoke archive is independently
verified:

```bash
bash scripts/run_transformer_publication.sh
```

The aggregate must include `analysis_plan.yaml`, omnibus
`primary_analysis.csv`, `primary_run_deltas.csv`, secondary
`primary_analysis_by_task.csv`, exact-cost paired rows, run-level deltas,
run-cluster deltas, and run-cluster summaries. The confirmatory estimator first
averages budgets within each task/seed run, averages independent runs within
each task family, and then weights the two task-family means equally. Its
bootstrap resamples runs separately within each task; its exact sign-flip test
uses the same task-stratified statistic. This resolves the discrete-test issue
that would make a separate two-sided five-run test incapable of producing
`p < 0.05`. The inference is explicitly conditional on the two pre-specified
task families, not a claim over an unspecified population of transformer tasks.
The release and aggregate manifests carry explicit schema versions. The release
validator does not trust the aggregate's run-cluster table: it reconstructs the
primary candidate/exact-uniform pairs from every released raw
`budget_results.csv`, averages adaptation replicates and budgets in the declared
order, and then recomputes the task summaries, omnibus estimate, interval, and
exact test. Task-specific five-run p-values are secondary and cannot be below
0.0625 under two-sided exact enumeration.
