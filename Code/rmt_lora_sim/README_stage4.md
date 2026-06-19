# Stage4 useful-rank replication

Stage4 predicts practical adapter-rank targets from early-gradient spectral
statistics. The publication estimand is fixed in `rmt_lora/targets.py`:

```text
Vmin = minimum validation loss over the tested rank grid
DeltaV = V0 - Vmin
near-best(gamma) = smallest r with V(r) <= Vmin + gamma DeltaV
recovery(rho) = smallest r with (V0 - V(r)) / DeltaV >= rho
penalized(lambda) = argmin_r V(r) + lambda DeltaV r / rmax
```

The simulation oracle loss is diagnostic only. It cannot affect a target
threshold or penalty. Curves with nonpositive observed improvement have
undefined useful-rank targets and are excluded from target fits. Fits use
`log2(rank)` against `log2(1 + spectral_score)`, exactly as stated in the
manuscript.

The pre-rerun analysis plan is machine-readable in
`rmt_lora/stage4_analysis.py`. The hard-knee `near_best_rank_gap_0.1` versus
`gradient_effective_rank` pair is primary; the sample-limited counterpart is a
prespecified robustness check. Other combinations are secondary or exploratory.

## End-to-end smoke check

Run from `Code/rmt_lora_sim`:

```bash
bash scripts/run_stage4_smoke.sh
```

This executes one seed in each condition with reduced dimensions, independently
recomputes the targets, aggregates the exact source runs, validates every nested
checksum, and creates a portable archive under:

```text
runs/stage4_releases/
```

A smoke or fast release is a pipeline check only. The paper importer rejects it.

## Publication run

```bash
bash scripts/run_stage4.sh
```

The paper profile runs:

```text
5 seeds x 48 synthetic layers x 2 conditions
```

It refuses a dirty Git worktree and writes a self-contained release containing:

```text
configs/                         exact generated configs
source_runs/                     raw per-rank metrics for every seed
aggregate/stage4_key_table.csv   publication aggregate
aggregate/stage4_source_runs.csv source paths and hashes
aggregate/aggregate_manifest.json
release_manifest.json
SHA256SUMS.txt                   recursive release manifest
```

Validate a release independently with:

```bash
PYTHONPATH=. python3 scripts/validate_stage4_release.py \
  runs/stage4_releases/<release-id> \
  --expected-seeds 101,103,107,109,113 \
  --expected-conditions hard_knee,sample_limited
```

## Importing a corrected release into the manuscript

After the five-seed release validates:

```bash
cd ..
make stage4-to-paper
make paper
```

`make stage4-to-paper` follows `runs/stage4_releases/LATEST`, verifies the
aggregate checksums and observed-best metadata, and requires exactly five seed
runs in each condition. The old compact aggregate under
`results/released/stage4_aggregate/` used oracle-gap targets and is superseded;
the importer intentionally rejects it.
