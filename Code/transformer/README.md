# Synthetic Transformer Rank Allocation

This package is the transformer-module bridge for the RMT/LoRA useful-rank project.
It runs small synthetic sequence tasks, computes early module gradients, estimates
regularized-whitened spectral ranks, and tests whether these spectra predict useful LoRA rank.

The package is intentionally self-contained and uses synthetic data only.

## Quick smoke tests

From the package root:

```bash
# Backward-compatible single-policy smoke
PYTHONPATH=. python3 scripts/run_synthetic_transformer.py --config configs/smoke_modular.yaml

# Publication repair: fixed, standard, and rsLoRA-style policies
bash scripts/run_rank_scaling_smoke.sh
```

The legacy command writes a single run directory. The corrected smoke writes a
self-contained release and checksum-verified archive under
`runs/transformer_releases/`. A source run contains:

```text
source_runs/<run_id>/
  config.yaml
  base/base_metrics.csv
  calibration/module_stats.csv
  calibration/singular_values.npz
  calibration/null_maxima.npz
  sweeps/site_rank_sweep_metrics.csv
  sweeps/site_target_summary.csv
  sweeps/site_prediction_fit.csv
  budget/budget_results.csv
  budget/allocation_comparison.csv
  figures/*.png
```

## Dependencies

Required Python packages:

```text
torch
numpy
pandas
matplotlib
pyyaml
```

On Ubuntu without virtual environments, try:

```bash
sudo apt update
sudo apt install -y python3-numpy python3-pandas python3-matplotlib python3-yaml python3-torch
```

If your Ubuntu repository does not provide `python3-torch`, install PyTorch using your preferred system policy.

## Rank-scaling controls

The runner supports three explicit LoRA multipliers: conventional `alpha/r`, a
rank-independent fixed update scale, and a reference-matched `1/sqrt(r)`
rsLoRA-style sensitivity condition. All conditions share the same base model,
initialization bank, data batches, validation set, and dropout stream. Useful-
rank targets and allocation summaries are kept separate by scaling policy.

The corrected mechanistic analysis pre-specifies `fixed_update_scale` as the
primary condition; `standard` and `rslora` are sensitivity conditions. See
`docs/RANK_SCALING_PROTOCOL.md` for formulas and required reporting.


## Budget and edge controls

The corrected runner saves every permutation-null maximum, reports Monte Carlo
uncertainty and quantile sensitivity for the spectral edge, evaluates modular
arithmetic over all input pairs, and creates an exactly same-realized-cost
uniform comparator for pre-specified allocation rules. See
`docs/SPECTRAL_EDGE_AND_BUDGET_PROTOCOL.md`.

## Release verification

The smoke wrapper emits a recursive-checksum release containing raw source data,
the exact code snapshot, environment records, and an archive checksum. See
`docs/TRANSFORMER_RELEASE_PROTOCOL.md`.

## Pre-specified multi-task evidence driver

The confirmatory transformer study is defined before execution in
`configs/transformer_publication_plan.yaml`. It fixes the task families, seeds,
primary scaling condition, allocation rule, exact-cost reference, metric,
independent unit, within-run budget weighting, equal weighting across the two
pre-specified task families, confidence level, bootstrap count, and two-sided
task-stratified sign-flip test. The driver refuses to start from a dirty Git
worktree or from a plan that does not meet the publication design gate.

Validate the plans without running experiments:

```bash
PYTHONPATH=. python3 scripts/run_transformer_publication.py \
  --plan configs/transformer_publication_smoke_plan.yaml \
  --validate-plan-only

PYTHONPATH=. python3 scripts/run_transformer_publication.py \
  --plan configs/transformer_publication_plan.yaml \
  --validate-plan-only
```

Exercise the complete two-task release pipeline with non-evidential smoke
settings:

```bash
bash scripts/run_transformer_publication_smoke.sh
```

Only after that archive passes independent verification, run the full design:

```bash
bash scripts/run_transformer_publication.sh
```

The driver uses exact source-run paths rather than glob discovery, validates
every source run before aggregation, rejects duplicate task/seed units, binds
generated configs and the aggregate to the analysis-plan SHA-256, and emits
an omnibus `primary_analysis.csv`, the underlying `primary_run_deltas.csv`, and
secondary `primary_analysis_by_task.csv` from exact-cost run-cluster summaries.
The omnibus estimator first averages budgets within a task/seed run, then runs
within each task, and finally gives each task family equal weight. Its interval
resamples runs within task using a versioned, outcome-independent deterministic
seed derived from the analysis identity and stratum sizes; values are sorted
within task so validation is invariant to row order and harmless CSV
round-tripping. Its exact test flips run signs while retaining task strata. The
full design enumerates all 1,024 sign patterns over ten independent
runs at a pre-specified two-sided alpha of 0.05 (nominal minimum two-sided
p-value 0.001953125). Task-specific five-run results are secondary and retain
their coarser exact-test resolution. Inference is conditional on the two named
task families. The release and aggregate carry explicit schema versions, and
the release validator reconstructs the confirmatory run rows directly from
each source run's raw exact-cost budget table before recomputing all primary
statistics. Existing validated source runs may be reused only with `--resume`;
invalid or partial source runs are rejected. An incomplete release directory
without its archive/sidecar is removed and rebuilt from the validated source
runs and regenerated aggregate.
Subprocesses use Matplotlib's noninteractive `Agg` backend and an ignored local
font/config cache, so headless publication runs do not depend on desktop display
state or pollute Git provenance with cache files.

## What it tests

For each candidate transformer module site `j`, the package estimates

```text
M_hat(j, λ) = −G_hat(j) @ (C_hat(j) + λ I)^(−1/2)
```

where `G_j` is the early full gradient and `C_j` is the module input covariance.
It computes:

- effective rank;
- stable rank;
- hard detectable rank;
- soft spectral dimension;
- top singular values.

Then it runs single-site LoRA rank sweeps and budgeted multi-site allocations.

## Main scripts

```text
scripts/run_synthetic_transformer.py    # raw source-run producer
scripts/validate_rank_scaling_run.py    # scientific/protocol validator
scripts/run_transformer_publication.py  # pre-specified multi-task driver
scripts/aggregate_step4_tasks.py        # task/seed run-cluster aggregation
scripts/build_transformer_release.py    # self-contained release builder
scripts/validate_transformer_release.py # checksum/release validator
scripts/run_rank_scaling_smoke.sh       # complete corrected smoke gate
scripts/run_transformer_publication_smoke.sh # multi-task pipeline gate
scripts/run_transformer_publication.sh  # full confirmatory experiment
```

## Design boundary

This is not a large language-model benchmark. It is a controlled bridge from the
matrix simulations to actual transformer modules: attention Q/K/V/O and MLP projections.

## Reproducibility notes

The main runner writes a deterministic `condition_seed` for every single-site sweep and budget comparison block. Allocation-rule and scaling-policy labels are deliberately excluded, so those conditions share adapter initialization, training data, evaluation data, and dropout streams. Rule order therefore cannot determine stochastic inputs.

Calibration hooks accumulate PyTorch autograd gradients of the mean token loss.  The released paper defines the empirical gradient estimator in the same convention: the per-token output gradients collected by hooks already include the batch mean-loss normalization, and the code averages the accumulated full-gradient matrix over calibration backward calls.
