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

## What it tests

For each candidate transformer module site `j`, the package estimates

\[
\widehat M_{j,\lambda}
= -\widehat G_j(\widehat C_j+\lambda I)^{-1/2},
\]

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
scripts/build_transformer_release.py    # self-contained release builder
scripts/validate_transformer_release.py # checksum/release validator
scripts/run_rank_scaling_smoke.sh       # complete corrected smoke gate
```

## Design boundary

This is not a large language-model benchmark. It is a controlled bridge from the
matrix simulations to actual transformer modules: attention Q/K/V/O and MLP projections.

## Reproducibility notes

The main runner writes a deterministic `condition_seed` for every single-site sweep and budget comparison block. Allocation-rule and scaling-policy labels are deliberately excluded, so those conditions share adapter initialization, training data, evaluation data, and dropout streams. Rule order therefore cannot determine stochastic inputs.

Calibration hooks accumulate PyTorch autograd gradients of the mean token loss.  The released paper defines the empirical gradient estimator in the same convention: the per-token output gradients collected by hooks already include the batch mean-loss normalization, and the code averages the accumulated full-gradient matrix over calibration backward calls.
