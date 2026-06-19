# Synthetic Transformer Rank Allocation

This package is the transformer-module bridge for the RMT/LoRA useful-rank project.
It runs small synthetic sequence tasks, computes early module gradients, estimates
regularized-whitened spectral ranks, and tests whether these spectra predict useful LoRA rank.

The package is intentionally self-contained and uses synthetic data only.

## Quick smoke test

From the package root:

```bash
PYTHONPATH=. python3 scripts/run_synthetic_transformer.py --config configs/smoke_modular.yaml
```

This writes a run directory like:

```text
runs/YYYYMMDD_HHMMSS_smoke_modular/
  config.yaml
  base/base_metrics.csv
  calibration/module_stats.csv
  calibration/singular_values.npz
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
scripts/run_synthetic_transformer.py   # full smoke/small experiment
scripts/summarize_run.py               # compact summary for a completed run
scripts/run_smoke.sh                   # convenience wrapper
```

## Design boundary

This is not a large language-model benchmark. It is a controlled bridge from the
matrix simulations to actual transformer modules: attention Q/K/V/O and MLP projections.

## Reproducibility notes

The main runner writes a deterministic `condition_seed` for every single-site sweep and every budgeted allocation condition.  These seeds are derived from the run seed and the condition labels, so rule order does not silently determine LoRA initialization or minibatch order in new runs.

Calibration hooks accumulate PyTorch autograd gradients of the mean token loss.  The released paper defines the empirical gradient estimator in the same convention: the per-token output gradients collected by hooks already include the batch mean-loss normalization, and the code averages the accumulated full-gradient matrix over calibration backward calls.
