# RMT-LoRA Simulation Package

Pure numerical simulations for testing whether low-rank transformer adaptation is better explained by **detectable spectral rank** than by nominal LoRA rank.

This package deliberately uses generated matrices and generated linear adaptation tasks. It does not require external datasets or pretrained model weights. The purpose is to validate the physics/mathematics hypotheses before moving to real transformer adapters.

## Research target

The central model is a spiked random-matrix approximation to an early fine-tuning gradient or update:

```text
G_l = S_l + Z_l
```

where `S_l` is a low-rank task signal and `Z_l` is a high-dimensional random bulk/noise component. The simulation asks whether useful adaptation directions are the singular directions that separate from a layer-specific null bulk edge.

The package tests four hypotheses:

1. **BBP detectability:** planted low-rank signals become useful only when their singular values separate from the random bulk.
2. **Rank saturation:** nominal LoRA rank stops helping once detectable task directions are represented.
3. **Alpha/intruder behavior:** overly aggressive LoRA scaling can create large directions poorly aligned with the planted task signal, producing a forgetting proxy.
4. **Task-vector interference:** merging degradation is predicted by overlap among detectable singular directions.

## Installation

From `Code/rmt_lora_sim`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

or, without editable installation:

```bash
pip install -r requirements.txt
```

## Quick smoke test

```bash
python scripts/run_lora_rank.py --config configs/smoke.yaml --plot
```

This writes a timestamped directory under `runs/`, containing:

```text
config.yaml
metrics.csv
figures/*.png
```

## Main experiments

### 1. BBP / spiked matrix transition

```bash
python scripts/run_bbp.py --config configs/bbp.yaml --plot
```

Expected outputs:

- top singular value versus planted spike strength;
- singular-vector overlap versus planted spike strength;
- detectable rank versus planted spike strength.

This validates the basic RMT mechanism without LoRA.

### 2. Synthetic LoRA rank sweep

```bash
python scripts/run_lora_rank.py --config configs/lora_rank.yaml --plot
```

Generated task:

```text
y = (W_base + Delta_true) x + noise
```

The adapter is trained as:

```text
Delta_LoRA = (alpha / rank) B A
```

The key plots are:

- validation loss vs nominal rank;
- validation loss vs detectable rank;
- detectable rank vs nominal rank;
- intruder directions vs forgetting proxy.

The intended positive result is a curve where validation loss saturates when detectable rank saturates, not merely when nominal rank increases.

### 3. Alpha sweep

```bash
python scripts/run_alpha_sweep.py --config configs/alpha_sweep.yaml --plot
```

This tests whether scaling controls spike amplification and whether very large alpha values produce unstable or poorly aligned outlier directions.

### 4. Task-vector merging simulation

```bash
python scripts/run_merging.py --config configs/merging.yaml --plot
```

This generates two synthetic task updates with controllable singular-subspace overlap and tests whether a spectral-interference score predicts merge degradation.


## Hypothesis fitting / model comparison

After any run, fit transparent linear predictor models to the generated `metrics.csv`:

```bash
python scripts/fit_hypotheses.py runs/<RUN_NAME>/metrics.csv
```

This writes:

```text
hypothesis_fit.csv
```

For LoRA rank and alpha simulations, it compares predictors such as nominal rank, activation-whitened early-gradient detectable/effective rank, raw early-gradient ablations, final-adapter detectable rank, stable rank, Frobenius norm, and intruder count. For merging simulations, it compares planted overlap against the spectral-interference score. Use this file to decide which hypothesis is fitting the numerical evidence rather than judging plots by eye.

## Early-gradient estimator

The default matrix estimator is now activation-whitened:

```text
M_hat = (-dL/dDelta at Delta=0) @ C_hat^{-1/2}
```

where `C_hat = X^T X / n` is the empirical activation second moment from the generated calibration inputs. In the noiseless population linear model, the raw negative gradient is proportional to `Delta_* C`, while the activation-whitened matrix is proportional to `Delta_* C^{1/2}`. The latter is the operator whose singular tail energy determines useful rank in the paper's reduced-rank theorem.

Run CSVs contain:

```text
gradient_*            # config-selected estimator; publication configs use activation_whitened
whitened_gradient_*   # explicit alias when the selected estimator is activation_whitened
raw_gradient_*        # unwhitened early-gradient ablation
gradient_matrix_mode  # selected estimator name
```

Set `detect.gradient_matrix_mode: raw` in a config only to reproduce older unwhitened diagnostics.

## Package layout

```text
rmt_lora_sim/
  configs/
    bbp.yaml
    lora_rank.yaml
    alpha_sweep.yaml
    merging.yaml
    smoke.yaml
  scripts/
    run_bbp.py
    run_lora_rank.py
    run_alpha_sweep.py
    run_merging.py
    make_plots.py
  rmt_lora/
    spectra.py       # singular values, MP edge, stable/effective rank, entropy profile
    nulls.py         # permutation/sign-flip/Gaussian null edges
    spiked.py        # spiked random matrices and subspace-overlap utilities
    lora_sim.py      # generated linear adaptation task and NumPy LoRA training
    experiments.py   # experiment drivers
    plotting.py      # matplotlib figures
    cli.py           # command-line entry points
```

## Main observables

For a matrix `M`, the package computes:

- singular values;
- Marchenko-Pastur edge under Gaussian scaling;
- bootstrap null edge from entry permutation or sign flips;
- detectable rank: number of singular values above the chosen edge;
- stable rank;
- entropy effective rank;
- planted-subspace overlap;
- intruder count: detectable singular directions poorly aligned with the planted task signal;
- forgetting proxy: output drift on the base mapping caused by the adapter.

## Interpreting results

A simulation run supports the proposed research line if the following patterns appear:

1. The BBP experiment shows a sharp increase in singular-vector alignment near the spectral edge.
2. In rank sweeps, validation loss correlates better with detectable rank than with nominal rank.
3. In alpha sweeps, very small alpha underfits while very large alpha increases output drift or intruder count.
4. In merging simulations, merge degradation grows with spectral interference among detectable singular directions.

A simulation run weakens the line if:

- nominal rank predicts validation loss as well as detectable rank;
- detectable-rank estimates are unstable under null choice;
- intruder count does not correlate with forgetting proxy;
- merging degradation is unrelated to spectral interference.

## Next extension toward a transformer paper

The pure-simulation package is a phase-0 validation tool. If the hypotheses survive here, the same API can be extended to real adapters by replacing generated matrices with actual layerwise tensors:

```text
Delta W_Q, Delta W_K, Delta W_V, Delta W_O, MLP up/down/gate updates
```

The spectral analysis code can be reused unchanged.


## Released paper results

The cleaned publication package stores paper-supporting matrix outputs under:

```text
results/released/
```

The folders use semantic names such as `bbp`, `lora_rank_clean`, `alpha_noise_intruder`, `merge_conflict`, `rank_alpha_grid`, and `stage4_aggregate`. New experiments continue to write timestamped directories under `runs/`; generated `runs/` directories are ignored by Git.
