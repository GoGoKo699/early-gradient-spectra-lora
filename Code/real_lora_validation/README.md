# Controlled real-model LoRA validation

This directory contains the real-pretrained-model protocol used to test whether
early gradient spectra support nonuniform LoRA rank allocation. The current
protocol supersedes the historical real-model runs bundled with earlier project
snapshots.

## What the corrected protocol controls

`run_real_lora_validation.py` now enforces:

- calibration in `model.eval()` mode while retaining gradients, so dropout does
  not contaminate spectral estimates;
- named SHA-256-derived random streams for model loading, adapter initialization,
  calibration data, training data, evaluation data, training dropout, and
  evaluation;
- a canonical maximum-rank LoRA-A initialization per module, with every smaller
  rank receiving an exact prefix of the same tensor;
- a training RNG reset after model, adapter, and optimizer construction, so
  allocation-dependent tensor shapes cannot shift dropout masks;
- exact trainable-parameter equality to the uniform-rank reference through a
  bounded dynamic-programming allocator;
- fixed validation data and fresh, identically seeded training loaders for every
  strategy;
- an optional duplicate-uniform identity control; and
- automatic failure if identical allocations produce different training traces,
  final adapter states, or evaluation metrics beyond the declared tolerance; and
- an active-adapter gate that records the first-step LoRA gradient, saves the
  final adapter tensors, independently reconstructs the zero-B initial state,
  and rejects any run with no parameter movement or no effective low-rank update.

A completed run includes raw calibration batches, activation components,
allocation and initialization hashes, per-step training histories, final adapter
state files, named seeds, input provenance, protocol checks, and a complete
recursive SHA-256 manifest. The saved adapter states let the validator recompute
parameter movement and the effective update matrix rather than trusting a logged
boolean.

## Allocation strategies

The runner supports:

1. `uniform`: exact-cost uniform rank;
2. `gradient_norm`: early full-weight gradient Frobenius norm;
3. `spectral_effective`: activation-whitened gradient effective rank;
4. `eva_activation`: activation explained-variance components;
5. `fim_gradient_variance`: mean squared initial LoRA-B gradient; and
6. `gora_sensitivity`: mean absolute full-weight `W * gradient` sensitivity.

The last three are **allocation-only controls**. They use the same nested random
LoRA initialization as every other strategy. They do not claim to reproduce the
full initialization or optimization procedures of EVA (arXiv:2410.07170),
FIM-LoRA (arXiv:2605.16800), or GoRA (arXiv:2502.12171).

## Environment

For the validated AMD setup:

```bash
source "$HOME/venvs/lora-rocm721/bin/activate"
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.hip)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

The tested core environment is recorded in
`requirements-tested-rocm721.txt`. Before a long run, also capture the resolved
local environment:

```bash
python -m pip freeze > requirements-local-freeze.txt
```

## Unit tests

```bash
cd Code/real_lora_validation
python -m pytest -q
```

## Publication-protocol smoke release

```bash
cd Code/real_lora_validation
bash scripts/run_protocol_smoke.sh 2>&1 | tee real_protocol_smoke.log
```

The smoke uses `sshleifer/tiny-gpt2`, a built-in non-evidential corpus, all six
allocation strategies, a duplicate uniform control, deterministic algorithms,
and exact cost matching. It validates and packages the result under:

```text
runs/real_lora_releases/<run_id>.tar.gz
runs/real_lora_releases/<run_id>.tar.gz.sha256
```

Required final messages are:

```text
Real LoRA run validation: PASS
Real LoRA release validation: PASS
Real LoRA release build: PASS
Real LoRA protocol smoke release: PASS
```

Smoke results verify execution only. They must not be imported into the paper.
The tiny smoke may show no visible validation-loss change at printed precision,
but it must now show nonzero first-step LoRA-B gradients, nonzero saved-parameter
movement, and a nonzero effective update.

## Manual source-run validation

```bash
python scripts/validate_real_lora_run.py \
  runs/real_lora_source_runs/<run_id> \
  --expected-kind smoke
```

## Manual release validation

```bash
PYTHONDONTWRITEBYTECODE=1 python -B scripts/validate_real_lora_release.py \
  runs/real_lora_releases/<run_id> \
  --expected-kind smoke

(
  cd runs/real_lora_releases
  sha256sum -c <run_id>.tar.gz.sha256
)
```

## Publication-run requirements

Passing `--run_kind publication` activates hard input and protocol gates. A
publication run must use:

- a local pinned model path;
- local train and validation text files;
- `--local_files_only`;
- `--deterministic_algorithms`; and
- `--include_identity_control`.

Example shape only—the full multi-seed publication driver is maintained
separately from this single-source-run command:

```bash
python run_real_lora_validation.py \
  --run-id real_gpt2_seed101 \
  --run-kind publication \
  --model models/gpt2_local \
  --local_files_only \
  --dataset_mode local \
  --train_text_file data/wikitext2_local/train.txt \
  --val_text_file data/wikitext2_local/validation.txt \
  --seed 101 \
  --uniform_rank 4 \
  --min_rank 1 \
  --max_rank 16 \
  --strategies uniform,gradient_norm,spectral_effective,eva_activation,fim_gradient_variance,gora_sensitivity \
  --include_identity_control \
  --deterministic_algorithms \
  --out_dir runs/real_lora_source_runs
```

Do not treat a single source run as an independent multi-seed result.

## Legacy commands

The older scripts named `run_gpt2_*`, `run_publication_correction.py`, and the
historical released real-model tables predate the full stochastic-control and
exact-cost protocol. They are retained only for provenance. Do not use their
outputs as current publication evidence. `scripts/run_smoke.sh` now delegates to
`scripts/run_protocol_smoke.sh`.
