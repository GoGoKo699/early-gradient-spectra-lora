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

## Frozen multi-source publication driver

The publication driver binds every source run to a committed JSON plan, runs all
model work offline, aggregates only at the independent suite-seed-run level, and
builds one checksum-complete portable release. Proxy variables are removed only
from child processes. The invoking shell and VPN configuration are never
modified.

Validate the two committed plans before using GPU time:

```bash
python scripts/run_real_lora_publication.py \
  --plan configs/real_lora_publication_smoke_plan.json \
  --validate-plan-only

python scripts/run_real_lora_publication.py \
  --plan configs/real_lora_publication_plan.json \
  --validate-plan-only
```

The smoke plan contains four independent tiny-GPT-2 source runs (two seeds in
both the core and attention-output-projection suites) and is execution-only.
Run it with:

```bash
bash scripts/run_real_lora_publication_smoke.sh 2>&1 \
  | tee real_lora_publication_smoke.log
```

The frozen publication plan contains 11 source runs:

- eight primary `c_attn,c_fc` runs at seeds 101, 103, 107, 109, 113, 127,
  131, and 137;
- three attention-output-projection boundary runs at seeds 101, 103, and 107.

The primary estimand is the paired final-validation-loss difference,
`spectral_effective - uniform_r4`, across the eight primary runs. Eight primary
units are required because a two-sided exact sign-flip test with five units
cannot attain a p-value below 0.05. The boundary suite is descriptive and is
never pooled with the primary suite. Secondary allocation controls are adjusted
with Holm's procedure within each suite.

The long run is launched only after the publication-driver smoke release has
been independently validated:

```bash
RID="real_lora_publication_$(date -u +%Y%m%dT%H%M%SZ)"
printf '%s\n' "$RID" > "$HOME/real_lora_publication_active_id.txt"

bash scripts/run_real_lora_publication.sh \
  --release-id "$RID" 2>&1 \
  | tee "$HOME/${RID}.log"
```

Resume with the same release ID:

```bash
RID=$(cat "$HOME/real_lora_publication_active_id.txt")
bash scripts/run_real_lora_publication.sh \
  --release-id "$RID" --resume 2>&1 \
  | tee -a "$HOME/${RID}.log"
```

A valid release ends with `Real LoRA publication driver: PASS` and includes the
raw source runs, saved final adapter states, the frozen plan, independent
statistics, exact checksum coverage, code snapshot, environment record, and a
`.tar.gz.sha256` sidecar.

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

## Single-source publication gates

Passing `--run_kind publication` activates hard input and protocol gates. A
publication source run must use a pinned local model, local train and validation
text, offline loading, deterministic algorithms, the duplicate-uniform control,
and complete plan/release/suite bindings. Source runs should normally be created
only by `scripts/run_real_lora_publication.py`; direct invocation is retained for
diagnostics and validator development. A single source run is never treated as
independent multi-seed evidence.

## Legacy commands

The older scripts named `run_gpt2_*`, `run_publication_correction.py`, and the
historical released real-model tables predate the full stochastic-control and
exact-cost protocol. They are retained only for provenance. Do not use their
outputs as current publication evidence. `scripts/run_smoke.sh` now delegates to
`scripts/run_protocol_smoke.sh`.
