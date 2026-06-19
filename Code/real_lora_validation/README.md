# Real pretrained-transformer LoRA validation

This directory contains the corrected real-model test for the
gradient-spectrum rank-allocation idea. The publication rerun completed on June
19, 2026; released summaries and compact per-seed evidence are included.
It compares approximately budget-matched LoRA strategies on a real pretrained causal LM:

1. uniform LoRA rank,
2. early gradient-norm allocation,
3. activation-whitened early-gradient spectral-effective-rank allocation.

It is self-contained and does not require PEFT. It wraps `torch.nn.Linear` and
HuggingFace GPT-style `Conv1D` modules directly, so every module rank is explicit
in `allocations.csv`.

## Expected environment

For the validated AMD reproduction environment, use the ROCm venv:

```bash
source "$HOME/venvs/lora-rocm721/bin/activate"
python - <<'PY'
import torch
print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```

Expected: ROCm PyTorch, `cuda available: True`, AMD Radeon Graphics.

## Install into the project

Unpack/copy this directory to:

```text
Code/real_lora_validation
```

## Step 1: smoke test

```bash
cd Code/real_lora_validation
./scripts/run_smoke.sh 2>&1 | tee smoke_$(date +%Y%m%d_%H%M%S).log
```

This uses `sshleifer/tiny-gpt2` plus a built-in text corpus. It only verifies
that downloads, GPU execution, calibration, allocation, and training run end to end.
Do not use smoke-test numbers in the paper.

## Step 2: calibration-only GPT-2 test

```bash
cd Code/real_lora_validation
./scripts/run_gpt2_calibrate.sh 2>&1 | tee gpt2_calibrate_$(date +%Y%m%d_%H%M%S).log
```

This downloads GPT-2 and Wikitext-2, finds GPT-2 target modules, computes raw and
activation-whitened early-gradient spectral metrics, writes:

```text
real_lora_runs/<timestamp>_gpt2/calibration_metrics.csv
real_lora_runs/<timestamp>_gpt2/allocations.csv
```

## Step 3: first real validation run

```bash
cd Code/real_lora_validation
./scripts/run_gpt2_validation.sh 2>&1 | tee gpt2_validation_$(date +%Y%m%d_%H%M%S).log
```

Outputs:

```text
real_lora_runs/<timestamp>_gpt2/results.csv
real_lora_runs/<timestamp>_gpt2/summary.txt
real_lora_runs/<timestamp>_gpt2/train_history_*.csv
```

Interpretation:

- If `spectral_effective` beats `uniform_r4` and `gradient_norm` under the same
  trainable-parameter budget cap, the paper has real-model positive evidence.
- If it loses, this is still useful: it defines a boundary between synthetic and
  pretrained-LM behavior.
- If it ties with fewer or similar effective ranks concentrated in plausible
  modules, inspect `allocations.csv` and `calibration_metrics.csv` before judging.

## Useful monitoring

```bash
watch -n 10 'find real_lora_runs -maxdepth 2 -type f \( -name results.csv -o -name summary.txt -o -name "train_history_*.csv" \) -printf "%TY-%Tm-%Td %TH:%TM %p\n" | sort | tail -30'
```

## Notes

Default target suffixes are GPT-2-style:

```text
c_attn,c_proj,c_fc
```

For LLaMA/Qwen-style models, use suffixes such as:

```text
q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```


## Reproducing the released GPT-2/Wikitext results

The clean release does not bundle GPT-2 model weights, Wikitext local text files, or raw `real_lora_runs/` folders.  The released summaries under `results/released/` are the compact artifacts used by the paper.

To rerun the real validation, prepare local files with this layout:

```text
models/gpt2_local/
  config.json
  generation_config.json
  model.safetensors
  vocab.json
  merges.txt
  tokenizer_config.json
  special_tokens_map.json

data/wikitext2_local/
  train.txt
  validation.txt
```

The paper's reported real validation used Python 3.12 with ROCm PyTorch 2.9.1+rocm7.2.1, transformers 5.12.0, datasets 5.0.0, accelerate 1.14.0, peft 0.19.1, safetensors 0.8.0, tokenizers 0.22.2, and numpy 1.26.4. The tested core package list is recorded in `requirements-tested-rocm721.txt`; run `python -m pip freeze` in your environment to create a full local lockfile before reproducing long runs.

The released CSVs may contain historical local `run_path`/`run_dir` strings. Treat those path strings as provenance labels only; the semantic fields for paper reproduction are the seed, strategy, validation losses, parameter counts, ranks, and pairwise differences.

## Effective-rank correction workflow

The entropy effective-rank statistic now uses normalized spectral energy:

```text
p_i = sigma_i^2 / sum_j sigma_j^2
effective_rank = exp(-sum_i p_i log(p_i))
```

For a clean corrected publication rerun, use the pinned-input and full-rerun helpers:

```bash
python scripts/prepare_publication_inputs.py
python scripts/preflight_publication_rerun.py
python scripts/run_publication_correction.py
```

The final command reruns uniform rank 4, gradient norm, and spectral effective rank
for all five plus three paper seeds. Every strategy receives a fresh DataLoader
with the same per-seed shuffle seed, so batch order is matched within a seed. It
regenerates `results/released/`, imports the paper-facing table rows, and creates a
`CORRECTED_REAL_GPT2_RESULTS_*.zip` bundle at the project root. Do not combine new
spectral measurements with the historical control rows when a full rerun is
feasible.
