# Corrected real GPT-2 experiment: completed release

## Status

This project fixes entropy effective rank to use normalized **squared singular
values**. The complete five-seed plus three-seed GPT-2/Wikitext-2 study was
rerun on June 19, 2026 with all three strategies trained together under matched
per-seed batch order. Corrected released tables, per-seed evidence, provenance,
and checksums are included, and the manuscript has been reconciled with them.

The instructions below reproduce the corrected experiment from pinned inputs.
Do not use the original unpatched ZIP or the explicitly archived superseded
raw-singular-value tables.

The folder in this ZIP is already named `Lora_Project`. Put it in a short local
path such as:

```text
C:\research\Lora_Project
```

or:

```text
~/research/Lora_Project
```

Avoid running from a cloud-synced directory.

## What you must download manually

1. This patched `Lora_Project` ZIP.
2. A supported GPU environment:
   - NVIDIA GPU: a current NVIDIA driver and Python 3.12.
   - AMD GPU: Linux with ROCm-capable hardware and Docker; use the pinned AMD
     container below.

You do **not** manually download GPT-2 or WikiText-2. The preparation script
fetches immutable, pinned revisions and verifies the dataset checksums.

A CPU-only machine, macOS, or AMD GPU on Windows is not accepted by the
publication runner. Use a Linux/NVIDIA GPU machine or the AMD Linux route.

---

# Route A — NVIDIA GPU on Windows

Open PowerShell and confirm that the NVIDIA driver sees the GPU:

```powershell
nvidia-smi
```

Then run the following commands exactly, changing only the first path:

```powershell
cd C:\research\Lora_Project

py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r Code\real_lora_validation\requirements.txt
python -m pip check

cd Code\real_lora_validation
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
python -m pytest -q tests
python scripts\prepare_publication_inputs.py
python scripts\preflight_publication_rerun.py
python scripts\run_publication_correction.py
```

Do not close PowerShell or let the computer sleep while the final command is
running.

---

# Route B — NVIDIA GPU on Linux

Confirm that the NVIDIA driver sees the GPU:

```bash
nvidia-smi
```

Then run, changing only the first path:

```bash
cd ~/research/Lora_Project

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r Code/real_lora_validation/requirements.txt
python -m pip check

cd Code/real_lora_validation
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); assert torch.cuda.is_available()'
python -m pytest -q tests
python scripts/prepare_publication_inputs.py
python scripts/preflight_publication_rerun.py
python scripts/run_publication_correction.py
```

---

# Route C — AMD ROCm GPU on Linux (validated route)

The completed publication run used Ubuntu, Python 3.12, ROCm 7.2.1, and AMD's
PyTorch 2.9.1 wheels. After confirming that `/dev/kfd` and a render device are
available, run:

```bash
cd ~/research/Lora_Project
python3.12 -m venv "$HOME/venvs/lora-rocm721"
source "$HOME/venvs/lora-rocm721/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir \
  torch==2.9.1 torchvision==0.24.0 torchaudio==2.9.0 \
  -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/
python -m pip install -r Code/real_lora_validation/requirements.txt
python -m pip check

cd Code/real_lora_validation
python -c 'import torch; print(torch.__version__); print(torch.version.hip); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); assert torch.cuda.is_available()'
python -m pytest -q tests
python scripts/prepare_publication_inputs.py
python scripts/preflight_publication_rerun.py
python scripts/run_publication_correction.py
```

---

# What each final command does

`prepare_publication_inputs.py` downloads and records:

- `openai-community/gpt2` at commit
  `607a30d783dfa663caf39e06633721c8d4cfcd7e`;
- `Salesforce/wikitext`, subset `wikitext-2-raw-v1`, at commit
  `b08601e04326c79dfdd32d625aee71d232d685c3`.

`preflight_publication_rerun.py` refuses to proceed if the Python/PyTorch/core
package versions, model identity, dataset checksums, input files, formula, or GPU
are wrong.

`run_publication_correction.py` performs the complete replacement experiment:

- five seeds for `c_attn,c_fc`;
- three seeds for `c_attn,attn.c_proj,c_fc`;
- uniform rank 4, gradient norm, and corrected spectral effective rank for every
  seed;
- matched training batch order within each seed;
- regenerated aggregate CSVs and LaTeX table rows;
- archived superseded real-GPT-2 tables and full provenance/checksums.

# Successful completion

The last command must end with:

```text
PUBLICATION CORRECTION RUN COMPLETE
```

It creates this file at the project root:

```text
CORRECTED_REAL_GPT2_RESULTS_corrected_full_<timestamp>.zip
```

For a new reproduction, retain that resulting ZIP together with the console log.
The corrected release supplied here has already completed the manuscript and PDF
reconciliation step.

# Resume after an interruption

The exact run directory is stored in:

```text
Code/real_lora_validation/LATEST_CORRECTION_RUN.txt
```

Resume from the same computer and unchanged virtual environment.

Windows PowerShell:

```powershell
$run = Get-Content .\LATEST_CORRECTION_RUN.txt
python scripts\run_publication_correction.py --resume $run
```

Linux:

```bash
python scripts/run_publication_correction.py --resume "$(cat LATEST_CORRECTION_RUN.txt)"
```

The resume command rejects changed code, packages, model files, dataset files, or
GPU environment instead of silently mixing runs.
