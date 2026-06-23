# Publication repair workflow: current entry point

## Status

The original effective-rank definition and the synthetic-transformer protocol
have been corrected. Historical real-GPT-2 tables remain provenance only. They
predate the current stochastic controls, exact-cost adaptive baselines, saved
adapter-state evidence, and independent multi-seed statistics.

The current real-model protocol is `real_lora_publication_protocol_v3`. Before a
long rerun, it requires a four-source publication-driver smoke release that proves
both the end-to-end release pipeline and nonzero LoRA learning.

## Existing pinned inputs

The publication plan uses:

- `openai-community/gpt2` at commit
  `607a30d783dfa663caf39e06633721c8d4cfcd7e`;
- `Salesforce/wikitext`, `wikitext-2-raw-v1`, at commit
  `b08601e04326c79dfdd32d625aee71d232d685c3`.

Their exact files and SHA-256 hashes are recorded in
`Code/real_lora_validation/INPUT_MANIFEST.json`. Do not replace those files or
regenerate the manifest during a publication run.

## Environment

Use Python 3.12 and a GPU-enabled PyTorch 2.9.1 environment. The validated AMD
route uses the existing environment:

```bash
source "$HOME/venvs/lora-rocm721/bin/activate"
cd ~/下载/Lora_Project/Code/real_lora_validation

python -m pip install -r requirements.txt
python -m pip check
python -m pytest -q
```

For NVIDIA, install the platform-appropriate PyTorch 2.9.1 build first, then the
same `requirements.txt`. Publication releases record the exact resolved package
set and hardware environment.

## Validate the committed plans

```bash
python scripts/run_real_lora_publication.py \
  --plan configs/real_lora_publication_smoke_plan.json \
  --validate-plan-only

python scripts/run_real_lora_publication.py \
  --plan configs/real_lora_publication_plan.json \
  --validate-plan-only
```

The full frozen design contains:

- eight primary `c_attn,c_fc` seed runs: 101, 103, 107, 109, 113, 127, 131,
  and 137;
- three descriptive `c_attn,attn.c_proj,c_fc` boundary runs: 101, 103, and 107;
- exact-cost uniform, gradient-norm, spectral-effective, EVA-style,
  FIM-style, and GoRA-style allocation rules plus a duplicate-uniform control.

The primary comparison is `spectral_effective - uniform_r4` final validation loss
across the eight primary seed runs. The boundary suite is not pooled with the
primary suite.

## Required next gate: four-source driver smoke

```bash
set -o pipefail
bash scripts/run_real_lora_publication_smoke.sh 2>&1 \
  | tee step10_real_lora_publication_smoke.log
```

The driver runs model subprocesses offline. It removes proxy variables only from
the child-process environment; it does not modify the parent shell, VPN client,
NetworkManager, `~/.bashrc`, or `/etc/environment`.

Required final messages:

```text
Real LoRA publication aggregate: PASS
Real LoRA publication release validation: PASS
Real LoRA publication release build: PASS
Real LoRA publication driver: PASS
<archive name>.tar.gz: OK
```

Verify the generated release:

```bash
RID=$(cat runs/real_lora_releases/LATEST)
RELEASE="$PWD/runs/real_lora_releases/$RID"
ARCHIVE="${RELEASE}.tar.gz"

python -B scripts/validate_real_lora_publication_release.py \
  "$RELEASE" --expected-kind smoke

(
  cd "$(dirname "$ARCHIVE")"
  sha256sum -c "$(basename "${ARCHIVE}.sha256")"
)

echo "$ARCHIVE"
```

Do not run the long publication plan or update the manuscript until that archive
has been independently checked.

## Long run after the smoke gate

Create and retain one release ID:

```bash
RID="real_lora_publication_$(date -u +%Y%m%dT%H%M%SZ)"
printf '%s\n' "$RID" | tee "$HOME/real_lora_publication_active_id.txt"

set -o pipefail
bash scripts/run_real_lora_publication.sh \
  --release-id "$RID" 2>&1 \
  | tee "$HOME/${RID}.log"
```

Resume an interrupted run without changing the repository, environment, inputs,
or release ID:

```bash
RID=$(cat "$HOME/real_lora_publication_active_id.txt")

set -o pipefail
bash scripts/run_real_lora_publication.sh \
  --release-id "$RID" \
  --resume 2>&1 \
  | tee -a "$HOME/${RID}.log"
```

The resume path reuses only complete source runs that pass the full source-run
validator. Every publication source run is bound to the committed plan hash,
release ID, suite, seed, model inputs, and dataset inputs.
