# Reproducibility guide

## Publication build lock

The reviewer-facing PDF build is frozen independently of the experimental
runtimes. Its release gate is defined by:

- `.python-version`: CPython 3.12.3;
- `Paper/requirements-publication.lock.txt`: exact linux/amd64 wheels with
  SHA-256 hashes, including Matplotlib 3.11.0;
- `Paper/publication-toolchain.lock.json`: environment constants and expected
  hashes for every regenerated table, vector figure, and `paper.pdf`;
- `Dockerfile.publication`: an immutable Ubuntu base, a UTC package snapshot,
  and pdfTeX 1.40.25;
- `Paper/verify_publication_reproducibility.py`: two independent clean builds,
  byte comparisons against each other and the lock, plus a Type 3 font check.

Run the canonical full gate from the clean repository root:

```bash
docker build --platform linux/amd64 \
  -f Dockerfile.publication \
  -t early-gradient-spectra-publication .
```

The build fails unless all 13 Python-generated artifacts and the nine-page
manuscript reproduce their committed SHA-256 values. The Ubuntu archive is
fixed at `20260622T120000Z`; `SOURCE_DATE_EPOCH`, locale, timezone, Matplotlib
backend/configuration, and Python hash seed are also fixed.

A local Python-only check is available when CPython 3.12.3 is installed:

```bash
python3.12 -m venv .venv-publication
.venv-publication/bin/python -m pip install --require-hashes \
  -r Paper/requirements-publication.lock.txt
.venv-publication/bin/python \
  Paper/verify_publication_reproducibility.py --scope python
```

Ordinary `make -C Paper paper` compiles the committed assets, but byte identity
is guaranteed only by the locked full gate above. Overleaf uses a rolling TeX
environment and is intended for source review, not byte-identical PDF output.

## Levels of verification

### 1. Manuscript and compact-evidence verification

The repository includes all paper-facing tables, figures, compact aggregates,
analysis plans, release manifests, and validation code. Start with:

```bash
sha256sum -c SHA256SUMS.txt
```

The aggregate evidence is under:

- `evidence/stage4/`
- `evidence/transformer/`
- `evidence/real_lora/`

### 2. Full raw-release validation

The full releases are intentionally excluded from Git. Obtain the three
archives whose names and SHA-256 hashes are listed in
`RELEASE_ARTIFACTS.json`, extract each archive, then run:

```bash
python Code/rmt_lora_sim/scripts/validate_stage4_release.py \
  /path/to/stage4_paper_20260622T061351 \
  --expected-seeds 101,103,107,109,113 \
  --expected-conditions hard_knee,sample_limited

PYTHONPATH=Code/transformer \
python Code/transformer/scripts/validate_transformer_release.py \
  /path/to/transformer_publication_20260622T101458Z \
  --expected-kind publication

python Code/real_lora_validation/scripts/validate_real_lora_publication_release.py \
  /path/to/real_lora_publication_20260623T074520Z \
  --expected-kind publication
```

Each archive contains source runs, recursive checksum manifests, a frozen code
snapshot, environment provenance, analysis plans, and aggregate outputs.

### 3. Full reruns

The matrix and synthetic-transformer experiments can be reconstructed from
their frozen plans and scripts. The GPT-2 study additionally requires the
pinned offline GPT-2 and WikiText-2 inputs recorded in
`Code/real_lora_validation/INPUT_MANIFEST.json` and a compatible GPU runtime.

For the packaged local artifact, restore those inputs with:

```bash
cp -a ../Local/inputs/real_lora_validation/. \
  Code/real_lora_validation/
```

The real-model publication driver is network-offline after preflight.
