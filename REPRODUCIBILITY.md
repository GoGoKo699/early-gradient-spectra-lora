# Reproducibility guide

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
