# Publication repair workflow: current entry point

## Status

The effective-rank estimand, synthetic-transformer protocol, and controlled
real-model protocol have been corrected. The frozen Stage4, synthetic-
transformer, and GPT-2/Wikitext-2 publication releases have completed their
independent numerical and provenance audits.

The manuscript must now be generated only through the checksum-bound importers.
Legacy pre-CRN transformer summaries and pre-v3 real-model tables are audit
history, not paper inputs.

## Frozen real-model release

The current real-model protocol is `real_lora_publication_protocol_v3` and the
paper release is:

- release ID: `real_lora_publication_20260623T074520Z`;
- archive SHA-256: `4368ca32afd32fde04c68e1b889bcba2add57a801d8360db4e99557953c991de`;
- analysis-plan SHA-256: `e407ba9f34946cfaa6a1246ebd55247b4fdc208d22de5bc1bd279d0c63a7da7b`;
- source Git commit: `9d8547ca172ffcc8e059351b8d91a4fc92d36b7e`.

It contains eight primary `c_attn,c_fc` seed runs and three separate
`c_attn,attn.c_proj,c_fc` boundary runs. The primary comparison is
`spectral_effective - uniform_r4` final validation loss. The boundary suite is
not pooled with the primary suite.

## Verify and import the real-model release

Activate the validated environment, then run:

```bash
source "$HOME/venvs/lora-rocm721/bin/activate"
cd ~/下载/Lora_Project/Code/real_lora_validation

RID="real_lora_publication_20260623T074520Z"
RELEASE="$PWD/runs/real_lora_releases/$RID"
ARCHIVE="${RELEASE}.tar.gz"

PYTHONDONTWRITEBYTECODE=1 python -B \
  scripts/validate_real_lora_publication_release.py \
  "$RELEASE" --expected-kind publication

(
  cd "$(dirname "$ARCHIVE")"
  sha256sum -c "$(basename "${ARCHIVE}.sha256")"
)

cd ~/下载/Lora_Project
python Paper/import_real_lora_results.py --check-only
python Paper/make_paper_artifacts.py --validate-only
```

Required final messages include:

```text
Real LoRA publication release validation: PASS
real_lora_publication_20260623T074520Z.tar.gz: OK
Real-model manuscript import check: PASS
paper artifact input validation: PASS
```

## Build the manuscript

```bash
cd ~/下载/Lora_Project/Paper
make paper-artifacts

SOURCE_DATE_EPOCH=1782123298 FORCE_SOURCE_DATE=1 \
  pdflatex -interaction=nonstopmode -halt-on-error paper.tex
SOURCE_DATE_EPOCH=1782123298 FORCE_SOURCE_DATE=1 \
  pdflatex -interaction=nonstopmode -halt-on-error paper.tex

cp paper.pdf ../paper.pdf
```

Run both code test suites before freezing the submission:

```bash
cd ~/下载/Lora_Project/Code/transformer
source .venv/bin/activate
PYTHONPATH=. python -m pytest -q

deactivate
cd ../rmt_lora_sim
source .venv/bin/activate
PYTHONPATH=. python -m pytest -q

deactivate
source "$HOME/venvs/lora-rocm721/bin/activate"
cd ../real_lora_validation
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```

Do not regenerate the root `SHA256SUMS.txt`, `PATCH_MANIFEST.json`, or final
submission archive until the rendered PDF and all tracked files have passed the
last preflight.
