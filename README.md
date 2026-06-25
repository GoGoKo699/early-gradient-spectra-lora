# Early Gradient Spectra Predict Useful Low-Rank Adaptation

This repository is the reviewer-facing artifact for the anonymous manuscript
**“Early Gradient Spectra Predict Useful Low-Rank Adaptation.”**

## Evidence summary

The final evidence supports three deliberately scoped conclusions:

1. In the controlled matrix study, early gradient-spectrum summaries predict
   useful low-rank targets across the hard-knee and sample-limited regimes.
2. In the preregistered synthetic-transformer allocation study, the primary
   soft-dimension rule did **not** outperform its exact-cost uniform comparator:
   mean loss delta `+0.014638`, 95% bootstrap interval
   `[-0.007751, +0.035215]`, exact two-sided sign-flip `p=0.240234`.
3. In GPT-2/WikiText-2, spectral allocation improved the preregistered primary
   `c_attn,c_fc` suite relative to exact-cost uniform allocation:
   mean validation-loss delta `-0.007026`, 95% bootstrap interval
   `[-0.008343, -0.006026]`, exact sign-flip `p=0.0078125`, with 8/8 seed
   wins. The separate `attn.c_proj` boundary suite reversed direction and is
   reported separately; the paper does not claim universal superiority.

## Repository layout

- `Code/rmt_lora_sim/`: matrix and Stage4 simulations.
- `Code/transformer/`: controlled synthetic-transformer experiments.
- `Code/real_lora_validation/`: controlled GPT-2/WikiText-2 experiments.
- `Paper/`: manuscript source, generated tables, and figures.
- `evidence/`: compact, checksum-bound aggregate evidence and release manifests.
- `paper.pdf`: the manuscript PDF corresponding to this source tree.
- `RELEASE_ARTIFACTS.json`: identities and hashes of the full raw releases.
- `SHA256SUMS.txt`: checksums for every non-Git file in this repository.

The scientific source tree was frozen at commit
`bbfa459a4ceca34abac2dba3d522eaab970b9f2e`. Later packaging-only changes add
compact evidence, byte-reproducible publication build locks, and repository
checksums; they do not change the analyses or claims. Full raw releases and
pinned offline model/data inputs are kept outside Git because of size; their
exact identities are recorded here.

## Fast reviewer checks

```bash
sha256sum -c SHA256SUMS.txt

cd Code/rmt_lora_sim
python -m pip install -r requirements.txt pytest
PYTHONPATH=. python -m pytest -q

cd ../transformer
python -m pip install -r requirements.txt pytest
PYTHONPATH=. python -m pytest -q
```

The real-model suite requires a compatible PyTorch GPU installation:

```bash
cd Code/real_lora_validation
python -m pip install -r requirements.txt
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```

Expected test counts for the frozen artifact are 19, 38, and 37 respectively.

## Build and verify the paper

The committed tables and figures allow ordinary compilation without the large
raw releases:

```bash
cd Paper
make paper
```

For the release-grade, byte-for-byte check, build the locked linux/amd64 image
from the repository root. The image build performs two independent clean
rebuilds and rejects any figure, table, font, TeX, or manuscript hash drift:

```bash
docker build --platform linux/amd64 \
  -f Dockerfile.publication \
  -t early-gradient-spectra-publication .
```

The canonical environment is CPython 3.12.3, Matplotlib 3.11.0, the exact
hash-pinned wheels in `Paper/requirements-publication.lock.txt`, and pdfTeX
1.40.25 from a timestamped Ubuntu package snapshot. See `REPRODUCIBILITY.md`
for local checks, full-release validation, and rerun instructions.
