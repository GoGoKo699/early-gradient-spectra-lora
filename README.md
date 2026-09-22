# Early gradient spectra for low-rank adaptation

Can a gradient measured before adapter training tell us how much low-rank
capacity a task needs? This research repository studies that question in a
linear model, controlled matrix simulations, synthetic transformers, and one
GPT-2/WikiText-2 protocol.

**Outcome: this attempt did not establish the hoped-for broad, reliable benefit.**

The aim was to turn early-gradient spectra into a useful guide for LoRA rank
allocation. The evidence fell short of that broader expectation: the primary
synthetic-transformer comparison showed no demonstrated advantage, and a small
GPT-2 gain reversed when the eligible module families changed. The linear
calculation remains valid as a consequence of classical reduced-rank
approximation, and the matrix simulations support prediction within their
constructed families. These findings do not establish a generally superior
allocator or a distinct new theoretical contribution.

This repository records a good-faith research attempt, including the
corrections and results that did not meet expectations. The code, analysis,
and experimental evidence remain public so others can inspect what was tried,
understand its limitations, and build on it. The broader goal was not achieved
by this study; the underlying idea has not been proved impossible.

All maintained research documents and generated tables are Markdown. No
document compilation is required.

## What the evidence says

Loss differences below are candidate minus the matched uniform baseline;
negative values favor the candidate.

| Study | Supported observation | Important limit |
|---|---|---|
| Linear population model | A covariance-whitened full gradient reveals the singular spectrum governing optimal rank-constrained squared-error risk. | Classical truncated-SVD reasoning; not a theorem about nonlinear transformer training or the initial LoRA factor gradient. |
| Matrix simulations | Five seeds per regime show an association between early-gradient effective rank and rank-sweep targets; mean in-sample R² is 0.777 (hard-knee) and 0.687 (sample-limited). | Synthetic layers and fitted relationships; not evidence of prediction on new real tasks. |
| Synthetic transformers | Primary mean loss difference **+0.014638**, bootstrap 95% interval **[−0.007751, +0.035215]**, exact sign-flip **p = 0.240234**. | No demonstrated advantage in ten runs across two fixed task families. |
| GPT-2, `c_attn` / `c_fc` | Mean loss difference **−0.007026**, bootstrap 95% interval **[−0.008343, −0.006026]**, exact **p = 0.0078125**, within-suite Holm **p = 0.0390625**, 8/8 seed wins. | About 0.70% lower perplexity in one checkpoint/dataset/short-training protocol. |
| GPT-2 boundary suite, adding `attn.c_proj` | Direction reverses: **+0.001087**, 0/3 seed wins. | Three seeds give limited inference; report separately. |

The EVA-, GoRA-, and FIM-LoRA-style controls compare allocation scores under
this project's shared training protocol. They are **not full implementations
or head-to-head benchmarks of those methods**. Stored analysis plans document
the chosen protocol; file hashes alone do not prove prospective registration.

## Read the project

1. [Research report](Paper/paper.md): question, assumptions, derivations,
   methods, results, limitations, and references.
2. [Scientific sanity check](docs/SCIENTIFIC_AUDIT.md): what was checked,
   substantive corrections, and unresolved research questions.
3. [Reproducibility guide](REPRODUCIBILITY.md): lightweight checks, tests,
   artifact regeneration, and full rerun requirements.
4. [Cleanup record](docs/CLEANUP.md): scope and verification of this revision.

## Check the included evidence

From the repository root, using Python 3.12 or later:

```bash
python verify.py
python -m unittest discover -s tests -v
```

The first command needs only the Python standard library. It checks repository
checksums and independently reconstructs the primary compact-evidence
statistics. It does not download data or retrain a model. The three experiment
test suites have additional dependencies; see the reproducibility guide.

## Repository map

| Path | Purpose |
|---|---|
| `Code/rmt_lora_sim/` | Matrix and Stage4 simulations |
| `Code/transformer/` | Synthetic-transformer allocation experiments |
| `Code/real_lora_validation/` | GPT-2/WikiText-2 experiments and release validators |
| `Paper/paper.md` | Maintained research report |
| `Paper/tables/`, `Paper/figures/` | Derived tables and figures; status notes identify legacy outputs |
| `evidence/` | Compact evidence from the recorded releases |
| `docs/provenance/` | Historical correction and provenance records |
| `RELEASE_ARTIFACTS.json` | External archive identities and current report location |

The cleanup preserves experiment evidence and the existing licenses. Archived
and superseded results remain explicitly labeled for provenance; they are not
additional support for the current claims.

## Historical release and data

The earlier software release is
[v1.0.4-publication](https://github.com/GoGoKo699/early-gradient-spectra-lora/releases/tag/v1.0.4-publication).
The recorded companion data is
[Zenodo 10.5281/zenodo.21061917](https://doi.org/10.5281/zenodo.21061917).
Those identifiers refer to historical artifacts, not a new release of this
cleanup. Full raw archives and pinned model/data inputs are outside this Git
checkout; their hashes remain in [the artifact ledger](ARTIFACT_LEDGER.md).

Code is MIT licensed; project-generated evidence and documentation are covered
by [LICENSE-DATA.md](LICENSE-DATA.md). Third-party inputs retain their own terms.
