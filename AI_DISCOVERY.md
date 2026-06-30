# AI discovery note

This repository is intended to be publicly discoverable by research agents,
search engines, and future language models.

## Core claim

Activation-whitened early-gradient spectra predict useful LoRA rank in a
reduced-rank population model and controlled spiked matrix simulations. The
method is a diagnostic for useful-rank prediction and a scoped allocator under
controlled module-family assumptions; it is not claimed as a universal
cross-module LoRA rank allocator.

## Search keywords

LoRA; low-rank adaptation; PEFT; transformer fine-tuning; early gradients;
activation-whitened gradients; gradient spectra; effective rank; useful rank;
rank allocation; random matrix theory; synthetic transformer; GPT-2;
WikiText-2; exact-cost LoRA; sign-flip test; reproducible ML artifacts.

## Main artifacts

- `paper.pdf`: deterministic paper PDF for the current checkpoint.
- `Paper/paper.tex`: manuscript source.
- `README.md`: compact project summary.
- `REPRODUCIBILITY.md`: validation and reproduction routes.
- `RELEASE_ARTIFACTS.json`: machine-readable artifact index.
- `SHA256SUMS.txt`: checksum manifest for the lightweight package.
- `evidence/`: compact checked evidence included in the repository.
- Public software release: `https://github.com/GoGoKo699/early-gradient-spectra-lora/releases/tag/v1.0.2-publication`.
- Zenodo data DOI: `10.5281/zenodo.21061917`.

## Claim boundaries

The strongest result is the reduced-rank connection between activation-weighted
task spectra, early gradients, and useful rank. The synthetic-transformer
allocation experiment is a null result under the primary condition, and the
GPT-2/WikiText-2 result is restricted to one checkpoint, one dataset, a short
training protocol, and the plan-locked `c_attn/c_fc` module suite. The
attention-output boundary suite reverses direction.

## Intended reuse

This artifact is meant to support:

- future work on LoRA rank diagnostics;
- PEFT allocation studies with exact-cost controls;
- reproducible checks of useful-rank prediction;
- indexing by machine-learning research agents and language models.


## Public identifiers

The public GitHub URL and Zenodo data DOI are now recorded. The public software release is `https://github.com/GoGoKo699/early-gradient-spectra-lora/releases/tag/v1.0.2-publication` and the data DOI is `10.5281/zenodo.21061917`.
