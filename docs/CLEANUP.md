# Cleanup and verification record

Date: 2026-09-22. Baseline commit:
`5bf03dc1c5472b9297208a2df5da8bb44854f5da`.

## Changes

- Converted the complete research report, displayed derivations, proofs,
  references, and generated tables to Markdown using plain/Unicode notation.
  Active figures are PNG. Removed the manuscript PDF, TeX sources, compiler
  Makefile, and obsolete document-build lock/container.
- Retained report regeneration from the committed CSVs and retained the full
  release importers' validation logic. Historical derived tables are explicitly
  marked superseded.
- Rewrote the entry points and artifact map around the current research
  question, evidence, reproducibility levels, and limitations. Removed stale
  release setup tasks and stopped labeling the current tree as the old release.
- Corrected the global allocation argument: gain/cost ordering does not solve
  unequal-cost integer rank allocation. The linear population result is
  explicitly a classical low-rank approximation corollary.
- Distinguished in-sample Stage4 fits and within-family leave-one-out fits from
  a frozen predictor tested on independent task families. Kept the synthetic
  null, narrow GPT-2 gain, and boundary reversal visible.
- Fixed real-model evaluation to weight batch losses by predicted-token count.
  The previous equal-batch average overweighted a short final batch. Two
  regression tests cover partition invariance and the evaluation batch cap.
  The recorded publication plan uses fixed-length blocks, batch size 4, and
  32 evaluation batches from 256 blocks, so its evaluated batches are full;
  historical result tables were not changed.
- Added a dependency-free compact verifier, failure-case tests, a small CI
  workflow, and contributor guidance. Added the missing transformer test
  dependency extra.

## Verification performed

The baseline root checksum check passed for **523 files** before edits.

| Check | Scope and result |
|---|---|
| Matrix test suite | 19 tests passed |
| Synthetic-transformer test suite | 38 tests passed |
| Real-model protocol test suite | 39 tests passed, including two new regressions |
| Compact verifier regression tests | 17 tests passed; cover malformed paths, altered evidence, missing runs, non-finite values, wrong costs, and optimized-Python validation |
| Python source compilation | Passed |
| Report input validation | Passed on the committed imported tables |
| Markdown/PNG regeneration | Passed using the plotting versions below |
| Scientific recalculation | Headline Stage4 associations and leave-one-out fits, paired means, and exact sign-flip p-values agree with committed evidence |
| Preservation review | Original licenses, all CSV data, and all files under `evidence/` remain byte-identical to the baseline |

The test environment was Python 3.12.14, NumPy 2.3.5, pandas 2.2.3,
SciPy 1.17.0, PyTorch 2.14.0+cpu, and pytest 9.1.1. Report generation used
Matplotlib 3.10.8. This is a CPU maintenance environment, not a recreation of
the original ROCm training environment.

The current root manifest replaces the baseline inventory after intentional
documentation/code changes. The nested legacy transformer table manifest also
changes because two readable derived tables moved from TeX to Markdown; the
underlying CSVs remain unchanged. Original manifests remain in Git history.

## Limits and remaining research work

No full model training, model download, external raw-release download, new
release, or DOI publication was performed. Bootstrap confidence intervals were
retained from the recorded outputs rather than independently regenerated in
the compact audit. A checksum establishes content integrity, not correctness,
prospective preregistration, or novelty.

[The scientific audit](SCIENTIFIC_AUDIT.md) gives the substantive assessment.
Current evidence is internally consistent within the tested scope. A precise
new contribution relative to existing gradient/activation-based LoRA work and
generalization beyond the restricted protocols remain research questions.
