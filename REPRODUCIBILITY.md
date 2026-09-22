# Reproducibility guide

The current research report is [Paper/paper.md](Paper/paper.md). Reading it
requires no build. These checks have different scopes; passing a cheaper check
does not establish a more expensive one.

## 1. Check the compact repository

From the root, with Python 3.12 or later:

```bash
python verify.py
python -m unittest discover -s tests -v
```

The verifier needs only the standard library and does not write into evidence
directories. It checks the root checksum manifest and reconstructs the primary
compact-evidence comparisons. Its output states the specific checks performed.
The unit tests exercise verifier failures as well as valid examples.

Alternatively, `sha256sum -c SHA256SUMS.txt` checks byte integrity only. The root
manifest covers maintained repository files except itself; experiment-local
manifests continue to identify their historical artifacts.

## 2. Run the experiment unit tests

Use a separate Python environment. For the matrix and synthetic-transformer
suites, install the declared package dependencies and pytest. For the
real-model **unit tests**, CPU PyTorch is sufficient; full training has a
separate model/data stack.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r Code/rmt_lora_sim/requirements.txt pytest
python -m pip install -r Code/transformer/requirements.txt
make -C Code test
```

These tests cover numerical helpers, allocation controls, analysis and release
validation, and runner behavior. They do not retrain GPT-2. For the versions
and observed test counts in this cleanup, read [the cleanup record](docs/CLEANUP.md).
The historical GPU environment record is
[requirements-tested-rocm721.txt](Code/real_lora_validation/requirements-tested-rocm721.txt);
it is not a complete transitive lockfile.

## 3. Regenerate Markdown tables and figures

```bash
python -m pip install -r Paper/requirements.txt
python Paper/make_paper_artifacts.py
```

This renders the committed, previously imported analysis tables into Markdown
and PNG. It does not revalidate the missing raw releases or rerun experiments.
Review generated changes before updating the root checksum manifest. Numerical
tables are the comparison target; plot bytes can vary with plotting versions.

The old document compiler, PDF, and publication-build lock remain retrievable
from the historical Git tag. They are not part of the maintained workflow.

## 4. Validate full external releases

The full releases are outside Git. Obtain the archives identified by
[RELEASE_ARTIFACTS.json](RELEASE_ARTIFACTS.json), verify their recorded archive
hashes, extract them, then run:

```bash
PYTHONPATH=Code/rmt_lora_sim \
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

The Stage4 validator checks structure, hashes, seed/condition coverage, and
metadata; it does not independently reproduce every fit. The transformer and
real-model validators additionally reconstruct their analysis outputs. None of
these validators reruns training.

The Stage4 archive relies on source commit
`e78df5898c949b003dfde4a8ac568465a5188b6d`; its recorded source sidecar supplies
the corresponding code. Current-source fixes and old release-source checks are
different operations. Use the recorded snapshot when reproducing the exact
historical release.

The import scripts in `Paper/` validate supplied full releases before updating
the report-facing tables. Their `--help` output documents path arguments.
Do not substitute superseded `Code/*/results/released/` tables for the current
publication aggregates under `evidence/`.

## 5. Retrain from the recorded protocols

Read each experiment's README and frozen analysis plan. GPT-2 training needs
the pinned model and WikiText-2 inputs in
[INPUT_MANIFEST.json](Code/real_lora_validation/INPUT_MANIFEST.json), plus a
compatible PyTorch/model runtime.

```bash
python Code/real_lora_validation/scripts/prepare_publication_inputs.py
```

That preparation may download third-party inputs. The real-model driver runs
offline after preflight. The recorded data companion is
[Zenodo 10.5281/zenodo.21061917](https://doi.org/10.5281/zenodo.21061917);
archive identities are preserved in [ARTIFACT_LEDGER.md](ARTIFACT_LEDGER.md).

This cleanup did not download those raw archives, independently verify their
remote availability, or repeat model training. No new release or DOI was
created.
