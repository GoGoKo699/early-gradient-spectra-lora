# Run me first

This is the short entry point for reviewers.

## Compact package check

```bash
sha256sum -c SHA256SUMS.txt
python -m compileall Code
```

Run the three unit-test suites as described in `REPRODUCIBILITY.md`.

## Full evidence check

The full raw releases and pinned real-model inputs are not stored in Git. They
belong in the Zenodo data record and are indexed by `RELEASE_ARTIFACTS.json`.

## Artifact map

Read these files before attempting full validation:

```text
ARTIFACT_LEDGER.md
REPRODUCIBILITY.md
RELEASE_ARTIFACTS.json
PROVENANCE.md
```
