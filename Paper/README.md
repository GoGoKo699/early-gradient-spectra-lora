# Research report

Read [paper.md](paper.md). The report retains the derivations, proofs, experimental protocols, results, limitations, and references in ordinary Markdown. Equations use Unicode and plain text; figures use PNG. Numerical results describe existing experiments, not fresh model training.

The reduced-rank result is a classical approximation corollary. Matrix useful-rank prediction, synthetic-transformer allocation, and the narrow GPT-2 allocation result are different claims. The [scientific audit](../docs/SCIENTIFIC_AUDIT.md) records the current assessment.

## Regenerate readable artifacts

From the repository root, using an environment with the packages in [requirements.txt](requirements.txt):

```bash
python Paper/make_paper_artifacts.py --validate-only
make -C Code report
```

This checks the imported current tables, writes five Markdown tables under `tables/generated/`, refreshes their marked copies inside `paper.md`, and renders eight PNG figures from the committed CSV files. No training, download, compiler, or large release bundle is needed. The report's narrative and equations are maintained manually; changing data does not silently rewrite their claims.

The listed package versions were used for the cleanup sanity check. PNG bytes may differ across plotting-library or font versions. Source CSV values and generated Markdown numbers are the scientific reproducibility targets; same-environment regeneration should reproduce the figures.

The legacy `make -C Code paper` and `paper-artifacts` commands are aliases for this Markdown workflow.

## Refresh imports from full releases

The importers retain their archive, manifest, protocol, cost, and statistical checks:

```bash
python Paper/import_stage4_aggregate.py
python Paper/import_transformer_publication.py
python Paper/import_real_lora_results.py
```

These operations need the original companion releases indexed by [RELEASE_ARTIFACTS.json](../RELEASE_ARTIFACTS.json). They are separate from regeneration from committed tables. Passing the lightweight artifact check does not mean absent raw archives or external model inputs were reverified.

## Historical artifacts

The current five generated tables are `stage4_rows.md`, `transformer_primary_rows.md`, `transformer_scaling_rows.md`, `transformer_sitewise_rows.md`, and `real_lora_rows.md`.

Other generated table summaries explicitly say **superseded**. They retain earlier values for traceability, not for current inference. The `fig_*` PNG files are also historical pre-control allocation/whitening figures and are not used in the current report. See [tables/TRANSFORMER_RESULTS_STATUS.md](tables/TRANSFORMER_RESULTS_STATUS.md). CSV and JSON evidence remains unchanged by the Markdown conversion. Historical source and compiled manuscript versions remain available in Git history.
