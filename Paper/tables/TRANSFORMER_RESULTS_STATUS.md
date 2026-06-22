# Transformer results status

## Current publication evidence

The only transformer inputs used by the manuscript are under
`tables/transformer_publication/`. They were imported from:

- release: `transformer_publication_20260622T101458Z`
- archive SHA-256: `b1deb63807b04a4cd1032a7152d9664ff4b47929dd68ce99afc4317d500482e3`
- frozen analysis-plan SHA-256: `89c74a8d3e773cd20950bf2a9854bcdf39cd6f0a38a8eed78ca0d3345c85be6d`
- protocol: `synthetic_transformer_publication_protocol_v4`

`../import_transformer_publication.py` verifies the recursive checksums and
archive sidecar, rebuilds the confirmatory run effects from raw source rows,
checks exact-cost and common-random-number invariants, and recomputes the
confirmatory statistics before writing these paper-facing files.

## Superseded material

The older transformer CSVs directly under `tables/`, the old generated files
`transformer_allocation_rows.tex`, `transformer_cluster_rows.tex`,
`by_task_transformer_rows.tex`, and `whitening_rows.tex`, and the legacy PNG
figures beginning with `fig_` were generated before the corrected randomization,
exact-cost, and scaling-control protocol. They are retained only for audit
history. They are not called by `make_paper_artifacts.py`, are not referenced by
`paper.tex`, and must not be used for publication claims.
