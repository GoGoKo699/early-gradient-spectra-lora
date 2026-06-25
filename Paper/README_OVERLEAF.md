# Overleaf package

Upload the contents of this `Paper/` directory to Overleaf.

Main file:

```text
paper.tex
```

Required subdirectories:

```text
figures/
tables/
```

The compiled reviewer-facing PDF is provided outside this directory as:

```text
../paper.pdf
```

To compile the committed tables and figures locally from the clean package:

```bash
cd Paper
make paper
```

This target does not require the large raw releases. `make paper-from-releases`
refreshes imported evidence first and therefore requires the external releases
listed in `../RELEASE_ARTIFACTS.json`.

Overleaf's rolling TeX environment may produce a visually equivalent but
byte-different PDF. The canonical byte-for-byte release gate is the
`../Dockerfile.publication` build documented in `../REPRODUCIBILITY.md`.
