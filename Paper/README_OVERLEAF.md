# Overleaf Package

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

The compiled PDF is provided outside this directory as:

```text
../paper.pdf
```

To rebuild locally from the clean package root:

```bash
cd Paper
make paper
```

The paper-generation scripts read released CSV artifacts from the sibling `../Code/` directory.
