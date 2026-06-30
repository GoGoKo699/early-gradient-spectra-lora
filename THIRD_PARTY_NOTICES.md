# Third-party notices

This project includes code, generated evidence, documentation, and references
to third-party software, models, datasets, and TeX tooling. This notice is
intended to identify major upstream resources; verify exact upstream license
versions before a formal public release.

## Models and datasets

- **GPT-2 model and tokenizer files.** Used only as checksum-pinned local inputs
  for the real-model validation. These files are not stored in the lightweight
  GitHub package. They are indexed by `INPUT_MANIFEST.json` and belong in the
  large data artifact.
- **WikiText-2.** Used as checksum-pinned local source/derived text files for
  validation. These files are not stored in the lightweight GitHub package.
  They are indexed by `INPUT_MANIFEST.json` and belong in the large data
  artifact.

Third-party model and dataset files remain governed by their upstream licenses
and terms. The project-generated manifests, tables, and validation outputs do
not change those upstream terms.

## Software dependencies

Major Python/runtime dependencies include:

- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- NumPy
- pandas
- SciPy
- Matplotlib
- PyYAML
- pytest
- threadpoolctl
- Pillow
- TeX Live / pdfTeX / LaTeX packages used for manuscript compilation

These dependencies are governed by their own upstream licenses. The lightweight
package records exact publication-toolchain versions in:

```text
Paper/requirements-publication.lock.txt
Paper/publication-toolchain.lock.json
```

## Project-generated materials

Unless otherwise stated, original code in this repository is released under the
MIT License. Original generated evidence, compact tables, plots, manifests,
audit metadata, and documentation authored for this project are released under
CC BY 4.0; see `LICENSE-DATA.md`.

## Exclusions

This notice does not grant rights to third-party resources beyond their upstream
terms. It also does not replace the license files for the project code or
project-generated data/evidence.
