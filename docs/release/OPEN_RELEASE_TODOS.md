# Open release TODOs

These items must be closed before public release or artifact submission:

- Select and add the final code license in `LICENSE`.
- Select and add the final data/evidence license in `LICENSE-DATA.md`.
- Add `CITATION.cff` with final author, title, version, and DOI metadata.
- Add `THIRD_PARTY_NOTICES.md` covering GPT-2, WikiText-2, PyTorch,
  Transformers, datasets, NumPy, pandas, SciPy, and bundled TeX/style files.
- Decide the Stage4 strategy: rebuild archive with `code_snapshot/` or publish
  a source sidecar.
- Apply the transformer scalar-canonicalization patch or pin pandas/numpy
  exactly in the validator environment.
- Insert final Zenodo software and data DOIs.
- Rebuild or re-export the final manuscript after wording edits.
- Run the Docker publication gate in a Docker-capable environment.
