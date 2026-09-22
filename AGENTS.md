# Working on this research repository

- Read `README.md`, `docs/SCIENTIFIC_AUDIT.md`, and `REPRODUCIBILITY.md` first.
- Maintain research prose, derivations, references, and generated tables in
  Markdown with readable plain or Unicode notation. Do not add a document
  compiler, manuscript PDF, or TeX source. Keep executable code and structured
  data in their native formats.
- Preserve `LICENSE`, `LICENSE-DATA.md`, recorded raw evidence, input identities,
  and historical release hashes. Corrections to an analysis must be explicit;
  do not overwrite a historical result to make a check pass.
- Distinguish mathematical identities, synthetic associations, protocol-specific
  empirical results, and untested claims. Do not claim new theory from classical
  truncated SVD, or universal allocation gains from the restricted GPT-2 result.
- Keep the synthetic null and module-boundary reversal visible. Name the
  independent experimental unit and distinguish allocation-score controls from
  full reproductions of published methods.
- A stored plan is not proof of prospective registration. A checksum is not
  proof of scientific validity. Unit tests are not model retraining.
- Run `python verify.py` and the relevant test suite after changes. Refresh
  `SHA256SUMS.txt` only after reviewing the intended file changes. Its coverage
  is the maintained tracked files other than the checksum manifest itself.
- Use small local checks for maintenance. Full training, external uploads, and
  new releases require task scope that calls for them.
