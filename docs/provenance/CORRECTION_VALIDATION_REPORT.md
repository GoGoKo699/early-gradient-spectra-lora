# Effective-rank correction validation report

## Status

The real GPT-2/Wikitext-2 effective-rank correction is complete. Entropy
effective rank is computed from normalized squared singular values:

```text
p_i = sigma_i^2 / sum_j sigma_j^2
effective_rank = exp(-sum_i p_i log(p_i))
```

All three allocation strategies were rerun together for all paper seeds. The
corrected tables, compact per-seed evidence, provenance, and manuscript are
included in this release.

## Rerun provenance

- Run identifier: `corrected_full_20260619_165707`
- Main suite: seeds 101, 103, 107, 109, 123
- Attention-output ablation: seeds 101, 103, 107
- Strategies per seed: uniform rank 4, gradient norm, spectral effective rank
- Total completed strategy runs: 24
- Model: `openai-community/gpt2` at revision
  `607a30d783dfa663caf39e06633721c8d4cfcd7e`
- Dataset: `Salesforce/wikitext`, `wikitext-2-raw-v1`, at revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`
- Runtime: Python 3.12.3, PyTorch 2.9.1 with ROCm 7.2.1
- Device recorded by the run: AMD Radeon Graphics

The complete package/environment record is in
`Code/real_lora_validation/results/corrected_full_runs/corrected_full_20260619_165707/RERUN_CONTEXT.json`.
Input file hashes and revisions are in `Code/real_lora_validation/INPUT_MANIFEST.json`.

## Independent validation performed during integration

- Verified all 13 nested `SHA256SUMS.txt` manifests: 157/157 entries passed.
- Verified the five recorded source-code hashes against the executed code.
- Verified every released aggregate row against the corresponding per-seed
  `results.csv` file.
- Independently recomputed means, sample SEMs, per-seed winners, pairwise
  differences, rank summaries, and trainable-parameter totals.
- Verified `val_loss_delta = final_val_loss - initial_val_loss` and
  `perplexity = exp(final_val_loss)` for every strategy run.
- Verified finite calibration, allocation, training-history, and result values.
- Verified rank bounds, parameter-cost identities, protocol fields, target
  module sets, seed sets, and matched per-strategy loader configuration.
- Regenerated the paper-facing CSV and LaTeX rows from released results.
- Ran project tests: transformer 5 passed; RMT 9 passed; real validation 7
  passed. Python compile checks passed.
- Rebuilt the 12-page manuscript twice with no undefined references, citation
  failures, overfull boxes, or Type 3/unembedded fonts.
- Rendered and visually inspected the corrected GPT-2 prose and table pages.

## Corrected result summary

### `c_attn,c_fc`, five seeds

| Strategy | Final validation loss, mean (SEM) | PPL | Mean parameters | Wins |
|---|---:|---:|---:|---:|
| Spectral effective | 3.4573 (0.0016) | 31.73 | 331,776 | 4/5 |
| Uniform rank 4 | 3.4587 (0.0017) | 31.78 | 331,776 | 1/5 |
| Gradient norm | 3.5047 (0.0014) | 33.27 | 330,086 | 0/5 |

The paired mean spectral-minus-uniform loss difference is -0.0015 with SEM
0.0013. The manuscript therefore treats this comparison as descriptive rather
than decisive. The paired mean spectral-minus-gradient difference is -0.0475
with SEM 0.0010.

### `c_attn,attn.c_proj,c_fc`, three seeds

| Strategy | Final validation loss, mean (SEM) | PPL | Mean parameters | Wins |
|---|---:|---:|---:|---:|
| Uniform rank 4 | 3.4479 (0.0006) | 31.44 | 405,504 | 3/3 |
| Spectral effective | 3.4530 (0.0026) | 31.60 | 404,992 | 0/3 |
| Gradient norm | 3.4847 (0.0014) | 32.61 | 405,248 | 0/3 |

This remains the reported module-family boundary condition.

## Superseded artifacts

The earlier real-model summaries based on normalized raw singular values are
retained only under
`Code/real_lora_validation/results/archive_raw_singular_value_effective_rank/`.
They are not imported by the paper build.

## Remaining project work

This report closes the effective-rank correction. Other publication issues found
in the earlier sanity check, including first-page and float-layout cleanup, are
separate tasks and are not claimed as fixed here.
