# Effective-rank correction notes

## Corrected definition

The real GPT-2 validation now computes entropy effective rank from normalized
spectral energy:

```text
energy_i = sigma_i^2
p_i = energy_i / sum_j energy_j
effective_rank = exp(-sum_i p_i log(p_i))
```

The former real-validation implementation normalized raw singular values. Those historical real-GPT-2 tables have been replaced by the corrected full
rerun and are retained only under
`results/archive_raw_singular_value_effective_rank/` as explicitly superseded
artifacts.

## Comparison protocol

The corrected workflow reruns all three strategies for every paper seed. It does
not splice new spectral rows into old control rows. Each strategy also receives a
fresh DataLoader with the same seed, which matches the training mini-batch order
within a seed.

## Guardrails

- Unit tests cover the squared-singular-value formula, scale invariance, zero and
  invalid spectra, numerical extremes, and matched loader order.
- GPT-2 and WikiText-2 are downloaded at immutable revisions.
- WikiText parquet SHA-256 values and all prepared-file hashes are verified.
- The full runner requires a visible CUDA/ROCm GPU.
- Completed runs record package versions, `pip freeze`, hardware/runtime details,
  input manifests, code hashes, per-seed evidence, and output checksums.
- Resume is accepted only when the code, environment, hardware, and inputs match
  the original run context.


## Completed corrected result

The corrected publication rerun completed on June 19, 2026. In the
`c_attn,c_fc` setting, spectral effective rank obtained mean validation loss
3.4573, versus 3.4587 for uniform rank 4 and 3.5047 for gradient norm, and won
4/5 seeds. With `attn.c_proj` added, uniform rank 4 won 3/3 seeds; spectral
effective rank remained better than gradient norm. The paper reports these as
descriptive, limited real-model evidence.
