# Spectral-edge, evaluation, and budget-fairness protocol

## Permutation edge

For each calibrated module, the runner saves the complete vector of top singular
values obtained after entry permutation of the whitened early-gradient matrix.
The primary edge is the configured empirical quantile of this vector.
`calibration/module_stats.csv` records:

- `null_bootstrap` and `null_quantile`;
- `edge`;
- `edge_mc_ci_low` and `edge_mc_ci_high`, obtained by resampling the null
  maxima; and
- sensitivity values at quantiles 0.99, 0.995, and 0.999.

The raw maxima are released in `calibration/null_maxima.npz`; summary columns are
not a substitute for that source evidence. Calibration is performed with
model dropout disabled. Calibration examples and permutation streams are named,
separate random streams and are recorded per module.

The lightweight smoke configuration uses 16 permutations only to exercise the
pipeline. It must not be used for a numerical claim. A publication run must use
at least 4,096 permutation maxima per module, retain the 0.995 primary quantile,
and use at least 2,000 uncertainty resamples. Whitening and quantile sensitivity
results remain secondary analyses.

## Exhaustive modular evaluation

When `protocol.exact_modular_evaluation: true`, validation enumerates every
ordered pair `(a,b)` in the modular task exactly once. For modulus `p`, each
reported validation metric therefore uses exactly `p^2` labeled examples rather
than a random validation sample. Batch boundaries do not affect the mean loss:
losses are weighted by the number of labeled examples in each batch.

Associative-recall evaluation remains a fixed, pre-generated sample because its
input space is much larger. Its evaluation seed and sample count are released.

## Budget cap versus realized cost

Every budget row reports both:

- `requested_budget`, the upper cap; and
- `actual_cost`, the exact number of trainable LoRA parameters realized by the
  discrete rank vector.

A cap comparison against `uniform_fill` is retained for continuity, but it is
not an exact-cost comparison. For every rule named in
`budgets.cost_matched_rules`, the runner constructs a deterministic
`uniform_exact_cost` allocation with exactly the candidate's realized cost.

Among feasible rank vectors at that exact cost, the comparator minimizes

```text
n * sum_i r_i^2 - (sum_i r_i)^2,
```

which is the sum of pairwise squared rank differences. Equal-objective
solutions use a deterministic lexicographic site/rank tie break. This makes the
comparator as uniform in nominal rank as the discrete rank grid and
heterogeneous module costs permit.

The source rows identify exact-cost comparators through `condition_id`,
`comparison_role`, `matched_cost`, `matched_to_rules`,
`exact_cost_match_required`, and `matched_baseline_condition_id`. The
pre-specified candidate is marked by `is_primary_allocation_rule`. Publication
analysis must report exact-cost paired deltas and performance as a function of
realized cost. Win counts under a common cap are descriptive only.

## Acceptance checks

`validate_rank_scaling_run.py` recomputes each allocation cost from module
shapes, verifies that trainable parameter counts agree, verifies every declared
candidate has exactly one same-cost comparator, checks exhaustive modular sample
counts, and checks the raw null-array lengths. A run that fails any check must
not enter an aggregate.
