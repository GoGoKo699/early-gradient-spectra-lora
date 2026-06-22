# Reference-matched LoRA rank-scaling protocol

## Scientific role

Changing a LoRA rank under standard scaling also changes the multiplier applied
to the factor product. A rank comparison can therefore mix two effects:

1. the number of available low-rank directions; and
2. the update scale used to optimize those directions.

The corrected synthetic-transformer protocol treats scaling policy as an
explicit experimental factor. It does not pool outcomes or useful-rank targets
across scaling policies.

## Conditions

Let `alpha_ref` be the alpha used by the original standard experiment and let
`r_ref` be a declared reference rank. Define

```text
s_ref = alpha_ref / r_ref.
```

For a module assigned rank `r > 0`, the runner implements:

```text
standard:           s(r) = alpha_ref / r
fixed_update_scale: s(r) = s_ref
rslora:             s(r) = s_ref * sqrt(r_ref / r)
```

The adapter update is always

```text
Delta W = s(r) * B @ A.
```

The last condition is a reference-matched rsLoRA-style sensitivity analysis: it
has the `1/sqrt(r)` dependence of rank-stabilized scaling, but is normalized to
match the other two conditions at `r_ref`. All three therefore have exactly the
same multiplier at the declared reference rank.

The CSV column `effective_alpha` is `r * s(r)`. It is the per-module alpha that
would produce the same multiplier in an `alpha/r` implementation. The input
configuration value is recorded as `reference_alpha`; it must not be interpreted
as the per-module alpha in the fixed or rsLoRA-style conditions.

## Pre-specified analysis role

For the corrected publication experiment:

```yaml
lora:
  alpha: 16.0
  scaling_modes: [fixed_update_scale, standard, rslora]
  scale_reference_rank: 8
protocol:
  primary_scaling_mode: fixed_update_scale
```

`fixed_update_scale` is the primary mechanistic condition because changing rank
does not change the adapter multiplier. `standard` measures performance under
the conventional LoRA policy. `rslora` is a sensitivity condition. Claims and
figures must label these roles and must not select the primary condition after
observing the outcomes.

## Randomization

Scaling mode is excluded from the comparison seed. For a fixed task, budget,
adaptation replicate, allocation, and rank, all scaling conditions receive the
same:

- frozen base model;
- maximum-rank initialization bank;
- ordered training batches;
- fixed validation batches; and
- dropout RNG stream.

Only the scalar multiplier differs. Allocation rules are likewise excluded from
the comparison seed.

## Outputs

The single-site and allocation CSVs record:

- `scaling_mode`;
- `is_primary_scaling_mode`;
- `reference_alpha`;
- `scale_reference_rank`;
- `scale_at_reference_rank`;
- `effective_alpha`; and
- `lora_scale`.

Budget-level rows additionally record the minimum, maximum, and mean active-site
multiplier. Exact per-site values are in `budget/allocation_comparison.csv`.
Useful-rank targets, prediction fits, paired deltas, winners, whitening
summaries, and plots are all stratified by `scaling_mode`.

## Acceptance test

From `Code/transformer`:

```bash
PYTHONPATH=. python3 -m pytest -q
bash scripts/run_rank_scaling_smoke.sh
```

The smoke validator checks formulas, reference-rank scale and metric identity,
common random streams across scaling modes, separation of target estimates,
pre-specified primary flags, and the identical-allocation invariant within each
scaling policy.
