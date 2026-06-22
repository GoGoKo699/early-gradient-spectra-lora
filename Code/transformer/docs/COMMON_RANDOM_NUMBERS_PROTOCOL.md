# Common-random-number protocol for allocation comparisons

## Status

This protocol replaces the rule-dependent seed design. Results produced before
this change must not be mixed with results produced after it.

## Comparison block

A budget comparison block is identified by:

```text
(base seed, task, parameter budget, adaptation replicate)
```

The allocation-rule name and rank-scaling policy are deliberately excluded.
Every rule and scaling condition in the same block receives the same:

- frozen base-model state;
- maximum-rank LoRA initialization bank;
- ordered training batches;
- validation batches; and
- dropout RNG stream.

Separate deterministic streams are derived for adapter initialization, training
data, evaluation data, and dropout. Their seeds are recorded in every raw result
row.

## Nested initialization

For each module and comparison block, one maximum-rank pair `(A_max, B_max)` is
created. A rank-`r` condition receives prefix slices:

```text
A = A_max[:r, :]
B = B_max[:, :r]
```

Thus shared coordinates are identical when rank changes. The existing zero-`B`
initialization is retained.

## Replication and inference

`protocol.adaptation_replicates` controls within-base-run adaptation repeats.
These repeats reduce adaptation noise but are not independent base-model runs.
The aggregation scripts therefore average adaptation repeats inside each base
run before computing standard errors across base runs.

Recommended publication configuration:

```yaml
protocol:
  adaptation_replicates: 3
  sweep_replicates: 1
```

Use at least five independent base seeds for confirmatory claims; more are
preferable when feasible.

## Runtime invariant

The runner groups result rows by scaling policy, budget, adaptation replicate,
and complete allocation signature. If two rule names produce the same
allocation under one scaling policy but yield
different metrics under the shared streams, the run aborts. This is a protocol
failure, not a result to average away.

## Required raw columns

`budget_results.csv` records:

- `protocol_version`;
- `adaptation_replicate`;
- `comparison_seed` and component stream seeds;
- `allocation_signature`;
- requested and realized budget/cost; and
- raw train/validation metrics.

## Acceptance checks

From `Code/transformer`:

```bash
PYTHONPATH=. python3 -m pytest -q tests
bash scripts/run_rank_scaling_smoke.sh
```

For every group with an identical `allocation_signature` inside one comparison
block, the loss and accuracy ranges must be zero within numerical tolerance.
Rank-zero conditions and all conditions at the declared common reference rank
are also required to produce identical metrics across scaling policies.
