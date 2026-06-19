# Publication repair progress

## Step 1 — synthetic-transformer comparison protocol

Implemented in this patch:

- common random numbers across allocation rules;
- fixed training and validation batches;
- separate model/adapter/data/evaluation/dropout streams;
- nested maximum-rank LoRA initialization;
- within-base-run adaptation replicates;
- run-level aggregation that does not treat nested repeats as independent; and
- an abort-on-failure invariant for identical allocations.

All previously released synthetic-transformer allocation and whitening results
are superseded. They must be rerun before the manuscript's allocation claims or
win counts are restored.

## Remaining submission blockers

1. Reconcile the Stage4 useful-rank estimand between paper and code, then rerun
   all source seeds and release per-layer/per-rank raw data.
2. Add scaling-controlled transformer conditions before making a mechanistic
   rank-demand claim.
3. Correct the factorized gradient-flow statement in the appendix.
4. Complete the remaining statistical, spectral-edge, budget-matching,
   real-model stochastic-control, baseline, breadth, and release-engineering
   repairs listed in the publication audit.
