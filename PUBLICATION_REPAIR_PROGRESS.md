# Publication repair progress

## Step 1 — synthetic-transformer comparison protocol

Status: **smoke-validated; full evidence rerun still required.**

Implemented and independently verified in the uploaded smoke artifact:

- common random numbers across allocation rules;
- fixed training and validation batches;
- separate model, adapter, data, evaluation, and dropout streams;
- nested maximum-rank LoRA initialization;
- within-base-run adaptation replicates;
- run-level aggregation that does not treat nested repeats as independent; and
- an abort-on-failure invariant for identical allocations.

Duplicate allocations produced exactly identical metrics in the smoke release.
All earlier synthetic-transformer allocation and whitening tables remain
superseded until the full corrected experiment is rerun.

## Step 2 — Stage4 useful-rank estimand and provenance

Status: **code/test repair complete; end-to-end smoke and five-seed rerun
required.**

Implemented:

- one canonical observed-best target implementation shared by producer and
  analyzer;
- explicit exclusion of oracle loss from target thresholds and penalties;
- undefined-target handling for nonpositive observed improvement;
- smallest-rank tie breaking;
- target-estimand metadata in source, fit, aggregate, and paper-facing tables;
- independent producer/analyzer agreement checks;
- complete per-rank source-run packaging;
- portable source paths, source hashes, recursive checksum manifests, and an
  independent release validator;
- a paper importer that rejects legacy, mixed, smoke, fast, or fewer-than-five-
  seed aggregates; and
- manuscript equations aligned with the executable estimand.

The bundled legacy Stage4 aggregate is now explicitly superseded. Its numerical
claims must not be used. Run `Code/rmt_lora_sim/scripts/run_stage4_smoke.sh`
before starting the five-seed publication run.

## Remaining submission blockers after Step 2

1. Run and validate the full corrected Stage4 five-seed release, then update all
   Stage4 numerical claims and the compiled manuscript.
2. Run the full corrected synthetic-transformer experiment and replace all
   allocation, whitening, and win-count claims.
3. Add scaling-controlled transformer conditions before making a mechanistic
   rank-demand claim.
4. Correct the factorized gradient-flow statement in the appendix.
5. Complete the remaining statistical, spectral-edge, budget-matching,
   real-model stochastic-control, baseline, breadth, and release-engineering
   repairs listed in the publication audit.
