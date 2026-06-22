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

Status: **completed; five-seed release validated and imported.**

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

The bundled legacy Stage4 aggregate is explicitly superseded. The corrected
five-seed evidence release is the manuscript source; the legacy numerical
claims must not be reused.

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

## Step 3 — corrected Stage4 evidence import

Status: **completed by the project owner.**

The validated five-seed Stage4 release was imported into the manuscript and the
paper-facing Stage4 artifacts were rebuilt and committed. Root release checksums
remain intentionally deferred until all scientific repairs are complete.

## Step 4 — corrected synthetic-transformer experiment protocol

Status: **implementation and local end-to-end release smoke complete; owner
smoke verification and full evidence rerun still required.**

Implemented:

- explicit `standard`, `fixed_update_scale`, and reference-matched `rslora`
  policies, with fixed update scale pre-specified as the primary mechanistic
  condition;
- common initialization, training data, exhaustive/fixed evaluation data, and
  dropout streams across scaling conditions and allocation rules;
- exact per-module scale/effective-alpha provenance and scaling-stratified
  useful-rank targets, fits, aggregates, whitening summaries, and figures;
- exhaustive evaluation of all modular-arithmetic input pairs;
- raw permutation-null maxima, edge Monte Carlo intervals, and 0.99/0.995/0.999
  quantile sensitivity fields;
- a publication requirement of at least 4,096 null maxima per module, replacing
  the former eight-sample edge;
- same-realized-cost deterministic uniform comparators for pre-specified rules,
  explicit candidate-to-baseline links, pre-specified primary-analysis flags,
  and explicit cap utilization;
- portable raw-run validation, code/environment snapshots, recursive SHA-256
  manifests, archive sidecars, and tamper-detecting release validation; and
- 22 passing transformer unit tests plus a complete local packaged smoke.

The smoke configuration is only an execution gate; its metrics are not evidence.
Do not update the manuscript or run the expensive multi-seed study until the
owner-generated smoke archive has been independently verified. The next gate is
the pre-specified multi-task publication driver and run-cluster inference.
