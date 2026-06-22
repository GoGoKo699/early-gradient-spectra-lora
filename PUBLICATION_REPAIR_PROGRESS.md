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

1. Run the full corrected synthetic-transformer experiment and replace all
   allocation, whitening, and win-count claims.
2. Correct the factorized gradient-flow statement in the appendix.
3. Complete the real-model stochastic-control rerun.
4. Update practical baselines and related work.
5. Finish release metadata, manuscript claim narrowing, and rendered-PDF review.

## Step 3 — corrected Stage4 evidence import

Status: **completed by the project owner.**

The validated five-seed Stage4 release was imported into the manuscript and the
paper-facing Stage4 artifacts were rebuilt and committed. Root release checksums
remain intentionally deferred until all scientific repairs are complete.

## Step 4 — corrected synthetic-transformer experiment protocol

Status: **owner smoke archive independently verified; full evidence rerun still
required.**

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

The owner-generated smoke archive passed its recursive checksums, source-run and
release validators, all 22 bundled tests, exact-cost links, common-randomness
invariants, and independent recomputation. Its Git provenance recorded an
untracked `Paper/__pycache__/`; the next patch adds root ignore rules and the
full driver requires a clean committed worktree. The smoke configuration is
only an execution gate; its metrics are not evidence.

## Step 5 — pre-specified multi-task publication driver

Status: **implemented and locally test-validated; owner multi-task smoke gate
required before the full experiment.**

Implemented:

- immutable full and smoke analysis plans with task families, seeds, primary
  comparison, independent unit, budget weighting, confidence level, bootstrap
  count, and exact-test direction declared before execution;
- a publication design gate requiring two task families, five independent
  seeds per task, all three scaling policies, exact-cost comparators, three
  adaptation replicates, exhaustive modular evaluation, 4,096 permutation
  maxima, and 2,000 null-edge uncertainty resamples;
- a clean-Git requirement and root cache/build ignore rules;
- deterministic generated configs bound to the plan version, plan SHA-256, and
  release identifier;
- exact source-run enumeration, per-run validation, duplicate task/seed
  rejection, and validator-gated resumption;
- task-stratified run-cluster inference after averaging adaptation replicates
  and pre-specified budgets within each independent run, followed by equal
  weighting of the two pre-specified task-family means;
- a confirmatory omnibus `primary_analysis.csv`, released underlying
  `primary_run_deltas.csv`, and task-specific secondary
  `primary_analysis_by_task.csv`;
- a within-task cluster bootstrap and exact task-stratified sign-flip test over
  all ten task/seed runs, with pre-specified alpha 0.05 and all 1,024 sign
  patterns enumerated, avoiding the `p >= 0.0625` resolution floor of a
  separate two-sided five-run test;
- independent reconstruction of confirmatory run deltas from each released raw
  exact-cost budget table before any primary statistic is accepted;
- versioned aggregate/release schemas and manifests bound to the copied
  analysis-plan SHA-256; and
- complete driver/script/config inclusion in the immutable code snapshot.

All 33 transformer tests and both plan-only validation gates pass locally. The
two-task smoke release must now be generated on the owner machine and verified
before launching the ten source-run publication experiment.
