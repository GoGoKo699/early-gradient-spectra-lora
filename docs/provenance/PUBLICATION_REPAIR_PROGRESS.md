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

## Step 5 owner smoke verification

Status: **completed.**

The two-task smoke release passed recursive checksums, all four source-run
validators, both frozen-plan hashes, exact-cost pairing, common-random-number
invariants, and the publication release validator. Its numerical output was
correctly treated as non-evidential.

## Step 6 full transformer publication release

Status: **completed and independently verified.**

The full release contains two task families and five independent base-model
seeds per task, with three adaptation replicates per budget. All 350 recursive
checksums, ten source runs, 2,925 exact-cost candidate pairs, identical-
allocation invariants, and 38 bundled tests pass. The prespecified fixed-scale
soft-dimension comparison has mean loss delta `+0.014638`, 95% task-stratified
bootstrap interval `[-0.007751, +0.035215]`, exact two-sided sign-flip
`p=0.240234`, 3/10 run wins, and 0/2 task-family wins. This does not support the
former positive transformer-allocation claim.

## Step 7 manuscript synchronization and theory correction

Status: **implemented and validation-gated.**

Implemented:

- a transformer manuscript importer that verifies the archive and recursive
  manifests, rejects stale/smoke releases, reconstructs primary run effects
  from raw exact-cost rows, checks common-random-number invariants, and
  recomputes inference;
- paper-facing primary, task-specific, scaling-sensitivity, site-prediction,
  and run-level provenance tables;
- replacement of the invalid 33/39-win and whitening narratives with the
  prespecified ten-run null result and explicit scale/task heterogeneity;
- exclusion of all pre-CRN transformer files from current artifact generation;
- correction of the factorized gradient-flow equation, including the distinction
  between general zero-B dynamics and the balanced logistic special case; and
- `transformer-to-paper` build targets in both Makefiles.

## Remaining submission work after Step 7

1. Complete the Step 10 real-model publication-driver smoke and frozen 11-run rerun.
2. Import the corrected real-model result and finish related-work positioning.
3. Perform final rendered-PDF, metadata, archive, and root-checksum preflight.

## Step 8 — controlled real-model protocol and adaptive allocation controls

Status: **owner smoke release verified; superseded by the Step 9 learning gate.**

Implemented:

- evaluation-mode calibration with gradients retained and dropout disabled;
- named, independent model, adapter, data, dropout, and evaluation RNG streams;
- a training RNG reset after allocation-dependent adapter and optimizer
  construction;
- module-name-derived maximum-rank LoRA initialization with exact nested prefixes;
- exact trainable-parameter-cost allocation with explicit infeasibility errors;
- raw calibration-batch, activation-component, initialization, seed, allocation,
  and input-provenance artifacts;
- duplicate-uniform and generalized identical-allocation invariants;
- allocation-only EVA-style activation, FIM-LoRA-style LoRA-B-gradient, and
  GoRA-style weight-gradient sensitivity controls, all under the same common
  initialization and training protocol;
- source-run and release validators with exact recursive checksum coverage;
- deterministic portable release archives and SHA-256 sidecars; and
- 20 passing real-model protocol unit tests.

The owner smoke archive passed recursive checksum, source-run, and release
validation. It remains non-evidential: the Step 8 result schema did not preserve
first-step LoRA gradients or final adapter tensors, so identical printed losses
could not prove that the adapters had moved. Step 9 closes that instrumentation
gap before any long rerun.

## Step 9 — active-adapter gate

Status: **completed; owner activity-gated smoke release independently verified.**

Implemented and verified:

- first-step full-adapter and LoRA-B gradient norms for every strategy;
- saved final adapter-state artifacts and serialization-independent tensor hashes;
- independent reconstruction of the nested zero-B initial state;
- exact recomputation of parameter movement and the effective low-rank update;
- hard rejection of any inactive adapter or identical allocation that diverges
  in training trace, final state, or evaluation result; and
- positive gradients, parameter movement, and effective updates for all seven
  strategies in the owner smoke archive.

## Step 10 — frozen real-model publication driver

Status: **implemented and locally validated; owner four-source driver smoke
release required.**

Implemented:

- committed, checksum-bound four-source publication-driver smoke and 8+3 full
  analysis plans;
- eight primary GPT-2/WikiText-2 runs, allowing an exact two-sided sign-flip
  p-value below 0.05 even with up to two exact ties, plus a separate three-run
  descriptive attention-output-projection boundary suite;
- allocation-only gradient-norm, EVA-style, FIM-style, and GoRA-style secondary
  controls under exact parameter-cost matching;
- deterministic bootstrap seeds derived only from the committed plan hash;
- exact sign-flip inference on nonzero paired differences, t and bootstrap
  intervals, attainable p-value reporting, and within-suite Holm adjustment;
- immutable SHA-256-bound GPT-2 and WikiText-2 input specifications;
- a resumable offline child-process driver that strips proxy variables only in
  child processes and never changes the parent shell or VPN configuration;
- quarantine of invalid partial source runs before safe regeneration; and
- multi-source release construction with independent semantic recomputation of
  source bindings, raw aggregate tables, paired effects, inference, Git/input
  provenance, and exact checksum coverage.

The long 11-run experiment remains blocked until the four-source Step 10 smoke
archive passes independent verification.

## Step 10 owner design-smoke verification

Status: **completed.**

The four-source driver smoke passed all source-run and release checksums,
exact-cost and stochastic controls, identity controls, adapter-activity gates,
and independent aggregate recomputation. Its numerical output was correctly
classified as non-evidential.

## Step 11 full real-model publication release

Status: **completed and independently verified.**

The frozen release contains eight primary `c_attn,c_fc` runs and three separate
`c_attn,attn.c_proj,c_fc` boundary runs. All 354 root checksums, nine aggregate
checksums, 11 source-run manifests, 77 result rows, 2,100 allocation rows, and
55 paired candidate effects pass independent reconstruction. Every strategy is
exact-cost matched, every adapter passes the activity gate, and each duplicate
uniform control is identical in assignment, training trace, final adapter
state, and metrics.

For the prespecified primary comparison, `spectral_effective - uniform_r4`
final validation loss is `-0.007025766`, with deterministic bootstrap 95%
interval `[-0.008342654, -0.006025875]`, exact two-sided sign-flip
`p=0.0078125`, within-suite Holm-adjusted `p=0.0390625`, and 8/8 wins. The
three-run attention-output boundary reverses direction (`+0.001087166`, 0/3
wins) and has exact-test resolution only `0.25`.

## Step 12 checksum-bound real-model manuscript import

Status: **implemented and validation-gated.**

Implemented:

- recursive archive, aggregate, and source-run verification in the manuscript
  importer;
- independent reconstruction of all paired effects and prespecified inference;
- immutable paper-facing tables with release, plan, protocol, archive, and Git
  provenance columns;
- removal of superseded pre-v3 real-model tables from manuscript inputs;
- exact-cost comparison against uniform, gradient norm, and allocation-only
  EVA-, GoRA-, and FIM-LoRA-style controls;
- manuscript synchronization with the positive restricted-scope primary result
  and the reversed module-family boundary; and
- artifact-generation gates that reject stale, mixed, or modified real-model
  tables.

## Remaining submission work after Step 12

1. Apply and validate the manuscript-import patch on the owner machine.
2. Perform the final rendered-PDF, metadata, archive, and root-checksum preflight.
3. Regenerate `SHA256SUMS.txt` and the top-level submission manifest only after
   all final artifacts are frozen.
