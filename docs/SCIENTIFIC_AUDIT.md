# Scientific sanity check

Audit date: 2026-09-22. Baseline: `5bf03dc1c5472b9297208a2df5da8bb44854f5da`.

This audit reviewed the original manuscript, the current experiment and analysis code, the compact evidence, and selected primary literature. It independently recomputed the main Stage4 associations and transformer paired statistics. It did not rerun training, download the full raw releases, reproduce bootstrap intervals, or complete an exhaustive novelty review.

**Assessment:** the compact results are internally consistent, and the single-site population calculation is correct under its stated model. The strongest defensible research story is a controlled study of when spectral rank diagnostics do and do not transfer to allocation. The original presentation overstated the exactness of one allocation argument and did not make the classical nature of the population result sufficiently prominent. Cleaning up the repository does not establish publication novelty or broad practical usefulness.

## 1. Results independently recomputed

All loss differences below are candidate minus uniform; negative favors the candidate.

| Evidence | Independent calculation | Result | Interpretation |
| --- | --- | --- | --- |
| Stage4 hard-knee | Refit the specified transformed one-predictor regression separately within each seed; recompute Spearman with average ranks for ties and leave-one-layer-out predictions | Mean in-sample R² 0.7766385244; mean Spearman 0.8515174799; mean leave-one-out R² 0.7546958834 | Strong association and within-family predictive performance |
| Stage4 sample-limited | Same calculation | Mean in-sample R² 0.6869321809; mean Spearman 0.7857586671; mean leave-one-out R² 0.6617029342 | Association and within-family prediction persist in the specified noisier family |
| Synthetic transformer | Average the 10 released task–seed effects; enumerate all 1,024 sign assignments | Mean loss difference +0.01463835418; two-sided p = 0.240234375; 3/10 wins | No demonstrated advantage under the primary condition |
| GPT-2 primary suite | Reconstruct spectral-minus-uniform differences directly from strategy results for eight seeds; enumerate all 256 sign assignments | Mean loss difference −0.007025766186; two-sided p = 0.0078125; 8/8 wins | Small conditional gain in the fixed primary protocol |
| GPT-2 boundary suite | Reconstruct three paired differences; enumerate all eight sign assignments | Mean loss difference +0.001087166369; two-sided p = 0.25; 0/3 wins | Descriptive reversal, with very limited inferential resolution |

The Stage4 calculation used all 480 layer-summary rows: 48 simulated layers in each of ten condition–seed runs. The reported regression uses `log2(rank)` as the response and `log2(1 + score)` as the predictor, with an intercept. The headline in-sample fit is assessed on the same 48 rows used for fitting. The separate leave-one-out calculation refits on 47 rows and predicts the omitted layer, repeats this for all 48 layers, and computes R² from the held-out residuals against the full within-run target sum of squares. Run-level scores are then averaged across five seeds. The in-sample statistics agree with the compact summary tables to floating-point precision; the leave-one-out means reproduce the original manuscript's rounded values of 0.7547 and 0.6617.

The real-model calculation used the 77 strategy-result rows: seven strategies, including a duplicate-uniform control, in 11 runs. Spectral and uniform trainable-parameter counts match within every run. Duplicate-uniform final losses match the corresponding uniform losses. These aggregate checks do not independently verify the underlying adapter states or training traces.

Source files:

- `evidence/stage4/aggregate/stage4_all_layer_summaries.csv`
- `evidence/stage4/aggregate/stage4_all_fits.csv`
- `evidence/stage4/aggregate/stage4_key_table.csv`
- `evidence/transformer/aggregate/primary_run_deltas.csv`
- `evidence/real_lora/aggregate/all_results.csv`
- `evidence/real_lora/aggregate/primary_seed_deltas.csv`

## 2. Theory: valid foundation, limited novelty

The population model assumes a fixed linear teacher, squared loss, finite second moments, zero conditional mean noise, and positive-definite activation covariance C. Write the activation-weighted task update as `M = Delta_star @ sqrt(C)`.

Under these assumptions:

1. Excess risk is half the squared Frobenius norm of the error in activation-weighted coordinates.
2. Multiplication by the invertible square root of C preserves rank.
3. A truncated singular-value decomposition therefore gives a rank-constrained optimum, and the residual risk is half the discarded squared singular-value sum.
4. The full update gradient at the frozen model is `G0 = -Delta_star @ C`; right multiplication by the inverse square root of C recovers `-M`.

This derivation is correct. It is a direct application of classical low-rank approximation and reduced-rank regression, followed by differentiation of quadratic loss. It should be presented as the mathematical foundation of the study, not as a newly established general law of LoRA. Repeated singular values can make the optimizer nonunique. Singular covariance requires a separate support/pseudoinverse statement and is outside the theorem as written.

The complete spectrum determines the optimal population risk curve. Its entropy effective-rank summary does not uniquely determine that curve or every recovery-rank threshold. Association of this scalar with an empirical useful-rank target is an additional empirical observation.

The repository uses **energy-entropy effective rank**, computed from normalized squared singular values. This definition must remain explicit because other literature uses entropy of normalized singular values instead.

The factor-gradient equations are valid for the explicitly whitened quadratic objective. A change of coordinates preserves the rank-constrained optimum but generally changes Euclidean gradient-descent dynamics. The whitened-coordinate equations are not automatically the dynamics of the original factors with anisotropic activations. The balanced scalar logistic equation is a special case; it is not the zero-B initialization trajectory.

## 3. Confirmed allocation overclaim

The original manuscript describes selecting singular directions by benefit divided by parameter cost and calls the population interpretation exact. A direction's single-site benefit is exact in the linear model. Global integer allocation requires additional qualifications.

Even with independent sites and additive risks, sorting by benefit/cost is not generally optimal when rank increments have different costs:

| Available direction | Parameter cost | Risk reduction | Reduction per cost |
| --- | ---: | ---: | ---: |
| A | 2 | 3 | 1.5 |
| B | 3 | 4 | 1.333… |

With budget 3, density ordering takes A and obtains reduction 3. The optimum takes B and obtains reduction 4. A single ratio threshold cannot select B while excluding A.

Exact integer allocation is a budget-constrained discrete optimization problem; nested singular directions add prefix constraints. Sorting individual gains is exact for additive sites with equal increment costs and an unrestricted consecutive rank grid. Density ordering characterizes an appropriate fractional relaxation. A sparse allowed rank grid introduces grouped increments even when underlying per-rank costs agree.

For transformer modules, additivity itself is unproved: modifying one module changes the inputs and downstream effects of others. The valid single-site theorem therefore does not prove global optimality of the implemented transformer allocator.

## 4. What the empirical evidence establishes

**Stage4 includes within-family prediction checks.** The headline scores are in-sample, and the manuscript also reports leave-one-layer-out R² values, reproduced above. These checks hold out one of the 48 simulated layers while fitting on the other 47 within the same seed and task family. They support prediction within the constructed families. They do not evaluate one frozen calibration on independent unseen task families or establish transfer to pretrained transformer modules. The target is the smallest tested rank near the best observed validation loss on a finite rank grid. It depends on the optimization protocol, evaluation noise, and available ranks. It is not an intrinsic or globally optimal rank. Near-best and recovery targets with complementary thresholds are algebraically equivalent; those columns are not independent replications.

**The synthetic-transformer primary result is null.** Its interval includes zero and its point estimate favors uniform allocation. This does not prove equivalence or rule out all benefits. Budget rows and adaptation replicates are nested within task–seed runs and must not be counted as independent samples. The inference is conditional on two named task families.

**The GPT-2 gain is small and narrow.** The primary mean log-loss difference corresponds to approximately 0.70% lower perplexity as a geometric ratio. It concerns one checkpoint, one dataset, selected modules, fixed scale, and 200 training steps. The evaluation loader uses a fixed validation prefix under the capped evaluation protocol. Seeds vary training randomness; they do not supply independent datasets, model families, or a fresh held-out benchmark.

**The boundary failure must remain visible.** Adding attention output projections changes the observed direction. Three seeds are insufficient for a conventional two-sided 5% exact sign-flip test: its smallest attainable p-value is 0.25. A small-sample bootstrap interval excluding zero does not remove this limitation. The suite is a scope warning, not a powered test of a module-family interaction.

**The named controls are score-family controls.** EVA-, GoRA-, and FIM-LoRA-style allocation rows share the repository's initialization and training protocol. They are not full reproductions of those methods and cannot establish superiority over the complete published algorithms. A lower mean than another score control also does not itself establish a statistically resolved pairwise difference.

## 5. Inference and provenance limits

The enumerated sign-flip p-values are exact for the assumed sign-exchangeability null; independence of runs alone is not sufficient to justify that null. The bootstrap intervals summarize variation among the available runs. Neither procedure accounts for benchmark selection, historical tuning, or transfer to other datasets.

The released primary GPT-2 p-value and the within-suite Holm-adjusted value of 0.0390625 refer to the recorded comparison family. That adjustment does not cover arbitrary earlier experiments or later-selected comparisons. Several unrelated candidate comparisons have the same minimum exact p-value because all eight differences share a sign, including consistently harmful strategies; the sign and effect size must accompany p-values.

Analysis plans, recorded source revisions, and checksums are useful integrity controls. They do not independently prove that hypotheses were specified before any relevant pilot outcomes were seen. The provenance directory records earlier corrected studies of related suites. The current material supports the phrase **frozen-protocol analysis**; independent preregistration or a clean separation from prior exploration has not been established by this audit. Original machine-readable fields such as `confirmatory_primary` are historical plan labels, not new audit certification.

Empirical ridge whitening and diagonal whitening are also different from exact population whitening. The singular-value perturbation inequality is valid conditional on an error bound, but the study does not supply a calibrated finite-sample guarantee incorporating covariance estimation, regularization bias, nonlinear loss geometry, and the final useful-rank decision. The nonlinear experiments should retain their empirical status.

## 6. Targeted primary-literature check

The following neighboring works and identifiers were verified on primary publication or author-submitted pages. This is a targeted check, not an exhaustive novelty clearance.

| Work | Relevance to the repository |
| --- | --- |
| [Eckart and Young, 1936](https://doi.org/10.1007/BF02288367) | Classical optimal low-rank matrix approximation underlying the population proof |
| [Izenman, 1975, reduced-rank regression](https://www.sciencedirect.com/science/article/pii/0047259X75900421) | Classical statistical setting for rank-constrained multivariate regression |
| [LoRA-One, ICML 2025](https://proceedings.mlr.press/v267/zhang25ax.html) | One-step full-gradient singular subspaces already guide low-rank adaptation |
| [EVA, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/41d33bd41fd44bd9dba0e092047cf213-Abstract-Conference.html) | Activation-spectrum initialization and adaptive rank allocation |
| [GoRA, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a5e4907a40c0dcb8433a35c714ba9d79-Abstract-Conference.html) | Gradient-driven rank adaptation and initialization |
| [FIM-LoRA, arXiv:2605.16800](https://arxiv.org/abs/2605.16800) | Calibration-time gradient-variance/empirical-Fisher rank allocation |
| [ShapLoRA, arXiv:2601.17921](https://arxiv.org/abs/2601.17921) | Sensitivity-based allocation with a validation and retraining protocol |

The existence of these works rules out broad novelty claims such as being the first to use pre-training gradients or activation spectra to guide LoRA ranks. A narrower contribution would need a precise comparison of assumptions, estimands, algorithms, and guarantees.

## 7. Remaining research decisions

Before presenting this as a new broadly useful method, establish a distinct contribution relative to the neighboring work; evaluate a frozen predictor on independent unseen task families beyond the existing within-family leave-one-out check; and distinguish a module's attainable gain from the rank needed to realize that gain. Any further real-model claim would require a separately justified validation design, including the observed boundary failure. These are research tasks, not issues that a format conversion or passing unit tests can resolve.

No additional training or external publication was performed as part of this audit. The compact evidence remains a valuable record of positive, null, and boundary results within its stated scope.
