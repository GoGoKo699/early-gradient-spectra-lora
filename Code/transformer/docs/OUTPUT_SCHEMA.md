# Output schema

## `base/base_metrics.csv`

Base-model metrics before LoRA adaptation. Every raw CSV records the shared
`protocol_version`.

Columns include:

- `train_loss_last`
- `base_task_loss`, `base_task_accuracy`
- `adapt_task_loss_before_lora`, `adapt_task_accuracy_before_lora`

## `calibration/module_stats.csv`

One row per candidate module site.

Important columns:

- `site_name`
- `d_in`, `d_out`
- `n_tokens`, `n_batches`
- `whitening`, `lambda_scale`
- `gradient_norm`
- `cov_trace`, `cov_effective_rank`
- `edge`, `edge_mc_ci_low`, `edge_mc_ci_high`
- `edge_q_0_99`, `edge_q_0_995`, `edge_q_0_999`
- `null_bootstrap`, `null_quantile`
- `top_sv`
- `frobenius`, `nuclear`
- `effective_rank`
- `stable_rank`
- `hard_detectable_rank`
- `soft_dimension`

## `calibration/null_maxima.npz`

One raw permutation-null maximum vector per candidate site. Array keys replace
`.` with `__`; vector length must equal `null_bootstrap`.

## `sweeps/site_rank_sweep_metrics.csv`

One row per single-site LoRA rank run.

Important columns:

- `site_name`
- `rank`
- `scaling_mode`, `is_primary_scaling_mode`
- `reference_alpha`, `scale_reference_rank`, `scale_at_reference_rank`
- `effective_alpha`, `lora_scale`
- `diverged`
- `final_train_loss`
- `final_val_loss`
- `final_val_accuracy`
- `trainable_params`
- `evaluation_examples`, `evaluation_batches`

## `sweeps/site_target_summary.csv`

Useful-rank targets computed from each site's validation curve. Curves from different `scaling_mode` values are never pooled.

Important columns:

- `best_rank`
- `near_best_rank_gap_*`
- `recovery_rank_*`
- `penalized_rank_lambda_*`

## `sweeps/site_prediction_fit.csv`

Spearman correlations between early-gradient spectral predictors and useful-rank targets.

## `budget/budget_results.csv`

One row per allocation rule and parameter budget.

Important columns:

- `condition_id`, `rule`, `comparison_role`
- `matched_cost`, `matched_to_rules`
- `exact_cost_match_required`, `matched_baseline_condition_id`
- `is_primary_allocation_rule`
- `scaling_mode`, `is_primary_scaling_mode`
- `reference_alpha`, `scale_reference_rank`, `scale_at_reference_rank`
- `min_lora_scale`, `max_lora_scale`, `mean_lora_scale`
- `budget`, `requested_budget`
- `actual_cost`, `cost_utilization`
- `final_val_loss`
- `final_val_accuracy`
- `trainable_params`
- `evaluation_examples`, `evaluation_batches`

## `budget/allocation_comparison.csv`

Assigned rank per module for each rule, budget, and scaling policy, including exact per-site `effective_alpha` and `lora_scale`.
