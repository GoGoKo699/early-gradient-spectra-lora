# Output schema

## `base/base_metrics.csv`

Base-model metrics before LoRA adaptation.

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
- `edge`
- `top_sv`
- `frobenius`, `nuclear`
- `effective_rank`
- `stable_rank`
- `hard_detectable_rank`
- `soft_dimension`

## `sweeps/site_rank_sweep_metrics.csv`

One row per single-site LoRA rank run.

Important columns:

- `site_name`
- `rank`
- `alpha`
- `diverged`
- `final_train_loss`
- `final_val_loss`
- `final_val_accuracy`
- `trainable_params`

## `sweeps/site_target_summary.csv`

Useful-rank targets computed from each site's validation curve.

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

- `rule`
- `budget`
- `actual_cost`
- `final_val_loss`
- `final_val_accuracy`
- `trainable_params`

## `budget/allocation_comparison.csv`

Assigned rank per module for each rule/budget.
