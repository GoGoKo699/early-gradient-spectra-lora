#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from strank.allocation import (
    allocation_cost,
    gradient_norm_allocation,
    marginal_gain_allocation,
    module_cost,
    score_allocation,
    uniform_allocation,
    uniform_exact_cost_allocations,
    uniform_fill_allocation,
)
from strank.protocol import (
    SYNTHETIC_TRANSFORMER_PROTOCOL_VERSION,
    validate_synthetic_transformer_config,
)
from strank.scaling import resolve_lora_scaling
from strank.spectral import (
    effective_rank_from_s,
    hard_rank_from_s,
    quantile_bootstrap_interval,
    soft_dimension_from_s,
    stable_rank_from_s,
)
from strank.targets import compute_targets, prediction_fit
from strank.tasks import make_task_spec
from strank.utils import load_yaml, stable_seed

EXPECTED_PROTOCOL = SYNTHETIC_TRANSFORMER_PROTOCOL_VERSION
SEED_COLUMNS = ["adapter_seed", "train_data_seed", "eval_data_seed", "dropout_seed"]
METRIC_COLUMNS = ["final_train_loss", "final_val_loss", "final_val_accuracy"]
NUMERIC_TOLERANCE = 1e-10
METRIC_TOLERANCE = 1e-6


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing required JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"missing required CSV file: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"required CSV is empty: {path}")
    return frame


def _require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _require_unique(df: pd.DataFrame, columns: list[str], label: str) -> None:
    duplicated = df.duplicated(columns, keep=False)
    if duplicated.any():
        sample = df.loc[duplicated, columns].head(10).to_dict(orient="records")
        raise ValueError(f"{label} has duplicate keys {columns}: {sample}")


def _assert_finite(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{label}.{column} contains non-finite values")


def _validate_protocol_column(df: pd.DataFrame, label: str) -> None:
    _require_columns(df, ["protocol_version"], label)
    observed = set(df["protocol_version"].astype(str))
    if observed != {EXPECTED_PROTOCOL}:
        raise ValueError(
            f"{label}.protocol_version={sorted(observed)}; expected {EXPECTED_PROTOCOL!r}"
        )


def _validate_primary_scaling_flags(
    df: pd.DataFrame,
    label: str,
    primary_scaling_mode: str,
) -> None:
    _require_columns(df, ["scaling_mode", "is_primary_scaling_mode"], label)
    recorded = df["is_primary_scaling_mode"].eq(True)
    expected = df["scaling_mode"].astype(str).eq(primary_scaling_mode)
    if not recorded.equals(expected):
        raise ValueError(f"{label}.is_primary_scaling_mode does not match the pre-specified mode")


def _assert_close(actual: float, expected: float, label: str, atol: float = NUMERIC_TOLERANCE) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=atol, equal_nan=True):
        raise ValueError(f"{label}: recorded={actual!r}, recomputed={expected!r}")


def _task_key(task) -> str:
    return json.dumps(
        {"name": task.name, "params": task.params},
        sort_keys=True,
        separators=(",", ":"),
    )


def _seed_bundle(comparison_seed: int) -> dict[str, int]:
    return {
        "comparison_seed": int(comparison_seed),
        "adapter_seed": int(stable_seed(comparison_seed, "adapter_init")),
        "train_data_seed": int(stable_seed(comparison_seed, "train_data")),
        "eval_data_seed": int(stable_seed(comparison_seed, "eval_data")),
        "dropout_seed": int(stable_seed(comparison_seed, "dropout")),
    }


def _validate_seed_row(row, expected: Mapping[str, int], label: str) -> None:
    if int(row.condition_seed) != int(expected["comparison_seed"]):
        raise ValueError(f"{label}: condition_seed does not equal expected comparison seed")
    for column, value in expected.items():
        if int(getattr(row, column)) != int(value):
            raise ValueError(f"{label}: {column} does not match the deterministic protocol")


def _validate_scale_rows(df: pd.DataFrame, label: str) -> None:
    required = [
        "rank",
        "scaling_mode",
        "reference_alpha",
        "scale_reference_rank",
        "effective_alpha",
        "lora_scale",
    ]
    _require_columns(df, required, label)
    for row in df.itertuples(index=False):
        expected = resolve_lora_scaling(
            int(row.rank),
            float(row.reference_alpha),
            str(row.scaling_mode),
            int(row.scale_reference_rank),
        )
        _assert_close(
            float(row.effective_alpha),
            expected.effective_alpha,
            f"{label}: effective_alpha mode={row.scaling_mode}, rank={row.rank}",
        )
        _assert_close(
            float(row.lora_scale),
            expected.lora_scale,
            f"{label}: lora_scale mode={row.scaling_mode}, rank={row.rank}",
        )


def _check_shared_streams(df: pd.DataFrame, group_columns: list[str], label: str) -> None:
    _require_columns(df, group_columns + ["comparison_seed"] + SEED_COLUMNS, label)
    counts = df.groupby(group_columns, dropna=False)[["comparison_seed"] + SEED_COLUMNS].nunique(
        dropna=False
    )
    if (counts.to_numpy() != 1).any():
        raise ValueError(f"{label} does not share every random stream across compared conditions")


def _check_metric_identity(
    df: pd.DataFrame,
    group_columns: list[str],
    label: str,
    atol: float = METRIC_TOLERANCE,
) -> None:
    for keys, group in df.groupby(group_columns, dropna=False):
        if len(group) < 2:
            continue
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(group[metric], errors="raise").to_numpy(dtype=float)
            spread = float(values.max() - values.min())
            if spread > atol:
                raise ValueError(f"{label}: {keys}, {metric}, range={spread}")


def _normalise_scalar(value):
    if pd.isna(value) or value == "":
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return str(value)


def _assert_frames_equivalent(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    keys: list[str],
    label: str,
    atol: float = NUMERIC_TOLERANCE,
) -> None:
    _require_columns(actual, list(expected.columns), label)
    _require_unique(actual, keys, label)
    _require_unique(expected, keys, f"recomputed {label}")
    actual_sorted = actual[list(expected.columns)].sort_values(keys).reset_index(drop=True)
    expected_sorted = expected.sort_values(keys).reset_index(drop=True)
    if len(actual_sorted) != len(expected_sorted):
        raise ValueError(
            f"{label} row count {len(actual_sorted)} != recomputed {len(expected_sorted)}"
        )
    for column in expected_sorted.columns:
        left = actual_sorted[column]
        right = expected_sorted[column]
        if pd.api.types.is_numeric_dtype(right) and not pd.api.types.is_bool_dtype(right):
            left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
            right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
            if not np.allclose(left_values, right_values, rtol=0.0, atol=atol, equal_nan=True):
                indices = np.where(~np.isclose(left_values, right_values, rtol=0.0, atol=atol, equal_nan=True))[0]
                index = int(indices[0])
                raise ValueError(
                    f"{label}.{column} mismatch at row {index}: "
                    f"recorded={left_values[index]!r}, recomputed={right_values[index]!r}"
                )
        else:
            left_values = [_normalise_scalar(value) for value in left]
            right_values = [_normalise_scalar(value) for value in right]
            if left_values != right_values:
                index = next(i for i, pair in enumerate(zip(left_values, right_values)) if pair[0] != pair[1])
                raise ValueError(
                    f"{label}.{column} mismatch at row {index}: "
                    f"recorded={left_values[index]!r}, recomputed={right_values[index]!r}"
                )


def _validate_base(
    base: pd.DataFrame,
    cfg: dict,
    base_task,
    adapt_task,
    base_seed: int,
) -> None:
    if len(base) != 1:
        raise ValueError(f"base_metrics must contain one row, found {len(base)}")
    row = base.iloc[0]
    expected_seeds = {
        "base_model_seed": stable_seed(base_seed, _task_key(base_task), "base_model"),
        "base_train_data_seed": stable_seed(base_seed, _task_key(base_task), "base_train_data"),
        "base_dropout_seed": stable_seed(base_seed, _task_key(base_task), "base_dropout"),
        "base_eval_seed": stable_seed(base_seed, _task_key(base_task), "base_eval"),
        "adapt_eval_seed": stable_seed(base_seed, _task_key(adapt_task), "pre_adapt_eval"),
    }
    _require_columns(base, list(expected_seeds) + ["base_eval_examples", "adapt_eval_examples"], "base")
    for column, expected in expected_seeds.items():
        if int(row[column]) != int(expected):
            raise ValueError(f"base.{column} does not match the deterministic protocol")
    _assert_finite(
        base,
        [
            "train_loss_last",
            "base_task_loss",
            "base_task_accuracy",
            "adapt_task_loss_before_lora",
            "adapt_task_accuracy_before_lora",
        ],
        "base",
    )
    exact_modular = bool(cfg.get("protocol", {}).get("exact_modular_evaluation", False))
    if exact_modular and adapt_task.name == "modular":
        expected_base = int(base_task.params["modulus"]) ** 2
        expected_adapt = int(adapt_task.params["modulus"]) ** 2
        if int(row["base_eval_examples"]) != expected_base:
            raise ValueError("base-task modular evaluation is not exhaustive")
        if int(row["adapt_eval_examples"]) != expected_adapt:
            raise ValueError("pre-adaptation modular evaluation is not exhaustive")


def _validate_calibration(
    run_dir: Path,
    module_stats: pd.DataFrame,
    sites: list[str],
    cfg: dict,
    adapt_task,
    base_seed: int,
) -> dict:
    _require_unique(module_stats, ["site_name"], "module_stats")
    if set(module_stats["site_name"].astype(str)) != set(sites):
        raise ValueError("module_stats sites do not match config")
    required = [
        "d_in",
        "d_out",
        "n_tokens",
        "n_batches",
        "calibration_data_seed",
        "permutation_seed",
        "dropout_disabled_during_calibration",
        "null_bootstrap",
        "null_quantile",
        "null_quantile_method",
        "null_uncertainty_resamples",
        "edge_mc_confidence",
        "edge",
        "edge_mc_ci_low",
        "edge_mc_ci_high",
        "edge_q_0_99",
        "edge_q_0_995",
        "edge_q_0_999",
        "top_sv",
        "frobenius",
        "nuclear",
        "effective_rank",
        "stable_rank",
        "hard_detectable_rank",
        "soft_dimension",
        "gradient_norm",
        "cov_trace",
        "cov_effective_rank",
    ]
    _require_columns(module_stats, required, "module_stats")
    _assert_finite(
        module_stats,
        [column for column in required if column not in {"null_quantile_method", "dropout_disabled_during_calibration"}],
        "module_stats",
    )
    if not module_stats["dropout_disabled_during_calibration"].eq(True).all():
        raise ValueError("calibration dropout was not disabled")
    if set(module_stats["null_quantile_method"].astype(str)) != {"linear"}:
        raise ValueError("unsupported or mixed null quantile method")

    calibration_cfg = cfg["calibration"]
    expected_bootstrap = int(calibration_cfg.get("null_bootstrap", 8))
    expected_quantile = float(calibration_cfg.get("null_quantile", 0.995))
    expected_resamples = int(calibration_cfg.get("null_uncertainty_resamples", 1000))
    expected_confidence = float(calibration_cfg.get("null_uncertainty_confidence", 0.95))
    if set(module_stats["null_bootstrap"].astype(int)) != {expected_bootstrap}:
        raise ValueError("module_stats null_bootstrap does not match config")
    if not np.allclose(module_stats["null_quantile"], expected_quantile, rtol=0.0, atol=0.0):
        raise ValueError("module_stats null_quantile does not match config")
    if set(module_stats["null_uncertainty_resamples"].astype(int)) != {expected_resamples}:
        raise ValueError("module_stats uncertainty-resample count does not match config")
    if not np.allclose(module_stats["edge_mc_confidence"], expected_confidence, rtol=0.0, atol=0.0):
        raise ValueError("module_stats edge confidence does not match config")

    null_path = run_dir / "calibration" / "null_maxima.npz"
    sv_path = run_dir / "calibration" / "singular_values.npz"
    if not null_path.is_file() or not sv_path.is_file():
        raise ValueError("calibration arrays are missing")
    expected_keys = {site.replace(".", "__") for site in sites}
    calibration_data_seed = stable_seed(base_seed, _task_key(adapt_task), "calibration_data")
    permutation_seed_base = stable_seed(base_seed, _task_key(adapt_task), "calibration_null")
    with np.load(null_path, allow_pickle=False) as null_arrays, np.load(
        sv_path, allow_pickle=False
    ) as sv_arrays:
        if set(null_arrays.files) != expected_keys or set(sv_arrays.files) != expected_keys:
            raise ValueError("calibration array keys do not match configured sites")
        by_site = module_stats.set_index("site_name")
        for site in sites:
            key = site.replace(".", "__")
            maxima = np.asarray(null_arrays[key], dtype=float).ravel()
            singular = np.asarray(sv_arrays[key], dtype=float).ravel()
            if len(maxima) != expected_bootstrap:
                raise ValueError(f"null maxima length mismatch for {site}")
            if len(singular) == 0:
                raise ValueError(f"empty singular-value array for {site}")
            if not np.isfinite(maxima).all() or not np.isfinite(singular).all():
                raise ValueError(f"non-finite calibration array for {site}")
            if (maxima < 0).any() or (singular < 0).any():
                raise ValueError(f"negative singular-value diagnostic for {site}")
            if not np.all(singular[:-1] >= singular[1:] - NUMERIC_TOLERANCE):
                raise ValueError(f"singular values are not sorted for {site}")

            row = by_site.loc[site]
            expected_site_seed = stable_seed(permutation_seed_base, "permutation_null", site)
            if int(row["calibration_data_seed"]) != int(calibration_data_seed):
                raise ValueError(f"calibration data seed mismatch for {site}")
            if int(row["permutation_seed"]) != int(expected_site_seed):
                raise ValueError(f"permutation seed mismatch for {site}")

            edge = float(np.quantile(maxima, expected_quantile, method="linear")) if len(maxima) else 0.0
            ci_low, ci_high = quantile_bootstrap_interval(
                maxima,
                expected_quantile,
                n_resamples=expected_resamples,
                seed=int(expected_site_seed) + 1_000_003,
                confidence=expected_confidence,
            )
            recomputed = {
                "edge": edge,
                "edge_mc_ci_low": ci_low,
                "edge_mc_ci_high": ci_high,
                "edge_q_0_99": float(np.quantile(maxima, 0.99, method="linear")) if len(maxima) else 0.0,
                "edge_q_0_995": float(np.quantile(maxima, 0.995, method="linear")) if len(maxima) else 0.0,
                "edge_q_0_999": float(np.quantile(maxima, 0.999, method="linear")) if len(maxima) else 0.0,
                "top_sv": float(singular[0]),
                "frobenius": float(np.linalg.norm(singular)),
                "nuclear": float(np.sum(singular)),
                "effective_rank": effective_rank_from_s(singular),
                "stable_rank": stable_rank_from_s(singular),
                "hard_detectable_rank": hard_rank_from_s(singular, edge),
                "soft_dimension": soft_dimension_from_s(singular, edge),
            }
            for column, expected in recomputed.items():
                _assert_close(float(row[column]), expected, f"module_stats[{site}].{column}")

    return {
        "null_bootstrap": expected_bootstrap,
        "null_quantile": expected_quantile,
        "null_uncertainty_resamples": expected_resamples,
        "edge_mc_confidence": expected_confidence,
    }


def _validate_sweeps(
    sweeps: pd.DataFrame,
    cfg: dict,
    module_stats: pd.DataFrame,
    adapt_task,
    base_seed: int,
    scaling: dict,
    sites: list[str],
) -> None:
    modes = list(scaling["scaling_modes"])
    ranks = [int(value) for value in cfg["lora"]["ranks"]]
    replicates = int(cfg.get("protocol", {}).get("sweep_replicates", 1))
    keys = ["scaling_mode", "site_name", "rank", "adaptation_replicate"]
    _require_unique(sweeps, keys, "sweeps")
    expected_rows = len(modes) * len(sites) * len(ranks) * replicates
    if len(sweeps) != expected_rows:
        raise ValueError(f"expected {expected_rows} sweep rows, found {len(sweeps)}")
    if set(sweeps["scaling_mode"].astype(str)) != set(modes):
        raise ValueError("sweep scaling modes do not match config")
    if set(sweeps["site_name"].astype(str)) != set(sites):
        raise ValueError("sweep sites do not match config")
    if set(sweeps["rank"].astype(int)) != set(ranks):
        raise ValueError("sweep ranks do not match config")
    if set(sweeps["adaptation_replicate"].astype(int)) != set(range(replicates)):
        raise ValueError("sweep replicate IDs are incomplete")
    _validate_scale_rows(sweeps, "sweeps")
    _assert_finite(
        sweeps,
        METRIC_COLUMNS + ["trainable_params", "evaluation_examples", "evaluation_batches"],
        "sweeps",
    )
    _check_shared_streams(sweeps, ["site_name", "rank", "adaptation_replicate"], "sweeps")
    costs = {
        str(row.site_name): int(module_cost(row))
        for row in module_stats.itertuples(index=False)
    }
    for row in sweeps.itertuples(index=False):
        expected_seed = stable_seed(
            base_seed,
            _task_key(adapt_task),
            "site_sweep",
            str(row.site_name),
            int(row.adaptation_replicate),
        )
        _validate_seed_row(
            row,
            _seed_bundle(expected_seed),
            f"sweep {row.site_name}, rank={row.rank}, replicate={row.adaptation_replicate}",
        )
        expected_params = costs[str(row.site_name)] * int(row.rank)
        if int(row.trainable_params) != expected_params:
            raise ValueError(
                f"sweep trainable_params mismatch for {row.site_name}, rank={row.rank}"
            )
    _check_metric_identity(
        sweeps[sweeps["rank"].astype(int) == 0],
        ["site_name", "adaptation_replicate"],
        "rank-zero scaling invariant",
    )
    reference_rank = int(scaling["reference_rank"])
    at_reference = sweeps[sweeps["rank"].astype(int) == reference_rank]
    if at_reference.empty:
        raise ValueError("rank grid does not contain scale_reference_rank")
    spread = at_reference.groupby(["site_name", "adaptation_replicate"])["lora_scale"].agg(
        lambda values: float(values.max() - values.min())
    )
    if (spread > NUMERIC_TOLERANCE).any():
        raise ValueError("scaling modes are not matched at the reference rank")
    _check_metric_identity(
        at_reference,
        ["site_name", "adaptation_replicate"],
        "reference-rank scaling invariant",
    )


def _validate_targets_and_fits(
    targets: pd.DataFrame,
    fits: pd.DataFrame,
    sweeps: pd.DataFrame,
    module_stats: pd.DataFrame,
    cfg: dict,
    modes: list[str],
    sites: list[str],
) -> None:
    expected_targets = compute_targets(
        sweeps,
        near_gaps=[float(value) for value in cfg.get("targets", {}).get("near_best_gaps", [0.1, 0.2])],
        recovery_fracs=[float(value) for value in cfg.get("targets", {}).get("recovery_fracs", [0.7, 0.8])],
        lambdas=[float(value) for value in cfg.get("targets", {}).get("penalty_lambdas", [0.2, 0.3])],
    )
    _assert_frames_equivalent(
        targets,
        expected_targets,
        ["scaling_mode", "site_name"],
        "targets",
    )
    if len(targets) != len(modes) * len(sites):
        raise ValueError("target row count does not match scaling-mode/site design")
    expected_fits = prediction_fit(expected_targets, module_stats)
    _assert_frames_equivalent(
        fits,
        expected_fits,
        ["scaling_mode", "target", "predictor"],
        "prediction fits",
    )



def _allocation_for_rule(
    rule: str,
    module_stats: pd.DataFrame,
    singular_values: Mapping[str, np.ndarray],
    rank_grid: list[int],
    budget: int,
) -> dict[str, int]:
    if rule == "uniform":
        return uniform_allocation(module_stats, rank_grid, budget)
    if rule == "uniform_fill":
        return uniform_fill_allocation(module_stats, rank_grid, budget)
    if rule == "effective_rank":
        return score_allocation(module_stats, rank_grid, budget, "effective_rank")
    if rule == "soft_dimension":
        return score_allocation(module_stats, rank_grid, budget, "soft_dimension")
    if rule == "gradient_norm":
        return gradient_norm_allocation(module_stats, rank_grid, budget)
    if rule == "marginal_gain_raw":
        return marginal_gain_allocation(
            module_stats, singular_values, rank_grid, budget, mode="raw"
        )
    if rule == "marginal_gain_edge":
        return marginal_gain_allocation(
            module_stats, singular_values, rank_grid, budget, mode="edge"
        )
    if rule == "marginal_gain_soft":
        return marginal_gain_allocation(
            module_stats, singular_values, rank_grid, budget, mode="soft"
        )
    raise ValueError(f"unsupported allocation rule in validator: {rule}")


def _recompute_condition_plan(
    cfg: dict,
    module_stats: pd.DataFrame,
    singular_values: Mapping[str, np.ndarray],
    budget: int,
) -> dict[str, dict]:
    rank_grid = [int(value) for value in cfg["lora"]["ranks"]]
    rules = [str(value) for value in cfg.get("budgets", {}).get("rules", [])]
    matched_rules = [
        str(value) for value in cfg.get("budgets", {}).get("cost_matched_rules", [])
    ]
    allocations = {
        rule: _allocation_for_rule(
            rule, module_stats, singular_values, rank_grid, int(budget)
        )
        for rule in rules
    }
    cost_to_rules: dict[int, list[str]] = {}
    for rule in matched_rules:
        cost = allocation_cost(module_stats, allocations[rule])
        cost_to_rules.setdefault(int(cost), []).append(rule)
    exact = uniform_exact_cost_allocations(
        module_stats, rank_grid, sorted(cost_to_rules)
    )

    primary_rule = str(cfg.get("protocol", {}).get("primary_allocation_rule", ""))
    plan: dict[str, dict] = {}
    for rule, allocation in allocations.items():
        realized_cost = allocation_cost(module_stats, allocation)
        exact_required = rule in matched_rules
        plan[rule] = {
            "condition_id": rule,
            "rule": rule,
            "comparison_role": (
                "cap_baseline" if rule in {"uniform", "uniform_fill"} else "candidate"
            ),
            "matched_cost": int(realized_cost),
            "matched_to_rules": "[]",
            "exact_cost_match_required": bool(exact_required),
            "matched_baseline_condition_id": (
                f"uniform_exact_cost__{realized_cost}" if exact_required else ""
            ),
            "is_primary_allocation_rule": rule == primary_rule,
            "allocation": {str(site): int(rank) for site, rank in allocation.items()},
        }
    for cost, allocation in exact.items():
        condition_id = f"uniform_exact_cost__{int(cost)}"
        plan[condition_id] = {
            "condition_id": condition_id,
            "rule": "uniform_exact_cost",
            "comparison_role": "exact_cost_baseline",
            "matched_cost": int(cost),
            "matched_to_rules": json.dumps(
                sorted(cost_to_rules[int(cost)]), separators=(",", ":")
            ),
            "exact_cost_match_required": False,
            "matched_baseline_condition_id": "",
            "is_primary_allocation_rule": False,
            "allocation": {str(site): int(rank) for site, rank in allocation.items()},
        }
    return plan


def _validate_recomputed_condition_plans(
    allocations: pd.DataFrame,
    budgets: pd.DataFrame,
    cfg: dict,
    module_stats: pd.DataFrame,
    singular_values: Mapping[str, np.ndarray],
) -> None:
    metadata_columns = [
        "rule",
        "comparison_role",
        "matched_cost",
        "matched_to_rules",
        "exact_cost_match_required",
        "matched_baseline_condition_id",
        "is_primary_allocation_rule",
    ]
    for budget in sorted(set(budgets["budget"].astype(int))):
        expected = _recompute_condition_plan(
            cfg, module_stats, singular_values, int(budget)
        )
        observed_budget = budgets[budgets["budget"].astype(int) == int(budget)]
        observed_ids = set(observed_budget["condition_id"].astype(str))
        if observed_ids != set(expected):
            raise ValueError(
                f"budget {budget} condition IDs differ from recomputed plan; "
                f"recorded={sorted(observed_ids)}, expected={sorted(expected)}"
            )
        for keys, row_group in observed_budget.groupby(
            ["scaling_mode", "adaptation_replicate", "condition_id"],
            dropna=False,
        ):
            condition_id = str(keys[2])
            spec = expected[condition_id]
            if len(row_group) != 1:
                raise ValueError(f"budget condition is not unique: {keys}")
            row = row_group.iloc[0]
            for column in metadata_columns:
                actual = _normalise_scalar(row[column])
                expected_value = _normalise_scalar(spec[column])
                if actual != expected_value:
                    raise ValueError(
                        f"budget {keys}.{column}: recorded={actual!r}, "
                        f"recomputed={expected_value!r}"
                    )

            allocation_group = allocations[
                (allocations["budget"].astype(int) == int(budget))
                & (allocations["scaling_mode"].astype(str) == str(keys[0]))
                & (
                    allocations["adaptation_replicate"].astype(int)
                    == int(keys[1])
                )
                & (allocations["condition_id"].astype(str) == condition_id)
            ]
            recorded_allocation = {
                str(item.site_name): int(item.rank)
                for item in allocation_group.itertuples(index=False)
            }
            if recorded_allocation != spec["allocation"]:
                raise ValueError(
                    f"allocation {keys} differs from recomputed deterministic plan; "
                    f"recorded={recorded_allocation}, expected={spec['allocation']}"
                )

def _validate_allocations(
    allocations: pd.DataFrame,
    budgets: pd.DataFrame,
    module_stats: pd.DataFrame,
    sites: list[str],
) -> None:
    budget_keys = ["scaling_mode", "budget", "adaptation_replicate", "condition_id"]
    allocation_keys = budget_keys + ["site_name"]
    _require_unique(allocations, allocation_keys, "allocations")
    expected_sites = set(sites)
    costs = {
        str(row.site_name): int(module_cost(row))
        for row in module_stats.itertuples(index=False)
    }
    for keys, group in allocations.groupby(budget_keys, dropna=False):
        observed_sites = set(group["site_name"].astype(str))
        if observed_sites != expected_sites:
            raise ValueError(
                f"allocation {keys} sites {sorted(observed_sites)} != {sorted(expected_sites)}"
            )
        realized = int(
            sum(costs[str(row.site_name)] * int(row.rank) for row in group.itertuples(index=False))
        )
        recorded = set(pd.to_numeric(group["actual_cost"], errors="raise").astype(int))
        if recorded != {realized}:
            raise ValueError(f"allocation {keys} has recorded costs {recorded}, recomputed {realized}")
        expected_signature = json.dumps(
            sorted((str(row.site_name), int(row.rank)) for row in group.itertuples(index=False)),
            separators=(",", ":"),
        )
        if set(group["allocation_signature"].astype(str)) != {expected_signature}:
            raise ValueError(f"allocation signature mismatch for {keys}")

        budget_row = budgets[
            (budgets["scaling_mode"].astype(str) == str(keys[0]))
            & (budgets["budget"].astype(int) == int(keys[1]))
            & (budgets["adaptation_replicate"].astype(int) == int(keys[2]))
            & (budgets["condition_id"].astype(str) == str(keys[3]))
        ]
        if len(budget_row) != 1:
            raise ValueError(f"allocation {keys} does not have exactly one budget row")
        active = group[group["rank"].astype(int) > 0]
        scales = active["lora_scale"].to_numpy(dtype=float)
        expected_scale_summary = {
            "active_sites": int(len(active)),
            "min_lora_scale": float(np.min(scales)) if len(scales) else 0.0,
            "max_lora_scale": float(np.max(scales)) if len(scales) else 0.0,
            "mean_lora_scale": float(np.mean(scales)) if len(scales) else 0.0,
        }
        for column, expected_value in expected_scale_summary.items():
            _assert_close(
                float(budget_row.iloc[0][column]),
                expected_value,
                f"budget scale summary {keys}.{column}",
            )

    joined = budgets.merge(
        allocations.groupby(budget_keys, as_index=False).agg(
            recomputed_cost=("actual_cost", "first"),
            n_sites=("site_name", "nunique"),
            allocation_signature_from_rows=("allocation_signature", "first"),
        ),
        on=budget_keys,
        how="left",
        validate="one_to_one",
    )
    if joined["recomputed_cost"].isna().any():
        raise ValueError("some budget rows have no allocation rows")
    if not (
        joined["actual_cost"].astype(int) == joined["recomputed_cost"].astype(int)
    ).all():
        raise ValueError("budget and allocation actual_cost values disagree")
    if not (joined["n_sites"].astype(int) == len(sites)).all():
        raise ValueError("some budget allocations omit candidate sites")
    if not (
        joined["trainable_params"].astype(int) == joined["actual_cost"].astype(int)
    ).all():
        raise ValueError("trainable_params does not equal realized LoRA parameter cost")
    if not (
        joined["allocation_signature"].astype(str)
        == joined["allocation_signature_from_rows"].astype(str)
    ).all():
        raise ValueError("budget and allocation signatures disagree")


def _validate_exact_cost_pairing(
    budgets: pd.DataFrame,
    matched_rules: list[str],
    *,
    primary_rule: str = "",
    primary_reference_rule: str = "uniform_exact_cost",
) -> None:
    if not matched_rules:
        return
    if primary_reference_rule != "uniform_exact_cost":
        raise ValueError("primary_reference_rule must be uniform_exact_cost")
    candidates = budgets[budgets["rule"].astype(str).isin(matched_rules)]
    if candidates.empty:
        raise ValueError("no rows exist for configured cost-matched rules")
    if not candidates["exact_cost_match_required"].eq(True).all():
        raise ValueError("a configured cost-matched candidate is not marked as requiring a match")
    if not candidates["comparison_role"].astype(str).eq("candidate").all():
        raise ValueError("a configured cost-matched rule is not labeled as a candidate")
    if not (candidates["matched_cost"].astype(int) == candidates["actual_cost"].astype(int)).all():
        raise ValueError("candidate matched_cost differs from actual_cost")
    if primary_rule:
        if primary_rule not in matched_rules:
            raise ValueError("primary allocation rule is not cost matched")
        expected_primary = candidates["rule"].astype(str).eq(primary_rule)
        recorded_primary = candidates["is_primary_allocation_rule"].eq(True)
        if not recorded_primary.equals(expected_primary):
            raise ValueError("is_primary_allocation_rule flags do not match the pre-specified primary rule")
    for candidate in candidates.itertuples(index=False):
        expected_id = f"uniform_exact_cost__{int(candidate.actual_cost)}"
        if str(candidate.matched_baseline_condition_id) != expected_id:
            raise ValueError(
                f"candidate {candidate.rule} points to {candidate.matched_baseline_condition_id!r}; "
                f"expected {expected_id!r}"
            )
        baseline = budgets[
            (budgets["scaling_mode"] == candidate.scaling_mode)
            & (budgets["budget"] == candidate.budget)
            & (budgets["adaptation_replicate"] == candidate.adaptation_replicate)
            & (budgets["condition_id"] == expected_id)
        ]
        if len(baseline) != 1:
            raise ValueError(
                "expected exactly one exact-cost comparator for "
                f"mode={candidate.scaling_mode}, budget={candidate.budget}, "
                f"replicate={candidate.adaptation_replicate}, rule={candidate.rule}; "
                f"found {len(baseline)}"
            )
        baseline_row = baseline.iloc[0]
        if str(baseline_row["comparison_role"]) != "exact_cost_baseline":
            raise ValueError(f"{expected_id} is not labeled exact_cost_baseline")
        if str(baseline_row["rule"]) != "uniform_exact_cost":
            raise ValueError(f"{expected_id} does not use the uniform_exact_cost rule")
        if bool(baseline_row["exact_cost_match_required"]):
            raise ValueError(f"{expected_id} is incorrectly marked as requiring another match")
        if int(baseline_row["actual_cost"]) != int(candidate.actual_cost):
            raise ValueError("exact-cost baseline has unequal realized cost")
        matched_to = json.loads(str(baseline_row["matched_to_rules"]))
        if str(candidate.rule) not in matched_to:
            raise ValueError(
                f"exact-cost baseline metadata does not list candidate rule {candidate.rule}"
            )


def _validate_budgets(
    budgets: pd.DataFrame,
    allocations: pd.DataFrame,
    cfg: dict,
    module_stats: pd.DataFrame,
    singular_values: Mapping[str, np.ndarray],
    adapt_task,
    base_seed: int,
    scaling: dict,
    sites: list[str],
) -> list[str]:
    modes = list(scaling["scaling_modes"])
    budget_values = [int(value) for value in cfg.get("budgets", {}).get("param_budgets", [])]
    replicates = int(cfg.get("protocol", {}).get("adaptation_replicates", 1))
    keys = ["scaling_mode", "budget", "adaptation_replicate", "condition_id"]
    _require_unique(budgets, keys, "budgets")
    required = [
        "comparison_role",
        "matched_cost",
        "matched_to_rules",
        "exact_cost_match_required",
        "matched_baseline_condition_id",
        "allocation_signature",
        "requested_budget",
        "actual_cost",
        "cost_utilization",
        "trainable_params",
        "evaluation_examples",
        "evaluation_batches",
        "active_sites",
        "min_lora_scale",
        "max_lora_scale",
        "mean_lora_scale",
    ]
    _require_columns(budgets, required, "budgets")
    if set(budgets["scaling_mode"].astype(str)) != set(modes):
        raise ValueError("budget scaling modes do not match config")
    if set(budgets["budget"].astype(int)) != set(budget_values):
        raise ValueError("budget values do not match config")
    if set(budgets["adaptation_replicate"].astype(int)) != set(range(replicates)):
        raise ValueError("budget replicate IDs are incomplete")
    if not (budgets["requested_budget"].astype(int) == budgets["budget"].astype(int)).all():
        raise ValueError("requested_budget differs from budget")
    if not (budgets["actual_cost"].astype(int) <= budgets["requested_budget"].astype(int)).all():
        raise ValueError("an allocation exceeds its requested budget cap")
    expected_utilization = np.divide(
        budgets["actual_cost"].to_numpy(dtype=float),
        budgets["requested_budget"].to_numpy(dtype=float),
        out=np.zeros(len(budgets), dtype=float),
        where=budgets["requested_budget"].to_numpy(dtype=float) > 0,
    )
    if not np.allclose(
        budgets["cost_utilization"].to_numpy(dtype=float),
        expected_utilization,
        rtol=0.0,
        atol=NUMERIC_TOLERANCE,
    ):
        raise ValueError("cost_utilization is inconsistent with realized cost")
    _assert_finite(
        budgets,
        METRIC_COLUMNS
        + ["actual_cost", "cost_utilization", "trainable_params", "evaluation_examples"],
        "budgets",
    )
    _check_shared_streams(
        budgets,
        ["budget", "adaptation_replicate", "condition_id"],
        "budget scaling comparisons",
    )
    for row in budgets.itertuples(index=False):
        expected_seed = stable_seed(
            base_seed,
            _task_key(adapt_task),
            "budget",
            int(row.budget),
            int(row.adaptation_replicate),
        )
        _validate_seed_row(
            row,
            _seed_bundle(expected_seed),
            f"budget={row.budget}, replicate={row.adaptation_replicate}, condition={row.condition_id}",
        )
    _check_metric_identity(
        budgets,
        ["scaling_mode", "budget", "adaptation_replicate", "allocation_signature"],
        "identical-allocation CRN invariant",
    )
    _validate_scale_rows(allocations, "allocations")
    _validate_allocations(allocations, budgets, module_stats, sites)
    _validate_recomputed_condition_plans(
        allocations, budgets, cfg, module_stats, singular_values
    )
    matched_rules = [str(value) for value in cfg.get("budgets", {}).get("cost_matched_rules", [])]
    protocol_cfg = cfg.get("protocol", {})
    _validate_exact_cost_pairing(
        budgets,
        matched_rules,
        primary_rule=str(protocol_cfg.get("primary_allocation_rule", "")),
        primary_reference_rule=str(
            protocol_cfg.get("primary_reference_rule", "uniform_exact_cost")
        ),
    )
    return matched_rules


def validate(run_dir: Path, *, fail_on_divergence: bool = True) -> dict:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise ValueError(f"run directory does not exist: {run_dir}")
    if any(path.is_symlink() for path in run_dir.rglob("*")):
        raise ValueError("source run must not contain symbolic links")

    cfg = load_yaml(run_dir / "config.yaml")
    validated_config = validate_synthetic_transformer_config(cfg)
    info = _read_json(run_dir / "run_info.json")
    success = _read_json(run_dir / "_SUCCESS.json")
    if info.get("protocol_version") != EXPECTED_PROTOCOL:
        raise ValueError(
            f"unexpected protocol {info.get('protocol_version')!r}; expected {EXPECTED_PROTOCOL!r}"
        )
    if success.get("status") != "PASS" or success.get("protocol_version") != EXPECTED_PROTOCOL:
        raise ValueError("_SUCCESS.json does not certify the expected protocol")
    if info.get("run_id") != run_dir.name:
        raise ValueError("run_info run_id does not match directory name")
    if success.get("base_seed") != info.get("base_seed") or success.get("task_name") != info.get("task_name"):
        raise ValueError("_SUCCESS metadata does not match run_info")
    if not (run_dir / "base" / "base_model.pt").is_file():
        raise ValueError("missing base/base_model.pt")

    scaling = {
        key: validated_config[key]
        for key in (
            "scaling_modes",
            "primary_scaling_mode",
            "reference_alpha",
            "reference_rank",
            "scale_at_reference_rank",
        )
    }
    modes = list(scaling["scaling_modes"])
    if list(info.get("scaling_modes", [])) != modes:
        raise ValueError("run_info scaling modes do not match config")
    if info.get("primary_scaling_mode") != scaling["primary_scaling_mode"]:
        raise ValueError("run_info primary scaling mode does not match config")
    if int(info.get("scale_reference_rank")) != int(scaling["reference_rank"]):
        raise ValueError("run_info reference rank does not match config")
    for flag in (
        "common_random_numbers",
        "fixed_training_batches",
        "fixed_validation_batches",
        "nested_max_rank_initialization",
        "calibration_dropout_disabled",
    ):
        if info.get(flag) is not True:
            raise ValueError(f"run_info protocol flag is not true: {flag}")
    if info.get("rule_in_comparison_seed") is not False:
        raise ValueError("allocation rule is included in the comparison seed")
    if info.get("scaling_mode_in_comparison_seed") is not False:
        raise ValueError("scaling mode is included in the comparison seed")

    protocol_cfg = cfg.get("protocol", {})
    matched_rules_cfg = [
        str(value) for value in cfg.get("budgets", {}).get("cost_matched_rules", [])
    ]
    expected_protocol_metadata = {
        "primary_allocation_rule": str(protocol_cfg.get("primary_allocation_rule", "")),
        "primary_reference_rule": str(
            protocol_cfg.get("primary_reference_rule", "uniform_exact_cost")
        ),
        "primary_metric": str(protocol_cfg.get("primary_metric", "final_val_loss")),
        "independent_unit": str(protocol_cfg.get("independent_unit", "task_seed_run")),
        "cost_matched_rules": matched_rules_cfg,
        "exact_cost_baselines": bool(matched_rules_cfg),
    }
    for key, expected in expected_protocol_metadata.items():
        if info.get(key) != expected:
            raise ValueError(
                f"run_info {key}={info.get(key)!r} does not match config {expected!r}"
            )
    if expected_protocol_metadata["primary_metric"] != "final_val_loss":
        raise ValueError("publication protocol primary_metric must be final_val_loss")
    if expected_protocol_metadata["independent_unit"] != "task_seed_run":
        raise ValueError("publication protocol independent_unit must be task_seed_run")

    base_seed = int(cfg["run"].get("seed", 0))
    if int(info.get("base_seed")) != base_seed:
        raise ValueError("run_info base seed does not match config")
    base_task = make_task_spec(cfg["task"], split="base")
    adapt_task = make_task_spec(cfg["task"], split="adapt")
    if str(info.get("task_name")) != adapt_task.name:
        raise ValueError("run_info task name does not match config")

    base = _read_csv(run_dir / "base" / "base_metrics.csv")
    module_stats = _read_csv(run_dir / "calibration" / "module_stats.csv")
    sweeps = _read_csv(run_dir / "sweeps" / "site_rank_sweep_metrics.csv")
    targets = _read_csv(run_dir / "sweeps" / "site_target_summary.csv")
    fits = _read_csv(run_dir / "sweeps" / "site_prediction_fit.csv")
    budgets = _read_csv(run_dir / "budget" / "budget_results.csv")
    allocations = _read_csv(run_dir / "budget" / "allocation_comparison.csv")
    frames = {
        "base": base,
        "module_stats": module_stats,
        "sweeps": sweeps,
        "targets": targets,
        "fits": fits,
        "budgets": budgets,
        "allocations": allocations,
    }
    for label, frame in frames.items():
        _validate_protocol_column(frame, label)
    for label, frame in {
        "sweeps": sweeps,
        "targets": targets,
        "fits": fits,
        "budgets": budgets,
        "allocations": allocations,
    }.items():
        _validate_primary_scaling_flags(
            frame, label, str(scaling["primary_scaling_mode"])
        )
    sites = [str(value) for value in cfg.get("sites", {}).get("include", [])]
    if not sites:
        raise ValueError("config contains no candidate sites")

    _validate_base(base, cfg, base_task, adapt_task, base_seed)
    calibration_summary = _validate_calibration(
        run_dir,
        module_stats,
        sites,
        cfg,
        adapt_task,
        base_seed,
    )
    with np.load(
        run_dir / "calibration" / "singular_values.npz", allow_pickle=False
    ) as arrays:
        singular_values = {
            site: np.asarray(arrays[site.replace(".", "__")], dtype=float).copy()
            for site in sites
        }
    _validate_sweeps(
        sweeps,
        cfg,
        module_stats,
        adapt_task,
        base_seed,
        scaling,
        sites,
    )
    _validate_targets_and_fits(targets, fits, sweeps, module_stats, cfg, modes, sites)
    matched_rules = _validate_budgets(
        budgets,
        allocations,
        cfg,
        module_stats,
        singular_values,
        adapt_task,
        base_seed,
        scaling,
        sites,
    )

    exact_modular = bool(cfg.get("protocol", {}).get("exact_modular_evaluation", False))
    if exact_modular and adapt_task.name == "modular":
        expected_examples = int(adapt_task.params["modulus"]) ** 2
        if set(sweeps["evaluation_examples"].astype(int)) != {expected_examples}:
            raise ValueError("single-site modular evaluation is not exhaustive")
        if set(budgets["evaluation_examples"].astype(int)) != {expected_examples}:
            raise ValueError("budget modular evaluation is not exhaustive")

    if fail_on_divergence:
        for label, frame in (("sweeps", sweeps), ("budgets", budgets)):
            if "diverged" in frame and frame["diverged"].eq(True).any():
                raise ValueError(f"{label} contains diverged adaptation runs")

    summary = {
        "status": "PASS",
        "run_id": run_dir.name,
        "protocol_version": EXPECTED_PROTOCOL,
        "task_name": adapt_task.name,
        "base_seed": base_seed,
        "scaling_modes": modes,
        "primary_scaling_mode": str(info["primary_scaling_mode"]),
        "scale_reference_rank": int(scaling["reference_rank"]),
        "n_sites": len(sites),
        "n_sweep_rows": len(sweeps),
        "n_budget_rows": len(budgets),
        "n_allocation_rows": len(allocations),
        "adaptation_replicates": int(cfg.get("protocol", {}).get("adaptation_replicates", 1)),
        "sweep_replicates": int(cfg.get("protocol", {}).get("sweep_replicates", 1)),
        "exact_modular_evaluation": exact_modular,
        "exact_cost_matched_rules": matched_rules,
        "primary_allocation_rule": str(info.get("primary_allocation_rule", "")),
        "primary_reference_rule": str(info.get("primary_reference_rule", "")),
        "primary_metric": str(info.get("primary_metric", "")),
        "independent_unit": str(info.get("independent_unit", "")),
        **calibration_summary,
    }
    print("Rank-scaling run validation: PASS")
    for key, value in summary.items():
        if key != "status":
            print(f"{key}: {value}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a corrected rank-scaling synthetic-transformer run."
    )
    parser.add_argument("run_dir")
    parser.add_argument(
        "--allow-divergence",
        action="store_true",
        help="Do not reject finite rows marked diverged (publication smoke rejects them).",
    )
    args = parser.parse_args()
    validate(Path(args.run_dir), fail_on_divergence=not args.allow_divergence)


if __name__ == "__main__":
    main()
