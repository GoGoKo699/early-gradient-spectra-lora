#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aggregate_step3_replicates import (
    cluster_bootstrap_interval,
    exact_sign_flip_p,
)
from scripts.validate_rank_scaling_run import validate as validate_source_run
from strank.inference import (
    task_stratified_bootstrap_interval,
    task_stratified_mean,
    task_stratified_sign_flip_p,
)
from strank.protocol import (
    SYNTHETIC_TRANSFORMER_PROTOCOL_VERSION,
    TRANSFORMER_AGGREGATE_SCHEMA_VERSION,
    TRANSFORMER_RELEASE_SCHEMA_VERSION,
)

PUBLICATION_MIN_TASKS = 2
PUBLICATION_MIN_RUNS_PER_TASK = 5
PUBLICATION_MIN_NULL_MAXIMA = 4096
PUBLICATION_MIN_NULL_UNCERTAINTY_RESAMPLES = 2000
PUBLICATION_PRIMARY_NULL_QUANTILE = 0.995
EXPECTED_SCALING_MODES = {"fixed_update_scale", "standard", "rslora"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing required JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _safe_relative_path(value: str, label: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe {label}: {value!r}")
    return Path(*pure.parts)


def _read_checksums(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"missing checksum manifest: {path}")
    entries: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            digest, relative = raw.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"invalid checksum line {line_number}: {raw!r}") from exc
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"invalid SHA-256 digest on line {line_number}")
        _safe_relative_path(relative, f"checksum path on line {line_number}")
        if relative in entries:
            raise ValueError(f"duplicate checksum entry: {relative}")
        entries[relative] = digest
    if not entries:
        raise ValueError("checksum manifest is empty")
    return entries


def _common_summary_value(summaries: list[dict], key: str):
    values = [summary.get(key) for summary in summaries]
    canonical = [json.dumps(value, sort_keys=True) for value in values]
    if not values or len(set(canonical)) != 1:
        raise ValueError(f"source runs disagree on {key}: {values!r}")
    return values[0]



PRIMARY_RUN_COLUMNS = [
    "task_family",
    "run",
    "seed",
    "scaling_mode",
    "condition_id",
    "rule",
    "allocation_rule",
    "is_primary_allocation_rule",
    "n_budgets",
    "mean_adaptation_replicates",
    "loss_delta_vs_exact_uniform",
    "acc_delta_vs_exact_uniform",
    "mean_actual_cost",
]


def _boolean_mask(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    normalised = series.astype(str).str.strip().str.lower()
    if not normalised.isin({"true", "false"}).all():
        bad = sorted(set(normalised) - {"true", "false"})
        raise ValueError(f"{label} contains non-boolean values: {bad}")
    return normalised.eq("true")


def _recompute_primary_run_deltas(
    manifest: dict,
    release_dir: Path,
    release_by_id: dict[str, dict],
    summary_by_id: dict[str, dict],
) -> pd.DataFrame:
    """Recompute the confirmatory run rows directly from released raw budgets."""

    rows: list[dict[str, object]] = []
    primary_mode = str(manifest["primary_scaling_mode"])
    primary_rule = str(manifest["primary_allocation_rule"])
    for run_id in sorted(release_by_id):
        release_entry = release_by_id[run_id]
        summary = summary_by_id[run_id]
        source_dir = release_dir / _safe_relative_path(
            str(release_entry["relative_path"]), "source-run path"
        )
        budget_path = source_dir / "budget" / "budget_results.csv"
        raw = pd.read_csv(budget_path)
        required = {
            "scaling_mode",
            "condition_id",
            "rule",
            "comparison_role",
            "matched_baseline_condition_id",
            "is_primary_allocation_rule",
            "budget",
            "adaptation_replicate",
            "actual_cost",
            "final_val_loss",
            "final_val_accuracy",
        }
        if not required.issubset(raw.columns):
            raise ValueError(
                f"raw budget table for {run_id} is missing columns: "
                f"{sorted(required - set(raw.columns))}"
            )
        primary_flags = _boolean_mask(
            raw["is_primary_allocation_rule"],
            f"{run_id}.is_primary_allocation_rule",
        )
        candidates = raw[
            raw["scaling_mode"].astype(str).eq(primary_mode)
            & raw["rule"].astype(str).eq(primary_rule)
            & raw["comparison_role"].astype(str).eq("candidate")
            & primary_flags
        ].copy()
        if candidates.empty:
            raise ValueError(f"raw budget table for {run_id} has no primary candidate rows")
        candidate_keys = ["budget", "adaptation_replicate"]
        if candidates.duplicated(candidate_keys).any():
            raise ValueError(f"raw primary candidate rows are not unique in {run_id}")
        if candidates["matched_baseline_condition_id"].isna().any():
            raise ValueError(f"raw primary candidate lacks an exact-cost link in {run_id}")

        references = raw[
            raw["scaling_mode"].astype(str).eq(primary_mode)
            & raw["comparison_role"].astype(str).eq("exact_cost_baseline")
        ][
            [
                "budget",
                "adaptation_replicate",
                "condition_id",
                "actual_cost",
                "final_val_loss",
                "final_val_accuracy",
            ]
        ].rename(
            columns={
                "condition_id": "matched_baseline_condition_id",
                "actual_cost": "exact_uniform_cost",
                "final_val_loss": "exact_uniform_loss",
                "final_val_accuracy": "exact_uniform_accuracy",
            }
        )
        reference_keys = candidate_keys + ["matched_baseline_condition_id"]
        if references.duplicated(reference_keys).any():
            raise ValueError(f"raw exact-cost references are not unique in {run_id}")
        paired = candidates.merge(
            references,
            on=reference_keys,
            how="left",
            validate="one_to_one",
        )
        paired_numeric = [
            "actual_cost",
            "exact_uniform_cost",
            "final_val_loss",
            "exact_uniform_loss",
            "final_val_accuracy",
            "exact_uniform_accuracy",
        ]
        for column in paired_numeric:
            values = pd.to_numeric(paired[column], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"raw primary pairing {run_id}.{column} is non-finite")
        if not (
            paired["actual_cost"].astype(int)
            == paired["exact_uniform_cost"].astype(int)
        ).all():
            raise ValueError(f"raw primary exact-cost pairs have unequal costs in {run_id}")
        paired["loss_delta_vs_exact_uniform"] = (
            paired["final_val_loss"] - paired["exact_uniform_loss"]
        )
        paired["acc_delta_vs_exact_uniform"] = (
            paired["final_val_accuracy"] - paired["exact_uniform_accuracy"]
        )
        expected_replicates = int(summary["adaptation_replicates"])
        per_budget = paired.groupby("budget", as_index=False, dropna=False).agg(
            n_adaptation_replicates=("adaptation_replicate", "nunique"),
            loss_delta_vs_exact_uniform=("loss_delta_vs_exact_uniform", "mean"),
            acc_delta_vs_exact_uniform=("acc_delta_vs_exact_uniform", "mean"),
            actual_cost=("actual_cost", "mean"),
        )
        if set(per_budget["n_adaptation_replicates"].astype(int)) != {
            expected_replicates
        }:
            raise ValueError(
                f"raw primary rows for {run_id} do not contain every adaptation replicate"
            )
        condition_ids = set(candidates["condition_id"].astype(str))
        rules = set(candidates["rule"].astype(str))
        if len(condition_ids) != 1 or rules != {primary_rule}:
            raise ValueError(f"raw primary rows have inconsistent identities in {run_id}")
        rows.append(
            {
                "task_family": str(summary["task_name"]),
                "run": run_id,
                "seed": int(summary["base_seed"]),
                "scaling_mode": primary_mode,
                "condition_id": next(iter(condition_ids)),
                "rule": primary_rule,
                "allocation_rule": primary_rule,
                "is_primary_allocation_rule": True,
                "n_budgets": int(len(per_budget)),
                "mean_adaptation_replicates": float(
                    per_budget["n_adaptation_replicates"].mean()
                ),
                "loss_delta_vs_exact_uniform": float(
                    per_budget["loss_delta_vs_exact_uniform"].mean()
                ),
                "acc_delta_vs_exact_uniform": float(
                    per_budget["acc_delta_vs_exact_uniform"].mean()
                ),
                "mean_actual_cost": float(per_budget["actual_cost"].mean()),
            }
        )
    return (
        pd.DataFrame(rows, columns=PRIMARY_RUN_COLUMNS)
        .sort_values(["task_family", "seed", "run"])
        .reset_index(drop=True)
    )


def _recompute_primary_by_task(primary_run: pd.DataFrame) -> pd.DataFrame:
    """Recompute task-specific secondary summaries from verified run rows."""

    rows: list[dict[str, object]] = []
    identity_columns = [
        "task_family",
        "scaling_mode",
        "condition_id",
        "rule",
        "allocation_rule",
        "is_primary_allocation_rule",
    ]
    for keys, group in primary_run.groupby(identity_columns, dropna=False):
        identity = dict(zip(identity_columns, keys))
        loss = group["loss_delta_vs_exact_uniform"].to_numpy(dtype=float)
        accuracy = group["acc_delta_vs_exact_uniform"].to_numpy(dtype=float)
        ci_low, ci_high = cluster_bootstrap_interval(
            loss, seed_parts=tuple(identity.values()) + ("exact_uniform",)
        )
        rows.append(
            {
                **identity,
                "n_independent_runs": int(group["run"].nunique()),
                "mean_loss_delta": float(np.mean(loss)),
                "sem_loss_delta": (
                    float(np.std(loss, ddof=1) / np.sqrt(len(loss)))
                    if len(loss) > 1
                    else 0.0
                ),
                "median_loss_delta": float(np.median(loss)),
                "cluster_bootstrap_ci_low": ci_low,
                "cluster_bootstrap_ci_high": ci_high,
                "exact_sign_flip_p_two_sided": exact_sign_flip_p(loss),
                "run_wins_loss": int(np.sum(loss < 0)),
                "run_ties_loss": int(np.sum(loss == 0)),
                "mean_accuracy_delta": float(np.mean(accuracy)),
                "sem_accuracy_delta": (
                    float(np.std(accuracy, ddof=1) / np.sqrt(len(accuracy)))
                    if len(accuracy) > 1
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("task_family").reset_index(drop=True)


def _validate_analysis_plan_roles(plan: object, expected_roles: dict[str, object]) -> None:
    if not isinstance(plan, dict) or not isinstance(plan.get("analysis"), dict):
        raise ValueError("aggregate analysis plan is not a valid plan mapping")
    if plan.get("protocol_version") != expected_roles["protocol_version"]:
        raise ValueError(
            "analysis plan protocol_version="
            f"{plan.get('protocol_version')!r}; expected "
            f"{expected_roles['protocol_version']!r}"
        )
    analysis = plan["analysis"]
    for key, expected in expected_roles.items():
        if key == "protocol_version":
            continue
        if analysis.get(key) != expected:
            raise ValueError(
                f"analysis plan {key}={analysis.get(key)!r}; expected {expected!r}"
            )


def _validate_aggregate_source_binding(
    manifest: dict,
    aggregate_manifest: dict,
    aggregate_dir: Path,
    release_dir: Path,
    summaries: list[dict],
) -> None:
    aggregate_entries = aggregate_manifest.get("source_runs")
    if not isinstance(aggregate_entries, list) or not aggregate_entries:
        raise ValueError("aggregate manifest does not bind outputs to source-run hashes")
    release_entries = manifest.get("source_runs", [])
    release_by_id = {
        Path(str(entry["relative_path"])).name: entry for entry in release_entries
    }
    summary_by_id = {str(summary["run_id"]): summary for summary in summaries}
    aggregate_by_id: dict[str, dict] = {}
    for entry in aggregate_entries:
        if not isinstance(entry, dict):
            raise ValueError("aggregate source-run entry is not an object")
        run_id = str(entry.get("run_id", ""))
        if not run_id or run_id in aggregate_by_id:
            raise ValueError(f"invalid or duplicate aggregate source run: {run_id!r}")
        aggregate_by_id[run_id] = entry
    if set(aggregate_by_id) != set(release_by_id):
        raise ValueError(
            "aggregate source-run IDs do not match release source runs; "
            f"aggregate={sorted(aggregate_by_id)}, release={sorted(release_by_id)}"
        )

    required_hashed_files = {
        "config.yaml",
        "run_info.json",
        "_SUCCESS.json",
        "calibration/module_stats.csv",
        "calibration/singular_values.npz",
        "calibration/null_maxima.npz",
        "sweeps/site_rank_sweep_metrics.csv",
        "sweeps/site_target_summary.csv",
        "sweeps/site_prediction_fit.csv",
        "budget/budget_results.csv",
        "budget/allocation_comparison.csv",
    }
    metadata_keys = (
        "task_name",
        "base_seed",
        "protocol_version",
        "primary_scaling_mode",
        "primary_allocation_rule",
        "primary_reference_rule",
        "primary_metric",
        "independent_unit",
    )
    for run_id, aggregate_entry in aggregate_by_id.items():
        release_entry = release_by_id[run_id]
        summary = summary_by_id[run_id]
        expected_metadata = {
            "task_name": str(summary["task_name"]),
            "base_seed": int(summary["base_seed"]),
            "protocol_version": str(summary["protocol_version"]),
            "primary_scaling_mode": str(summary["primary_scaling_mode"]),
            "primary_allocation_rule": str(summary["primary_allocation_rule"]),
            "primary_reference_rule": str(summary["primary_reference_rule"]),
            "primary_metric": str(summary["primary_metric"]),
            "independent_unit": str(summary["independent_unit"]),
        }
        for key in metadata_keys:
            if aggregate_entry.get(key) != expected_metadata[key]:
                raise ValueError(
                    f"aggregate source metadata mismatch for {run_id}.{key}: "
                    f"recorded={aggregate_entry.get(key)!r}, "
                    f"expected={expected_metadata[key]!r}"
                )
        hashes = aggregate_entry.get("sha256")
        if not isinstance(hashes, dict) or set(hashes) != required_hashed_files:
            raise ValueError(
                f"aggregate source hashes for {run_id} must cover exactly the required raw files"
            )
        source_dir = release_dir / _safe_relative_path(
            str(release_entry["relative_path"]), "source-run path"
        )
        for relative_text, expected_digest in hashes.items():
            relative = _safe_relative_path(str(relative_text), "aggregate source hash path")
            path = source_dir / relative
            if not path.is_file():
                raise ValueError(f"aggregate-bound source file is missing: {run_id}/{relative_text}")
            if _sha256(path) != str(expected_digest):
                raise ValueError(f"aggregate source hash mismatch: {run_id}/{relative_text}")

    expected_roles = {
        "protocol_version": manifest["protocol_version"],
        "primary_scaling_mode": manifest["primary_scaling_mode"],
        "primary_allocation_rule": manifest["primary_allocation_rule"],
        "primary_reference_rule": manifest["primary_reference_rule"],
        "primary_metric": manifest["primary_metric"],
        "independent_unit": manifest["unit_of_inference"],
    }
    for key, expected in expected_roles.items():
        if aggregate_manifest.get(key) != expected:
            raise ValueError(
                f"aggregate manifest {key}={aggregate_manifest.get(key)!r}; expected {expected!r}"
            )
    if aggregate_manifest.get("scaling_conditions_analyzed_separately") is not True:
        raise ValueError("aggregate pooled distinct scaling conditions")
    if aggregate_manifest.get("exact_cost_primary_outputs") is not True:
        raise ValueError("aggregate does not certify exact-cost primary outputs")
    if set(aggregate_manifest.get("scaling_modes", [])) != EXPECTED_SCALING_MODES:
        raise ValueError("aggregate scaling modes do not match the publication design")

    expected_task_counts = Counter(str(item["task_name"]) for item in summaries)
    recorded_task_counts = aggregate_manifest.get("independent_runs_by_task")
    if not isinstance(recorded_task_counts, dict):
        raise ValueError("aggregate manifest lacks independent_runs_by_task")
    normalised_task_counts = {
        str(task): int(count) for task, count in recorded_task_counts.items()
    }
    if normalised_task_counts != dict(expected_task_counts):
        raise ValueError(
            "aggregate task/run counts do not match source runs; "
            f"recorded={normalised_task_counts}, expected={dict(expected_task_counts)}"
        )

    required_outputs = (
        "all_budget_results.csv",
        "run_level_budget_results.csv",
        "deltas_exact_cost.csv",
        "run_level_deltas_exact_cost.csv",
        "delta_summary_exact_cost.csv",
        "run_cluster_deltas_exact_cost.csv",
        "run_cluster_summary_exact_cost.csv",
        "primary_run_deltas.csv",
        "primary_analysis_by_task.csv",
        "primary_analysis.csv",
        "analysis_plan.yaml",
    )
    for name in required_outputs:
        if not (aggregate_dir / name).is_file():
            raise ValueError(f"publication aggregate is missing required output: {name}")

    plan_path = aggregate_dir / "analysis_plan.yaml"
    recorded_plan_hash = str(aggregate_manifest.get("analysis_plan_sha256", ""))
    if len(recorded_plan_hash) != 64 or _sha256(plan_path) != recorded_plan_hash:
        raise ValueError("aggregate analysis-plan hash is missing or incorrect")
    aggregate_entry = manifest.get("aggregate")
    if not isinstance(aggregate_entry, dict):
        raise ValueError("release manifest is missing aggregate metadata")
    if aggregate_entry.get("analysis_plan_sha256") != recorded_plan_hash:
        raise ValueError("release manifest aggregate analysis-plan hash is stale")
    if (
        aggregate_entry.get("analysis_plan_version")
        != aggregate_manifest.get("analysis_plan_version")
    ):
        raise ValueError("release manifest aggregate analysis-plan version is stale")
    if aggregate_manifest.get("publication_release_id") != manifest.get("release_id"):
        raise ValueError("aggregate publication_release_id does not match release_id")
    if (
        aggregate_entry.get("publication_release_id")
        != aggregate_manifest.get("publication_release_id")
    ):
        raise ValueError("release manifest aggregate publication_release_id is stale")
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    _validate_analysis_plan_roles(plan, expected_roles)
    analysis = plan["analysis"]
    if aggregate_manifest.get("analysis_plan_version") != plan.get("plan_version"):
        raise ValueError("aggregate analysis-plan version does not match copied plan")
    for key in (
        "primary_estimand",
        "budget_weighting",
        "two_sided_alpha",
        "cluster_bootstrap_resamples",
        "exact_sign_flip_test",
        "primary_scope",
        "task_weighting",
        "generalization_scope",
    ):
        if aggregate_manifest.get(key) != analysis.get(key):
            raise ValueError(
                f"aggregate manifest {key}={aggregate_manifest.get(key)!r}; "
                f"plan declares {analysis.get(key)!r}"
            )
    expected_analysis_design = {
        "primary_scope": "task_stratified_omnibus",
        "task_weighting": "equal_weight_across_task_families",
        "generalization_scope": "conditional_on_prespecified_task_families",
        "secondary_task_specific": True,
        "budget_weighting": "equal_weight_within_run",
        "exact_sign_flip_test": "two_sided_task_stratified",
    }
    for key, expected in expected_analysis_design.items():
        if analysis.get(key) != expected:
            raise ValueError(
                f"analysis plan {key}={analysis.get(key)!r}; expected {expected!r}"
            )
    if abs(float(analysis.get("two_sided_alpha", 0.0)) - 0.05) > 1e-15:
        raise ValueError("analysis plan two-sided alpha must be 0.05")
    if abs(float(analysis.get("confidence", 0.0)) - 0.95) > 1e-15:
        raise ValueError("analysis plan confidence must be 0.95")
    if int(analysis.get("cluster_bootstrap_resamples", 0)) != 10_000:
        raise ValueError("analysis plan cluster bootstrap must use 10000 resamples")
    expected_plan_version = str(plan.get("plan_version", ""))
    for run_id, release_entry in release_by_id.items():
        source_dir = release_dir / _safe_relative_path(
            str(release_entry["relative_path"]), "source-run path"
        )
        source_cfg = yaml.safe_load(
            (source_dir / "config.yaml").read_text(encoding="utf-8")
        )
        if not isinstance(source_cfg, dict):
            raise ValueError(f"source config is not a mapping: {run_id}")
        protocol_cfg = source_cfg.get("protocol")
        run_cfg = source_cfg.get("run")
        task_cfg = source_cfg.get("task")
        if not isinstance(protocol_cfg, dict) or not isinstance(run_cfg, dict):
            raise ValueError(f"source config lacks protocol/run mapping: {run_id}")
        if not isinstance(task_cfg, dict):
            raise ValueError(f"source config lacks task mapping: {run_id}")
        if protocol_cfg.get("publication_plan_sha256") != recorded_plan_hash:
            raise ValueError(f"source config plan hash mismatch: {run_id}")
        if protocol_cfg.get("publication_plan_version") != expected_plan_version:
            raise ValueError(f"source config plan version mismatch: {run_id}")
        if protocol_cfg.get("publication_release_id") != manifest.get("release_id"):
            raise ValueError(f"source config release ID mismatch: {run_id}")
        if not run_id.startswith(str(manifest.get("release_id")) + "_"):
            raise ValueError(f"source run is not namespaced by its release ID: {run_id}")
        if str(run_cfg.get("name", "")) != run_id:
            raise ValueError(f"source config run.name mismatch: {run_id}")
        if int(run_cfg.get("seed", -1)) != int(summary_by_id[run_id]["base_seed"]):
            raise ValueError(f"source config seed mismatch: {run_id}")
        if str(task_cfg.get("name", "")) != str(summary_by_id[run_id]["task_name"]):
            raise ValueError(f"source config task mismatch: {run_id}")
    plan_tasks = {str(item.get("task_name")) for item in plan.get("tasks", [])}
    if plan_tasks != set(expected_task_counts):
        raise ValueError("analysis-plan tasks do not match release task families")
    plan_seeds = {int(value) for value in plan.get("seeds", [])}
    release_seeds_by_task = {
        task: {
            int(item["base_seed"])
            for item in summaries
            if str(item["task_name"]) == task
        }
        for task in expected_task_counts
    }
    if any(values != plan_seeds for values in release_seeds_by_task.values()):
        raise ValueError("analysis-plan seeds do not match release source runs")

    cluster = pd.read_csv(aggregate_dir / "run_cluster_summary_exact_cost.csv")
    required_columns = {
        "task_family",
        "scaling_mode",
        "allocation_rule",
        "is_primary_allocation_rule",
        "n_independent_runs",
        "mean_loss_delta",
        "cluster_bootstrap_ci_low",
        "cluster_bootstrap_ci_high",
        "exact_sign_flip_p_two_sided",
    }
    if not required_columns.issubset(cluster.columns):
        raise ValueError(
            "run-cluster summary is missing columns: "
            f"{sorted(required_columns - set(cluster.columns))}"
        )
    primary_flags = cluster["is_primary_allocation_rule"].astype(str).str.lower().eq("true")
    primary = cluster[
        cluster["scaling_mode"].astype(str).eq(manifest["primary_scaling_mode"])
        & cluster["allocation_rule"].astype(str).eq(manifest["primary_allocation_rule"])
        & primary_flags
    ]
    if len(primary) != len(expected_task_counts):
        raise ValueError(
            "run-cluster summary must contain exactly one pre-specified primary row per task"
        )
    if set(primary["task_family"].astype(str)) != set(expected_task_counts):
        raise ValueError("primary run-cluster rows do not cover every task family")
    for row in primary.itertuples(index=False):
        expected_n = int(expected_task_counts[str(row.task_family)])
        if int(row.n_independent_runs) != expected_n:
            raise ValueError(
                f"primary aggregate run count mismatch for {row.task_family}: "
                f"recorded={row.n_independent_runs}, expected={expected_n}"
            )
    primary_file = pd.read_csv(aggregate_dir / "primary_analysis_by_task.csv")

    run_cluster = pd.read_csv(aggregate_dir / "run_cluster_deltas_exact_cost.csv")
    run_required = set(PRIMARY_RUN_COLUMNS)
    if not run_required.issubset(run_cluster.columns):
        raise ValueError(
            "run-cluster delta table is missing columns: "
            f"{sorted(run_required - set(run_cluster.columns))}"
        )
    run_primary_flags = _boolean_mask(
        run_cluster["is_primary_allocation_rule"],
        "run_cluster_deltas_exact_cost.is_primary_allocation_rule",
    )
    aggregate_run_primary = run_cluster[
        run_cluster["scaling_mode"].astype(str).eq(manifest["primary_scaling_mode"])
        & run_cluster["allocation_rule"].astype(str).eq(
            manifest["primary_allocation_rule"]
        )
        & run_primary_flags
    ][PRIMARY_RUN_COLUMNS].sort_values(
        ["task_family", "seed", "run"]
    ).reset_index(drop=True)
    if len(aggregate_run_primary) != len(summaries):
        raise ValueError(
            "primary run-level table does not contain one row per source run"
        )
    if aggregate_run_primary[["task_family", "run"]].duplicated().any():
        raise ValueError("primary run-level table contains duplicate task/run rows")

    expected_run_primary = _recompute_primary_run_deltas(
        manifest, release_dir, release_by_id, summary_by_id
    )
    recorded_run_primary = pd.read_csv(aggregate_dir / "primary_run_deltas.csv")
    missing_primary_columns = set(PRIMARY_RUN_COLUMNS) - set(recorded_run_primary.columns)
    if missing_primary_columns:
        raise ValueError(
            "primary_run_deltas.csv is missing columns: "
            f"{sorted(missing_primary_columns)}"
        )
    recorded_run_primary = recorded_run_primary[PRIMARY_RUN_COLUMNS].sort_values(
        ["task_family", "seed", "run"]
    ).reset_index(drop=True)
    for label, observed in (
        ("run_cluster_deltas_exact_cost.csv", aggregate_run_primary),
        ("primary_run_deltas.csv", recorded_run_primary),
    ):
        try:
            pd.testing.assert_frame_equal(
                expected_run_primary,
                observed,
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
        except AssertionError as exc:
            raise ValueError(
                f"{label} does not recompute from released raw exact-cost budget rows"
            ) from exc

    expected_primary_by_task = _recompute_primary_by_task(expected_run_primary)
    task_summary_columns = expected_primary_by_task.columns.tolist()
    if not set(task_summary_columns).issubset(primary.columns):
        raise ValueError("run-cluster primary rows lack required task-summary columns")
    if not set(task_summary_columns).issubset(primary_file.columns):
        raise ValueError(
            "primary_analysis_by_task.csv is missing required secondary columns"
        )
    aggregate_primary_by_task = primary[task_summary_columns].sort_values(
        "task_family"
    ).reset_index(drop=True)
    recorded_primary_by_task = primary_file[task_summary_columns].sort_values(
        "task_family"
    ).reset_index(drop=True)
    for label, observed in (
        ("run_cluster_summary_exact_cost.csv", aggregate_primary_by_task),
        ("primary_analysis_by_task.csv", recorded_primary_by_task),
    ):
        try:
            pd.testing.assert_frame_equal(
                expected_primary_by_task,
                observed,
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
        except AssertionError as exc:
            raise ValueError(
                f"{label} does not recompute from verified primary run rows"
            ) from exc

    overall = pd.read_csv(aggregate_dir / "primary_analysis.csv")
    if len(overall) != 1:
        raise ValueError("primary_analysis.csv must contain one omnibus row")
    required_omnibus_columns = {
        "analysis_scope",
        "task_weighting",
        "generalization_scope",
        "scaling_mode",
        "condition_id",
        "rule",
        "allocation_rule",
        "reference_rule",
        "metric",
        "n_task_families",
        "n_independent_runs",
        "min_runs_per_task",
        "max_runs_per_task",
        "min_budgets_per_run",
        "max_budgets_per_run",
        "mean_loss_delta",
        "median_task_mean_loss_delta",
        "cluster_bootstrap_ci_low",
        "cluster_bootstrap_ci_high",
        "two_sided_alpha",
        "confidence",
        "cluster_bootstrap_resamples",
        "exact_sign_flip_n_units",
        "exact_sign_flip_n_patterns",
        "nominal_min_two_sided_p",
        "exact_sign_flip_p_two_sided",
        "run_wins_loss",
        "run_ties_loss",
        "task_wins_loss",
        "task_ties_loss",
        "mean_accuracy_delta",
    }
    if not required_omnibus_columns.issubset(overall.columns):
        raise ValueError(
            "primary_analysis.csv is missing columns: "
            f"{sorted(required_omnibus_columns - set(overall.columns))}"
        )
    row = overall.iloc[0]
    expected_scope = {
        "analysis_scope": "task_stratified_omnibus",
        "task_weighting": "equal_weight_across_task_families",
        "generalization_scope": "conditional_on_prespecified_task_families",
    }
    for key, expected in expected_scope.items():
        if analysis.get(
            {
                "analysis_scope": "primary_scope",
                "task_weighting": "task_weighting",
                "generalization_scope": "generalization_scope",
            }[key]
        ) != expected or str(row[key]) != expected:
            raise ValueError(f"primary omnibus {key} is not pre-specified")
    loss_values = expected_run_primary[
        "loss_delta_vs_exact_uniform"
    ].to_numpy(dtype=float)
    accuracy_values = expected_run_primary[
        "acc_delta_vs_exact_uniform"
    ].to_numpy(dtype=float)
    task_labels = expected_run_primary["task_family"].astype(str).tolist()
    alpha = float(analysis.get("two_sided_alpha", 0.0))
    confidence = float(analysis.get("confidence", 0.0))
    n_resamples = int(analysis.get("cluster_bootstrap_resamples", 0))
    ci_low, ci_high = task_stratified_bootstrap_interval(
        loss_values,
        task_labels,
        confidence=confidence,
        n_resamples=n_resamples,
        seed_parts=(
            "primary",
            str(manifest["primary_scaling_mode"]),
            str(manifest["primary_allocation_rule"]),
        ),
    )
    task_means = expected_run_primary.groupby("task_family")[
        "loss_delta_vs_exact_uniform"
    ].mean()
    task_counts = expected_run_primary.groupby("task_family")["run"].nunique()
    expected_numeric = {
        "n_task_families": len(task_counts),
        "n_independent_runs": len(expected_run_primary),
        "min_runs_per_task": int(task_counts.min()),
        "max_runs_per_task": int(task_counts.max()),
        "min_budgets_per_run": int(expected_run_primary["n_budgets"].min()),
        "max_budgets_per_run": int(expected_run_primary["n_budgets"].max()),
        "mean_loss_delta": task_stratified_mean(loss_values, task_labels),
        "median_task_mean_loss_delta": float(np.median(task_means)),
        "cluster_bootstrap_ci_low": ci_low,
        "cluster_bootstrap_ci_high": ci_high,
        "two_sided_alpha": alpha,
        "confidence": confidence,
        "cluster_bootstrap_resamples": n_resamples,
        "exact_sign_flip_n_units": len(expected_run_primary),
        "exact_sign_flip_n_patterns": 2 ** len(expected_run_primary),
        "nominal_min_two_sided_p": 2.0 / (2 ** len(expected_run_primary)),
        "exact_sign_flip_p_two_sided": task_stratified_sign_flip_p(
            loss_values, task_labels
        ),
        "run_wins_loss": int(np.sum(loss_values < 0)),
        "run_ties_loss": int(np.sum(loss_values == 0)),
        "task_wins_loss": int(np.sum(task_means.to_numpy() < 0)),
        "task_ties_loss": int(np.sum(task_means.to_numpy() == 0)),
        "mean_accuracy_delta": task_stratified_mean(accuracy_values, task_labels),
    }
    for key, expected in expected_numeric.items():
        observed = float(row[key])
        if not np.isclose(observed, float(expected), rtol=1e-12, atol=1e-12):
            raise ValueError(
                f"primary omnibus {key}={observed}; expected {expected}"
            )
    expected_text = {
        "scaling_mode": manifest["primary_scaling_mode"],
        "condition_id": expected_run_primary["condition_id"].iloc[0],
        "rule": expected_run_primary["rule"].iloc[0],
        "allocation_rule": manifest["primary_allocation_rule"],
        "reference_rule": manifest["primary_reference_rule"],
        "metric": manifest["primary_metric"],
    }
    for key, expected in expected_text.items():
        if str(row[key]) != str(expected):
            raise ValueError(
                f"primary omnibus {key}={row[key]!r}; expected {expected!r}"
            )


def _validate_attached_aggregate(
    manifest: dict, summaries: list[dict], release_dir: Path
) -> None:
    aggregate = manifest.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError("release is missing its aggregate directory")
    aggregate_relative = _safe_relative_path(
        str(aggregate.get("relative_path", "")), "aggregate path"
    )
    if len(aggregate_relative.parts) != 2 or aggregate_relative.parts[0] != "aggregate":
        raise ValueError("aggregate must be a direct child of aggregate/")
    aggregate_dir = release_dir / aggregate_relative
    aggregate_manifest = _read_json(aggregate_dir / "manifest.json")
    if (
        aggregate_manifest.get("aggregate_schema_version")
        != TRANSFORMER_AGGREGATE_SCHEMA_VERSION
    ):
        raise ValueError("aggregate schema version is missing or stale")
    if (
        aggregate.get("aggregate_schema_version")
        != aggregate_manifest.get("aggregate_schema_version")
    ):
        raise ValueError("release manifest aggregate schema version is stale")
    aggregate_count = int(aggregate_manifest.get("n_independent_runs", -1))
    if aggregate_count != len(summaries):
        raise ValueError("aggregate independent-run count does not match source runs")
    if aggregate.get("n_independent_runs") != aggregate_count:
        raise ValueError("release manifest aggregate run count is stale")
    unit = str(aggregate_manifest.get("unit_of_inference", ""))
    if aggregate.get("unit_of_inference") != unit:
        raise ValueError("release manifest aggregate inference unit is stale")
    if "run" not in unit or "adaptation" not in unit:
        raise ValueError("aggregate manifest does not declare run-cluster inference")
    git_info = _read_json(release_dir / "provenance" / "git.json")
    if git_info.get("status_porcelain"):
        raise ValueError("plan-bound releases must be built from a clean committed worktree")
    head = str(git_info.get("head") or "")
    if len(head) != 40 or any(ch not in "0123456789abcdef" for ch in head.lower()):
        raise ValueError("plan-bound release provenance lacks a valid Git commit")
    _validate_aggregate_source_binding(
        manifest, aggregate_manifest, aggregate_dir, release_dir, summaries
    )


def _validate_publication_design(manifest: dict, summaries: list[dict], release_dir: Path) -> None:
    task_counts = Counter(str(item["task_name"]) for item in summaries)
    if len(task_counts) < PUBLICATION_MIN_TASKS:
        raise ValueError(
            f"publication release needs at least {PUBLICATION_MIN_TASKS} task families; "
            f"found {dict(task_counts)}"
        )
    underpowered = {
        task: count for task, count in task_counts.items() if count < PUBLICATION_MIN_RUNS_PER_TASK
    }
    if underpowered:
        raise ValueError(
            "publication release has too few independent runs per task; "
            f"minimum={PUBLICATION_MIN_RUNS_PER_TASK}, found={underpowered}"
        )
    if len(summaries) > 20:
        raise ValueError(
            "publication release has more than 20 independent units, so the "
            "declared sign-flip test would not be exactly enumerated"
        )
    if 2.0 / (2 ** len(summaries)) > 0.05:
        raise ValueError("publication release has insufficient exact-test resolution")
    pairs = [(str(item["task_name"]), int(item["base_seed"])) for item in summaries]
    if len(set(pairs)) != len(pairs):
        raise ValueError("publication release contains duplicate task/base-seed units")
    for summary in summaries:
        if set(summary["scaling_modes"]) != EXPECTED_SCALING_MODES:
            raise ValueError(
                f"source run {summary['run_id']} lacks the full pre-specified scaling design"
            )
        if summary["primary_scaling_mode"] != "fixed_update_scale":
            raise ValueError("publication primary scaling mode must be fixed_update_scale")
        if int(summary["null_bootstrap"]) < PUBLICATION_MIN_NULL_MAXIMA:
            raise ValueError(
                f"source run {summary['run_id']} has only {summary['null_bootstrap']} null maxima; "
                f"publication minimum is {PUBLICATION_MIN_NULL_MAXIMA}"
            )
        if abs(float(summary["null_quantile"]) - PUBLICATION_PRIMARY_NULL_QUANTILE) > 1e-15:
            raise ValueError("publication primary null quantile must be 0.995")
        if int(summary["null_uncertainty_resamples"]) < PUBLICATION_MIN_NULL_UNCERTAINTY_RESAMPLES:
            raise ValueError(
                f"source run {summary['run_id']} has too few null uncertainty resamples"
            )
        if summary["task_name"] == "modular" and not summary["exact_modular_evaluation"]:
            raise ValueError("publication modular runs must use exhaustive evaluation")
        if not summary["exact_cost_matched_rules"]:
            raise ValueError("publication runs must include pre-specified exact-cost comparators")
        if summary["primary_reference_rule"] != "uniform_exact_cost":
            raise ValueError("publication primary reference must be uniform_exact_cost")
        if summary["primary_allocation_rule"] not in summary["exact_cost_matched_rules"]:
            raise ValueError("primary allocation rule lacks an exact-cost comparator")

    if not isinstance(manifest.get("aggregate"), dict):
        raise ValueError("publication release is missing its aggregate directory")

    git_info = _read_json(release_dir / "provenance" / "git.json")
    if git_info.get("status_porcelain"):
        raise ValueError("publication release must be built from a clean committed worktree")


def validate_release(release_dir: Path, *, expected_kind: str | None = None) -> dict:
    release_dir = release_dir.resolve()
    if not release_dir.is_dir():
        raise ValueError(f"release directory does not exist: {release_dir}")
    if any(path.is_symlink() for path in release_dir.rglob("*")):
        raise ValueError("release must not contain symbolic links")

    checksum_path = release_dir / "SHA256SUMS.txt"
    recorded = _read_checksums(checksum_path)
    actual_files = {
        path.relative_to(release_dir).as_posix()
        for path in release_dir.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if set(recorded) != actual_files:
        unlisted = sorted(actual_files - set(recorded))
        missing = sorted(set(recorded) - actual_files)
        raise ValueError(f"checksum file set mismatch; unlisted={unlisted}, missing={missing}")
    for relative, expected_digest in recorded.items():
        actual_digest = _sha256(release_dir / _safe_relative_path(relative, "checksum path"))
        if actual_digest != expected_digest:
            raise ValueError(
                f"checksum mismatch for {relative}: expected {expected_digest}, got {actual_digest}"
            )

    manifest = _read_json(release_dir / "release_manifest.json")
    if manifest.get("release_id") != release_dir.name:
        raise ValueError("release_manifest release_id does not match directory name")
    release_kind = str(manifest.get("release_kind"))
    if release_kind not in {"smoke", "publication"}:
        raise ValueError(f"unsupported release kind: {release_kind!r}")
    if expected_kind is not None and release_kind != expected_kind:
        raise ValueError(f"release kind {release_kind!r} != expected {expected_kind!r}")
    if manifest.get("release_schema_version") != TRANSFORMER_RELEASE_SCHEMA_VERSION:
        raise ValueError("release manifest schema version is missing or stale")
    if manifest.get("protocol_version") != SYNTHETIC_TRANSFORMER_PROTOCOL_VERSION:
        raise ValueError("release manifest protocol version is not current")
    if manifest.get("checksum_algorithm") != "sha256":
        raise ValueError("release manifest does not declare SHA-256")
    if manifest.get("primary_scaling_mode") != "fixed_update_scale":
        raise ValueError("release manifest primary scaling mode is not fixed_update_scale")

    source_entries = manifest.get("source_runs")
    if not isinstance(source_entries, list) or not source_entries:
        raise ValueError("release manifest has no source runs")
    summaries = []
    seen_paths: set[str] = set()
    for entry in source_entries:
        if not isinstance(entry, dict):
            raise ValueError("source-run manifest entry is not an object")
        relative_text = str(entry.get("relative_path", ""))
        if relative_text in seen_paths:
            raise ValueError(f"duplicate source-run manifest path: {relative_text}")
        seen_paths.add(relative_text)
        relative = _safe_relative_path(relative_text, "source-run path")
        if len(relative.parts) != 2 or relative.parts[0] != "source_runs":
            raise ValueError(
                f"source run must be a direct child of source_runs/: {relative_text}"
            )
        source_dir = release_dir / relative
        summary = validate_source_run(source_dir, fail_on_divergence=True)
        checks = {
            "protocol_version": str(summary["protocol_version"]),
            "base_seed": int(summary["base_seed"]),
            "task_name": str(summary["task_name"]),
            "scaling_modes": list(summary["scaling_modes"]),
            "primary_scaling_mode": str(summary["primary_scaling_mode"]),
            "scale_reference_rank": int(summary["scale_reference_rank"]),
            "primary_allocation_rule": str(summary["primary_allocation_rule"]),
            "primary_reference_rule": str(summary["primary_reference_rule"]),
            "primary_metric": str(summary["primary_metric"]),
            "independent_unit": str(summary["independent_unit"]),
            "n_sites": int(summary["n_sites"]),
            "n_sweep_rows": int(summary["n_sweep_rows"]),
            "n_budget_rows": int(summary["n_budget_rows"]),
            "n_allocation_rows": int(summary["n_allocation_rows"]),
            "adaptation_replicates": int(summary["adaptation_replicates"]),
            "null_bootstrap": int(summary["null_bootstrap"]),
            "null_quantile": float(summary["null_quantile"]),
            "null_uncertainty_resamples": int(summary["null_uncertainty_resamples"]),
            "exact_modular_evaluation": bool(summary["exact_modular_evaluation"]),
            "exact_cost_matched_rules": list(summary["exact_cost_matched_rules"]),
        }
        for key, expected in checks.items():
            recorded_value = entry.get(key)
            if recorded_value != expected:
                raise ValueError(
                    f"source-run manifest mismatch for {relative_text}.{key}: "
                    f"recorded={recorded_value!r}, validated={expected!r}"
                )
        summaries.append(summary)

    common_protocol = {
        key: _common_summary_value(summaries, key)
        for key in (
            "protocol_version",
            "scaling_modes",
            "primary_scaling_mode",
            "scale_reference_rank",
            "primary_allocation_rule",
            "primary_reference_rule",
            "primary_metric",
            "independent_unit",
            "exact_cost_matched_rules",
        )
    }
    expected_manifest = {
        "protocol_version": common_protocol["protocol_version"],
        "unit_of_inference": common_protocol["independent_unit"],
        "cluster_structure_note": (
            "adaptation replicates and budgets are nested within each independent "
            "task/base-model seed run"
        ),
        "primary_scaling_mode": common_protocol["primary_scaling_mode"],
        "sensitivity_scaling_modes": sorted(
            set(common_protocol["scaling_modes"])
            - {common_protocol["primary_scaling_mode"]}
        ),
        "scale_reference_rank": common_protocol["scale_reference_rank"],
        "primary_allocation_rule": common_protocol["primary_allocation_rule"],
        "primary_reference_rule": common_protocol["primary_reference_rule"],
        "primary_metric": common_protocol["primary_metric"],
        "exact_cost_matched_rules": common_protocol["exact_cost_matched_rules"],
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"release manifest mismatch for {key}: "
                f"recorded={manifest.get(key)!r}, validated={expected!r}"
            )

    required_snapshot = [
        "code_snapshot/strank/protocol.py",
        "code_snapshot/strank/scaling.py",
        "code_snapshot/strank/allocation.py",
        "code_snapshot/strank/calibrate.py",
        "code_snapshot/strank/spectral.py",
        "code_snapshot/strank/inference.py",
        "code_snapshot/scripts/run_synthetic_transformer.py",
        "code_snapshot/scripts/aggregate_step4_tasks.py",
        "code_snapshot/scripts/validate_rank_scaling_run.py",
        "code_snapshot/scripts/build_transformer_release.py",
        "code_snapshot/scripts/validate_transformer_release.py",
        "code_snapshot/scripts/run_transformer_publication.py",
        "code_snapshot/scripts/run_transformer_publication.sh",
        "code_snapshot/configs/transformer_publication_plan.yaml",
        "code_snapshot/configs/transformer_publication_smoke_plan.yaml",
        "code_snapshot/requirements.txt",
        "code_snapshot/pyproject.toml",
        "code_snapshot/tests/test_core.py",
        "code_snapshot/tests/test_publication_driver.py",
        "code_snapshot/tests/test_task_stratified_inference.py",
        "provenance/python_environment.txt",
        "provenance/system.json",
        "provenance/git.json",
    ]
    for relative in required_snapshot:
        if not (release_dir / relative).is_file():
            raise ValueError(f"release is missing required provenance file: {relative}")
    protocol_text = (release_dir / "code_snapshot" / "strank" / "protocol.py").read_text(
        encoding="utf-8"
    )
    if SYNTHETIC_TRANSFORMER_PROTOCOL_VERSION not in protocol_text:
        raise ValueError("code snapshot does not declare the release protocol version")

    if manifest.get("aggregate") is not None:
        _validate_attached_aggregate(manifest, summaries, release_dir)
    if release_kind == "publication":
        _validate_publication_design(manifest, summaries, release_dir)

    summary = {
        "status": "PASS",
        "release_id": release_dir.name,
        "release_kind": release_kind,
        "release_schema_version": TRANSFORMER_RELEASE_SCHEMA_VERSION,
        "protocol_version": SYNTHETIC_TRANSFORMER_PROTOCOL_VERSION,
        "n_source_runs": len(summaries),
        "tasks": sorted({item["task_name"] for item in summaries}),
        "seeds": sorted({int(item["base_seed"]) for item in summaries}),
        "scaling_modes": sorted({mode for item in summaries for mode in item["scaling_modes"]}),
        "n_files": len(recorded) + 1,
    }
    print("Transformer release validation: PASS")
    for key, value in summary.items():
        if key != "status":
            print(f"{key}: {value}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a self-contained transformer release.")
    parser.add_argument("release_dir")
    parser.add_argument("--expected-kind", choices=("smoke", "publication"), default=None)
    args = parser.parse_args()
    validate_release(Path(args.release_dir), expected_kind=args.expected_kind)


if __name__ == "__main__":
    main()
