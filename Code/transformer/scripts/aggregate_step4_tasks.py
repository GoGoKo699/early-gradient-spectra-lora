#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from scripts.aggregate_step3_replicates import (
    _common_non_null,
    _normalise_protocol_columns,
    _run_level_budget,
    _source_manifest_entry,
    _write_delta_outputs,
    collect_runs,
    infer_seed_from_run,
    sem,
)
from scripts.validate_rank_scaling_run import validate as validate_source_run
from strank.inference import (
    task_stratified_bootstrap_interval,
    task_stratified_mean,
    task_stratified_sign_flip_p,
)
from strank.protocol import TRANSFORMER_AGGREGATE_SCHEMA_VERSION


def _task_family(run_dir: Path) -> str:
    info_path = run_dir / "run_info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        return str(info.get("task_name", "unknown"))
    name = run_dir.name.lower()
    if "assoc" in name:
        return "associative_recall"
    if "modular" in name:
        return "modular"
    return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_analysis_plan(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"analysis plan must be a YAML mapping: {path}")
    if not isinstance(value.get("analysis"), dict):
        raise ValueError("analysis plan is missing its analysis mapping")
    return value


def _load_yaml_mapping(path: Path, label: str) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a YAML mapping: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate multiple transformer tasks using independent task/seed run clusters."
    )
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--out", default="step4_aggregate")
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=None,
        help="Legacy glob discovery. Publication runs should use repeated --run paths.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        dest="explicit_runs",
        help="Exact source-run directory; repeat once per independent task/seed run.",
    )
    parser.add_argument("--analysis-plan", default=None)
    args = parser.parse_args()

    analysis_plan_path: Path | None = None
    analysis_plan: dict | None = None
    if args.analysis_plan is not None:
        analysis_plan_path = Path(args.analysis_plan).resolve()
        if not analysis_plan_path.is_file():
            raise ValueError(f"analysis plan does not exist: {analysis_plan_path}")
        analysis_plan = _load_analysis_plan(analysis_plan_path)

    root = Path(args.runs_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.explicit_runs and args.patterns:
        raise SystemExit("Use either repeated --run paths or --patterns, not both")
    if args.explicit_runs:
        selected = [Path(value).resolve() for value in args.explicit_runs]
        if len(set(selected)) != len(selected):
            raise ValueError("duplicate explicit --run path")
        run_dirs = sorted(selected, key=lambda path: path.name)
    else:
        patterns = args.patterns or [
            "*_modular_small_step2",
            "*_modular_small_step3_s*",
            "*_assoc_small_step4_s*",
        ]
        run_dirs = collect_runs(root, patterns)
    if not run_dirs:
        raise SystemExit("No source runs were selected")
    summaries = [validate_source_run(path, fail_on_divergence=True) for path in run_dirs]
    units = [(str(item["task_name"]), int(item["base_seed"])) for item in summaries]
    if len(set(units)) != len(units):
        raise ValueError(f"duplicate independent task/base-seed units: {units}")
    summary_by_path = {path: summary for path, summary in zip(run_dirs, summaries)}
    source_entries = [_source_manifest_entry(path) for path in run_dirs]
    bound_publication_release_id: str | None = None
    if analysis_plan is not None and analysis_plan_path is not None:
        expected_plan_sha256 = _sha256(analysis_plan_path)
        expected_plan_version = str(analysis_plan.get("plan_version", ""))
        release_ids: set[str] = set()
        for run_dir, summary in zip(run_dirs, summaries):
            source_cfg = _load_yaml_mapping(run_dir / "config.yaml", "source config")
            protocol_cfg = source_cfg.get("protocol")
            run_cfg = source_cfg.get("run")
            task_cfg = source_cfg.get("task")
            if not isinstance(protocol_cfg, dict) or not isinstance(run_cfg, dict):
                raise ValueError(f"source config lacks protocol/run mapping: {run_dir}")
            if not isinstance(task_cfg, dict):
                raise ValueError(f"source config lacks task mapping: {run_dir}")
            if protocol_cfg.get("publication_plan_sha256") != expected_plan_sha256:
                raise ValueError(
                    f"source run {run_dir.name} is not bound to the supplied analysis plan"
                )
            if protocol_cfg.get("publication_plan_version") != expected_plan_version:
                raise ValueError(
                    f"source run {run_dir.name} has a stale publication-plan version"
                )
            release_id = str(protocol_cfg.get("publication_release_id", ""))
            if not release_id:
                raise ValueError(
                    f"source run {run_dir.name} lacks publication_release_id"
                )
            release_ids.add(release_id)
            if str(run_cfg.get("name", "")) != run_dir.name:
                raise ValueError(f"source config run.name mismatch: {run_dir.name}")
            if int(run_cfg.get("seed", -1)) != int(summary["base_seed"]):
                raise ValueError(f"source config seed mismatch: {run_dir.name}")
            if str(task_cfg.get("name", "")) != str(summary["task_name"]):
                raise ValueError(f"source config task mismatch: {run_dir.name}")
            if not run_dir.name.startswith(release_id + "_"):
                raise ValueError(
                    f"source run {run_dir.name} is not namespaced by release {release_id}"
                )
        if len(release_ids) != 1:
            raise ValueError(
                f"source runs disagree on publication_release_id: {sorted(release_ids)}"
            )
        bound_publication_release_id = next(iter(release_ids))

    budget_frames: list[pd.DataFrame] = []
    site_fit_frames: list[pd.DataFrame] = []
    module_frames: list[pd.DataFrame] = []
    base_frames: list[pd.DataFrame] = []
    target_frames: list[pd.DataFrame] = []
    allocation_frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        validated_summary = summary_by_path[run_dir]
        seed = int(validated_summary["base_seed"])
        run_name = run_dir.name
        task_family = str(validated_summary["task_name"])

        def add_meta(frame: pd.DataFrame) -> pd.DataFrame:
            frame = frame.copy()
            frame.insert(0, "run", run_name)
            frame.insert(1, "seed", seed)
            frame.insert(2, "task_family", task_family)
            return frame

        budget_frames.append(add_meta(pd.read_csv(run_dir / "budget" / "budget_results.csv")))
        optional = [
            ("budget/allocation_comparison.csv", allocation_frames),
            ("sweeps/site_prediction_fit.csv", site_fit_frames),
            ("sweeps/site_target_summary.csv", target_frames),
            ("calibration/module_stats.csv", module_frames),
            ("base/base_metrics.csv", base_frames),
        ]
        for relative, destination in optional:
            path = run_dir / relative
            if path.exists():
                destination.append(add_meta(pd.read_csv(path)))

    budget = _normalise_protocol_columns(pd.concat(budget_frames, ignore_index=True))
    budget.to_csv(out / "all_budget_results.csv", index=False)
    for frames, filename in [
        (base_frames, "all_base_metrics.csv"),
        (module_frames, "all_module_stats.csv"),
        (site_fit_frames, "all_site_prediction_fit.csv"),
        (target_frames, "all_site_target_summary.csv"),
        (allocation_frames, "all_allocation_comparison.csv"),
    ]:
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out / filename, index=False)

    run_budget = _run_level_budget(budget, extra_keys=["task_family"])
    run_budget.to_csv(out / "run_level_budget_results.csv", index=False)
    summary = run_budget.groupby(
        [
            "task_family",
            "scaling_mode",
            "condition_id",
            "rule",
            "allocation_rule",
            "comparison_role",
            "is_primary_allocation_rule",
            "budget",
        ],
        dropna=False,
        as_index=False,
    ).agg(
        n_runs=("run", "nunique"),
        mean_adaptation_replicates=("n_adaptation_replicates", "mean"),
        mean_actual_cost=("actual_cost", "mean"),
        mean_requested_budget=("requested_budget", "mean"),
        mean_val_loss=("final_val_loss", "mean"),
        sem_val_loss=("final_val_loss", sem),
        median_val_loss=("final_val_loss", "median"),
        mean_val_accuracy=("final_val_accuracy", "mean"),
        sem_val_accuracy=("final_val_accuracy", sem),
        median_val_accuracy=("final_val_accuracy", "median"),
        diverged_count=("diverged_count", "sum"),
    )
    summary.to_csv(out / "budget_summary.csv", index=False)
    delta_outputs = _write_delta_outputs(budget, out, extra_keys=["task_family"])

    winner_source = run_budget[
        ~run_budget["comparison_role"].astype(str).eq("exact_cost_baseline")
    ]
    winners = []
    for (task_family, run, scaling_mode, budget_id), group in winner_source.groupby(
        ["task_family", "run", "scaling_mode", "budget"]
    ):
        best_loss = group.loc[group["final_val_loss"].idxmin()]
        best_accuracy = group.loc[group["final_val_accuracy"].idxmax()]
        winners.append(
            {
                "task_family": task_family,
                "run": run,
                "seed": best_loss.get("seed"),
                "scaling_mode": scaling_mode,
                "budget": budget_id,
                "best_loss_condition_id": best_loss["condition_id"],
                "best_loss_rule": best_loss["allocation_rule"],
                "best_loss": best_loss["final_val_loss"],
                "best_accuracy_condition_id": best_accuracy["condition_id"],
                "best_accuracy_rule": best_accuracy["allocation_rule"],
                "best_accuracy": best_accuracy["final_val_accuracy"],
            }
        )
    winner_frame = pd.DataFrame(winners)
    winner_frame.to_csv(out / "winner_by_budget_run.csv", index=False)
    winner_frame.groupby(
        ["task_family", "scaling_mode", "budget", "best_loss_rule"], as_index=False
    ).size().rename(columns={"size": "count"}).to_csv(
        out / "winner_counts_by_budget.csv", index=False
    )

    if site_fit_frames:
        fits = pd.concat(site_fit_frames, ignore_index=True)
        if "scaling_mode" not in fits:
            fits["scaling_mode"] = "standard"
        fits.groupby(
            ["task_family", "scaling_mode", "target", "predictor"], as_index=False
        ).agg(
            n_runs=("run", "nunique"),
            mean_spearman=("spearman", "mean"),
            sem_spearman=("spearman", sem),
            median_spearman=("spearman", "median"),
        ).sort_values(
            ["task_family", "scaling_mode", "target", "mean_spearman"],
            ascending=[True, True, True, False],
        ).to_csv(out / "site_prediction_summary.csv", index=False)

    figure_dir = out / "figures"
    figure_dir.mkdir(exist_ok=True)
    cap_summary = summary[
        ~summary["comparison_role"].astype(str).eq("exact_cost_baseline")
    ]
    for (task_family, scaling_mode), task_summary in cap_summary.groupby(
        ["task_family", "scaling_mode"]
    ):
        plt.figure(figsize=(8, 5))
        for condition_id, group in task_summary.groupby("condition_id"):
            group = group.sort_values("budget")
            plt.errorbar(
                group["budget"],
                group["mean_val_loss"],
                yerr=group["sem_val_loss"],
                marker="o",
                capsize=3,
                label=condition_id,
            )
        plt.xlabel("parameter budget cap")
        plt.ylabel("mean validation loss across independent runs")
        plt.title(f"Budgeted allocation: {task_family}, {scaling_mode}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            figure_dir / f"budget_loss_summary_{task_family}_{scaling_mode}.png",
            dpi=160,
        )
        plt.close()

    task_counts = {
        task: int(count)
        for task, count in pd.Series(
            [str(item["task_name"]) for item in summaries]
        ).value_counts().items()
    }
    primary_scaling_mode = _common_non_null(source_entries, "primary_scaling_mode")
    primary_allocation_rule = _common_non_null(source_entries, "primary_allocation_rule")
    primary_reference_rule = _common_non_null(source_entries, "primary_reference_rule")
    cluster_summary = delta_outputs.get("cluster_summary_exact")
    if cluster_summary is None or cluster_summary.empty:
        raise ValueError("aggregate lacks exact-cost run-cluster outputs")
    primary_flags = (
        cluster_summary["is_primary_allocation_rule"]
        .astype(str)
        .str.lower()
        .eq("true")
    )
    primary_by_task = cluster_summary[
        cluster_summary["scaling_mode"].astype(str).eq(str(primary_scaling_mode))
        & cluster_summary["allocation_rule"].astype(str).eq(
            str(primary_allocation_rule)
        )
        & primary_flags
    ].copy()
    if len(primary_by_task) != len(task_counts):
        raise ValueError(
            "primary analysis must contain exactly one run-cluster row per task; "
            f"found {len(primary_by_task)}, expected {len(task_counts)}"
        )
    for row in primary_by_task.itertuples(index=False):
        expected = task_counts[str(row.task_family)]
        if int(row.n_independent_runs) != expected:
            raise ValueError(
                f"primary run count mismatch for {row.task_family}: "
                f"{row.n_independent_runs} != {expected}"
            )
    primary_by_task.sort_values("task_family").to_csv(
        out / "primary_analysis_by_task.csv", index=False
    )

    primary_run = delta_outputs["run_cluster_exact"]
    primary_run_flags = (
        primary_run["is_primary_allocation_rule"]
        .astype(str)
        .str.lower()
        .eq("true")
    )
    primary_run = primary_run[
        primary_run["scaling_mode"].astype(str).eq(str(primary_scaling_mode))
        & primary_run["allocation_rule"].astype(str).eq(
            str(primary_allocation_rule)
        )
        & primary_run_flags
    ].copy()
    if len(primary_run) != len(run_dirs):
        raise ValueError(
            "primary run-level analysis must contain exactly one row per independent "
            f"task/seed run; found {len(primary_run)}, expected {len(run_dirs)}"
        )
    if primary_run[["task_family", "run"]].duplicated().any():
        raise ValueError("primary run-level analysis contains duplicate task/run rows")
    primary_run = primary_run.sort_values(["task_family", "seed", "run"])
    primary_run.to_csv(out / "primary_run_deltas.csv", index=False)

    analysis_settings = (
        analysis_plan["analysis"] if analysis_plan is not None else {}
    )
    alpha = float(analysis_settings.get("two_sided_alpha", 0.05))
    confidence = float(analysis_settings.get("confidence", 0.95))
    bootstrap_resamples = int(
        analysis_settings.get("cluster_bootstrap_resamples", 10_000)
    )
    loss_values = primary_run["loss_delta_vs_exact_uniform"].to_numpy(dtype=float)
    accuracy_values = primary_run["acc_delta_vs_exact_uniform"].to_numpy(dtype=float)
    task_labels = primary_run["task_family"].astype(str).tolist()
    loss_ci_low, loss_ci_high = task_stratified_bootstrap_interval(
        loss_values,
        task_labels,
        confidence=confidence,
        n_resamples=bootstrap_resamples,
        seed_parts=(
            "primary",
            str(primary_scaling_mode),
            str(primary_allocation_rule),
        ),
    )
    task_loss_means = primary_run.groupby("task_family")[
        "loss_delta_vs_exact_uniform"
    ].mean()
    task_run_counts = primary_run.groupby("task_family")["run"].nunique()
    task_budget_counts = primary_run["n_budgets"].astype(int)
    if primary_run["condition_id"].nunique() != 1 or primary_run["rule"].nunique() != 1:
        raise ValueError("primary run-level rows do not share one condition identity")
    overall_primary = pd.DataFrame(
        [
            {
                "analysis_scope": analysis_settings.get(
                    "primary_scope", "task_stratified_omnibus"
                ),
                "task_weighting": analysis_settings.get(
                    "task_weighting", "equal_weight_across_task_families"
                ),
                "generalization_scope": analysis_settings.get(
                    "generalization_scope",
                    "conditional_on_prespecified_task_families",
                ),
                "scaling_mode": primary_scaling_mode,
                "condition_id": primary_run["condition_id"].iloc[0],
                "rule": primary_run["rule"].iloc[0],
                "allocation_rule": primary_allocation_rule,
                "reference_rule": primary_reference_rule,
                "metric": _common_non_null(source_entries, "primary_metric"),
                "n_task_families": len(task_counts),
                "n_independent_runs": len(primary_run),
                "min_runs_per_task": int(task_run_counts.min()),
                "max_runs_per_task": int(task_run_counts.max()),
                "min_budgets_per_run": int(task_budget_counts.min()),
                "max_budgets_per_run": int(task_budget_counts.max()),
                "mean_loss_delta": task_stratified_mean(loss_values, task_labels),
                "median_task_mean_loss_delta": float(np.median(task_loss_means)),
                "cluster_bootstrap_ci_low": loss_ci_low,
                "cluster_bootstrap_ci_high": loss_ci_high,
                "two_sided_alpha": alpha,
                "confidence": confidence,
                "cluster_bootstrap_resamples": bootstrap_resamples,
                "exact_sign_flip_n_units": len(primary_run),
                "exact_sign_flip_n_patterns": 2 ** len(primary_run),
                "nominal_min_two_sided_p": 2.0 / (2 ** len(primary_run)),
                "exact_sign_flip_p_two_sided": task_stratified_sign_flip_p(
                    loss_values, task_labels
                ),
                "run_wins_loss": int(np.sum(loss_values < 0)),
                "run_ties_loss": int(np.sum(loss_values == 0)),
                "task_wins_loss": int(np.sum(task_loss_means.to_numpy() < 0)),
                "task_ties_loss": int(np.sum(task_loss_means.to_numpy() == 0)),
                "mean_accuracy_delta": task_stratified_mean(
                    accuracy_values, task_labels
                ),
            }
        ]
    )
    overall_primary.to_csv(out / "primary_analysis.csv", index=False)

    plan_metadata: dict[str, object] = {}
    if analysis_plan is not None and analysis_plan_path is not None:
        plan_path = analysis_plan_path
        plan = analysis_plan
        analysis = plan["analysis"]
        expected_roles = {
            "primary_scaling_mode": primary_scaling_mode,
            "primary_allocation_rule": primary_allocation_rule,
            "primary_reference_rule": primary_reference_rule,
            "primary_metric": _common_non_null(source_entries, "primary_metric"),
            "independent_unit": _common_non_null(source_entries, "independent_unit"),
        }
        for key, expected in expected_roles.items():
            if analysis.get(key) != expected:
                raise ValueError(
                    f"analysis plan {key}={analysis.get(key)!r}; expected {expected!r}"
                )
        plan_tasks = {str(item["task_name"]) for item in plan.get("tasks", [])}
        if plan_tasks != set(task_counts):
            raise ValueError(
                f"analysis-plan tasks {sorted(plan_tasks)} do not match source tasks "
                f"{sorted(task_counts)}"
            )
        plan_seeds = {int(value) for value in plan.get("seeds", [])}
        source_seeds_by_task = {
            task: {
                int(item["base_seed"])
                for item in summaries
                if str(item["task_name"]) == task
            }
            for task in task_counts
        }
        if any(values != plan_seeds for values in source_seeds_by_task.values()):
            raise ValueError(
                "analysis-plan seeds do not match every task's source-run seeds: "
                f"plan={sorted(plan_seeds)}, source={source_seeds_by_task}"
            )
        copied_plan = out / "analysis_plan.yaml"
        shutil.copy2(plan_path, copied_plan)
        plan_metadata = {
            "analysis_plan_file": copied_plan.name,
            "analysis_plan_sha256": _sha256(copied_plan),
            "analysis_plan_version": plan.get("plan_version"),
            "primary_estimand": analysis.get("primary_estimand"),
            "budget_weighting": analysis.get("budget_weighting"),
            "two_sided_alpha": analysis.get("two_sided_alpha"),
            "cluster_bootstrap_resamples": analysis.get(
                "cluster_bootstrap_resamples"
            ),
            "exact_sign_flip_test": analysis.get("exact_sign_flip_test"),
            "primary_scope": analysis.get("primary_scope"),
            "task_weighting": analysis.get("task_weighting"),
            "generalization_scope": analysis.get("generalization_scope"),
            "publication_release_id": bound_publication_release_id,
        }

    manifest = {
        "aggregate_schema_version": TRANSFORMER_AGGREGATE_SCHEMA_VERSION,
        "n_independent_runs": len(run_dirs),
        "independent_runs_by_task": task_counts,
        "run_ids": [path.name for path in run_dirs],
        "source_runs": source_entries,
        "protocol_version": _common_non_null(source_entries, "protocol_version"),
        "primary_scaling_mode": primary_scaling_mode,
        "primary_allocation_rule": primary_allocation_rule,
        "primary_reference_rule": primary_reference_rule,
        "primary_metric": _common_non_null(source_entries, "primary_metric"),
        "independent_unit": _common_non_null(source_entries, "independent_unit"),
        "unit_of_inference": (
            "task/base-model seed run; adaptation replicates are averaged within run and "
            "budgets are clustered within run for across-budget summaries"
        ),
        "scaling_modes": sorted(budget["scaling_mode"].astype(str).unique().tolist()),
        "scaling_conditions_analyzed_separately": True,
        "exact_cost_primary_outputs": bool(delta_outputs.get("exact") is not None),
        **plan_metadata,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"aggregated {len(run_dirs)} independent runs")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
