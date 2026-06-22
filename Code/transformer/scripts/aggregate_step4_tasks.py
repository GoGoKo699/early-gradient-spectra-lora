#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate multiple transformer tasks using independent task/seed run clusters."
    )
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--out", default="step4_aggregate")
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=[
            "*_modular_small_step2",
            "*_modular_small_step3_s*",
            "*_assoc_small_step4_s*",
        ],
    )
    args = parser.parse_args()

    root = Path(args.runs_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dirs = collect_runs(root, args.patterns)
    if not run_dirs:
        raise SystemExit(f"No matching runs found under {root} for {args.patterns}")
    source_entries = [_source_manifest_entry(path) for path in run_dirs]

    budget_frames: list[pd.DataFrame] = []
    site_fit_frames: list[pd.DataFrame] = []
    module_frames: list[pd.DataFrame] = []
    base_frames: list[pd.DataFrame] = []
    target_frames: list[pd.DataFrame] = []
    allocation_frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        seed = infer_seed_from_run(run_dir)
        run_name = run_dir.name
        task_family = _task_family(run_dir)

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
        for task, count in pd.Series([_task_family(path) for path in run_dirs]).value_counts().items()
    }
    manifest = {
        "n_independent_runs": len(run_dirs),
        "independent_runs_by_task": task_counts,
        "runs": [str(path) for path in run_dirs],
        "source_runs": source_entries,
        "protocol_version": _common_non_null(source_entries, "protocol_version"),
        "primary_scaling_mode": _common_non_null(source_entries, "primary_scaling_mode"),
        "primary_allocation_rule": _common_non_null(source_entries, "primary_allocation_rule"),
        "primary_reference_rule": _common_non_null(source_entries, "primary_reference_rule"),
        "primary_metric": _common_non_null(source_entries, "primary_metric"),
        "independent_unit": _common_non_null(source_entries, "independent_unit"),
        "unit_of_inference": (
            "task/base-model seed run; adaptation replicates are averaged within run and "
            "budgets are clustered within run for across-budget summaries"
        ),
        "scaling_modes": sorted(budget["scaling_mode"].astype(str).unique().tolist()),
        "scaling_conditions_analyzed_separately": True,
        "exact_cost_primary_outputs": bool(delta_outputs.get("exact") is not None),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"aggregated {len(run_dirs)} independent runs")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
