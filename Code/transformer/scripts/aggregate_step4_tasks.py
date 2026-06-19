#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def infer_seed_from_run(run_dir: Path) -> int | None:
    cfg = run_dir / "config.yaml"
    if cfg.exists():
        text = cfg.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"\n\s*seed:\s*(\d+)", "\n" + text)
        if m:
            return int(m.group(1))
    m = re.search(r"_s(\d+)", run_dir.name)
    return int(m.group(1)) if m else None


def collect_runs(root: Path, patterns: list[str]) -> list[Path]:
    runs = []
    for pat in patterns:
        for p in root.glob(pat):
            if (p / "budget" / "budget_results.csv").exists():
                runs.append(p)
    return sorted(set(runs), key=lambda p: p.name)


def sem(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) <= 1:
        return 0.0
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate synthetic-transformer tasks with nested adaptation replicates.")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out", default="step4_aggregate")
    ap.add_argument("--patterns", nargs="*", default=["*_modular_small_step2", "*_modular_small_step3_s*", "*_assoc_small_step4_s*"])
    args = ap.parse_args()

    root = Path(args.runs_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dirs = collect_runs(root, args.patterns)
    if not run_dirs:
        raise SystemExit(f"No matching runs found under {root} for {args.patterns}")

    budget_frames, site_fit_frames, module_frames = [], [], []
    base_frames, target_frames, alloc_frames = [], [], []
    for rd in run_dirs:
        seed = infer_seed_from_run(rd)
        run_name = rd.name
        task_family = "assoc" if "assoc" in run_name else ("modular" if "modular" in run_name else "unknown")

        def add_meta(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df.insert(0, "run", run_name)
            df.insert(1, "seed", seed)
            df.insert(2, "task_family", task_family)
            return df

        budget_frames.append(add_meta(pd.read_csv(rd / "budget" / "budget_results.csv")))
        if (rd / "budget" / "allocation_comparison.csv").exists():
            alloc_frames.append(add_meta(pd.read_csv(rd / "budget" / "allocation_comparison.csv")))
        if (rd / "sweeps" / "site_prediction_fit.csv").exists():
            site_fit_frames.append(add_meta(pd.read_csv(rd / "sweeps" / "site_prediction_fit.csv")))
        if (rd / "sweeps" / "site_target_summary.csv").exists():
            target_frames.append(add_meta(pd.read_csv(rd / "sweeps" / "site_target_summary.csv")))
        if (rd / "calibration" / "module_stats.csv").exists():
            module_frames.append(add_meta(pd.read_csv(rd / "calibration" / "module_stats.csv")))
        if (rd / "base" / "base_metrics.csv").exists():
            base_frames.append(add_meta(pd.read_csv(rd / "base" / "base_metrics.csv")))

    budget = pd.concat(budget_frames, ignore_index=True)
    if "adaptation_replicate" not in budget:
        budget["adaptation_replicate"] = 0
    budget["adaptation_replicate"] = pd.to_numeric(budget["adaptation_replicate"], errors="raise").astype(int)
    unique_keys = ["task_family", "run", "budget", "adaptation_replicate", "rule"]
    if budget.duplicated(unique_keys).any():
        raise ValueError(f"duplicate budget rows for keys {unique_keys}")
    budget.to_csv(out / "all_budget_results.csv", index=False)

    for frames, filename in [
        (base_frames, "all_base_metrics.csv"),
        (module_frames, "all_module_stats.csv"),
        (site_fit_frames, "all_site_prediction_fit.csv"),
        (target_frames, "all_site_target_summary.csv"),
        (alloc_frames, "all_allocation_comparison.csv"),
    ]:
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out / filename, index=False)

    run_budget = budget.groupby(["task_family", "run", "seed", "rule", "budget"], dropna=False, as_index=False).agg(
        n_adaptation_replicates=("adaptation_replicate", "nunique"),
        actual_cost=("actual_cost", "mean"),
        final_val_loss=("final_val_loss", "mean"),
        final_val_accuracy=("final_val_accuracy", "mean"),
        diverged_count=("diverged", "sum"),
    )
    run_budget.to_csv(out / "run_level_budget_results.csv", index=False)
    summary = run_budget.groupby(["task_family", "rule", "budget"], as_index=False).agg(
        n_runs=("run", "nunique"),
        mean_adaptation_replicates=("n_adaptation_replicates", "mean"),
        mean_actual_cost=("actual_cost", "mean"),
        mean_val_loss=("final_val_loss", "mean"),
        sem_val_loss=("final_val_loss", sem),
        median_val_loss=("final_val_loss", "median"),
        mean_val_accuracy=("final_val_accuracy", "mean"),
        sem_val_accuracy=("final_val_accuracy", sem),
        median_val_accuracy=("final_val_accuracy", "median"),
        diverged_count=("diverged_count", "sum"),
    )
    summary.to_csv(out / "budget_summary.csv", index=False)

    pair_keys = ["task_family", "run", "budget", "adaptation_replicate"]
    ref = budget[budget["rule"] == "uniform_fill"][pair_keys + ["final_val_loss", "final_val_accuracy", "actual_cost"]].rename(columns={
        "final_val_loss": "uniform_fill_loss",
        "final_val_accuracy": "uniform_fill_accuracy",
        "actual_cost": "uniform_fill_cost",
    })
    if ref.duplicated(pair_keys).any():
        raise ValueError("uniform_fill must have one row per task, run, budget, and adaptation replicate")
    deltas = budget[budget["rule"] != "uniform_fill"].merge(ref, on=pair_keys, how="inner", validate="many_to_one")
    deltas["loss_delta_vs_uniform_fill"] = deltas["final_val_loss"] - deltas["uniform_fill_loss"]
    deltas["acc_delta_vs_uniform_fill"] = deltas["final_val_accuracy"] - deltas["uniform_fill_accuracy"]
    deltas["loss_ratio_vs_uniform_fill"] = deltas["final_val_loss"] / deltas["uniform_fill_loss"].replace(0, np.nan)
    keep = [
        "task_family", "run", "seed", "budget", "adaptation_replicate", "rule",
        "loss_delta_vs_uniform_fill", "acc_delta_vs_uniform_fill", "loss_ratio_vs_uniform_fill",
        "actual_cost", "uniform_fill_cost",
    ]
    deltas = deltas[keep]
    if not deltas.empty:
        deltas.to_csv(out / "deltas_vs_uniform_fill.csv", index=False)
        run_deltas = deltas.groupby(["task_family", "run", "seed", "rule", "budget"], dropna=False, as_index=False).agg(
            n_adaptation_replicates=("adaptation_replicate", "nunique"),
            loss_delta_vs_uniform_fill=("loss_delta_vs_uniform_fill", "mean"),
            acc_delta_vs_uniform_fill=("acc_delta_vs_uniform_fill", "mean"),
            loss_ratio_vs_uniform_fill=("loss_ratio_vs_uniform_fill", "mean"),
            actual_cost=("actual_cost", "mean"),
            uniform_fill_cost=("uniform_fill_cost", "mean"),
        )
        run_deltas.to_csv(out / "run_level_deltas_vs_uniform_fill.csv", index=False)
        run_deltas.groupby(["task_family", "rule", "budget"], as_index=False).agg(
            n_runs=("run", "nunique"),
            mean_loss_delta=("loss_delta_vs_uniform_fill", "mean"),
            sem_loss_delta=("loss_delta_vs_uniform_fill", sem),
            median_loss_delta=("loss_delta_vs_uniform_fill", "median"),
            mean_acc_delta=("acc_delta_vs_uniform_fill", "mean"),
            sem_acc_delta=("acc_delta_vs_uniform_fill", sem),
            median_acc_delta=("acc_delta_vs_uniform_fill", "median"),
            mean_loss_ratio=("loss_ratio_vs_uniform_fill", "mean"),
        ).to_csv(out / "delta_summary_vs_uniform_fill.csv", index=False)

    winners = []
    for (task_family, run, budget_id), group in run_budget.groupby(["task_family", "run", "budget"]):
        best_loss = group.loc[group["final_val_loss"].idxmin()]
        best_acc = group.loc[group["final_val_accuracy"].idxmax()]
        winners.append({
            "task_family": task_family,
            "run": run,
            "seed": best_loss.get("seed"),
            "budget": budget_id,
            "best_loss_rule": best_loss["rule"],
            "best_loss": best_loss["final_val_loss"],
            "best_acc_rule": best_acc["rule"],
            "best_accuracy": best_acc["final_val_accuracy"],
        })
    win = pd.DataFrame(winners)
    win.to_csv(out / "winner_by_budget_run.csv", index=False)
    win.groupby(["task_family", "budget", "best_loss_rule"], as_index=False).size().rename(columns={"size": "count"}).to_csv(out / "winner_counts_by_budget.csv", index=False)

    if site_fit_frames:
        fits = pd.concat(site_fit_frames, ignore_index=True)
        fits.groupby(["task_family", "target", "predictor"], as_index=False).agg(
            n_runs=("run", "nunique"),
            mean_spearman=("spearman", "mean"),
            sem_spearman=("spearman", sem),
            median_spearman=("spearman", "median"),
        ).sort_values(["task_family", "target", "mean_spearman"], ascending=[True, True, False]).to_csv(out / "site_prediction_summary.csv", index=False)

    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    for task_family, task_summary in summary.groupby("task_family"):
        plt.figure(figsize=(8, 5))
        for rule, group in task_summary.groupby("rule"):
            group = group.sort_values("budget")
            plt.errorbar(group["budget"], group["mean_val_loss"], yerr=group["sem_val_loss"], marker="o", capsize=3, label=rule)
        plt.xlabel("parameter budget")
        plt.ylabel("mean validation loss across base runs")
        plt.title(f"Budgeted allocation: {task_family}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / f"budget_loss_summary_{task_family}.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8, 5))
        for rule, group in task_summary.groupby("rule"):
            group = group.sort_values("budget")
            plt.errorbar(group["budget"], group["mean_val_accuracy"], yerr=group["sem_val_accuracy"], marker="o", capsize=3, label=rule)
        plt.xlabel("parameter budget")
        plt.ylabel("mean validation accuracy across base runs")
        plt.title(f"Budgeted allocation accuracy: {task_family}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / f"budget_accuracy_summary_{task_family}.png", dpi=160)
        plt.close()

    (out / "manifest.json").write_text(json.dumps({
        "n_independent_runs": len(run_dirs),
        "runs": [str(p) for p in run_dirs],
        "unit_of_inference": "base-model run within task; adaptation replicates averaged within run",
    }, indent=2), encoding="utf-8")
    print(f"aggregated {len(run_dirs)} independent runs")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
