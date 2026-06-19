#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    ap = argparse.ArgumentParser(description="Aggregate synthetic-transformer rank-allocation runs.")
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

    budget_frames = []
    site_fit_frames = []
    module_frames = []
    base_frames = []
    target_frames = []
    alloc_frames = []

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
    budget.to_csv(out / "all_budget_results.csv", index=False)

    if base_frames:
        pd.concat(base_frames, ignore_index=True).to_csv(out / "all_base_metrics.csv", index=False)
    if module_frames:
        pd.concat(module_frames, ignore_index=True).to_csv(out / "all_module_stats.csv", index=False)
    if site_fit_frames:
        pd.concat(site_fit_frames, ignore_index=True).to_csv(out / "all_site_prediction_fit.csv", index=False)
    if target_frames:
        pd.concat(target_frames, ignore_index=True).to_csv(out / "all_site_target_summary.csv", index=False)
    if alloc_frames:
        pd.concat(alloc_frames, ignore_index=True).to_csv(out / "all_allocation_comparison.csv", index=False)

    summary = budget.groupby(["task_family", "rule", "budget"], as_index=False).agg(
        n_runs=("run", "nunique"),
        mean_actual_cost=("actual_cost", "mean"),
        mean_val_loss=("final_val_loss", "mean"),
        sem_val_loss=("final_val_loss", sem),
        median_val_loss=("final_val_loss", "median"),
        mean_val_accuracy=("final_val_accuracy", "mean"),
        sem_val_accuracy=("final_val_accuracy", sem),
        median_val_accuracy=("final_val_accuracy", "median"),
        diverged_count=("diverged", "sum"),
    )
    summary.to_csv(out / "budget_summary.csv", index=False)

    rows = []
    for (task_family, run, budget_id), g in budget.groupby(["task_family", "run", "budget"]):
        ref = g[g["rule"] == "uniform_fill"]
        if ref.empty:
            continue
        ref = ref.iloc[0]
        for _, row in g.iterrows():
            if row["rule"] == "uniform_fill":
                continue
            rows.append({
                "task_family": task_family,
                "run": run,
                "seed": row.get("seed"),
                "budget": budget_id,
                "rule": row["rule"],
                "loss_delta_vs_uniform_fill": row["final_val_loss"] - ref["final_val_loss"],
                "acc_delta_vs_uniform_fill": row["final_val_accuracy"] - ref["final_val_accuracy"],
                "loss_ratio_vs_uniform_fill": row["final_val_loss"] / ref["final_val_loss"] if ref["final_val_loss"] != 0 else np.nan,
                "actual_cost": row["actual_cost"],
                "uniform_fill_cost": ref["actual_cost"],
            })
    deltas = pd.DataFrame(rows)
    if not deltas.empty:
        deltas.to_csv(out / "deltas_vs_uniform_fill.csv", index=False)
        delta_summary = deltas.groupby(["task_family", "rule", "budget"], as_index=False).agg(
            n_runs=("run", "nunique"),
            mean_loss_delta=("loss_delta_vs_uniform_fill", "mean"),
            sem_loss_delta=("loss_delta_vs_uniform_fill", sem),
            median_loss_delta=("loss_delta_vs_uniform_fill", "median"),
            mean_acc_delta=("acc_delta_vs_uniform_fill", "mean"),
            sem_acc_delta=("acc_delta_vs_uniform_fill", sem),
            median_acc_delta=("acc_delta_vs_uniform_fill", "median"),
            mean_loss_ratio=("loss_ratio_vs_uniform_fill", "mean"),
        )
        delta_summary.to_csv(out / "delta_summary_vs_uniform_fill.csv", index=False)

    winners = []
    for (task_family, run, budget_id), g in budget.groupby(["task_family", "run", "budget"]):
        best_loss = g.loc[g["final_val_loss"].idxmin()]
        best_acc = g.loc[g["final_val_accuracy"].idxmax()]
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
    win.groupby(["task_family", "budget", "best_loss_rule"], as_index=False).size().rename(columns={"size":"count"}).to_csv(out / "winner_counts_by_budget.csv", index=False)

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
    for task_family, sdf in summary.groupby("task_family"):
        plt.figure(figsize=(8, 5))
        for rule, g in sdf.groupby("rule"):
            gg = g.sort_values("budget")
            plt.errorbar(gg["budget"], gg["mean_val_loss"], yerr=gg["sem_val_loss"], marker="o", capsize=3, label=rule)
        plt.xlabel("parameter budget")
        plt.ylabel("mean validation loss")
        plt.title(f"Budgeted allocation: {task_family}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / f"budget_loss_summary_{task_family}.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8, 5))
        for rule, g in sdf.groupby("rule"):
            gg = g.sort_values("budget")
            plt.errorbar(gg["budget"], gg["mean_val_accuracy"], yerr=gg["sem_val_accuracy"], marker="o", capsize=3, label=rule)
        plt.xlabel("parameter budget")
        plt.ylabel("mean validation accuracy")
        plt.title(f"Budgeted allocation accuracy: {task_family}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / f"budget_accuracy_summary_{task_family}.png", dpi=160)
        plt.close()

    (out / "manifest.json").write_text(json.dumps({"n_runs": len(run_dirs), "runs": [str(p) for p in run_dirs]}, indent=2), encoding="utf-8")
    print(f"aggregated {len(run_dirs)} runs")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
