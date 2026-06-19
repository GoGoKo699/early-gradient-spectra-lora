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


def _normalise_protocol_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "adaptation_replicate" not in df:
        df["adaptation_replicate"] = 0
    df["adaptation_replicate"] = pd.to_numeric(df["adaptation_replicate"], errors="raise").astype(int)
    keys = ["run", "budget", "adaptation_replicate", "rule"]
    duplicated = df.duplicated(keys, keep=False)
    if duplicated.any():
        raise ValueError(f"duplicate budget result rows for keys {keys}:\n{df.loc[duplicated, keys].to_string(index=False)}")
    return df


def _run_level_budget(budget: pd.DataFrame) -> pd.DataFrame:
    return budget.groupby(["run", "seed", "rule", "budget"], dropna=False, as_index=False).agg(
        n_adaptation_replicates=("adaptation_replicate", "nunique"),
        actual_cost=("actual_cost", "mean"),
        final_val_loss=("final_val_loss", "mean"),
        final_val_accuracy=("final_val_accuracy", "mean"),
        diverged_count=("diverged", "sum"),
    )


def _paired_deltas(budget: pd.DataFrame) -> pd.DataFrame:
    pair_keys = ["run", "budget", "adaptation_replicate"]
    ref = budget[budget["rule"] == "uniform_fill"][
        pair_keys + ["final_val_loss", "final_val_accuracy", "actual_cost"]
    ].rename(columns={
        "final_val_loss": "uniform_fill_loss",
        "final_val_accuracy": "uniform_fill_accuracy",
        "actual_cost": "uniform_fill_cost",
    })
    if ref.duplicated(pair_keys).any():
        raise ValueError("uniform_fill must have one row per run, budget, and adaptation replicate")
    other = budget[budget["rule"] != "uniform_fill"].copy()
    paired = other.merge(ref, on=pair_keys, how="inner", validate="many_to_one")
    paired["loss_delta_vs_uniform_fill"] = paired["final_val_loss"] - paired["uniform_fill_loss"]
    paired["acc_delta_vs_uniform_fill"] = paired["final_val_accuracy"] - paired["uniform_fill_accuracy"]
    paired["loss_ratio_vs_uniform_fill"] = paired["final_val_loss"] / paired["uniform_fill_loss"].replace(0, np.nan)
    return paired[[
        "run", "seed", "budget", "adaptation_replicate", "rule",
        "loss_delta_vs_uniform_fill", "acc_delta_vs_uniform_fill", "loss_ratio_vs_uniform_fill",
        "actual_cost", "uniform_fill_cost",
    ]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate modular-small replicate runs with nested adaptation replicates.")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out", default="step3_aggregate")
    ap.add_argument("--patterns", nargs="*", default=["*_modular_small_step2", "*_modular_small_step3_s*"])
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

        def add_meta(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df.insert(0, "run", run_name)
            df.insert(1, "seed", seed)
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

    budget = _normalise_protocol_columns(pd.concat(budget_frames, ignore_index=True))
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

    run_budget = _run_level_budget(budget)
    run_budget.to_csv(out / "run_level_budget_results.csv", index=False)
    summary = run_budget.groupby(["rule", "budget"], as_index=False).agg(
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

    deltas = _paired_deltas(budget)
    if not deltas.empty:
        deltas.to_csv(out / "deltas_vs_uniform_fill.csv", index=False)
        run_deltas = deltas.groupby(["run", "seed", "rule", "budget"], dropna=False, as_index=False).agg(
            n_adaptation_replicates=("adaptation_replicate", "nunique"),
            loss_delta_vs_uniform_fill=("loss_delta_vs_uniform_fill", "mean"),
            acc_delta_vs_uniform_fill=("acc_delta_vs_uniform_fill", "mean"),
            loss_ratio_vs_uniform_fill=("loss_ratio_vs_uniform_fill", "mean"),
            actual_cost=("actual_cost", "mean"),
            uniform_fill_cost=("uniform_fill_cost", "mean"),
        )
        run_deltas.to_csv(out / "run_level_deltas_vs_uniform_fill.csv", index=False)
        delta_summary = run_deltas.groupby(["rule", "budget"], as_index=False).agg(
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
    for (run, budget_id), group in run_budget.groupby(["run", "budget"]):
        best_loss = group.loc[group["final_val_loss"].idxmin()]
        best_acc = group.loc[group["final_val_accuracy"].idxmax()]
        winners.append({
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
    win.groupby(["budget", "best_loss_rule"], as_index=False).size().rename(columns={"size": "count"}).to_csv(out / "winner_counts_by_budget.csv", index=False)

    if site_fit_frames:
        fits = pd.concat(site_fit_frames, ignore_index=True)
        fits.groupby(["target", "predictor"], as_index=False).agg(
            n_runs=("run", "nunique"),
            mean_spearman=("spearman", "mean"),
            sem_spearman=("spearman", sem),
            median_spearman=("spearman", "median"),
        ).sort_values(["target", "mean_spearman"], ascending=[True, False]).to_csv(out / "site_prediction_summary.csv", index=False)

    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 5))
    for rule, group in summary.groupby("rule"):
        group = group.sort_values("budget")
        plt.errorbar(group["budget"], group["mean_val_loss"], yerr=group["sem_val_loss"], marker="o", capsize=3, label=rule)
    plt.xlabel("parameter budget")
    plt.ylabel("mean validation loss across base runs")
    plt.title("Budgeted allocation across independent runs")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "budget_loss_summary.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for rule, group in summary.groupby("rule"):
        group = group.sort_values("budget")
        plt.errorbar(group["budget"], group["mean_val_accuracy"], yerr=group["sem_val_accuracy"], marker="o", capsize=3, label=rule)
    plt.xlabel("parameter budget")
    plt.ylabel("mean validation accuracy across base runs")
    plt.title("Budgeted allocation accuracy across independent runs")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "budget_accuracy_summary.png", dpi=160)
    plt.close()

    manifest = {
        "n_independent_runs": len(run_dirs),
        "runs": [str(p) for p in run_dirs],
        "unit_of_inference": "base-model run; adaptation replicates averaged within run",
        "outputs": [p.name for p in out.iterdir()],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"aggregated {len(run_dirs)} independent runs")
    print(f"wrote {out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
