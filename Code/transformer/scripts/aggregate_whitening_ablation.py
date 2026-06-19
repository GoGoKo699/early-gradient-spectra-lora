#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def sem(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) <= 1:
        return 0.0
    return float(x.std(ddof=1) / (len(x) ** 0.5))


def _summarize(run_level: pd.DataFrame) -> pd.DataFrame:
    return run_level[run_level["label"] != "uniform_fill"].groupby(
        ["label", "whitening", "rule"], dropna=False
    ).agg(
        n_runs=("run", "nunique"),
        mean_loss_delta=("loss_delta_vs_uniform_fill", "mean"),
        sem_loss_delta=("loss_delta_vs_uniform_fill", sem),
        median_loss_delta=("loss_delta_vs_uniform_fill", "median"),
        mean_acc_delta=("acc_delta_vs_uniform_fill", "mean"),
        sem_acc_delta=("acc_delta_vs_uniform_fill", sem),
        run_wins=("loss_delta_vs_uniform_fill", lambda x: int((x < 0).sum())),
    ).reset_index().sort_values(["mean_loss_delta", "label"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate whitening ablations at the independent-run level.")
    ap.add_argument("runs", nargs="*", help="Ablation run directories. If omitted, uses runs/*whitening_ablation*")
    ap.add_argument("--out", default="whitening_ablation_aggregate")
    args = ap.parse_args()
    run_dirs = [Path(x) for x in args.runs] if args.runs else sorted(Path("runs").glob("*whitening_ablation*"))

    frames = []
    for directory in run_dirs:
        path = directory / "budget" / "whitening_ablation_deltas.csv"
        if not path.exists():
            print(f"skip missing {path}")
            continue
        df = pd.read_csv(path)
        if "adaptation_replicate" not in df:
            df["adaptation_replicate"] = 0
        df.insert(0, "run", directory.name)
        frames.append(df)
    if not frames:
        raise SystemExit("no whitening_ablation_deltas.csv files found")

    all_df = pd.concat(frames, ignore_index=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out / "all_whitening_ablation_deltas.csv", index=False)

    run_level = all_df.groupby(
        ["run", "label", "whitening", "rule", "budget"], dropna=False, as_index=False
    ).agg(
        n_adaptation_replicates=("adaptation_replicate", "nunique"),
        loss_delta_vs_uniform_fill=("loss_delta_vs_uniform_fill", "mean"),
        acc_delta_vs_uniform_fill=("acc_delta_vs_uniform_fill", "mean"),
    )
    run_level.to_csv(out / "run_level_whitening_ablation_deltas.csv", index=False)
    summary = _summarize(run_level)
    summary.to_csv(out / "whitening_ablation_summary.csv", index=False)

    low_run_level = run_level[run_level["budget"].isin([512, 1024, 1536, 2048])]
    _summarize(low_run_level).to_csv(out / "whitening_ablation_low_budget_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    plot_df = summary.sort_values("mean_loss_delta")
    plt.barh(plot_df["label"], plot_df["mean_loss_delta"])
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("mean run-level loss delta vs uniform_fill")
    plt.title("Whitening ablation aggregate")
    plt.tight_layout()
    plt.savefig(out / "fig_whitening_ablation_delta.png", dpi=160)
    plt.close()

    print("summary:")
    print(summary.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
