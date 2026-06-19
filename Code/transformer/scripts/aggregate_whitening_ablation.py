#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate whitening ablation runs.")
    ap.add_argument("runs", nargs="*", help="Ablation run directories. If omitted, uses runs/*whitening_ablation*")
    ap.add_argument("--out", default="whitening_ablation_aggregate")
    args = ap.parse_args()
    if args.runs:
        run_dirs = [Path(x) for x in args.runs]
    else:
        run_dirs = sorted(Path("runs").glob("*whitening_ablation*"))
    frames = []
    for d in run_dirs:
        p = d / "budget" / "whitening_ablation_deltas.csv"
        if not p.exists():
            print(f"skip missing {p}")
            continue
        df = pd.read_csv(p)
        df.insert(0, "run", d.name)
        frames.append(df)
    if not frames:
        raise SystemExit("no whitening_ablation_deltas.csv files found")
    all_df = pd.concat(frames, ignore_index=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out / "all_whitening_ablation_deltas.csv", index=False)
    summary = all_df[all_df["label"] != "uniform_fill"].groupby(["label", "whitening", "rule"], dropna=False).agg(
        mean_loss_delta=("loss_delta_vs_uniform_fill", "mean"),
        median_loss_delta=("loss_delta_vs_uniform_fill", "median"),
        mean_acc_delta=("acc_delta_vs_uniform_fill", "mean"),
        wins=("loss_delta_vs_uniform_fill", lambda x: int((x < 0).sum())),
        n=("loss_delta_vs_uniform_fill", "size"),
    ).reset_index().sort_values(["mean_loss_delta", "label"])
    summary.to_csv(out / "whitening_ablation_summary.csv", index=False)
    low = all_df[all_df["budget"].isin([512, 1024, 1536, 2048])]
    low_summary = low[low["label"] != "uniform_fill"].groupby(["label", "whitening", "rule"], dropna=False).agg(
        mean_loss_delta=("loss_delta_vs_uniform_fill", "mean"),
        median_loss_delta=("loss_delta_vs_uniform_fill", "median"),
        mean_acc_delta=("acc_delta_vs_uniform_fill", "mean"),
        wins=("loss_delta_vs_uniform_fill", lambda x: int((x < 0).sum())),
        n=("loss_delta_vs_uniform_fill", "size"),
    ).reset_index().sort_values(["mean_loss_delta", "label"])
    low_summary.to_csv(out / "whitening_ablation_low_budget_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    plot_df = summary.sort_values("mean_loss_delta")
    plt.barh(plot_df["label"], plot_df["mean_loss_delta"])
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("mean loss delta vs uniform_fill")
    plt.title("Whitening ablation aggregate")
    plt.tight_layout()
    plt.savefig(out / "fig_whitening_ablation_delta.png", dpi=160)
    plt.close()

    print("summary:")
    print(summary.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
