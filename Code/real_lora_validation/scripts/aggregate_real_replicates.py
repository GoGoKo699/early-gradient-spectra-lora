#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd


def sem(x: pd.Series) -> float:
    if len(x) <= 1:
        return float("nan")
    return float(x.std(ddof=1) / math.sqrt(len(x)))


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: aggregate_real_replicates.py <replicate_dir>")

    root = Path(sys.argv[1])
    rows = []

    for result_path in sorted(root.glob("*/results.csv")):
        run_dir = result_path.parent
        config_path = run_dir / "config.json"
        cfg = json.loads(config_path.read_text()) if config_path.exists() else {}

        df = pd.read_csv(result_path)
        for _, row in df.iterrows():
            d = row.to_dict()
            d["run_dir"] = str(run_dir)
            d["seed"] = cfg.get("seed", None)
            d["model"] = cfg.get("model", None)
            d["steps"] = cfg.get("steps", None)
            d["target_suffixes"] = cfg.get("target_suffixes", None)
            rows.append(d)

    if not rows:
        raise SystemExit(f"no results.csv files found under {root}")

    all_df = pd.DataFrame(rows)
    all_df.to_csv(root / "all_results.csv", index=False)

    winners = []
    for run_dir, g in all_df.groupby("run_dir"):
        best_idx = g["final_val_loss"].idxmin()
        best = all_df.loc[best_idx]
        winners.append({
            "run_dir": run_dir,
            "seed": best.get("seed", None),
            "winner": best["strategy"],
            "winner_final_val_loss": best["final_val_loss"],
        })
    winners_df = pd.DataFrame(winners)
    winners_df.to_csv(root / "per_run_winners.csv", index=False)

    summary = (
        all_df
        .groupby("strategy")
        .agg(
            n=("final_val_loss", "size"),
            mean_initial_val_loss=("initial_val_loss", "mean"),
            mean_final_val_loss=("final_val_loss", "mean"),
            sem_final_val_loss=("final_val_loss", sem),
            mean_val_loss_delta=("val_loss_delta", "mean"),
            sem_val_loss_delta=("val_loss_delta", sem),
            mean_perplexity=("perplexity", "mean"),
            sem_perplexity=("perplexity", sem),
            mean_trainable_params=("trainable_params", "mean"),
            min_rank=("rank_min", "mean"),
            mean_rank=("rank_mean", "mean"),
            max_rank=("rank_max", "mean"),
        )
        .reset_index()
        .sort_values("mean_final_val_loss")
    )

    win_counts = winners_df["winner"].value_counts().rename_axis("strategy").reset_index(name="wins")
    summary = summary.merge(win_counts, on="strategy", how="left")
    summary["wins"] = summary["wins"].fillna(0).astype(int)
    summary.to_csv(root / "summary_by_strategy.csv", index=False)

    # Pairwise spectral differences per run.
    pairs = []
    for run_dir, g in all_df.groupby("run_dir"):
        by = g.set_index("strategy")
        if "spectral_effective" not in by.index:
            continue
        spectral = by.loc["spectral_effective"]
        for other in ["uniform_r4", "gradient_norm"]:
            if other in by.index:
                pairs.append({
                    "run_dir": run_dir,
                    "seed": spectral.get("seed", None),
                    "comparison": f"spectral_minus_{other}",
                    "final_val_loss_diff": spectral["final_val_loss"] - by.loc[other, "final_val_loss"],
                    "perplexity_diff": spectral["perplexity"] - by.loc[other, "perplexity"],
                    "params_diff": spectral["trainable_params"] - by.loc[other, "trainable_params"],
                })
    pair_df = pd.DataFrame(pairs)
    pair_df.to_csv(root / "spectral_pairwise_diffs.csv", index=False)

    print("wrote", root / "all_results.csv")
    print("wrote", root / "per_run_winners.csv")
    print("wrote", root / "summary_by_strategy.csv")
    print("wrote", root / "spectral_pairwise_diffs.csv")
    print()
    print("summary_by_strategy:")
    print(summary.to_string(index=False))
    print()
    print("per_run_winners:")
    print(winners_df.to_string(index=False))
    print()
    if not pair_df.empty:
        print("spectral_pairwise_diffs:")
        print(pair_df.to_string(index=False))


if __name__ == "__main__":
    main()
