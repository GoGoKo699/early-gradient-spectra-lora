#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

KEY_TARGETS = [
    "near_best_rank_gap_0.1",
    "near_best_rank_gap_0.2",
    "recovery_rank_0.7",
    "recovery_rank_0.8",
    "penalized_rank_lambda_0.2",
    "penalized_rank_lambda_0.3",
]
KEY_PREDICTORS = [
    # ``gradient_*`` is the config-selected estimator.  Stage4 configs use
    # activation-whitened gradients after the publication-sanity patch, while
    # raw/whitened columns are kept for direct audit/ablation.
    "gradient_effective_rank",
    "gradient_detectable_rank",
    "predicted_rank_effective",
    "predicted_rank_detectable",
    "raw_gradient_effective_rank",
    "raw_gradient_detectable_rank",
    "predicted_rank_raw_effective",
    "predicted_rank_raw_detectable",
    "whitened_gradient_effective_rank",
    "whitened_gradient_detectable_rank",
    "predicted_rank_whitened_effective",
    "predicted_rank_whitened_detectable",
    "true_k_strong",
]


def _infer_condition(run_dir: Path) -> str:
    name = run_dir.name
    if "sample" in name:
        return "sample_limited"
    if "hard" in name:
        return "hard_knee"
    return "unknown"


def _infer_seed(run_dir: Path, summary: pd.DataFrame | None = None) -> int | None:
    # Prefer explicit config, fallback to run name suffix.
    cfg = run_dir / "config.yaml"
    if cfg.exists():
        try:
            import yaml
            data = yaml.safe_load(cfg.read_text()) or {}
            if "seed" in data:
                return int(data["seed"])
        except Exception:
            pass
    import re
    m = re.search(r"s(\d+)", run_dir.name)
    if m:
        return int(m.group(1))
    if summary is not None and "seed" in summary.columns and summary["seed"].notna().any():
        return int(summary["seed"].dropna().iloc[0])
    return None


def _sem(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) <= 1:
        return float("nan")
    return float(x.std(ddof=1) / math.sqrt(len(x)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="Stage4 run directories. If omitted, auto-detect runs/*stage4*.")
    ap.add_argument("--out", default="runs/stage4_aggregate", help="Output directory for aggregate tables and figures.")
    args = ap.parse_args()

    if args.runs:
        run_dirs = [Path(p) for p in args.runs]
    else:
        run_dirs = sorted(Path("runs").glob("*stage4*"))
        run_dirs = [p for p in run_dirs if p.is_dir() and p.name != "stage4_aggregate"]

    records = []
    summaries = []
    missing = []
    for d in run_dirs:
        fit_path = d / "layer_target_fit_augmented.csv"
        summary_path = d / "layer_summary_augmented.csv"
        if not fit_path.exists():
            missing.append(str(fit_path))
            continue
        fit = pd.read_csv(fit_path)
        summary = pd.read_csv(summary_path) if summary_path.exists() else None
        cond = _infer_condition(d)
        seed = _infer_seed(d, summary)
        fit["run_dir"] = str(d)
        fit["condition"] = cond
        fit["seed"] = seed
        records.append(fit)
        if summary is not None:
            s = summary.copy()
            s["run_dir"] = str(d)
            s["condition"] = cond
            s["seed"] = seed
            summaries.append(s)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if missing:
        (out_dir / "missing_files.txt").write_text("\n".join(missing) + "\n")
    if not records:
        raise SystemExit("No stage4 layer_target_fit_augmented.csv files found.")

    all_fit = pd.concat(records, ignore_index=True)
    all_fit.to_csv(out_dir / "stage4_all_fits.csv", index=False)

    key = all_fit[all_fit["target"].isin(KEY_TARGETS) & all_fit["predictor"].isin(KEY_PREDICTORS)].copy()
    agg = (key.groupby(["condition", "target", "predictor"], as_index=False)
             .agg(n_runs=("r2_log2_target", "count"),
                  mean_r2=("r2_log2_target", "mean"),
                  sem_r2=("r2_log2_target", _sem),
                  mean_spearman=("spearman", "mean"),
                  sem_spearman=("spearman", _sem),
                  mean_rmse_log2=("rmse_log2", "mean")))
    agg = agg.sort_values(["condition", "target", "mean_r2"], ascending=[True, True, False])
    agg.to_csv(out_dir / "stage4_key_table.csv", index=False)

    best = (all_fit.dropna(subset=["r2_log2_target"])
              .sort_values(["condition", "seed", "target", "r2_log2_target"], ascending=[True, True, True, False])
              .groupby(["condition", "seed", "target"], as_index=False)
              .head(1))
    best.to_csv(out_dir / "stage4_best_per_run_target.csv", index=False)

    best_agg = (best.groupby(["condition", "target", "predictor"], as_index=False)
                  .agg(n_wins=("r2_log2_target", "count"),
                       mean_r2=("r2_log2_target", "mean"),
                       mean_spearman=("spearman", "mean")))
    best_agg = best_agg.sort_values(["condition", "target", "n_wins", "mean_r2"], ascending=[True, True, False, False])
    best_agg.to_csv(out_dir / "stage4_best_predictor_counts.csv", index=False)

    if summaries:
        all_summary = pd.concat(summaries, ignore_index=True)
        all_summary.to_csv(out_dir / "stage4_all_layer_summaries.csv", index=False)

    # Simple publication-style figure: predictor performance over selected targets.
    for condition, dfc in agg.groupby("condition"):
        pivot = dfc.pivot_table(index="target", columns="predictor", values="mean_spearman")
        pivot = pivot.reindex(index=[t for t in KEY_TARGETS if t in pivot.index], columns=[p for p in KEY_PREDICTORS if p in pivot.columns])
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 4.8))
        im = ax.imshow(pivot.values, aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"Stage4 mean Spearman: {condition}")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="mean Spearman")
        fig.tight_layout()
        fig.savefig(out_dir / f"stage4_spearman_{condition}.png", dpi=200)
        plt.close(fig)

    print(f"wrote {out_dir / 'stage4_all_fits.csv'}")
    print(f"wrote {out_dir / 'stage4_key_table.csv'}")
    print(f"wrote {out_dir / 'stage4_best_predictor_counts.csv'}")
    if (out_dir / 'missing_files.txt').exists():
        print(f"warning: some files were missing; see {out_dir / 'missing_files.txt'}")
    print("\nTop key-table rows:")
    print(agg.sort_values("mean_r2", ascending=False).head(30).to_string(index=False))


if __name__ == "__main__":
    main()
