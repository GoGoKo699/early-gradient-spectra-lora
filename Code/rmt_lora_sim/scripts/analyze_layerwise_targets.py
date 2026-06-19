#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def _power2_ceiling(x: float, ranks: list[int]) -> int:
    x = max(1, int(np.ceil(float(x))))
    for r in sorted(ranks):
        if r >= x:
            return int(r)
    return int(max(ranks))


def _add_gradient_summary_fields(row: dict, g: pd.DataFrame, ranks: list[int]) -> None:
    prefixes = [
        ("", ""),
        ("raw_", "raw"),
        ("whitened_", "whitened"),
        ("oracle_whitened_", "oracle_whitened"),
    ]
    suffixes = [
        ("detectable_rank", "detectable", int),
        ("effective_rank", "effective", float),
        ("stable_rank", "stable", float),
    ]
    for col_prefix, label in prefixes:
        for suffix, metric, caster in suffixes:
            col = f"{col_prefix}gradient_{suffix}"
            if col not in g.columns:
                continue
            value = caster(g[col].iloc[0])
            row[col] = value
            pred_col = f"predicted_rank_{label}_{metric}" if label else f"predicted_rank_{metric}"
            row[pred_col] = _power2_ceiling(float(value), ranks)
    if "predicted_rank_detectable" in row:
        row.setdefault("predicted_rank_from_gradient", row["predicted_rank_detectable"])

def _fit_target(summary: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    predictor_candidates = [
        "gradient_detectable_rank",
        "predicted_rank_detectable" if "predicted_rank_detectable" in summary.columns else "predicted_rank_from_gradient",
        "gradient_effective_rank",
        "gradient_stable_rank",
        "raw_gradient_detectable_rank",
        "raw_gradient_effective_rank",
        "raw_gradient_stable_rank",
        "predicted_rank_raw_detectable",
        "predicted_rank_raw_effective",
        "predicted_rank_raw_stable",
        "whitened_gradient_detectable_rank",
        "whitened_gradient_effective_rank",
        "whitened_gradient_stable_rank",
        "predicted_rank_whitened_detectable",
        "predicted_rank_whitened_effective",
        "predicted_rank_whitened_stable",
        "true_k_strong",
    ]
    predictors = [p for p in predictor_candidates if p in summary.columns]
    rows = []
    for target in target_cols:
        for pred in predictors:
            sub = summary[[target, pred]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) < 4:
                continue
            y = np.log2(np.asarray(sub[target], dtype=float))
            x = np.log2(np.maximum(np.asarray(sub[pred], dtype=float), 1.0))
            sst = float(np.sum((y - y.mean()) ** 2))
            if sst <= 1e-12:
                r2 = np.nan
                rmse = 0.0
            else:
                A = np.column_stack([np.ones(len(sub)), x])
                beta, *_ = np.linalg.lstsq(A, y, rcond=None)
                resid = y - A @ beta
                r2 = 1.0 - float(np.sum(resid**2)) / sst
                rmse = float(np.sqrt(np.mean(resid**2)))
            sp = float(pd.Series(sub[pred]).corr(pd.Series(sub[target]), method="spearman"))
            rows.append({"target": target, "predictor": pred, "n": len(sub), "r2_log2_target": r2, "spearman": sp, "rmse_log2": rmse})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["target", "r2_log2_target"], ascending=[True, False], na_position="last")
    return out


def augment_summary_from_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    # Works with stage2 metrics even if v2 summary columns do not exist.
    rows = []
    ranks = sorted(metrics["rank"].unique())
    max_rank = max(ranks)
    for layer, g0 in metrics.groupby("layer"):
        g = g0.sort_values("rank").copy()
        base = float(g["base_val_loss"].iloc[0])
        oracle = float(g["oracle_val_loss"].iloc[0])
        gap = max(base - oracle, 1e-12)
        g["normalized_residual"] = (g["final_val_loss"] - oracle) / gap
        g["recovery_frac"] = (base - g["final_val_loss"]) / gap
        best = g.loc[g["final_val_loss"].idxmin()]
        row = {
            "layer": int(layer),
            "base_val_loss": base,
            "oracle_val_loss": oracle,
            "improvement_gap": gap,
            "best_rank": int(best["rank"]),
            "best_val_loss": float(best["final_val_loss"]),
            "true_k_strong": int(g["true_k_strong"].iloc[0]) if "true_k_strong" in g.columns else np.nan,
        }
        if "gradient_matrix_mode" in g.columns:
            row["gradient_matrix_mode"] = str(g["gradient_matrix_mode"].iloc[0])
        _add_gradient_summary_fields(row, g, ranks)
        for tol in [0.02, 0.05, 0.10, 0.20, 0.50]:
            ok = g[g["final_val_loss"] <= float(best["final_val_loss"]) + tol * gap].sort_values("rank")
            row[f"near_best_rank_gap_{tol:g}"] = int(ok["rank"].iloc[0]) if len(ok) else int(best["rank"])
        for target in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
            ok = g[g["recovery_frac"] >= target].sort_values("rank")
            row[f"recovery_rank_{target:g}"] = float(ok["rank"].iloc[0]) if len(ok) else np.nan
        for lam in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
            score = g["normalized_residual"] + lam * (g["rank"] / max_rank)
            idx = score.idxmin()
            row[f"penalized_rank_lambda_{lam:g}"] = int(g.loc[idx, "rank"])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="A run directory, metrics.csv, or layer_summary.csv")
    args = ap.parse_args()
    p = Path(args.path)
    if p.is_dir():
        metrics_path = p / "metrics.csv"
        summary_path = p / "layer_summary.csv"
        out_dir = p
    elif p.name == "metrics.csv":
        metrics_path = p
        summary_path = p.with_name("layer_summary.csv")
        out_dir = p.parent
    else:
        metrics_path = None
        summary_path = p
        out_dir = p.parent

    if metrics_path is not None and metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        summary = augment_summary_from_metrics(metrics)
    else:
        summary = pd.read_csv(summary_path)

    out_summary = out_dir / "layer_summary_augmented.csv"
    summary.to_csv(out_summary, index=False)
    target_cols = [c for c in summary.columns if c.startswith("near_best_rank_gap_") or c.startswith("recovery_rank_") or c.startswith("penalized_rank_lambda_")]
    target_cols = ["best_rank"] + target_cols
    fit = _fit_target(summary, target_cols)
    out_fit = out_dir / "layer_target_fit_augmented.csv"
    fit.to_csv(out_fit, index=False)

    print(f"wrote {out_summary}")
    print(f"wrote {out_fit}")
    if len(fit):
        print("\nTop target fits:")
        print(fit.sort_values("r2_log2_target", ascending=False).head(20).to_string(index=False))

    # A compact figure for the most useful existing-stage2 target.
    target = "penalized_rank_lambda_0.15" if "penalized_rank_lambda_0.15" in summary.columns else target_cols[0]
    fig, ax = plt.subplots()
    ax.scatter(summary["gradient_effective_rank"], summary[target])
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("early-gradient effective rank")
    ax.set_ylabel(target)
    ax.set_title("Layerwise rank target vs early spectrum")
    fig.tight_layout()
    fig_path = out_dir / "layer_target_vs_gradient_effective.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
