#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

EPS = 1e-12


def log2_series(s: pd.Series) -> np.ndarray:
    return np.log2(np.asarray(s, dtype=float) + EPS)


def log10_series(s: pd.Series) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(s, dtype=float), EPS))


def ols_r2(y: np.ndarray, x: np.ndarray) -> tuple[int, int, float, float, float, float]:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    y = y[mask]
    x = x[mask]
    n = len(y)
    if n == 0:
        return 0, x.shape[1] + 1, np.nan, np.nan, np.nan, np.nan
    xd = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(xd, y, rcond=None)
    pred = xd @ beta
    resid = y - pred
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - y.mean())**2))
    k = xd.shape[1]
    r2 = 1.0 - sse / (sst + EPS)
    adj = 1.0 - (1.0 - r2) * (n - 1) / max(n - k, 1)
    rmse = float(np.sqrt(sse / max(n, 1)))
    aic = float(n * np.log(sse / max(n, 1) + EPS) + 2 * k)
    return n, k, float(r2), float(adj), rmse, aic


def spearman(y: np.ndarray, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    return float(pd.Series(x[mask]).corr(pd.Series(y[mask]), method="spearman"))


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "rank" in df:
        df["log2_rank"] = log2_series(df["rank"])
    if "alpha" in df:
        df["log2_alpha"] = log2_series(df["alpha"])
    if "adapter_frobenius" in df:
        df["log10_adapter_frobenius"] = log10_series(df["adapter_frobenius"])
    if "final_val_loss" in df:
        df["log10_final_val_loss"] = log10_series(df["final_val_loss"])
    if {"final_val_loss", "oracle_val_loss"}.issubset(df.columns):
        excess = np.maximum(np.asarray(df["final_val_loss"], dtype=float) - np.asarray(df["oracle_val_loss"], dtype=float), EPS)
        df["excess_val_loss"] = excess
        df["log10_excess_val_loss"] = np.log10(excess)
    if "merge_degradation" in df:
        df["log10_merge_degradation"] = log10_series(df["merge_degradation"])
    return df


def available(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    return all(c in df.columns for c in cols)


def candidate_models(df: pd.DataFrame) -> list[tuple[str, list[str]]]:
    candidates = [
        ("rank", ["log2_rank"]),
        ("alpha", ["log2_alpha"]),
        ("rank_plus_alpha", ["log2_rank", "log2_alpha"]),
        ("gradient_detectable", ["gradient_detectable_rank"]),
        ("adapter_detectable", ["adapter_detectable_rank"]),
        ("adapter_detectable_plus_alpha", ["adapter_detectable_rank", "log2_alpha"]),
        ("adapter_stable_rank", ["adapter_stable_rank"]),
        ("adapter_effective_rank", ["adapter_effective_rank"]),
        ("adapter_frobenius", ["adapter_frobenius"]),
        ("log_adapter_frobenius", ["log10_adapter_frobenius"]),
        ("adapter_overlap_true", ["adapter_mean_overlap_true"]),
        ("intruder_count", ["intruder_count"]),
        ("intruder_energy", ["intruder_energy"]),
        ("overlap", ["overlap"]),
        ("spectral_interference", ["spectral_interference"]),
        ("signed_task_inner", ["signed_task_inner"]),
        ("conflict_score", ["conflict_score"]),
        ("interference_plus_conflict", ["spectral_interference", "conflict_score"]),
    ]
    return [(name, cols) for name, cols in candidates if available(df, cols)]


def subset_frames(df: pd.DataFrame, outcome: str) -> list[tuple[str, pd.DataFrame]]:
    frames = [("all", df)]
    clean = df.copy()
    if "diverged" in clean.columns:
        clean = clean[~clean["diverged"].astype(bool)]
    clean = clean[np.isfinite(clean[outcome].astype(float))]
    frames.append(("nondiverged", clean))
    if outcome in clean.columns and len(clean) >= 8:
        q95 = float(clean[outcome].quantile(0.95))
        frames.append(("nondiverged_drop_top5pct_outcome", clean[clean[outcome] <= q95]))
    if "final_val_loss" in clean.columns:
        frames.append(("nondiverged_val_loss_lt_1", clean[clean["final_val_loss"] < 1.0]))
    return [(name, sub) for name, sub in frames if len(sub) >= 3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", type=Path)
    ap.add_argument("--outcome", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    df = add_derived_columns(pd.read_csv(args.metrics))
    exp = str(df["experiment"].iloc[0]) if "experiment" in df.columns and len(df) else "unknown"
    if args.outcome is not None:
        outcomes = [args.outcome]
    elif exp == "merge" or "merge_degradation" in df.columns:
        outcomes = [c for c in ["merge_degradation", "log10_merge_degradation", "forgetting_merge"] if c in df.columns]
    else:
        outcomes = [c for c in ["final_val_loss", "log10_final_val_loss", "excess_val_loss", "log10_excess_val_loss", "forgetting_loss"] if c in df.columns]

    rows = []
    for outcome in outcomes:
        for subset_name, sub in subset_frames(df, outcome):
            y = np.asarray(sub[outcome], dtype=float)
            for model, cols in candidate_models(sub):
                x = np.column_stack([np.asarray(sub[c], dtype=float) for c in cols])
                n, k, r2, adj, rmse, aic = ols_r2(y, x)
                rho = spearman(y, x[:, 0]) if len(cols) == 1 else np.nan
                rows.append({
                    "experiment": exp,
                    "subset": subset_name,
                    "outcome": outcome,
                    "model": model,
                    "predictors": ",".join(cols),
                    "n": n,
                    "k": k,
                    "ols_r2": r2,
                    "adj_r2": adj,
                    "rmse": rmse,
                    "aic": aic,
                    "spearman_first_predictor": rho,
                    "abs_spearman_first_predictor": abs(rho) if np.isfinite(rho) else np.nan,
                })
    out = pd.DataFrame(rows).sort_values(["outcome", "subset", "aic", "rmse"])
    out_path = args.out or (args.metrics.parent / "robust_fit.csv")
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
