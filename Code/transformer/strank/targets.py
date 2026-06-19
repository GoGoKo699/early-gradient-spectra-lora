from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def compute_targets(metrics: pd.DataFrame, near_gaps: List[float], recovery_fracs: List[float], lambdas: List[float]) -> pd.DataFrame:
    rows = []
    for site, g in metrics.groupby("site_name"):
        h = g.sort_values("rank")
        if 0 in set(h["rank"]):
            v0 = float(h.loc[h["rank"] == 0, "final_val_loss"].iloc[0])
        else:
            v0 = float(h["final_val_loss"].max())
        vmin = float(h["final_val_loss"].min())
        best_rank = int(h.loc[h["final_val_loss"].idxmin(), "rank"])
        gap = max(v0 - vmin, 1e-12)
        row = {"site_name": site, "base_loss": v0, "best_loss": vmin, "best_rank": best_rank}
        for gamma in near_gaps:
            threshold = vmin + float(gamma) * gap
            ok = h[h["final_val_loss"] <= threshold]
            row[f"near_best_rank_gap_{gamma}"] = int(ok["rank"].min()) if len(ok) else best_rank
        for rho in recovery_fracs:
            threshold = v0 - float(rho) * gap
            ok = h[h["final_val_loss"] <= threshold]
            row[f"recovery_rank_{rho}"] = int(ok["rank"].min()) if len(ok) else best_rank
        max_rank = max(int(h["rank"].max()), 1)
        for lam in lambdas:
            obj = h["final_val_loss"].to_numpy() + float(lam) * gap * (h["rank"].to_numpy() / max_rank)
            idx = int(np.argmin(obj))
            row[f"penalized_rank_lambda_{lam}"] = int(h.iloc[idx]["rank"])
        rows.append(row)
    return pd.DataFrame(rows)


def spearman(x, y) -> float:
    xs = pd.Series(x, dtype="float64").rank(method="average")
    ys = pd.Series(y, dtype="float64").rank(method="average")
    if xs.std() == 0 or ys.std() == 0:
        return float("nan")
    return float(xs.corr(ys))


def prediction_fit(targets: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    df = targets.merge(stats, on="site_name", how="left")
    predictor_cols = ["effective_rank", "stable_rank", "hard_detectable_rank", "soft_dimension", "gradient_norm", "frobenius"]
    target_cols = [c for c in targets.columns if c.startswith("near_best") or c.startswith("recovery") or c.startswith("penalized")]
    rows = []
    for target in target_cols:
        for pred in predictor_cols:
            if pred not in df:
                continue
            sub = df[[target, pred]].dropna()
            if len(sub) < 2:
                rho = float("nan")
            else:
                rho = spearman(sub[pred], sub[target])
            rows.append({"target": target, "predictor": pred, "n": len(sub), "spearman": rho})
    return pd.DataFrame(rows)
