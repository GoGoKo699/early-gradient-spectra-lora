from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


_SCALING_METADATA = [
    "protocol_version",
    "scaling_mode",
    "is_primary_scaling_mode",
    "reference_alpha",
    "scale_reference_rank",
    "scale_at_reference_rank",
]


def _target_group_columns(metrics: pd.DataFrame) -> list[str]:
    columns = ["site_name"]
    if "scaling_mode" in metrics.columns:
        columns.insert(0, "scaling_mode")
    return columns


def compute_targets(metrics: pd.DataFrame, near_gaps: List[float], recovery_fracs: List[float], lambdas: List[float]) -> pd.DataFrame:
    rows = []
    # Multiple adaptation replicates are nested within each site/rank. Define
    # each rank target from their mean loss. Scaling conditions remain separate;
    # averaging them would reintroduce the rank/scale confound.
    group_columns = _target_group_columns(metrics)
    collapsed = (
        metrics.groupby(group_columns + ["rank"], dropna=False, as_index=False)
        .agg(final_val_loss=("final_val_loss", "mean"))
    )
    metadata = {
        column: metrics.groupby(group_columns, dropna=False)[column].first()
        for column in _SCALING_METADATA
        if column in metrics.columns
    }
    for keys, g in collapsed.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip(group_columns, keys))
        h = g.sort_values("rank")
        if 0 in set(h["rank"]):
            v0 = float(h.loc[h["rank"] == 0, "final_val_loss"].iloc[0])
        else:
            v0 = float(h["final_val_loss"].max())
        vmin = float(h["final_val_loss"].min())
        best_rank = int(h.loc[h["final_val_loss"].idxmin(), "rank"])
        gap = max(v0 - vmin, 1e-12)
        row = {**identity, "base_loss": v0, "best_loss": vmin, "best_rank": best_rank}
        lookup_key = keys[0] if len(keys) == 1 else keys
        for column, series in metadata.items():
            row[column] = series.loc[lookup_key]
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
    predictor_cols = ["effective_rank", "stable_rank", "hard_detectable_rank", "soft_dimension", "gradient_norm", "frobenius"]
    target_cols = [c for c in targets.columns if c.startswith("near_best") or c.startswith("recovery") or c.startswith("penalized")]
    fit_groups = ["scaling_mode"] if "scaling_mode" in targets.columns else []
    grouped = targets.groupby(fit_groups, dropna=False) if fit_groups else [((), targets)]
    rows = []
    for keys, target_group in grouped:
        if fit_groups and not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip(fit_groups, keys if fit_groups else ()))
        df = target_group.merge(stats, on="site_name", how="left")
        metadata = {
            column: target_group[column].iloc[0]
            for column in _SCALING_METADATA
            if column in target_group.columns and column not in identity
        }
        for target in target_cols:
            for pred in predictor_cols:
                if pred not in df:
                    continue
                sub = df[[target, pred]].dropna()
                if len(sub) < 2:
                    rho = float("nan")
                else:
                    rho = spearman(sub[pred], sub[target])
                rows.append({
                    **identity,
                    **metadata,
                    "target": target,
                    "predictor": pred,
                    "n": len(sub),
                    "spearman": rho,
                })
    return pd.DataFrame(rows)
