from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12


def _safe_log2(x: pd.Series) -> np.ndarray:
    return np.log2(np.asarray(x, dtype=float) + EPS)


def _fit_ols(y: np.ndarray, x: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    y = y[mask]
    x = x[mask]
    n = len(y)
    if n == 0:
        return {"n": 0, "k": x.shape[1] if x.ndim == 2 else 0, "r2": np.nan, "adj_r2": np.nan, "rmse": np.nan, "aic": np.nan}
    x_design = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    pred = x_design @ beta
    resid = y - pred
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    k = int(x_design.shape[1])
    r2 = 1.0 - sse / (sst + EPS)
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - k, 1)
    rmse = float(np.sqrt(sse / max(n, 1)))
    aic = float(n * np.log(sse / max(n, 1) + EPS) + 2 * k)
    return {"n": n, "k": k, "r2": float(r2), "adj_r2": float(adj_r2), "rmse": rmse, "aic": aic}


def _available(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    return all(c in df.columns for c in cols)


def fit_hypotheses(metrics_path: str | Path, outcome: str | None = None, out_path: str | Path | None = None) -> pd.DataFrame:
    """Fit simple predictor models for deciding which hypothesis is supported.

    This intentionally uses transparent OLS rather than a black-box ML model.
    Lower AIC/RMSE and higher adjusted R^2 indicate a better simple explanation.
    """
    metrics_path = Path(metrics_path)
    df = pd.read_csv(metrics_path)
    if len(df) == 0:
        raise ValueError(f"No rows in {metrics_path}")
    exp = str(df.get("experiment", pd.Series(["unknown"])).iloc[0])
    if outcome is None:
        outcome = "merge_degradation" if exp == "merge" else "final_val_loss"
    if outcome not in df.columns:
        raise ValueError(f"Outcome {outcome!r} not found in {metrics_path}")
    y = np.asarray(df[outcome], dtype=float)

    candidates: list[tuple[str, list[str], np.ndarray]] = []

    if _available(df, ["rank"]):
        candidates.append(("H_nominal_rank_log2", ["log2(rank)"], _safe_log2(df["rank"])[:, None]))
    if _available(df, ["alpha"]):
        candidates.append(("H_alpha_log2", ["log2(alpha)"], _safe_log2(df["alpha"])[:, None]))
    if _available(df, ["rank", "alpha"]):
        candidates.append(
            (
                "H_nominal_rank_plus_alpha",
                ["log2(rank)", "log2(alpha)"],
                np.column_stack([_safe_log2(df["rank"]), _safe_log2(df["alpha"])]),
            )
        )
    if _available(df, ["gradient_detectable_rank"]):
        candidates.append(("H_early_gradient_detectable_rank", ["gradient_detectable_rank"], np.asarray(df[["gradient_detectable_rank"]], dtype=float)))
    if _available(df, ["adapter_detectable_rank"]):
        candidates.append(("H_final_adapter_detectable_rank", ["adapter_detectable_rank"], np.asarray(df[["adapter_detectable_rank"]], dtype=float)))
    if _available(df, ["gradient_detectable_rank", "rank"]):
        candidates.append(
            (
                "H_gradient_detectable_plus_nominal_rank",
                ["gradient_detectable_rank", "log2(rank)"],
                np.column_stack([np.asarray(df["gradient_detectable_rank"], dtype=float), _safe_log2(df["rank"])]),
            )
        )
    if _available(df, ["adapter_detectable_rank", "alpha"]):
        candidates.append(
            (
                "H_adapter_detectable_plus_alpha",
                ["adapter_detectable_rank", "log2(alpha)"],
                np.column_stack([np.asarray(df["adapter_detectable_rank"], dtype=float), _safe_log2(df["alpha"])]),
            )
        )
    if _available(df, ["adapter_frobenius"]):
        candidates.append(("H_adapter_frobenius", ["adapter_frobenius"], np.asarray(df[["adapter_frobenius"]], dtype=float)))
    if _available(df, ["adapter_stable_rank"]):
        candidates.append(("H_adapter_stable_rank", ["adapter_stable_rank"], np.asarray(df[["adapter_stable_rank"]], dtype=float)))
    if _available(df, ["adapter_effective_rank"]):
        candidates.append(("H_adapter_effective_rank", ["adapter_effective_rank"], np.asarray(df[["adapter_effective_rank"]], dtype=float)))
    if _available(df, ["intruder_count"]):
        candidates.append(("H_intruder_count", ["intruder_count"], np.asarray(df[["intruder_count"]], dtype=float)))
    if _available(df, ["intruder_count", "adapter_detectable_rank"]):
        candidates.append(("H_detectable_rank_plus_intruders", ["adapter_detectable_rank", "intruder_count"], np.asarray(df[["adapter_detectable_rank", "intruder_count"]], dtype=float)))

    # Merging-specific hypotheses.
    if _available(df, ["overlap"]):
        candidates.append(("H_planted_overlap", ["overlap"], np.asarray(df[["overlap"]], dtype=float)))
    if _available(df, ["spectral_interference"]):
        candidates.append(("H_spectral_interference", ["spectral_interference"], np.asarray(df[["spectral_interference"]], dtype=float)))
    if _available(df, ["spectral_interference", "overlap"]):
        candidates.append(("H_interference_plus_overlap", ["spectral_interference", "overlap"], np.asarray(df[["spectral_interference", "overlap"]], dtype=float)))

    rows = []
    for name, predictors, x in candidates:
        stats = _fit_ols(y, np.asarray(x, dtype=float))
        rows.append(
            {
                "experiment": exp,
                "outcome": outcome,
                "model": name,
                "predictors": ",".join(predictors),
                **stats,
            }
        )
    result = pd.DataFrame(rows).sort_values(["aic", "rmse"], ascending=[True, True])
    if out_path is None:
        out_path = metrics_path.parent / "hypothesis_fit.csv"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    return result
