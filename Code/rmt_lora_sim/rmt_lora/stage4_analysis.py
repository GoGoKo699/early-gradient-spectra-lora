from __future__ import annotations

"""Pre-rerun Stage4 analysis plan and regression transform."""

from typing import Any

import numpy as np
import pandas as pd

ANALYSIS_PLAN_VERSION = "stage4_confirmatory_v1"
PRIMARY_PREDICTOR = "gradient_effective_rank"
PRIMARY_TARGET = "near_best_rank_gap_0.1"
PRIMARY_CONDITION = "hard_knee"
ROBUSTNESS_CONDITION = "sample_limited"
TARGET_TRANSFORM = "log2(rank)"
PREDICTOR_TRANSFORM = "log2(1+s)"


def transform_spectral_predictor(values: Any) -> np.ndarray:
    """Apply the manuscript-specified predictor transform."""

    array = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(array)):
        raise ValueError("spectral predictor values must be finite before transformation")
    if np.any(array < 0):
        raise ValueError("spectral predictor values must be nonnegative")
    return np.log2(1.0 + array)


def spearman_or_nan(x_values: Any, y_values: Any) -> float:
    """Return Spearman correlation, or NaN when it is not identifiable.

    Pandas delegates constant-input cases to SciPy, which otherwise emits a
    warning for every exploratory fit.  Detecting those cases explicitly keeps
    publication rerun logs readable without changing the statistical result.
    """

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if x.shape != y.shape:
        raise ValueError("Spearman inputs must have matching shapes")
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def analysis_role(condition: str, target: str, predictor: str) -> str:
    if target == PRIMARY_TARGET and predictor == PRIMARY_PREDICTOR:
        if condition == PRIMARY_CONDITION:
            return "confirmatory_primary"
        if condition == ROBUSTNESS_CONDITION:
            return "confirmatory_robustness"
        return "confirmatory_other_condition"
    if predictor == PRIMARY_PREDICTOR:
        return "secondary_target"
    return "exploratory"


def analysis_plan_document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "analysis_plan_version": ANALYSIS_PLAN_VERSION,
        "frozen_before_corrected_rerun": True,
        "primary_hypothesis": {
            "condition": PRIMARY_CONDITION,
            "target": PRIMARY_TARGET,
            "predictor": PRIMARY_PREDICTOR,
            "target_transform": TARGET_TRANSFORM,
            "predictor_transform": PREDICTOR_TRANSFORM,
            "model": "ordinary least squares with intercept within each seed run",
            "run_level_summary": "mean R2 and mean Spearman across independent seeds",
        },
        "robustness_condition": ROBUSTNESS_CONDITION,
        "secondary_targets": [
            "near_best_rank_gap_0.2",
            "recovery_rank_0.7",
            "recovery_rank_0.8",
            "penalized_rank_lambda_0.2",
            "penalized_rank_lambda_0.3",
        ],
        "multiplicity_policy": (
            "Only the hard-knee primary target/predictor pair is confirmatory. "
            "The sample-limited counterpart is a prespecified robustness check; "
            "all other target/predictor combinations are secondary or exploratory."
        ),
    }
