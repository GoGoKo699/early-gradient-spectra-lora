from __future__ import annotations

"""Canonical useful-rank target definitions.

The publication estimand is based only on the observed validation-loss curve over
its tested rank grid. Simulation oracle loss is retained as a diagnostic and is
never used to set a target threshold or rank penalty.
"""

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

TARGET_ESTIMAND = "observed_rank_sweep_minimum"
TARGET_ESTIMAND_VERSION = "observed_best_gap_v1"
TARGET_REFERENCE = "best_observed_validation_loss_on_tested_rank_grid"
TARGET_DENOMINATOR = "base_val_loss_minus_best_observed_val_loss"
TARGET_SCHEMA_VERSION = 1

TARGET_METADATA_COLUMNS = (
    "target_estimand",
    "target_estimand_version",
    "target_reference",
    "target_denominator",
)

_TARGET_PREFIXES = (
    "near_best_rank_gap_",
    "recovery_rank_",
    "penalized_rank_lambda_",
)


def target_definition_document() -> dict[str, Any]:
    """Return a machine-readable statement of the empirical target contract."""

    return {
        "schema_version": TARGET_SCHEMA_VERSION,
        "target_estimand": TARGET_ESTIMAND,
        "target_estimand_version": TARGET_ESTIMAND_VERSION,
        "target_reference": TARGET_REFERENCE,
        "target_denominator": TARGET_DENOMINATOR,
        "notation": {
            "base": "V0 = validation loss before adaptation",
            "best": "Vmin = min_r V(r) over the tested rank grid",
            "gap": "DeltaV = V0 - Vmin",
        },
        "targets": {
            "near_best": "min r such that V(r) <= Vmin + gamma * DeltaV",
            "recovery": "min r such that (V0 - V(r)) / DeltaV >= rho",
            "penalized": "argmin_r V(r) + lambda * DeltaV * r / rmax",
        },
        "oracle_policy": (
            "oracle_val_loss is diagnostic only and cannot affect empirical "
            "useful-rank targets"
        ),
        "degenerate_curve_policy": (
            "if DeltaV <= gap_epsilon, useful-rank targets are undefined (NaN)"
        ),
        "tie_breaking": "choose the smallest nominal rank",
    }


def useful_rank_target_columns(columns: Iterable[str]) -> list[str]:
    """Return canonical useful-rank target columns in their existing order."""

    return [str(c) for c in columns if str(c).startswith(_TARGET_PREFIXES)]


def _constant_finite_value(group: pd.DataFrame, column: str) -> float:
    if column not in group.columns:
        raise ValueError(f"rank curve is missing required column {column!r}")
    values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise ValueError(f"rank curve column {column!r} has no finite value")
    first = float(finite[0])
    if not np.allclose(finite, first, rtol=1e-10, atol=1e-12):
        raise ValueError(f"rank curve column {column!r} is not constant within a layer")
    return first


def _validated_parameters(
    recovery_targets: Sequence[float],
    gap_tolerances: Sequence[float],
    penalty_lambdas: Sequence[float],
) -> tuple[list[float], list[float], list[float]]:
    recovery = [float(v) for v in recovery_targets]
    tolerances = [float(v) for v in gap_tolerances]
    penalties = [float(v) for v in penalty_lambdas]
    if any(not np.isfinite(v) or v < 0.0 or v > 1.0 for v in recovery):
        raise ValueError("recovery targets must be finite and lie in [0, 1]")
    if any(not np.isfinite(v) or v < 0.0 or v > 1.0 for v in tolerances):
        raise ValueError("near-best gap tolerances must be finite and lie in [0, 1]")
    if any(not np.isfinite(v) or v < 0.0 for v in penalties):
        raise ValueError("penalty lambdas must be finite and nonnegative")
    return recovery, tolerances, penalties


def annotate_observed_best_curve(
    group: pd.DataFrame,
    *,
    gap_epsilon: float = 1e-12,
) -> pd.DataFrame:
    """Annotate one layer's rank sweep under the observed-best estimand.

    The returned frame is rank-sorted and includes the observed reference loss,
    gap, normalized residual, and recovery fraction. Oracle loss does not enter
    any of these calculations.
    """

    required = {"rank", "final_val_loss", "base_val_loss"}
    missing = sorted(required - set(group.columns))
    if missing:
        raise ValueError(f"rank curve is missing required columns: {missing}")
    if not np.isfinite(gap_epsilon) or gap_epsilon < 0:
        raise ValueError("gap_epsilon must be finite and nonnegative")

    out = group.copy()
    out["rank"] = pd.to_numeric(out["rank"], errors="raise")
    out = out.sort_values("rank", kind="stable").reset_index(drop=True)
    if out["rank"].duplicated().any():
        duplicates = out.loc[out["rank"].duplicated(keep=False), "rank"].tolist()
        raise ValueError(f"rank curve contains duplicate ranks: {duplicates}")

    ranks = out["rank"].to_numpy(dtype=float)
    if len(ranks) == 0 or not np.all(np.isfinite(ranks)):
        raise ValueError("rank curve must contain at least one finite rank")
    if np.any(ranks < 0):
        raise ValueError("nominal ranks must be nonnegative")

    base = _constant_finite_value(out, "base_val_loss")
    losses = pd.to_numeric(out["final_val_loss"], errors="coerce").to_numpy(dtype=float)
    finite_loss = np.isfinite(losses)

    if finite_loss.any():
        best_loss = float(np.min(losses[finite_loss]))
        observed_gap = float(base - best_loss)
        target_valid = bool(np.isfinite(observed_gap) and observed_gap > gap_epsilon)
        target_status = "ok" if target_valid else "nonpositive_observed_improvement"
    else:
        best_loss = float("nan")
        observed_gap = float("nan")
        target_valid = False
        target_status = "no_finite_rank_losses"

    out["target_estimand"] = TARGET_ESTIMAND
    out["target_estimand_version"] = TARGET_ESTIMAND_VERSION
    out["target_reference"] = TARGET_REFERENCE
    out["target_denominator"] = TARGET_DENOMINATOR
    out["best_observed_val_loss"] = best_loss
    out["observed_best_gap"] = observed_gap
    out["target_valid"] = target_valid
    out["target_status"] = target_status

    if target_valid:
        out["normalized_residual_observed_best"] = (losses - best_loss) / observed_gap
        out["recovery_fraction_observed_best"] = (base - losses) / observed_gap
    else:
        out["normalized_residual_observed_best"] = np.nan
        out["recovery_fraction_observed_best"] = np.nan
    return out


def summarize_observed_best_curve(
    group: pd.DataFrame,
    *,
    recovery_targets: Sequence[float],
    gap_tolerances: Sequence[float],
    penalty_lambdas: Sequence[float],
    gap_epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Compute the publication useful-rank targets for one rank sweep."""

    recovery, tolerances, penalties = _validated_parameters(
        recovery_targets, gap_tolerances, penalty_lambdas
    )
    curve = annotate_observed_best_curve(group, gap_epsilon=gap_epsilon)
    base = _constant_finite_value(curve, "base_val_loss")
    loss_values = pd.to_numeric(curve["final_val_loss"], errors="coerce")
    finite = curve[np.isfinite(loss_values)].copy()
    finite["final_val_loss"] = pd.to_numeric(finite["final_val_loss"], errors="coerce")

    best_loss = float(curve["best_observed_val_loss"].iloc[0])
    observed_gap = float(curve["observed_best_gap"].iloc[0])
    target_valid = bool(curve["target_valid"].iloc[0])
    target_status = str(curve["target_status"].iloc[0])
    max_rank = float(np.max(curve["rank"].to_numpy(dtype=float)))
    if penalties and max_rank <= 0:
        raise ValueError("penalized rank requires at least one positive nominal rank")

    if len(finite):
        best_candidates = finite[
            np.isclose(
                finite["final_val_loss"], best_loss, rtol=1e-12, atol=1e-15
            )
        ]
        best_rank: float | int = int(best_candidates["rank"].min())
    else:
        best_rank = float("nan")

    row: dict[str, Any] = {
        "target_estimand": TARGET_ESTIMAND,
        "target_estimand_version": TARGET_ESTIMAND_VERSION,
        "target_reference": TARGET_REFERENCE,
        "target_denominator": TARGET_DENOMINATOR,
        "target_valid": target_valid,
        "target_status": target_status,
        "n_rank_points": int(len(curve)),
        "n_finite_rank_points": int(len(finite)),
        "rank_grid": ";".join(f"{float(v):g}" for v in curve["rank"]),
        "base_val_loss": base,
        "best_rank": best_rank,
        "best_val_loss": best_loss,
        "best_observed_val_loss": best_loss,
        "observed_best_gap": observed_gap,
        # Backward-compatible alias, made unambiguous by metadata columns.
        "improvement_gap": observed_gap,
        "improvement_gap_reference": TARGET_REFERENCE,
    }

    if "oracle_val_loss" in curve.columns:
        oracle = _constant_finite_value(curve, "oracle_val_loss")
        row["oracle_val_loss"] = oracle
        row["oracle_improvement_gap"] = float(base - oracle)

    for tolerance in tolerances:
        name = f"near_best_rank_gap_{tolerance:g}"
        threshold_name = f"near_best_loss_threshold_gap_{tolerance:g}"
        if target_valid:
            threshold = best_loss + tolerance * observed_gap
            candidates = finite[finite["final_val_loss"] <= threshold + 1e-15]
            row[name] = int(candidates["rank"].min()) if len(candidates) else np.nan
            row[threshold_name] = float(threshold)
        else:
            row[name] = np.nan
            row[threshold_name] = np.nan

    for recovery_target in recovery:
        name = f"recovery_rank_{recovery_target:g}"
        threshold_name = f"recovery_loss_threshold_{recovery_target:g}"
        if target_valid:
            threshold = base - recovery_target * observed_gap
            candidates = finite[finite["final_val_loss"] <= threshold + 1e-15]
            row[name] = int(candidates["rank"].min()) if len(candidates) else np.nan
            row[threshold_name] = float(threshold)
        else:
            row[name] = np.nan
            row[threshold_name] = np.nan

    for penalty_lambda in penalties:
        name = f"penalized_rank_lambda_{penalty_lambda:g}"
        if target_valid:
            score = finite["final_val_loss"] + penalty_lambda * observed_gap * (
                finite["rank"] / max_rank
            )
            min_score = float(score.min())
            candidates = finite[
                np.isclose(score, min_score, rtol=1e-12, atol=1e-15)
            ]
            row[name] = int(candidates["rank"].min())
        else:
            row[name] = np.nan

    return row


def annotate_observed_best_curves(
    metrics: pd.DataFrame,
    *,
    group_column: str = "layer",
    gap_epsilon: float = 1e-12,
) -> pd.DataFrame:
    """Annotate all rank curves while preserving the grouping column."""

    if group_column not in metrics.columns:
        raise ValueError(f"metrics are missing grouping column {group_column!r}")
    groups: list[pd.DataFrame] = []
    for group_value, group in metrics.groupby(group_column, sort=True):
        annotated = annotate_observed_best_curve(group, gap_epsilon=gap_epsilon)
        annotated[group_column] = group_value
        groups.append(annotated)
    if not groups:
        return metrics.copy()
    return pd.concat(groups, ignore_index=True)


def validate_target_estimand_frame(frame: pd.DataFrame, *, context: str) -> None:
    """Reject missing, legacy, or mixed target-estimand metadata."""

    missing = [c for c in TARGET_METADATA_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{context} lacks target-estimand metadata columns {missing}; "
            "legacy oracle-gap artifacts are not publication-compatible"
        )
    expected = {
        "target_estimand": TARGET_ESTIMAND,
        "target_estimand_version": TARGET_ESTIMAND_VERSION,
        "target_reference": TARGET_REFERENCE,
        "target_denominator": TARGET_DENOMINATOR,
    }
    for column, value in expected.items():
        observed = {str(v) for v in frame[column].dropna().unique()}
        if observed != {value}:
            raise ValueError(
                f"{context} has invalid or mixed {column}: {sorted(observed)}; "
                f"expected {value!r}"
            )
