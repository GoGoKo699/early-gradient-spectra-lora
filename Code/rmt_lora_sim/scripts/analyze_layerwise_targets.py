#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from rmt_lora.provenance import write_json, write_sha256s
from rmt_lora.stage4_analysis import (
    ANALYSIS_PLAN_VERSION,
    PREDICTOR_TRANSFORM,
    PRIMARY_PREDICTOR,
    PRIMARY_TARGET,
    TARGET_TRANSFORM,
    analysis_plan_document,
    spearman_or_nan,
    transform_spectral_predictor,
)
from rmt_lora.targets import (
    TARGET_DENOMINATOR,
    TARGET_ESTIMAND,
    TARGET_ESTIMAND_VERSION,
    TARGET_REFERENCE,
    annotate_observed_best_curves,
    summarize_observed_best_curve,
    target_definition_document,
    useful_rank_target_columns,
    validate_target_estimand_frame,
)


def _power2_ceiling(x: float, ranks: list[int]) -> int:
    x = max(1, int(np.ceil(float(x))))
    for rank in sorted(ranks):
        if rank >= x:
            return int(rank)
    return int(max(ranks))


def _add_gradient_summary_fields(row: dict, group: pd.DataFrame, ranks: list[int]) -> None:
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
    for column_prefix, label in prefixes:
        for suffix, metric, caster in suffixes:
            column = f"{column_prefix}gradient_{suffix}"
            if column not in group.columns:
                continue
            value = caster(group[column].iloc[0])
            row[column] = value
            predicted = (
                f"predicted_rank_{label}_{metric}"
                if label
                else f"predicted_rank_{metric}"
            )
            row[predicted] = _power2_ceiling(float(value), ranks)
    if "predicted_rank_detectable" in row:
        row.setdefault("predicted_rank_from_gradient", row["predicted_rank_detectable"])


def _fit_target(summary: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    validate_target_estimand_frame(summary, context="augmented layer summary")
    predictor_candidates = [
        "gradient_detectable_rank",
        "predicted_rank_detectable"
        if "predicted_rank_detectable" in summary.columns
        else "predicted_rank_from_gradient",
        "predicted_rank_effective",
        "predicted_rank_stable",
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
        for predictor in predictors:
            sub = (
                summary[[target, predictor]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(sub) < 4:
                continue
            y = np.log2(np.asarray(sub[target], dtype=float))
            x = transform_spectral_predictor(np.asarray(sub[predictor], dtype=float))
            sst = float(np.sum((y - y.mean()) ** 2))
            if sst <= 1e-12:
                r2 = np.nan
                rmse = 0.0
            else:
                design = np.column_stack([np.ones(len(sub)), x])
                beta, *_ = np.linalg.lstsq(design, y, rcond=None)
                residual = y - design @ beta
                r2 = 1.0 - float(np.sum(residual**2)) / sst
                rmse = float(np.sqrt(np.mean(residual**2)))
            spearman = spearman_or_nan(sub[predictor], sub[target])
            rows.append(
                {
                    "target_estimand": TARGET_ESTIMAND,
                    "target_estimand_version": TARGET_ESTIMAND_VERSION,
                    "target_reference": TARGET_REFERENCE,
                    "target_denominator": TARGET_DENOMINATOR,
                    "analysis_plan_version": ANALYSIS_PLAN_VERSION,
                    "analysis_pair_role": (
                        "primary_pair"
                        if target == PRIMARY_TARGET and predictor == PRIMARY_PREDICTOR
                        else "secondary_or_exploratory"
                    ),
                    "target_transform": TARGET_TRANSFORM,
                    "predictor_transform": PREDICTOR_TRANSFORM,
                    "target": target,
                    "target_role": "diagnostic"
                    if target == "best_rank"
                    else "useful_rank",
                    "predictor": predictor,
                    "n": len(sub),
                    "r2_log2_target": r2,
                    "spearman": spearman,
                    "rmse_log2": rmse,
                }
            )
    columns = [
        "target_estimand",
        "target_estimand_version",
        "target_reference",
        "target_denominator",
        "analysis_plan_version",
        "analysis_pair_role",
        "target_transform",
        "predictor_transform",
        "target",
        "target_role",
        "predictor",
        "n",
        "r2_log2_target",
        "spearman",
        "rmse_log2",
    ]
    out = pd.DataFrame(rows, columns=columns)
    if len(out):
        out = out.sort_values(
            ["target", "r2_log2_target"],
            ascending=[True, False],
            na_position="last",
        )
    return out


def augment_summary_from_metrics(
    metrics: pd.DataFrame,
    summary_cfg: dict | None = None,
) -> pd.DataFrame:
    """Recompute targets from raw per-rank metrics using the canonical estimand."""

    cfg = summary_cfg or {}
    recovery_targets = [
        float(v)
        for v in cfg.get(
            "recovery_targets", [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
        )
    ]
    gap_tolerances = [
        float(v)
        for v in cfg.get("gap_tolerances", [0.02, 0.05, 0.10, 0.20, 0.50])
    ]
    penalty_lambdas = [
        float(v)
        for v in cfg.get(
            "penalty_lambdas", [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]
        )
    ]
    gap_epsilon = float(cfg.get("gap_epsilon", 1e-12))
    rows = []
    ranks = sorted(int(v) for v in metrics["rank"].unique())
    for layer, group0 in metrics.groupby("layer"):
        group = group0.sort_values("rank").copy()
        target_row = summarize_observed_best_curve(
            group,
            recovery_targets=recovery_targets,
            gap_tolerances=gap_tolerances,
            penalty_lambdas=penalty_lambdas,
            gap_epsilon=gap_epsilon,
        )
        row = {
            "layer": int(layer),
            **target_row,
            "true_k_strong": int(group["true_k_strong"].iloc[0])
            if "true_k_strong" in group.columns
            else np.nan,
        }
        if "gradient_matrix_mode" in group.columns:
            row["gradient_matrix_mode"] = str(group["gradient_matrix_mode"].iloc[0])
        _add_gradient_summary_fields(row, group, ranks)
        rows.append(row)
    return pd.DataFrame(rows)


def _load_summary_config(out_dir: Path) -> dict:
    config_path = out_dir / "config.yaml"
    if not config_path.exists():
        return {}
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    summary = config.get("summary", {})
    return dict(summary) if isinstance(summary, dict) else {}


def _assert_existing_summary_matches(
    existing: pd.DataFrame,
    recomputed: pd.DataFrame,
) -> None:
    """Fail if the producer and analyzer disagree on any canonical target field."""

    validate_target_estimand_frame(existing, context="producer layer_summary.csv")
    validate_target_estimand_frame(recomputed, context="recomputed layer summary")
    left = existing.sort_values("layer").reset_index(drop=True)
    right = recomputed.sort_values("layer").reset_index(drop=True)
    if left["layer"].tolist() != right["layer"].tolist():
        raise ValueError("producer and analyzer layer sets differ")

    columns = [
        "best_rank",
        "best_val_loss",
        "observed_best_gap",
        "target_valid",
        "target_status",
        *useful_rank_target_columns(right.columns),
        "target_estimand",
        "target_estimand_version",
        "target_reference",
        "target_denominator",
    ]
    missing = [c for c in columns if c not in left.columns]
    if missing:
        raise ValueError(f"producer layer summary is missing canonical fields: {missing}")
    for column in columns:
        if pd.api.types.is_numeric_dtype(right[column]):
            if not np.allclose(
                pd.to_numeric(left[column], errors="coerce"),
                pd.to_numeric(right[column], errors="coerce"),
                equal_nan=True,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(f"producer and analyzer disagree on {column}")
        elif left[column].astype(str).tolist() != right[column].astype(str).tolist():
            raise ValueError(f"producer and analyzer disagree on {column}")


def _assert_existing_fit_matches(existing: pd.DataFrame, recomputed: pd.DataFrame) -> None:
    """Fail if producer and analyzer use different transforms or regressions."""

    key = ["target", "predictor"]
    left = existing.sort_values(key).reset_index(drop=True)
    right = recomputed.sort_values(key).reset_index(drop=True)
    if left[key].astype(str).to_records(index=False).tolist() != right[key].astype(str).to_records(index=False).tolist():
        raise ValueError("producer and analyzer fit target/predictor sets differ")
    columns = [
        "target_estimand",
        "target_estimand_version",
        "target_reference",
        "target_denominator",
        "analysis_plan_version",
        "analysis_pair_role",
        "target_transform",
        "predictor_transform",
        "target_role",
        "n",
        "r2_log2_target",
        "spearman",
        "rmse_log2",
    ]
    missing = [column for column in columns if column not in left.columns]
    if missing:
        raise ValueError(f"producer layer_rank_fit.csv is missing fields: {missing}")
    for column in columns:
        if pd.api.types.is_numeric_dtype(right[column]):
            if not np.allclose(
                pd.to_numeric(left[column], errors="coerce"),
                pd.to_numeric(right[column], errors="coerce"),
                equal_nan=True,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(f"producer and analyzer fits disagree on {column}")
        elif left[column].astype(str).tolist() != right[column].astype(str).tolist():
            raise ValueError(f"producer and analyzer fits disagree on {column}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="A run directory, metrics.csv, or layer_summary.csv")
    parser.add_argument(
        "--check-existing",
        action="store_true",
        help="Require layer_summary.csv to match the independently recomputed targets.",
    )
    args = parser.parse_args()
    path = Path(args.path)
    if path.is_dir():
        metrics_path = path / "metrics.csv"
        summary_path = path / "layer_summary.csv"
        out_dir = path
    elif path.name == "metrics.csv":
        metrics_path = path
        summary_path = path.with_name("layer_summary.csv")
        out_dir = path.parent
    else:
        metrics_path = None
        summary_path = path
        out_dir = path.parent

    if metrics_path is not None and metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        summary_cfg = _load_summary_config(out_dir)
        summary = augment_summary_from_metrics(metrics, summary_cfg)
        annotated = annotate_observed_best_curves(
            metrics, gap_epsilon=float(summary_cfg.get("gap_epsilon", 1e-12))
        )
        annotated.to_csv(out_dir / "target_curve_metrics.csv", index=False)
        if args.check_existing and summary_path.exists():
            _assert_existing_summary_matches(pd.read_csv(summary_path), summary)
    else:
        summary = pd.read_csv(summary_path)
        validate_target_estimand_frame(summary, context=str(summary_path))

    out_summary = out_dir / "layer_summary_augmented.csv"
    summary.to_csv(out_summary, index=False)
    target_cols = ["best_rank"] + useful_rank_target_columns(summary.columns)
    fit = _fit_target(summary, target_cols)
    producer_fit_path = out_dir / "layer_rank_fit.csv"
    if args.check_existing and producer_fit_path.exists():
        _assert_existing_fit_matches(pd.read_csv(producer_fit_path), fit)
    out_fit = out_dir / "layer_target_fit_augmented.csv"
    fit.to_csv(out_fit, index=False)
    write_json(out_dir / "target_definition.json", target_definition_document())
    write_json(out_dir / "analysis_plan.json", analysis_plan_document())

    print(f"wrote {out_summary}")
    print(f"wrote {out_fit}")
    if len(fit):
        print("\nTop target fits:")
        print(fit.sort_values("r2_log2_target", ascending=False).head(20).to_string(index=False))

    target = (
        "penalized_rank_lambda_0.15"
        if "penalized_rank_lambda_0.15" in summary.columns
        else target_cols[0]
    )
    if "gradient_effective_rank" in summary.columns:
        figure, axis = plt.subplots()
        axis.scatter(summary["gradient_effective_rank"], summary[target])
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set_xlabel("early-gradient effective rank")
        axis.set_ylabel(target)
        axis.set_title("Layerwise rank target vs early spectrum")
        figure.tight_layout()
        figure_path = out_dir / "layer_target_vs_gradient_effective.png"
        figure.savefig(figure_path, dpi=180)
        plt.close(figure)
        print(f"wrote {figure_path}")

    write_sha256s(out_dir)


if __name__ == "__main__":
    main()
