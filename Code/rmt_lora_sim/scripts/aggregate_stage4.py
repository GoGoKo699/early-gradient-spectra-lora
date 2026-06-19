#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rmt_lora.provenance import (
    git_state,
    runtime_versions,
    sha256_file,
    utc_now_iso,
    verify_sha256s,
    write_json,
    write_sha256s,
)
from rmt_lora.stage4_analysis import (
    ANALYSIS_PLAN_VERSION,
    PREDICTOR_TRANSFORM,
    TARGET_TRANSFORM,
    analysis_plan_document,
    analysis_role,
)
from rmt_lora.targets import (
    TARGET_DENOMINATOR,
    TARGET_ESTIMAND,
    TARGET_ESTIMAND_VERSION,
    TARGET_REFERENCE,
    target_definition_document,
    validate_target_estimand_frame,
)

KEY_TARGETS = [
    "near_best_rank_gap_0.1",
    "near_best_rank_gap_0.2",
    "recovery_rank_0.7",
    "recovery_rank_0.8",
    "penalized_rank_lambda_0.2",
    "penalized_rank_lambda_0.3",
]
KEY_PREDICTORS = [
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

REQUIRED_SOURCE_FILES = (
    "config.yaml",
    "metrics.csv",
    "target_curve_metrics.csv",
    "layer_summary.csv",
    "layer_rank_fit.csv",
    "layer_summary_augmented.csv",
    "layer_target_fit_augmented.csv",
    "target_definition.json",
    "analysis_plan.json",
    "run_metadata.json",
    "SHA256SUMS.txt",
)


def _infer_condition(run_dir: Path) -> str:
    name = run_dir.name
    if "sample" in name:
        return "sample_limited"
    if "hard" in name:
        return "hard_knee"
    return "unknown"


def _infer_seed(run_dir: Path, summary: pd.DataFrame | None = None) -> int | None:
    config = run_dir / "config.yaml"
    if config.exists():
        try:
            import yaml

            data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
            if "seed" in data:
                return int(data["seed"])
        except Exception:
            pass
    import re

    match = re.search(r"s(\d+)", run_dir.name)
    if match:
        return int(match.group(1))
    if summary is not None and "seed" in summary.columns and summary["seed"].notna().any():
        return int(summary["seed"].dropna().iloc[0])
    return None


def _sem(series: pd.Series) -> float:
    values = series.dropna()
    if len(values) <= 1:
        return float("nan")
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def _parse_int_set(text: str | None) -> set[int] | None:
    if text is None:
        return None
    return {int(value.strip()) for value in text.split(",") if value.strip()}


def _parse_str_set(text: str | None) -> set[str] | None:
    if text is None:
        return None
    return {value.strip() for value in text.split(",") if value.strip()}


def _validate_source_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    missing = [name for name in REQUIRED_SOURCE_FILES if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(f"{run_dir} is missing required publication source files: {missing}")

    checksum_errors = verify_sha256s(run_dir, reject_unlisted=True)
    if checksum_errors:
        raise ValueError(
            f"{run_dir} failed checksum validation:\n  " + "\n  ".join(checksum_errors)
        )

    fit = pd.read_csv(run_dir / "layer_target_fit_augmented.csv")
    summary = pd.read_csv(run_dir / "layer_summary_augmented.csv")
    validate_target_estimand_frame(fit, context=str(run_dir / "layer_target_fit_augmented.csv"))
    validate_target_estimand_frame(summary, context=str(run_dir / "layer_summary_augmented.csv"))
    expected_fit_metadata = {
        "analysis_plan_version": ANALYSIS_PLAN_VERSION,
        "target_transform": TARGET_TRANSFORM,
        "predictor_transform": PREDICTOR_TRANSFORM,
    }
    for column, expected in expected_fit_metadata.items():
        if column not in fit.columns or set(fit[column].dropna().astype(str)) != {expected}:
            raise ValueError(
                f"{run_dir}/layer_target_fit_augmented.csv has invalid {column}; "
                f"expected {expected!r}"
            )

    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("target_estimand_version") != TARGET_ESTIMAND_VERSION:
        raise ValueError(
            f"{run_dir}/run_metadata.json has target version "
            f"{metadata.get('target_estimand_version')!r}; expected {TARGET_ESTIMAND_VERSION!r}"
        )
    return fit, summary, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "runs",
        nargs="*",
        help="Stage4 source run directories. Each directory must include raw per-rank metrics and checksums.",
    )
    parser.add_argument(
        "--out",
        default="runs/stage4_aggregate",
        help="Output directory for aggregate tables, figures, manifest, and hashes.",
    )
    parser.add_argument(
        "--expected-seeds",
        help="Optional comma-separated seed set required independently in every condition.",
    )
    parser.add_argument(
        "--expected-conditions",
        help="Optional comma-separated condition set, for example hard_knee,sample_limited.",
    )
    args = parser.parse_args()

    if args.runs:
        run_dirs = [Path(path).resolve() for path in args.runs]
    else:
        run_dirs = sorted(Path("runs").glob("*stage4*"))
        run_dirs = [
            path.resolve()
            for path in run_dirs
            if path.is_dir() and path.name != "stage4_aggregate"
        ]
    if not run_dirs:
        raise SystemExit("No Stage4 source run directories were provided or detected.")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_root = out_dir.parent

    fit_records = []
    summaries = []
    source_rows = []
    for run_dir in run_dirs:
        fit, summary, metadata = _validate_source_run(run_dir)
        condition = _infer_condition(run_dir)
        seed = _infer_seed(run_dir, summary)
        if condition == "unknown":
            raise ValueError(f"cannot infer Stage4 condition from source run {run_dir}")
        if seed is None:
            raise ValueError(f"cannot infer Stage4 seed from source run {run_dir}")

        fit = fit.copy()
        source_run_path = Path(os.path.relpath(run_dir, bundle_root)).as_posix()
        fit["source_run_id"] = run_dir.name
        fit["source_run_path"] = source_run_path
        fit["condition"] = condition
        fit["seed"] = seed
        fit_records.append(fit)

        summary = summary.copy()
        summary["source_run_id"] = run_dir.name
        summary["source_run_path"] = source_run_path
        summary["condition"] = condition
        summary["seed"] = seed
        summaries.append(summary)

        source_rows.append(
            {
                "condition": condition,
                "seed": seed,
                "source_run_id": run_dir.name,
                "source_run_path": source_run_path,
                "n_layers": int(summary["layer"].nunique()),
                "n_rank_rows": int(len(pd.read_csv(run_dir / "metrics.csv"))),
                "config_sha256": sha256_file(run_dir / "config.yaml"),
                "metrics_sha256": sha256_file(run_dir / "metrics.csv"),
                "summary_sha256": sha256_file(run_dir / "layer_summary_augmented.csv"),
                "fit_sha256": sha256_file(run_dir / "layer_target_fit_augmented.csv"),
                "git_revision": (metadata.get("git") or {}).get("git_revision"),
                "git_dirty": (metadata.get("git") or {}).get("git_dirty"),
                "target_estimand": TARGET_ESTIMAND,
                "target_estimand_version": TARGET_ESTIMAND_VERSION,
                "target_reference": TARGET_REFERENCE,
                "target_denominator": TARGET_DENOMINATOR,
                "analysis_plan_version": ANALYSIS_PLAN_VERSION,
                "target_transform": TARGET_TRANSFORM,
                "predictor_transform": PREDICTOR_TRANSFORM,
            }
        )

    sources = pd.DataFrame(source_rows).sort_values(["condition", "seed"])
    duplicate = sources.duplicated(["condition", "seed"], keep=False)
    if duplicate.any():
        raise ValueError(
            "duplicate Stage4 condition/seed source runs:\n"
            + sources.loc[duplicate, ["condition", "seed", "source_run_path"]].to_string(index=False)
        )

    expected_conditions = _parse_str_set(args.expected_conditions)
    observed_conditions = set(sources["condition"])
    if expected_conditions is not None and observed_conditions != expected_conditions:
        raise ValueError(
            f"condition set mismatch: observed {sorted(observed_conditions)}, "
            f"expected {sorted(expected_conditions)}"
        )
    expected_seeds = _parse_int_set(args.expected_seeds)
    if expected_seeds is not None:
        for condition, group in sources.groupby("condition"):
            observed = {int(value) for value in group["seed"]}
            if observed != expected_seeds:
                raise ValueError(
                    f"seed set mismatch for {condition}: observed {sorted(observed)}, "
                    f"expected {sorted(expected_seeds)}"
                )

    all_fit = pd.concat(fit_records, ignore_index=True)
    validate_target_estimand_frame(all_fit, context="combined Stage4 fits")
    all_fit.to_csv(out_dir / "stage4_all_fits.csv", index=False)

    all_summary = pd.concat(summaries, ignore_index=True)
    validate_target_estimand_frame(all_summary, context="combined Stage4 layer summaries")
    all_summary.to_csv(out_dir / "stage4_all_layer_summaries.csv", index=False)
    sources.to_csv(out_dir / "stage4_source_runs.csv", index=False)

    key = all_fit[
        all_fit["target"].isin(KEY_TARGETS)
        & all_fit["predictor"].isin(KEY_PREDICTORS)
    ].copy()
    if key.empty:
        raise ValueError("no publication key targets/predictors were found in Stage4 fits")

    group_columns = [
        "target_estimand",
        "target_estimand_version",
        "target_reference",
        "target_denominator",
        "condition",
        "target",
        "predictor",
    ]
    aggregate = (
        key.groupby(group_columns, as_index=False)
        .agg(
            n_runs=("r2_log2_target", "count"),
            mean_r2=("r2_log2_target", "mean"),
            sem_r2=("r2_log2_target", _sem),
            mean_spearman=("spearman", "mean"),
            sem_spearman=("spearman", _sem),
            mean_rmse_log2=("rmse_log2", "mean"),
        )
        .sort_values(["condition", "target", "mean_r2"], ascending=[True, True, False])
    )
    aggregate["analysis_role"] = [
        analysis_role(str(row.condition), str(row.target), str(row.predictor))
        for row in aggregate.itertuples(index=False)
    ]
    aggregate["analysis_plan_version"] = ANALYSIS_PLAN_VERSION
    aggregate["target_transform"] = TARGET_TRANSFORM
    aggregate["predictor_transform"] = PREDICTOR_TRANSFORM
    aggregate.to_csv(out_dir / "stage4_key_table.csv", index=False)

    best = (
        all_fit.dropna(subset=["r2_log2_target"])
        .sort_values(
            ["condition", "seed", "target", "r2_log2_target"],
            ascending=[True, True, True, False],
        )
        .groupby(["condition", "seed", "target"], as_index=False)
        .head(1)
    )
    best.to_csv(out_dir / "stage4_best_per_run_target.csv", index=False)

    best_aggregate = (
        best.groupby(
            [
                "target_estimand",
                "target_estimand_version",
                "target_reference",
                "target_denominator",
                "condition",
                "target",
                "predictor",
            ],
            as_index=False,
        )
        .agg(
            n_wins=("r2_log2_target", "count"),
            mean_r2=("r2_log2_target", "mean"),
            mean_spearman=("spearman", "mean"),
        )
        .sort_values(
            ["condition", "target", "n_wins", "mean_r2"],
            ascending=[True, True, False, False],
        )
    )
    best_aggregate["analysis_role"] = [
        analysis_role(str(row.condition), str(row.target), str(row.predictor))
        for row in best_aggregate.itertuples(index=False)
    ]
    best_aggregate["analysis_plan_version"] = ANALYSIS_PLAN_VERSION
    best_aggregate.to_csv(out_dir / "stage4_best_predictor_counts.csv", index=False)

    for condition, condition_frame in aggregate.groupby("condition"):
        pivot = condition_frame.pivot_table(
            index="target", columns="predictor", values="mean_spearman"
        )
        pivot = pivot.reindex(
            index=[target for target in KEY_TARGETS if target in pivot.index],
            columns=[predictor for predictor in KEY_PREDICTORS if predictor in pivot.columns],
        )
        if pivot.empty:
            continue
        figure, axis = plt.subplots(figsize=(9, 4.8))
        image = axis.imshow(pivot.values, aspect="auto", vmin=0.0, vmax=1.0)
        axis.set_xticks(np.arange(len(pivot.columns)))
        axis.set_xticklabels(pivot.columns, rotation=35, ha="right")
        axis.set_yticks(np.arange(len(pivot.index)))
        axis.set_yticklabels(pivot.index)
        axis.set_title(f"Stage4 mean Spearman: {condition}")
        for row_index in range(pivot.shape[0]):
            for column_index in range(pivot.shape[1]):
                value = pivot.values[row_index, column_index]
                if np.isfinite(value):
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
        figure.colorbar(image, ax=axis, label="mean Spearman")
        figure.tight_layout()
        figure.savefig(out_dir / f"stage4_spearman_{condition}.png", dpi=200)
        plt.close(figure)

    write_json(out_dir / "target_definition.json", target_definition_document())
    write_json(out_dir / "analysis_plan.json", analysis_plan_document())
    write_json(
        out_dir / "aggregate_manifest.json",
        {
            "schema_version": 1,
            "created_utc": utc_now_iso(),
            "target_estimand": TARGET_ESTIMAND,
            "target_estimand_version": TARGET_ESTIMAND_VERSION,
            "target_reference": TARGET_REFERENCE,
            "target_denominator": TARGET_DENOMINATOR,
            "analysis_plan_version": ANALYSIS_PLAN_VERSION,
            "target_transform": TARGET_TRANSFORM,
            "predictor_transform": PREDICTOR_TRANSFORM,
            "conditions": sorted(observed_conditions),
            "seeds_by_condition": {
                condition: sorted(int(value) for value in group["seed"])
                for condition, group in sources.groupby("condition")
            },
            "n_source_runs": int(len(sources)),
            "source_run_ids": sources["source_run_id"].tolist(),
            "git": git_state(Path(__file__)),
            "runtime": runtime_versions(),
        },
    )
    write_sha256s(out_dir)

    print(f"wrote {out_dir / 'stage4_all_fits.csv'}")
    print(f"wrote {out_dir / 'stage4_all_layer_summaries.csv'}")
    print(f"wrote {out_dir / 'stage4_source_runs.csv'}")
    print(f"wrote {out_dir / 'stage4_key_table.csv'}")
    print(f"wrote {out_dir / 'aggregate_manifest.json'}")
    print("\nTop key-table rows:")
    print(aggregate.sort_values("mean_r2", ascending=False).head(30).to_string(index=False))


if __name__ == "__main__":
    main()
