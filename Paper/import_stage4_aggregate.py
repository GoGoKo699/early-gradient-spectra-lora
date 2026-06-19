#!/usr/bin/env python3
"""Import a validated observed-best Stage4 release into the paper table.

The importer deliberately rejects the superseded oracle-gap aggregate and any
smoke/fast release with fewer than the required independent seed runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
RMT_PROJECT = ROOT / "Code" / "rmt_lora_sim"
if not RMT_PROJECT.exists():
    RMT_PROJECT = ROOT / "rmt_lora_sim"
sys.path.insert(0, str(RMT_PROJECT))

from rmt_lora.provenance import sha256_file, verify_sha256s  # noqa: E402
from rmt_lora.stage4_analysis import (  # noqa: E402
    ANALYSIS_PLAN_VERSION,
    PREDICTOR_TRANSFORM,
    TARGET_TRANSFORM,
)
from rmt_lora.targets import (  # noqa: E402
    TARGET_DENOMINATOR,
    TARGET_ESTIMAND,
    TARGET_ESTIMAND_VERSION,
    TARGET_REFERENCE,
    validate_target_estimand_frame,
)

TARGET_DISPLAY = {
    "near_best_rank_gap_0.1": "near-best 0.1 gap",
    "near_best_rank_gap_0.2": "near-best 0.2 gap",
    "recovery_rank_0.7": "70% recovery",
    "recovery_rank_0.8": "80% recovery",
    "penalized_rank_lambda_0.2": "penalty 0.2",
    "penalized_rank_lambda_0.3": "penalty 0.3",
}
TARGET_ORDER = {name: index for index, name in enumerate(TARGET_DISPLAY)}
EXPECTED_CONDITIONS = {"hard_knee", "sample_limited"}
METADATA_COLUMNS = [
    "target_estimand",
    "target_estimand_version",
    "target_reference",
    "target_denominator",
]
OUT_COLUMNS = [
    *METADATA_COLUMNS,
    "analysis_plan_version",
    "analysis_role",
    "target_transform",
    "predictor_transform",
    "source_release_id",
    "source_aggregate_sha256",
    "condition",
    "target_display",
    "n_runs",
    "mean_r2",
    "sem_r2",
    "mean_spearman",
    "sem_spearman",
    "mean_rmse_log2",
]


def _latest_release_aggregate() -> Path:
    release_root = RMT_PROJECT / "runs" / "stage4_releases"
    pointer = release_root / "LATEST"
    if not pointer.is_file():
        raise FileNotFoundError(
            "no corrected Stage4 release is available. Run "
            "`cd Code/rmt_lora_sim && bash scripts/run_stage4.sh` first. "
            "The bundled results/released/stage4_aggregate directory is superseded."
        )
    release_id = pointer.read_text(encoding="utf-8").strip()
    if not release_id or Path(release_id).name != release_id:
        raise ValueError(f"invalid Stage4 LATEST pointer: {pointer}")
    aggregate = release_root / release_id / "aggregate" / "stage4_key_table.csv"
    if not aggregate.is_file():
        raise FileNotFoundError(f"LATEST points to an incomplete release: {aggregate}")
    return aggregate


def default_aggregate_path() -> Path:
    return _latest_release_aggregate()


def _load_aggregate_manifest(aggregate: Path) -> tuple[dict, str]:
    aggregate_dir = aggregate.resolve().parent
    checksum_errors = verify_sha256s(aggregate_dir, reject_unlisted=True)
    if checksum_errors:
        raise ValueError(
            f"Stage4 aggregate checksum validation failed for {aggregate_dir}:\n  "
            + "\n  ".join(checksum_errors)
        )
    manifest_path = aggregate_dir / "aggregate_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"{aggregate_dir} lacks aggregate_manifest.json; legacy aggregate rejected"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "target_estimand": TARGET_ESTIMAND,
        "target_estimand_version": TARGET_ESTIMAND_VERSION,
        "target_reference": TARGET_REFERENCE,
        "target_denominator": TARGET_DENOMINATOR,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"{manifest_path} has {key}={manifest.get(key)!r}; expected {value!r}"
            )
    expected_analysis = {
        "analysis_plan_version": ANALYSIS_PLAN_VERSION,
        "target_transform": TARGET_TRANSFORM,
        "predictor_transform": PREDICTOR_TRANSFORM,
    }
    for key, value in expected_analysis.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"{manifest_path} has {key}={manifest.get(key)!r}; expected {value!r}"
            )
    release_manifest = aggregate_dir.parent / "release_manifest.json"
    release_id = aggregate_dir.parent.name
    if release_manifest.is_file():
        payload = json.loads(release_manifest.read_text(encoding="utf-8"))
        release_id = str(payload.get("release_id", release_id))
    return manifest, release_id


def import_stage4(
    aggregate: Path,
    out: Path,
    predictor: str,
    *,
    expected_runs: int = 5,
) -> pd.DataFrame:
    aggregate = aggregate.resolve()
    manifest, release_id = _load_aggregate_manifest(aggregate)
    frame = pd.read_csv(aggregate)
    required = {
        *METADATA_COLUMNS,
        "analysis_plan_version",
        "analysis_role",
        "target_transform",
        "predictor_transform",
        "condition",
        "target",
        "predictor",
        "n_runs",
        "mean_r2",
        "sem_r2",
        "mean_spearman",
        "sem_spearman",
        "mean_rmse_log2",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{aggregate} is missing required columns {missing}; "
            "legacy oracle-gap aggregate rejected"
        )
    validate_target_estimand_frame(frame, context=str(aggregate))
    expected_analysis_columns = {
        "analysis_plan_version": ANALYSIS_PLAN_VERSION,
        "target_transform": TARGET_TRANSFORM,
        "predictor_transform": PREDICTOR_TRANSFORM,
    }
    for column, expected in expected_analysis_columns.items():
        if set(frame[column].dropna().astype(str)) != {expected}:
            raise ValueError(f"{aggregate} has invalid or mixed {column}")

    conditions = set(frame["condition"].dropna().astype(str))
    if conditions != EXPECTED_CONDITIONS:
        raise ValueError(
            f"Stage4 condition set is {sorted(conditions)}, expected "
            f"{sorted(EXPECTED_CONDITIONS)}"
        )
    seeds_by_condition = manifest.get("seeds_by_condition") or {}
    for condition in EXPECTED_CONDITIONS:
        seeds = seeds_by_condition.get(condition)
        if not isinstance(seeds, list) or len(set(seeds)) != expected_runs:
            raise ValueError(
                f"{condition} has seed manifest {seeds!r}; publication import "
                f"requires exactly {expected_runs} independent seeds"
            )

    selected = frame[
        (frame["predictor"] == predictor) & frame["target"].isin(TARGET_DISPLAY)
    ].copy()
    if selected.empty:
        available = sorted(frame["predictor"].dropna().unique())
        raise ValueError(
            f"predictor {predictor!r} is unavailable for paper targets. "
            f"Available predictors: {available}"
        )
    expected_rows = len(EXPECTED_CONDITIONS) * len(TARGET_DISPLAY)
    if len(selected) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} condition-target rows for {predictor!r}, "
            f"found {len(selected)}"
        )
    if set(pd.to_numeric(selected["n_runs"], errors="raise")) != {expected_runs}:
        counts = sorted(set(selected["n_runs"]))
        raise ValueError(
            f"paper rows have n_runs={counts}; exactly {expected_runs} are required"
        )

    selected["target_display"] = selected["target"].map(TARGET_DISPLAY)
    selected["target_order"] = selected["target"].map(TARGET_ORDER)
    selected["source_release_id"] = release_id
    selected["source_aggregate_sha256"] = sha256_file(aggregate)
    selected = selected.sort_values(["condition", "target_order"]).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    selected[OUT_COLUMNS].to_csv(out, index=False)
    return selected[OUT_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a validated observed-best Stage4 aggregate into the paper table."
    )
    parser.add_argument(
        "--aggregate",
        type=Path,
        help="Path to stage4_key_table.csv. Default: latest validated local release.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tables/stage4_gradient_effective_summary.csv"),
    )
    parser.add_argument("--predictor", default="gradient_effective_rank")
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=5,
        help="Required independent seed-run count per condition.",
    )
    args = parser.parse_args()
    aggregate = args.aggregate if args.aggregate is not None else default_aggregate_path()
    written = import_stage4(
        aggregate, args.out, args.predictor, expected_runs=args.expected_runs
    )
    print(f"read {aggregate}")
    print(f"wrote {args.out}")
    print(written.to_string(index=False))


if __name__ == "__main__":
    main()
