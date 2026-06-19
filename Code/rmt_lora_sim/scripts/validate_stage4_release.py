#!/usr/bin/env python3
from __future__ import annotations

"""Validate a Stage4 release independently of the runner that created it."""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from rmt_lora.provenance import sha256_file, verify_sha256s
from rmt_lora.stage4_analysis import (
    ANALYSIS_PLAN_VERSION,
    PREDICTOR_TRANSFORM,
    TARGET_TRANSFORM,
)
from rmt_lora.targets import (
    TARGET_DENOMINATOR,
    TARGET_ESTIMAND,
    TARGET_ESTIMAND_VERSION,
    TARGET_REFERENCE,
    validate_target_estimand_frame,
)

REQUIRED_AGGREGATE_FILES = (
    "aggregate_manifest.json",
    "analysis_plan.json",
    "stage4_source_runs.csv",
    "stage4_all_fits.csv",
    "stage4_all_layer_summaries.csv",
    "stage4_key_table.csv",
    "target_definition.json",
    "SHA256SUMS.txt",
)


def _parse_int_set(text: str | None) -> set[int] | None:
    if text is None:
        return None
    return {int(value.strip()) for value in text.split(",") if value.strip()}


def _parse_str_set(text: str | None) -> set[str] | None:
    if text is None:
        return None
    return {value.strip() for value in text.split(",") if value.strip()}


def _require_target_metadata(payload: dict, *, context: str) -> None:
    expected = {
        "target_estimand": TARGET_ESTIMAND,
        "target_estimand_version": TARGET_ESTIMAND_VERSION,
        "target_reference": TARGET_REFERENCE,
        "target_denominator": TARGET_DENOMINATOR,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"{context} has {key}={payload.get(key)!r}; expected {value!r}"
            )


def validate_release(
    release_dir: Path,
    *,
    expected_seeds: set[int] | None = None,
    expected_conditions: set[str] | None = None,
) -> dict:
    release_dir = release_dir.resolve()
    root_errors = verify_sha256s(release_dir, reject_unlisted=True)
    if root_errors:
        raise ValueError("release checksum validation failed:\n  " + "\n  ".join(root_errors))

    manifest_path = release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_target_metadata(manifest, context=str(manifest_path))
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
    if not (release_dir / "analysis_plan.json").is_file():
        raise ValueError("release lacks analysis_plan.json")

    aggregate_dir = release_dir / str(manifest.get("aggregate", "aggregate"))
    missing = [name for name in REQUIRED_AGGREGATE_FILES if not (aggregate_dir / name).is_file()]
    if missing:
        raise ValueError(f"aggregate is missing required files: {missing}")
    aggregate_errors = verify_sha256s(aggregate_dir, reject_unlisted=True)
    if aggregate_errors:
        raise ValueError(
            "aggregate checksum validation failed:\n  " + "\n  ".join(aggregate_errors)
        )

    aggregate_manifest = json.loads(
        (aggregate_dir / "aggregate_manifest.json").read_text(encoding="utf-8")
    )
    _require_target_metadata(aggregate_manifest, context="aggregate_manifest.json")
    for key, value in expected_analysis.items():
        if aggregate_manifest.get(key) != value:
            raise ValueError(
                f"aggregate_manifest.json has {key}={aggregate_manifest.get(key)!r}; "
                f"expected {value!r}"
            )

    manifest_records = manifest.get("source_runs")
    if not isinstance(manifest_records, list):
        raise ValueError("release_manifest.json lacks a source_runs list")
    manifest_by_pair: dict[tuple[str, int], dict] = {}
    for record in manifest_records:
        if not isinstance(record, dict):
            raise ValueError("release_manifest source_runs entries must be mappings")
        pair = (str(record.get("condition")), int(record.get("seed")))
        if pair in manifest_by_pair:
            raise ValueError(f"duplicate release-manifest condition/seed pair: {pair}")
        manifest_by_pair[pair] = record

    source_table = pd.read_csv(aggregate_dir / "stage4_source_runs.csv")
    required_source_columns = {
        "condition",
        "seed",
        "source_run_path",
        "config_sha256",
        "metrics_sha256",
        "summary_sha256",
        "fit_sha256",
        "target_estimand",
        "target_estimand_version",
        "target_reference",
        "target_denominator",
        "analysis_plan_version",
        "target_transform",
        "predictor_transform",
    }
    missing_columns = sorted(required_source_columns - set(source_table.columns))
    if missing_columns:
        raise ValueError(f"stage4_source_runs.csv is missing columns: {missing_columns}")
    validate_target_estimand_frame(source_table, context="stage4_source_runs.csv")
    for column, expected in expected_analysis.items():
        if set(source_table[column].dropna().astype(str)) != {expected}:
            raise ValueError(f"stage4_source_runs.csv has invalid {column}")

    observed_pairs: set[tuple[str, int]] = set()
    for row in source_table.to_dict(orient="records"):
        relative = Path(str(row["source_run_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"non-portable source run path: {relative}")
        run_dir = (release_dir / relative).resolve()
        try:
            run_dir.relative_to(release_dir)
        except ValueError as error:
            raise ValueError(f"source run escapes release: {relative}") from error
        errors = verify_sha256s(run_dir, reject_unlisted=True)
        if errors:
            raise ValueError(
                f"source run checksum validation failed for {relative}:\n  "
                + "\n  ".join(errors)
            )
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
        seed = int(config["seed"])
        condition = str(row["condition"])
        if seed != int(row["seed"]):
            raise ValueError(f"seed mismatch for {relative}: config {seed}, table {row['seed']}")
        pair = (condition, seed)
        if pair in observed_pairs:
            raise ValueError(f"duplicate condition/seed pair: {pair}")
        observed_pairs.add(pair)
        if pair not in manifest_by_pair:
            raise ValueError(f"aggregate source {pair} is absent from release_manifest.json")
        record = manifest_by_pair[pair]
        if str(record.get("source_run")) != relative.as_posix():
            raise ValueError(f"source path mismatch for {pair}")
        config_copy = release_dir / str(record.get("config"))
        if not config_copy.is_file():
            raise ValueError(f"missing generated config copy for {pair}: {config_copy}")
        if config_copy.read_bytes() != (run_dir / "config.yaml").read_bytes():
            raise ValueError(f"generated config differs from materialized run config for {pair}")

        metrics = pd.read_csv(run_dir / "metrics.csv")
        required_metric_columns = {"layer", "rank", "base_val_loss", "final_val_loss"}
        missing_metric_columns = sorted(required_metric_columns - set(metrics.columns))
        if missing_metric_columns:
            raise ValueError(f"{relative}/metrics.csv lacks {missing_metric_columns}")
        if metrics.duplicated(["layer", "rank"]).any():
            raise ValueError(f"{relative}/metrics.csv has duplicate layer/rank rows")
        configured_ranks = [int(value) for value in config["layer_sweep"]["ranks"]]
        configured_layers = int(config["layer_sweep"]["n_layers"])
        expected_pairs = {
            (layer, rank)
            for layer in range(configured_layers)
            for rank in configured_ranks
        }
        observed_metric_pairs = {
            (int(layer), int(rank))
            for layer, rank in metrics[["layer", "rank"]].itertuples(index=False, name=None)
        }
        if observed_metric_pairs != expected_pairs:
            missing_pairs = sorted(expected_pairs - observed_metric_pairs)[:10]
            extra_pairs = sorted(observed_metric_pairs - expected_pairs)[:10]
            raise ValueError(
                f"{relative}/metrics.csv is incomplete: missing {missing_pairs}, extra {extra_pairs}"
            )
        target_curve = pd.read_csv(run_dir / "target_curve_metrics.csv")
        if len(target_curve) != len(metrics):
            raise ValueError(f"{relative}/target_curve_metrics.csv row count differs from metrics.csv")
        validate_target_estimand_frame(
            target_curve, context=f"{relative}/target_curve_metrics.csv"
        )

        hashes = {
            "config_sha256": sha256_file(run_dir / "config.yaml"),
            "metrics_sha256": sha256_file(run_dir / "metrics.csv"),
            "summary_sha256": sha256_file(run_dir / "layer_summary_augmented.csv"),
            "fit_sha256": sha256_file(run_dir / "layer_target_fit_augmented.csv"),
        }
        for key, actual in hashes.items():
            if str(row[key]) != actual:
                raise ValueError(f"aggregate source hash mismatch for {relative}/{key}")

        validate_target_estimand_frame(
            pd.read_csv(run_dir / "layer_summary_augmented.csv"),
            context=f"{relative}/layer_summary_augmented.csv",
        )
        fit_frame = pd.read_csv(run_dir / "layer_target_fit_augmented.csv")
        validate_target_estimand_frame(
            fit_frame,
            context=f"{relative}/layer_target_fit_augmented.csv",
        )
        for column, expected in expected_analysis.items():
            if column not in fit_frame.columns or set(fit_frame[column].dropna().astype(str)) != {expected}:
                raise ValueError(f"{relative}/layer_target_fit_augmented.csv has invalid {column}")

    if set(manifest_by_pair) != observed_pairs:
        missing = sorted(set(manifest_by_pair) - observed_pairs)
        extra = sorted(observed_pairs - set(manifest_by_pair))
        raise ValueError(
            f"release manifest/source table mismatch: manifest-only {missing}, table-only {extra}"
        )

    observed_conditions = {condition for condition, _ in observed_pairs}
    if expected_conditions is not None and observed_conditions != expected_conditions:
        raise ValueError(
            f"condition mismatch: observed {sorted(observed_conditions)}, "
            f"expected {sorted(expected_conditions)}"
        )
    if expected_seeds is not None:
        for condition in observed_conditions:
            observed = {seed for cond, seed in observed_pairs if cond == condition}
            if observed != expected_seeds:
                raise ValueError(
                    f"seed mismatch for {condition}: observed {sorted(observed)}, "
                    f"expected {sorted(expected_seeds)}"
                )

    for filename in (
        "stage4_all_fits.csv",
        "stage4_all_layer_summaries.csv",
        "stage4_key_table.csv",
    ):
        validate_target_estimand_frame(
            pd.read_csv(aggregate_dir / filename), context=f"aggregate/{filename}"
        )

    if int(manifest.get("n_source_runs", -1)) != len(observed_pairs):
        raise ValueError(
            f"release manifest lists {manifest.get('n_source_runs')} runs; "
            f"validated {len(observed_pairs)}"
        )
    return {
        "release_id": manifest.get("release_id"),
        "conditions": sorted(observed_conditions),
        "seeds": sorted({seed for _, seed in observed_pairs}),
        "n_source_runs": len(observed_pairs),
        "target_estimand_version": TARGET_ESTIMAND_VERSION,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release", type=Path)
    parser.add_argument("--expected-seeds")
    parser.add_argument("--expected-conditions")
    args = parser.parse_args()
    result = validate_release(
        args.release,
        expected_seeds=_parse_int_set(args.expected_seeds),
        expected_conditions=_parse_str_set(args.expected_conditions),
    )
    print("Stage4 release validation: PASS")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
