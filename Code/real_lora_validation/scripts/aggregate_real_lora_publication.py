#!/usr/bin/env python3
"""Aggregate a frozen 8+3 real-model LoRA plan at the seed-run level."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in [ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from publication_protocol import (  # noqa: E402
    BOOTSTRAP_SEED_SCHEME,
    PUBLICATION_AGGREGATE_VERSION,
    holm_adjust,
    sha256_file,
    stable_seed,
    summarize_paired_values,
    validate_input_provenance_against_manifest,
    validate_publication_plan,
)
from real_protocol import canonical_json_sha256  # noqa: E402
from validate_real_lora_run import validate_run  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, float_format="%.17g")


def write_checksums(directory: Path) -> None:
    target = directory / "AGGREGATE_SHA256SUMS.txt"
    rows = [
        f"{sha256_file(path)}  {path.relative_to(directory).as_posix()}"
        for path in sorted(
            item for item in directory.rglob("*") if item.is_file() and item != target
        )
    ]
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _same_value(actual: object, expected: object) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-15)
        except (TypeError, ValueError):
            return False
    return actual == expected


def expected_units(validated_plan: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (suite["suite_id"], int(seed)): suite
        for suite in validated_plan["suites"]
        for seed in suite["seeds"]
    }


def validate_source_binding(
    run_dir: Path,
    *,
    validated_plan: dict[str, Any],
    release_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    summary = validate_run(run_dir, expected_kind=validated_plan["release_kind"])
    config = read_json(run_dir / "config.json")
    require(
        config.get("publication_plan_version")
        == validated_plan["plan"]["plan_version"],
        f"source run {run_dir.name} has stale publication_plan_version",
    )
    require(
        config.get("publication_plan_sha256") == validated_plan["sha256"],
        f"source run {run_dir.name} is not bound to this plan",
    )
    bound_release = str(config.get("publication_release_id", ""))
    require(bound_release, f"source run {run_dir.name} lacks publication_release_id")
    source_commit = str(config.get("source_git_commit", ""))
    require(
        len(source_commit) == 40
        and all(ch in "0123456789abcdef" for ch in source_commit),
        f"source run {run_dir.name} lacks a valid source Git commit",
    )
    require(
        str(config.get("source_git_status_porcelain", "")) == "",
        f"source run {run_dir.name} was generated from a dirty worktree",
    )
    if release_id is not None:
        require(
            bound_release == release_id,
            f"source run {run_dir.name} release id {bound_release!r} != {release_id!r}",
        )
    suite_id = str(config.get("suite_id", ""))
    seed = int(config.get("seed", -1))
    unit = (suite_id, seed)
    units = expected_units(validated_plan)
    require(unit in units, f"source run {run_dir.name} is not a planned unit: {unit}")
    suite = units[unit]
    require(config.get("suite_role") == suite["role"], f"suite role mismatch for {unit}")
    require(
        config.get("target_suffixes") == suite["target_suffixes"],
        f"target_suffixes mismatch for {unit}",
    )
    require(int(config.get("max_targets", -1)) == int(suite["max_targets"]), f"max_targets mismatch for {unit}")
    require(str(config.get("run_id")) == run_dir.name, f"run_id/path mismatch for {run_dir}")
    require(run_dir.name == f"{bound_release}_{suite_id}_seed{seed}", f"run id is not canonically namespaced: {run_dir.name}")

    for key, expected in validated_plan["run"].items():
        if key == "strategies":
            require(list(config.get("requested_strategies", [])) == list(expected), f"strategy list mismatch for {unit}")
            continue
        require(_same_value(config.get(key), expected), f"run parameter {key} mismatch for {unit}: {config.get(key)!r} != {expected!r}")
    model = validated_plan["model"]
    dataset = validated_plan["dataset"]
    require(config.get("model") == model["argument"], f"model mismatch for {unit}")
    require(bool(config.get("local_files_only")) == bool(model["local_files_only"]), f"local_files_only mismatch for {unit}")
    require(config.get("dataset_mode") == dataset["mode"], f"dataset mode mismatch for {unit}")
    if dataset["mode"] == "local":
        require(config.get("train_text_file") == dataset["train_text_file"], f"train input mismatch for {unit}")
        require(config.get("val_text_file") == dataset["val_text_file"], f"validation input mismatch for {unit}")

    metadata = config.get("strategy_metadata")
    require(isinstance(metadata, dict), f"strategy metadata missing for {unit}")
    references = [
        name
        for name, row in metadata.items()
        if isinstance(row, dict) and row.get("requested_strategy") == "uniform"
    ]
    require(len(references) == 1, f"expected one uniform reference for {unit}; found {references}")
    reference = references[0]
    return summary, config, reference


def aggregate(
    *,
    run_dirs: list[Path],
    plan_path: Path,
    out_dir: Path,
    release_id: str | None = None,
) -> Path:
    validated_plan = validate_publication_plan(plan_path, ROOT)
    require(run_dirs, "at least one source run is required")
    resolved = [Path(path).resolve() for path in run_dirs]
    require(len(resolved) == len(set(resolved)), "duplicate source run path")
    expected = expected_units(validated_plan)
    require(len(resolved) == len(expected), f"expected {len(expected)} source runs; got {len(resolved)}")

    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"aggregate output already exists: {out_dir}")
    out_dir.mkdir(parents=True)

    all_results: list[pd.DataFrame] = []
    all_allocations: list[pd.DataFrame] = []
    paired_rows: list[dict[str, Any]] = []
    source_entries: list[dict[str, Any]] = []
    seen_units: set[tuple[str, int]] = set()
    bound_release_ids: set[str] = set()
    source_git_commits: set[str] = set()
    input_provenance_hashes: set[str] = set()
    environment_hashes: set[str] = set()
    input_manifest = (
        read_json(validated_plan["input_manifest_path"])
        if validated_plan["input_manifest_path"] is not None
        else None
    )

    for run_dir in sorted(resolved, key=lambda path: path.name):
        summary, config, reference = validate_source_binding(
            run_dir,
            validated_plan=validated_plan,
            release_id=release_id,
        )
        suite_id = str(config["suite_id"])
        suite_role = str(config["suite_role"])
        seed = int(config["seed"])
        unit = (suite_id, seed)
        require(unit not in seen_units, f"duplicate independent unit: {unit}")
        seen_units.add(unit)
        bound_release_ids.add(str(config["publication_release_id"]))
        source_git_commits.add(str(config["source_git_commit"]))
        input_provenance = read_json(run_dir / "input_provenance.json")
        if input_manifest is not None:
            validate_input_provenance_against_manifest(
                input_provenance, input_manifest
            )

        results = pd.read_csv(run_dir / "results.csv")
        allocations = pd.read_csv(run_dir / "allocations.csv")
        result_by = results.set_index("strategy", drop=False)
        require(reference in result_by.index, f"uniform reference missing for {unit}")
        reference_row = result_by.loc[reference]
        output_strategies = list(config["output_strategies"])
        require(set(results["strategy"]) == set(output_strategies), f"result strategy set mismatch for {unit}")

        result_copy = results.copy()
        result_copy.insert(0, "run_id", run_dir.name)
        result_copy.insert(1, "suite_id", suite_id)
        result_copy.insert(2, "suite_role", suite_role)
        result_copy.insert(3, "seed", seed)
        result_copy.insert(4, "reference_strategy", reference)
        all_results.append(result_copy)

        allocation_copy = allocations.copy()
        allocation_copy.insert(0, "run_id", run_dir.name)
        allocation_copy.insert(1, "suite_id", suite_id)
        allocation_copy.insert(2, "suite_role", suite_role)
        allocation_copy.insert(3, "seed", seed)
        all_allocations.append(allocation_copy)

        for candidate in output_strategies:
            if candidate in {reference, "uniform_identity_control"}:
                continue
            candidate_row = result_by.loc[candidate]
            paired_rows.append(
                {
                    "run_id": run_dir.name,
                    "suite_id": suite_id,
                    "suite_role": suite_role,
                    "seed": seed,
                    "candidate_strategy": candidate,
                    "reference_strategy": reference,
                    "candidate_minus_reference_final_val_loss": float(candidate_row["final_val_loss"]) - float(reference_row["final_val_loss"]),
                    "candidate_minus_reference_val_loss_delta": float(candidate_row["val_loss_delta"]) - float(reference_row["val_loss_delta"]),
                    "candidate_minus_reference_perplexity": float(candidate_row["perplexity"]) - float(reference_row["perplexity"]),
                    "candidate_trainable_params": int(candidate_row["trainable_params"]),
                    "reference_trainable_params": int(reference_row["trainable_params"]),
                    "parameter_difference": int(candidate_row["trainable_params"]) - int(reference_row["trainable_params"]),
                    "candidate_assignment_sha256": str(candidate_row["assignment_sha256"]),
                    "reference_assignment_sha256": str(reference_row["assignment_sha256"]),
                    "assignments_identical": str(candidate_row["assignment_sha256"]) == str(reference_row["assignment_sha256"]),
                    "candidate_adapter_parameter_delta_l2": float(candidate_row["adapter_parameter_delta_l2"]),
                    "reference_adapter_parameter_delta_l2": float(reference_row["adapter_parameter_delta_l2"]),
                    "candidate_effective_update_frobenius_l2": float(candidate_row["effective_update_frobenius_l2"]),
                    "reference_effective_update_frobenius_l2": float(reference_row["effective_update_frobenius_l2"]),
                    "candidate_final_adapter_state_sha256": str(candidate_row["final_adapter_state_sha256"]),
                    "reference_final_adapter_state_sha256": str(reference_row["final_adapter_state_sha256"]),
                }
            )

        input_hash = sha256_file(run_dir / "input_provenance.json")
        environment_hash = sha256_file(run_dir / "environment.json")
        input_provenance_hashes.add(input_hash)
        environment_hashes.add(environment_hash)
        source_entries.append(
            {
                "run_id": run_dir.name,
                "suite_id": suite_id,
                "suite_role": suite_role,
                "seed": seed,
                "path": run_dir.name,
                "config_sha256": sha256_file(run_dir / "config.json"),
                "run_checksums_sha256": sha256_file(run_dir / "RUN_SHA256SUMS.txt"),
                "input_provenance_sha256": input_hash,
                "environment_sha256": environment_hash,
                "source_git_commit": str(config["source_git_commit"]),
                "n_targets": int(summary["n_targets"]),
                "n_strategies": int(summary["n_strategies"]),
                "target_parameter_budget": int(summary["target_parameter_budget"]),
            }
        )

    require(seen_units == set(expected), f"source units do not match plan: missing={sorted(set(expected)-seen_units)}, extra={sorted(seen_units-set(expected))}")
    require(len(bound_release_ids) == 1, f"source runs disagree on release id: {sorted(bound_release_ids)}")
    bound_release = next(iter(bound_release_ids))
    if release_id is not None:
        require(bound_release == release_id, "aggregate release id mismatch")
    require(len(source_git_commits) == 1, "source runs do not share one Git commit")
    require(len(input_provenance_hashes) == 1, "source runs do not share identical input provenance")
    require(len(environment_hashes) == 1, "source runs do not share identical environment provenance")

    results_frame = pd.concat(all_results, ignore_index=True)
    allocations_frame = pd.concat(all_allocations, ignore_index=True)
    paired = pd.DataFrame(paired_rows).sort_values(["suite_id", "candidate_strategy", "seed"])
    require((paired["parameter_difference"] == 0).all(), "a paired comparison is not exact-cost matched")
    write_csv(out_dir / "all_results.csv", results_frame)
    write_csv(out_dir / "all_allocations.csv", allocations_frame)
    write_csv(out_dir / "paired_deltas.csv", paired)

    analysis = validated_plan["analysis"]
    confidence = float(analysis["confidence"])
    n_resamples = int(analysis["bootstrap_resamples"])
    summary_rows: list[dict[str, Any]] = []
    for (suite_id, candidate), group in paired.groupby(["suite_id", "candidate_strategy"], sort=True):
        values = group["candidate_minus_reference_final_val_loss"].to_numpy(dtype=float)
        seed = stable_seed(validated_plan["sha256"], f"suite={suite_id}|candidate={candidate}")
        stats = summarize_paired_values(
            values,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=seed,
        )
        summary_rows.append(
            {
                "suite_id": suite_id,
                "suite_role": str(group["suite_role"].iloc[0]),
                "candidate_strategy": candidate,
                "reference_strategy": str(group["reference_strategy"].iloc[0]),
                **stats,
                "exact_test_resolution_sufficient": bool(
                    float(stats["attainable_min_two_sided_p"])
                    <= float(analysis["two_sided_alpha"])
                ),
                "distinct_assignment_runs": int((~group["assignments_identical"].astype(bool)).sum()),
                "identical_assignment_runs": int(group["assignments_identical"].astype(bool).sum()),
                "confidence": confidence,
                "bootstrap_resamples": n_resamples,
                "bootstrap_seed": seed,
                "bootstrap_seed_scheme": BOOTSTRAP_SEED_SCHEME,
            }
        )
    summary_frame = pd.DataFrame(summary_rows).sort_values(["suite_id", "candidate_strategy"]).reset_index(drop=True)
    adjusted_parts: list[pd.DataFrame] = []
    for _, group in summary_frame.groupby("suite_id", sort=True):
        group = group.copy()
        group["holm_p_within_suite"] = holm_adjust(group["exact_sign_flip_p_two_sided"].tolist())
        adjusted_parts.append(group)
    summary_frame = pd.concat(adjusted_parts, ignore_index=True).sort_values(["suite_id", "candidate_strategy"])
    write_csv(out_dir / "analysis_by_suite_strategy.csv", summary_frame)

    primary_suite = str(analysis["primary_suite"])
    primary_candidate = str(analysis["primary_candidate_strategy"])
    primary_deltas = paired[
        (paired["suite_id"] == primary_suite)
        & (paired["candidate_strategy"] == primary_candidate)
    ].copy()
    expected_primary_n = len(next(row for row in validated_plan["suites"] if row["suite_id"] == primary_suite)["seeds"])
    require(len(primary_deltas) == expected_primary_n, f"primary seed row count {len(primary_deltas)} != {expected_primary_n}")
    write_csv(out_dir / "primary_seed_deltas.csv", primary_deltas)
    primary_summary = summary_frame[
        (summary_frame["suite_id"] == primary_suite)
        & (summary_frame["candidate_strategy"] == primary_candidate)
    ].copy()
    require(len(primary_summary) == 1, "primary analysis must have exactly one row")
    primary_summary.insert(0, "analysis_scope", "prespecified_primary_suite")
    primary_summary.insert(1, "primary_estimand", str(analysis["primary_estimand"]))
    primary_summary.insert(2, "delta_definition", str(analysis["delta_definition"]))
    primary_summary.insert(3, "lower_is_better", bool(analysis["lower_is_better"]))
    primary_summary.insert(4, "two_sided_alpha", float(analysis["two_sided_alpha"]))
    primary_summary.insert(
        5,
        "minimum_nonzero_primary_deltas_for_resolution",
        int(analysis["minimum_nonzero_primary_deltas_for_resolution"]),
    )
    primary_summary.insert(
        6, "planned_tie_tolerance", int(analysis["planned_tie_tolerance"])
    )
    write_csv(out_dir / "primary_analysis.csv", primary_summary)

    copied_plan = out_dir / "analysis_plan.json"
    shutil.copy2(validated_plan["path"], copied_plan)
    input_manifest_file = None
    input_manifest_sha = None
    if validated_plan["input_manifest_path"] is not None:
        input_manifest_file = "input_manifest.json"
        shutil.copy2(validated_plan["input_manifest_path"], out_dir / input_manifest_file)
        input_manifest_sha = sha256_file(out_dir / input_manifest_file)

    manifest = {
        "aggregate_schema_version": PUBLICATION_AGGREGATE_VERSION,
        "release_id": bound_release,
        "release_kind": validated_plan["release_kind"],
        "protocol_version": validated_plan["plan"]["protocol_version"],
        "analysis_plan_file": copied_plan.name,
        "analysis_plan_version": validated_plan["plan"]["plan_version"],
        "analysis_plan_sha256": sha256_file(copied_plan),
        "input_manifest_file": input_manifest_file,
        "input_manifest_sha256": input_manifest_sha,
        "n_source_runs": len(source_entries),
        "independent_unit": analysis["independent_unit"],
        "source_runs": sorted(source_entries, key=lambda row: (row["suite_id"], row["seed"])),
        "source_run_set_sha256": canonical_json_sha256(
            [
                {
                    key: row[key]
                    for key in [
                        "run_id",
                        "suite_id",
                        "seed",
                        "config_sha256",
                        "run_checksums_sha256",
                        "source_git_commit",
                    ]
                }
                for row in sorted(source_entries, key=lambda value: (value["suite_id"], value["seed"]))
            ]
        ),
        "primary_suite": primary_suite,
        "primary_candidate_strategy": primary_candidate,
        "primary_metric": analysis["primary_metric"],
        "delta_definition": analysis["delta_definition"],
        "confidence": confidence,
        "bootstrap_resamples": n_resamples,
        "bootstrap_seed_scheme": BOOTSTRAP_SEED_SCHEME,
        "exact_sign_flip_test": analysis["exact_sign_flip_test"],
        "secondary_multiplicity": analysis["secondary_multiplicity"],
        "minimum_nonzero_primary_deltas_for_resolution": int(
            analysis["minimum_nonzero_primary_deltas_for_resolution"]
        ),
        "planned_tie_tolerance": int(analysis["planned_tie_tolerance"]),
        "source_git_commit": next(iter(source_git_commits)),
        "input_provenance_sha256": next(iter(input_provenance_hashes)),
        "environment_sha256": next(iter(environment_hashes)),
    }
    write_json(out_dir / "manifest.json", manifest)
    write_checksums(out_dir)
    print("Real LoRA publication aggregate: PASS")
    print("release_id:", bound_release)
    print("release_kind:", validated_plan["release_kind"])
    print("n_source_runs:", len(source_entries))
    print("primary_suite:", primary_suite)
    print("primary_candidate:", primary_candidate)
    print("aggregate_dir:", out_dir)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, dest="runs")
    parser.add_argument("--analysis-plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--release-id", default=None)
    args = parser.parse_args()
    plan_path = Path(args.analysis_plan)
    if not plan_path.is_absolute():
        plan_path = ROOT / plan_path
    aggregate(
        run_dirs=[Path(value) for value in args.runs],
        plan_path=plan_path,
        out_dir=Path(args.out),
        release_id=args.release_id,
    )


if __name__ == "__main__":
    main()
