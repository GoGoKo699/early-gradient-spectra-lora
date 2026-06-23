#!/usr/bin/env python3
"""Validate a multi-source real-model LoRA smoke/publication release."""
from __future__ import annotations

import argparse
import json
import math
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in [ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from publication_protocol import (  # noqa: E402
    BOOTSTRAP_SEED_SCHEME,
    PUBLICATION_AGGREGATE_VERSION,
    PUBLICATION_CODE_SNAPSHOT_FILES,
    PUBLICATION_RELEASE_VERSION,
    holm_adjust,
    safe_package_path,
    sha256_file,
    stable_seed,
    summarize_paired_values,
    validate_input_provenance_against_manifest,
    validate_publication_plan,
)
from real_protocol import canonical_json_sha256  # noqa: E402
from aggregate_real_lora_publication import validate_source_binding  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def verify_checksum_file(directory: Path, filename: str) -> int:
    manifest = directory / filename
    require(manifest.is_file(), f"missing {filename}")
    recorded: set[str] = set()
    count = 0
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        require(len(parts) == 2, f"invalid {filename} line {line_number}")
        expected, relative = parts
        require(relative not in recorded, f"duplicate checksum path {relative}")
        recorded.add(relative)
        path = directory / relative
        require(path.is_file(), f"checksummed file is missing: {relative}")
        require(sha256_file(path) == expected, f"checksum mismatch for {relative}")
        count += 1
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path != manifest
    }
    require(
        recorded == actual,
        f"{filename} coverage mismatch: missing={sorted(actual-recorded)}, stale={sorted(recorded-actual)}",
    )
    return count


def _float_close(actual: object, expected: float, label: str, tolerance: float = 5e-12) -> None:
    value = float(actual)
    if math.isnan(expected):
        require(math.isnan(value), f"{label}={value}; expected NaN")
    else:
        require(
            math.isclose(value, float(expected), rel_tol=0.0, abs_tol=tolerance),
            f"{label}={value}; expected {expected}",
        )


def reconstruct_pairs(
    release_dir: Path,
    source_entries: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in source_entries:
        run_id = str(entry["run_id"])
        run_dir = release_dir / str(entry["path"])
        config = read_json(run_dir / "config.json")
        results = pd.read_csv(run_dir / "results.csv").set_index("strategy", drop=False)
        metadata = config.get("strategy_metadata")
        require(isinstance(metadata, dict), f"strategy metadata missing for {run_id}")
        references = [
            name
            for name, row in metadata.items()
            if isinstance(row, dict) and row.get("requested_strategy") == "uniform"
        ]
        require(len(references) == 1, f"expected one uniform reference for {run_id}")
        reference = references[0]
        require(reference in results.index, f"uniform reference missing for {run_id}")
        reference_row = results.loc[reference]
        for candidate in list(config["output_strategies"]):
            if candidate in {reference, "uniform_identity_control"}:
                continue
            require(candidate in results.index, f"candidate {candidate} missing for {run_id}")
            candidate_row = results.loc[candidate]
            rows.append(
                {
                    "run_id": run_id,
                    "suite_id": str(config["suite_id"]),
                    "suite_role": str(config["suite_role"]),
                    "seed": int(config["seed"]),
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
    return pd.DataFrame(rows).sort_values(["suite_id", "candidate_strategy", "seed"]).reset_index(drop=True)


def compare_pair_frames(recomputed: pd.DataFrame, released: pd.DataFrame) -> None:
    released = released.sort_values(["suite_id", "candidate_strategy", "seed"]).reset_index(drop=True)
    require(len(recomputed) == len(released), "paired_deltas row count mismatch")
    key_columns = [
        "run_id",
        "suite_id",
        "suite_role",
        "seed",
        "candidate_strategy",
        "reference_strategy",
        "candidate_trainable_params",
        "reference_trainable_params",
        "parameter_difference",
        "candidate_assignment_sha256",
        "reference_assignment_sha256",
        "candidate_final_adapter_state_sha256",
        "reference_final_adapter_state_sha256",
    ]
    for column in key_columns:
        require(
            recomputed[column].astype(str).tolist() == released[column].astype(str).tolist(),
            f"paired_deltas {column} mismatch",
        )
    bool_actual = recomputed["assignments_identical"].astype(bool).tolist()
    bool_released = released["assignments_identical"].map(
        lambda value: str(value).lower() in {"true", "1"}
    ).tolist()
    require(bool_actual == bool_released, "paired_deltas assignments_identical mismatch")
    numeric = [
        "candidate_minus_reference_final_val_loss",
        "candidate_minus_reference_val_loss_delta",
        "candidate_minus_reference_perplexity",
        "candidate_adapter_parameter_delta_l2",
        "reference_adapter_parameter_delta_l2",
        "candidate_effective_update_frobenius_l2",
        "reference_effective_update_frobenius_l2",
    ]
    for column in numeric:
        for index, expected in enumerate(recomputed[column].astype(float)):
            _float_close(released.loc[index, column], expected, f"paired_deltas[{index}].{column}")


def recompute_summary(
    pairs: pd.DataFrame,
    validated_plan: dict[str, Any],
) -> pd.DataFrame:
    analysis = validated_plan["analysis"]
    confidence = float(analysis["confidence"])
    n_resamples = int(analysis["bootstrap_resamples"])
    rows: list[dict[str, Any]] = []
    for (suite_id, candidate), group in pairs.groupby(["suite_id", "candidate_strategy"], sort=True):
        values = group["candidate_minus_reference_final_val_loss"].to_numpy(dtype=float)
        seed = stable_seed(validated_plan["sha256"], f"suite={suite_id}|candidate={candidate}")
        stats = summarize_paired_values(
            values,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=seed,
        )
        rows.append(
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
    frame = pd.DataFrame(rows).sort_values(["suite_id", "candidate_strategy"]).reset_index(drop=True)
    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby("suite_id", sort=True):
        group = group.copy()
        group["holm_p_within_suite"] = holm_adjust(group["exact_sign_flip_p_two_sided"].tolist())
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True).sort_values(["suite_id", "candidate_strategy"]).reset_index(drop=True)


def compare_summary_frames(recomputed: pd.DataFrame, released: pd.DataFrame) -> None:
    released = released.sort_values(["suite_id", "candidate_strategy"]).reset_index(drop=True)
    require(len(recomputed) == len(released), "analysis_by_suite_strategy row count mismatch")
    string_columns = [
        "suite_id",
        "suite_role",
        "candidate_strategy",
        "reference_strategy",
        "bootstrap_seed_scheme",
    ]
    integer_columns = [
        "n_independent_runs",
        "n_nonzero_deltas",
        "wins",
        "ties",
        "losses",
        "distinct_assignment_runs",
        "identical_assignment_runs",
        "bootstrap_resamples",
        "bootstrap_seed",
    ]
    for column in string_columns:
        require(recomputed[column].astype(str).tolist() == released[column].astype(str).tolist(), f"summary {column} mismatch")
    for column in integer_columns:
        require(recomputed[column].astype(int).tolist() == released[column].astype(int).tolist(), f"summary {column} mismatch")
    require(
        recomputed["exact_test_resolution_sufficient"].astype(bool).tolist()
        == released["exact_test_resolution_sufficient"].map(
            lambda value: str(value).lower() in {"true", "1"}
        ).tolist(),
        "summary exact_test_resolution_sufficient mismatch",
    )
    numeric_columns = [
        "mean_loss_delta",
        "median_loss_delta",
        "sd_loss_delta",
        "sem_loss_delta",
        "t_ci_low",
        "t_ci_high",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "exact_sign_flip_p_two_sided",
        "planned_no_tie_min_two_sided_p",
        "attainable_min_two_sided_p",
        "confidence",
        "holm_p_within_suite",
    ]
    for column in numeric_columns:
        for index, expected in enumerate(recomputed[column].astype(float)):
            _float_close(released.loc[index, column], expected, f"summary[{index}].{column}")


def compare_complete_frames(
    recomputed: pd.DataFrame,
    released: pd.DataFrame,
    *,
    sort_columns: list[str],
    label: str,
) -> None:
    require(
        list(recomputed.columns) == list(released.columns),
        f"{label} columns differ: recomputed={list(recomputed.columns)}, "
        f"released={list(released.columns)}",
    )
    recomputed = recomputed.sort_values(sort_columns).reset_index(drop=True)
    released = released.sort_values(sort_columns).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            recomputed,
            released,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=5e-12,
            check_like=False,
        )
    except AssertionError as exc:
        raise ValueError(f"{label} does not match source runs: {exc}") from exc


def validate_release(release_dir: Path, expected_kind: str | None = None) -> dict[str, Any]:
    release_dir = Path(release_dir).resolve()
    require(release_dir.is_dir(), f"release directory does not exist: {release_dir}")
    checksum_count = verify_checksum_file(release_dir, "SHA256SUMS.txt")
    manifest = read_json(release_dir / "release_manifest.json")
    require(
        manifest.get("schema_version") == PUBLICATION_RELEASE_VERSION,
        "publication release schema mismatch",
    )
    kind = str(manifest.get("release_kind", ""))
    require(kind in {"smoke", "publication"}, "invalid release kind")
    if expected_kind is not None:
        require(kind == expected_kind, f"release kind {kind!r} != {expected_kind!r}")
    release_id = str(manifest.get("release_id", ""))
    require(release_id == release_dir.name, "release_id does not match directory name")

    code_hashes = manifest.get("code_snapshot_sha256")
    require(isinstance(code_hashes, dict) and code_hashes, "release code snapshot is empty")
    require(
        set(code_hashes) == set(PUBLICATION_CODE_SNAPSHOT_FILES),
        "release code snapshot file set differs from the protocol",
    )
    code_root = release_dir / "code_snapshot"
    for relative, expected in sorted(code_hashes.items()):
        path = safe_package_path(code_root, relative, "code snapshot path")
        require(path.is_file(), f"missing code snapshot file {relative}")
        require(sha256_file(path) == expected, f"code snapshot hash mismatch for {relative}")
    require(
        canonical_json_sha256(code_hashes)
        == manifest.get("code_snapshot_manifest_sha256"),
        "code snapshot manifest hash mismatch",
    )

    plan_path = release_dir / "analysis_plan.json"
    require(plan_path.is_file(), "release analysis_plan.json is missing")
    require(
        sha256_file(plan_path) == manifest.get("analysis_plan_sha256"),
        "analysis plan hash mismatch",
    )
    validated_plan = validate_publication_plan(plan_path, code_root)
    require(validated_plan["release_kind"] == kind, "plan/release kind mismatch")
    require(
        manifest.get("analysis_plan_version")
        == validated_plan["plan"]["plan_version"],
        "release analysis plan version mismatch",
    )
    require(
        manifest.get("protocol_version")
        == validated_plan["plan"]["protocol_version"],
        "release protocol version mismatch",
    )
    require(
        int(manifest.get("n_source_runs", -1)) == validated_plan["n_source_runs"],
        "source-run count differs from plan",
    )

    source_entries = manifest.get("source_runs")
    require(isinstance(source_entries, list), "release source_runs must be a list")
    require(
        len(source_entries) == validated_plan["n_source_runs"],
        "release source-run entry count mismatch",
    )
    require(
        canonical_json_sha256(source_entries) == manifest.get("source_run_set_sha256"),
        "release source-run-set hash mismatch",
    )

    frozen_input_manifest = (
        read_json(validated_plan["input_manifest_path"])
        if validated_plan["input_manifest_path"] is not None
        else None
    )
    units: set[tuple[str, int]] = set()
    expected_release_entries: list[dict[str, Any]] = []
    expected_aggregate_entries: list[dict[str, Any]] = []
    all_results: list[pd.DataFrame] = []
    all_allocations: list[pd.DataFrame] = []
    source_git_commits: set[str] = set()
    input_provenance_hashes: set[str] = set()
    environment_hashes: set[str] = set()
    for entry in source_entries:
        require(isinstance(entry, dict), "source-run entry must be an object")
        run_id = str(entry.get("run_id", ""))
        expected_path = f"source_runs/{run_id}"
        require(entry.get("path") == expected_path, f"noncanonical source path for {run_id}")
        run_dir = safe_package_path(release_dir, expected_path, "source-run path")
        summary, config, reference = validate_source_binding(
            run_dir,
            validated_plan=validated_plan,
            release_id=release_id,
        )
        require(summary["run_id"] == run_id, "source-run id mismatch")
        unit = (str(config["suite_id"]), int(config["seed"]))
        require(unit not in units, f"duplicate source unit: {unit}")
        units.add(unit)

        config_hash = sha256_file(run_dir / "config.json")
        checksum_hash = sha256_file(run_dir / "RUN_SHA256SUMS.txt")
        input_provenance = read_json(run_dir / "input_provenance.json")
        if frozen_input_manifest is not None:
            validate_input_provenance_against_manifest(
                input_provenance, frozen_input_manifest
            )
        input_hash = sha256_file(run_dir / "input_provenance.json")
        environment_hash = sha256_file(run_dir / "environment.json")
        source_commit = str(config["source_git_commit"])
        source_git_commits.add(source_commit)
        input_provenance_hashes.add(input_hash)
        environment_hashes.add(environment_hash)
        expected_release_entries.append(
            {
                "run_id": run_id,
                "suite_id": str(config["suite_id"]),
                "suite_role": str(config["suite_role"]),
                "seed": int(config["seed"]),
                "path": expected_path,
                "config_sha256": config_hash,
                "run_checksums_sha256": checksum_hash,
                "source_git_commit": source_commit,
            }
        )
        expected_aggregate_entries.append(
            {
                "run_id": run_id,
                "suite_id": str(config["suite_id"]),
                "suite_role": str(config["suite_role"]),
                "seed": int(config["seed"]),
                "path": run_id,
                "config_sha256": config_hash,
                "run_checksums_sha256": checksum_hash,
                "input_provenance_sha256": input_hash,
                "environment_sha256": environment_hash,
                "source_git_commit": source_commit,
                "n_targets": int(summary["n_targets"]),
                "n_strategies": int(summary["n_strategies"]),
                "target_parameter_budget": int(summary["target_parameter_budget"]),
            }
        )

        result_frame = pd.read_csv(run_dir / "results.csv")
        result_frame.insert(0, "run_id", run_id)
        result_frame.insert(1, "suite_id", str(config["suite_id"]))
        result_frame.insert(2, "suite_role", str(config["suite_role"]))
        result_frame.insert(3, "seed", int(config["seed"]))
        result_frame.insert(4, "reference_strategy", reference)
        all_results.append(result_frame)

        allocation_frame = pd.read_csv(run_dir / "allocations.csv")
        allocation_frame.insert(0, "run_id", run_id)
        allocation_frame.insert(1, "suite_id", str(config["suite_id"]))
        allocation_frame.insert(2, "suite_role", str(config["suite_role"]))
        allocation_frame.insert(3, "seed", int(config["seed"]))
        all_allocations.append(allocation_frame)

    expected_units = {
        (suite["suite_id"], int(seed))
        for suite in validated_plan["suites"]
        for seed in suite["seeds"]
    }
    require(units == expected_units, f"release source units differ from plan: {units}")
    require(len(source_git_commits) == 1, "source runs do not share one Git commit")
    source_git_commit = next(iter(source_git_commits))
    require(
        manifest.get("source_git_commit") == source_git_commit,
        "release/source Git commit mismatch",
    )
    require(len(input_provenance_hashes) == 1, "source input provenance differs")
    require(len(environment_hashes) == 1, "source environments differ")
    expected_release_entries.sort(key=lambda row: row["run_id"])
    require(
        source_entries == expected_release_entries,
        "release source-run manifest entries do not match copied runs",
    )

    aggregate_dir = release_dir / "aggregate"
    aggregate_checksum_count = verify_checksum_file(
        aggregate_dir, "AGGREGATE_SHA256SUMS.txt"
    )
    aggregate_manifest = read_json(aggregate_dir / "manifest.json")
    require(
        aggregate_manifest.get("aggregate_schema_version")
        == PUBLICATION_AGGREGATE_VERSION,
        "aggregate schema mismatch",
    )
    require(aggregate_manifest.get("release_id") == release_id, "aggregate release id mismatch")
    require(aggregate_manifest.get("release_kind") == kind, "aggregate release kind mismatch")
    require(
        aggregate_manifest.get("protocol_version")
        == validated_plan["plan"]["protocol_version"],
        "aggregate protocol version mismatch",
    )
    require(
        aggregate_manifest.get("analysis_plan_version")
        == validated_plan["plan"]["plan_version"],
        "aggregate plan version mismatch",
    )
    require(
        aggregate_manifest.get("analysis_plan_sha256") == validated_plan["sha256"],
        "aggregate plan hash mismatch",
    )
    require(
        aggregate_manifest.get("analysis_plan_file") == "analysis_plan.json",
        "aggregate plan filename mismatch",
    )
    require(
        sha256_file(aggregate_dir / "analysis_plan.json") == validated_plan["sha256"],
        "aggregate plan copy differs from release plan",
    )
    require(
        int(aggregate_manifest.get("n_source_runs", -1))
        == validated_plan["n_source_runs"],
        "aggregate source-run count mismatch",
    )
    require(
        aggregate_manifest.get("independent_unit")
        == validated_plan["analysis"]["independent_unit"],
        "aggregate independent-unit mismatch",
    )
    require(
        aggregate_manifest.get("primary_suite")
        == validated_plan["analysis"]["primary_suite"],
        "aggregate primary suite mismatch",
    )
    require(
        aggregate_manifest.get("primary_candidate_strategy")
        == validated_plan["analysis"]["primary_candidate_strategy"],
        "aggregate primary candidate mismatch",
    )
    for key in [
        "primary_metric",
        "delta_definition",
        "exact_sign_flip_test",
        "secondary_multiplicity",
        "minimum_nonzero_primary_deltas_for_resolution",
        "planned_tie_tolerance",
    ]:
        require(
            aggregate_manifest.get(key) == validated_plan["analysis"][key],
            f"aggregate {key} mismatch",
        )
    require(
        int(aggregate_manifest.get("bootstrap_resamples", -1))
        == int(validated_plan["analysis"]["bootstrap_resamples"]),
        "aggregate bootstrap-resample count mismatch",
    )
    _float_close(
        aggregate_manifest.get("confidence"),
        float(validated_plan["analysis"]["confidence"]),
        "aggregate confidence",
    )
    require(
        aggregate_manifest.get("bootstrap_seed_scheme") == BOOTSTRAP_SEED_SCHEME,
        "aggregate bootstrap seed scheme mismatch",
    )
    expected_aggregate_entries.sort(key=lambda row: (row["suite_id"], row["seed"]))
    require(
        aggregate_manifest.get("source_runs") == expected_aggregate_entries,
        "aggregate source-run entries do not match copied runs",
    )
    aggregate_source_hash_rows = [
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
        for row in expected_aggregate_entries
    ]
    require(
        aggregate_manifest.get("source_run_set_sha256")
        == canonical_json_sha256(aggregate_source_hash_rows),
        "aggregate source-run-set hash mismatch",
    )
    require(
        aggregate_manifest.get("source_git_commit") == source_git_commit,
        "aggregate/source Git commit mismatch",
    )
    require(
        aggregate_manifest.get("input_provenance_sha256")
        == next(iter(input_provenance_hashes)),
        "aggregate input provenance binding mismatch",
    )
    require(
        aggregate_manifest.get("environment_sha256")
        == next(iter(environment_hashes)),
        "aggregate environment binding mismatch",
    )
    require(
        sha256_file(aggregate_dir / "manifest.json")
        == manifest.get("aggregate_manifest_sha256"),
        "aggregate manifest hash mismatch",
    )
    require(
        sha256_file(aggregate_dir / "AGGREGATE_SHA256SUMS.txt")
        == manifest.get("aggregate_checksums_sha256"),
        "aggregate checksum-manifest hash mismatch",
    )

    expected_input_manifest = validated_plan["input_manifest_path"]
    if kind == "publication":
        require(expected_input_manifest is not None, "publication input manifest is missing")
        expected_input_hash = sha256_file(expected_input_manifest)
        require((release_dir / "input_manifest.json").is_file(), "release input manifest is missing")
        require((aggregate_dir / "input_manifest.json").is_file(), "aggregate input manifest is missing")
        require(
            sha256_file(release_dir / "input_manifest.json") == expected_input_hash,
            "release input manifest differs from the pinned code snapshot",
        )
        require(
            sha256_file(aggregate_dir / "input_manifest.json") == expected_input_hash,
            "aggregate input manifest differs from the pinned code snapshot",
        )
        require(aggregate_manifest.get("input_manifest_file") == "input_manifest.json", "aggregate input manifest filename mismatch")
        require(aggregate_manifest.get("input_manifest_sha256") == expected_input_hash, "aggregate input manifest hash mismatch")
    else:
        require(expected_input_manifest is None, "smoke plan unexpectedly has an input manifest")
        require(not (release_dir / "input_manifest.json").exists(), "smoke release unexpectedly has an input manifest")
        require(not (aggregate_dir / "input_manifest.json").exists(), "smoke aggregate unexpectedly has an input manifest")
        require(aggregate_manifest.get("input_manifest_file") is None, "smoke aggregate input manifest filename must be null")
        require(aggregate_manifest.get("input_manifest_sha256") is None, "smoke aggregate input manifest hash must be null")

    recomputed_all_results = pd.concat(all_results, ignore_index=True)
    released_all_results = pd.read_csv(aggregate_dir / "all_results.csv")
    compare_complete_frames(
        recomputed_all_results,
        released_all_results,
        sort_columns=["run_id", "strategy"],
        label="all_results.csv",
    )
    recomputed_all_allocations = pd.concat(all_allocations, ignore_index=True)
    released_all_allocations = pd.read_csv(aggregate_dir / "all_allocations.csv")
    compare_complete_frames(
        recomputed_all_allocations,
        released_all_allocations,
        sort_columns=["run_id", "strategy", "module"],
        label="all_allocations.csv",
    )

    recomputed_pairs = reconstruct_pairs(release_dir, source_entries)
    released_pairs = pd.read_csv(aggregate_dir / "paired_deltas.csv")
    compare_pair_frames(recomputed_pairs, released_pairs)
    recomputed_summary = recompute_summary(recomputed_pairs, validated_plan)
    released_summary = pd.read_csv(aggregate_dir / "analysis_by_suite_strategy.csv")
    compare_summary_frames(recomputed_summary, released_summary)

    analysis = validated_plan["analysis"]
    primary_suite = str(analysis["primary_suite"])
    primary_candidate = str(analysis["primary_candidate_strategy"])
    primary_pairs = recomputed_pairs[
        (recomputed_pairs["suite_id"] == primary_suite)
        & (recomputed_pairs["candidate_strategy"] == primary_candidate)
    ].sort_values("seed").reset_index(drop=True)
    released_primary_pairs = (
        pd.read_csv(aggregate_dir / "primary_seed_deltas.csv")
        .sort_values("seed")
        .reset_index(drop=True)
    )
    compare_pair_frames(primary_pairs, released_primary_pairs)
    primary_summary = recomputed_summary[
        (recomputed_summary["suite_id"] == primary_suite)
        & (recomputed_summary["candidate_strategy"] == primary_candidate)
    ].reset_index(drop=True)
    released_primary = pd.read_csv(aggregate_dir / "primary_analysis.csv")
    require(
        len(released_primary) == 1 and len(primary_summary) == 1,
        "primary analysis row count mismatch",
    )
    compare_summary_frames(primary_summary, released_primary)
    require(
        str(released_primary.loc[0, "analysis_scope"])
        == "prespecified_primary_suite",
        "primary analysis scope mismatch",
    )
    require(
        str(released_primary.loc[0, "primary_estimand"])
        == str(analysis["primary_estimand"]),
        "primary estimand mismatch",
    )
    require(
        str(released_primary.loc[0, "delta_definition"])
        == str(analysis["delta_definition"]),
        "primary delta definition mismatch",
    )
    require(
        str(released_primary.loc[0, "lower_is_better"]).lower()
        == str(bool(analysis["lower_is_better"])).lower(),
        "primary lower_is_better mismatch",
    )
    _float_close(
        released_primary.loc[0, "two_sided_alpha"],
        float(analysis["two_sided_alpha"]),
        "primary two_sided_alpha",
    )
    require(
        int(released_primary.loc[0, "minimum_nonzero_primary_deltas_for_resolution"])
        == int(analysis["minimum_nonzero_primary_deltas_for_resolution"]),
        "primary minimum-nonzero resolution threshold mismatch",
    )
    require(
        int(released_primary.loc[0, "planned_tie_tolerance"])
        == int(analysis["planned_tie_tolerance"]),
        "primary planned tie tolerance mismatch",
    )

    environment = read_json(release_dir / "environment" / "environment.json")
    require(environment.get("git_commit") == manifest.get("git_commit"), "release Git commit mismatch")
    require(
        environment.get("git_commit") == source_git_commit,
        "release environment/source Git commit mismatch",
    )
    packages = environment.get("packages")
    require(isinstance(packages, dict), "release package provenance is missing")
    if kind == "publication":
        for package in [
            "torch",
            "numpy",
            "scipy",
            "pandas",
            "transformers",
            "datasets",
            "accelerate",
            "tokenizers",
            "safetensors",
        ]:
            require(packages.get(package), f"release package version missing: {package}")
        require(
            not str(environment.get("git_status_porcelain", "")),
            "publication release Git worktree was not clean",
        )
        require(
            validated_plan["n_source_runs"] == 11,
            "publication release must contain the frozen 8+3 design",
        )
    else:
        require(
            validated_plan["n_source_runs"] == 4,
            "smoke release must contain the frozen four-source design",
        )

    summary = {
        "release_id": release_id,
        "release_kind": kind,
        "protocol_version": manifest["protocol_version"],
        "n_source_runs": len(source_entries),
        "n_checksummed_files": checksum_count,
        "n_aggregate_checksummed_files": aggregate_checksum_count,
        "primary_suite": primary_suite,
        "primary_candidate_strategy": primary_candidate,
    }
    print("Real LoRA publication release validation: PASS")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("--expected-kind", choices=["smoke", "publication"], default=None)
    args = parser.parse_args()
    validate_release(args.release_dir, expected_kind=args.expected_kind)


if __name__ == "__main__":
    main()
