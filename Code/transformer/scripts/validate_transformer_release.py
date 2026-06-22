#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_rank_scaling_run import validate as validate_source_run
from strank.protocol import SYNTHETIC_TRANSFORMER_PROTOCOL_VERSION

PUBLICATION_MIN_TASKS = 2
PUBLICATION_MIN_RUNS_PER_TASK = 5
PUBLICATION_MIN_NULL_MAXIMA = 4096
PUBLICATION_MIN_NULL_UNCERTAINTY_RESAMPLES = 2000
PUBLICATION_PRIMARY_NULL_QUANTILE = 0.995
EXPECTED_SCALING_MODES = {"fixed_update_scale", "standard", "rslora"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing required JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _safe_relative_path(value: str, label: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe {label}: {value!r}")
    return Path(*pure.parts)


def _read_checksums(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"missing checksum manifest: {path}")
    entries: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            digest, relative = raw.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"invalid checksum line {line_number}: {raw!r}") from exc
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"invalid SHA-256 digest on line {line_number}")
        _safe_relative_path(relative, f"checksum path on line {line_number}")
        if relative in entries:
            raise ValueError(f"duplicate checksum entry: {relative}")
        entries[relative] = digest
    if not entries:
        raise ValueError("checksum manifest is empty")
    return entries


def _common_summary_value(summaries: list[dict], key: str):
    values = [summary.get(key) for summary in summaries]
    canonical = [json.dumps(value, sort_keys=True) for value in values]
    if not values or len(set(canonical)) != 1:
        raise ValueError(f"source runs disagree on {key}: {values!r}")
    return values[0]



def _validate_aggregate_source_binding(
    manifest: dict,
    aggregate_manifest: dict,
    aggregate_dir: Path,
    release_dir: Path,
    summaries: list[dict],
) -> None:
    aggregate_entries = aggregate_manifest.get("source_runs")
    if not isinstance(aggregate_entries, list) or not aggregate_entries:
        raise ValueError("aggregate manifest does not bind outputs to source-run hashes")
    release_entries = manifest.get("source_runs", [])
    release_by_id = {
        Path(str(entry["relative_path"])).name: entry for entry in release_entries
    }
    summary_by_id = {str(summary["run_id"]): summary for summary in summaries}
    aggregate_by_id: dict[str, dict] = {}
    for entry in aggregate_entries:
        if not isinstance(entry, dict):
            raise ValueError("aggregate source-run entry is not an object")
        run_id = str(entry.get("run_id", ""))
        if not run_id or run_id in aggregate_by_id:
            raise ValueError(f"invalid or duplicate aggregate source run: {run_id!r}")
        aggregate_by_id[run_id] = entry
    if set(aggregate_by_id) != set(release_by_id):
        raise ValueError(
            "aggregate source-run IDs do not match release source runs; "
            f"aggregate={sorted(aggregate_by_id)}, release={sorted(release_by_id)}"
        )

    required_hashed_files = {
        "config.yaml",
        "run_info.json",
        "_SUCCESS.json",
        "calibration/module_stats.csv",
        "calibration/singular_values.npz",
        "calibration/null_maxima.npz",
        "sweeps/site_rank_sweep_metrics.csv",
        "sweeps/site_target_summary.csv",
        "sweeps/site_prediction_fit.csv",
        "budget/budget_results.csv",
        "budget/allocation_comparison.csv",
    }
    metadata_keys = (
        "task_name",
        "base_seed",
        "protocol_version",
        "primary_scaling_mode",
        "primary_allocation_rule",
        "primary_reference_rule",
        "primary_metric",
        "independent_unit",
    )
    for run_id, aggregate_entry in aggregate_by_id.items():
        release_entry = release_by_id[run_id]
        summary = summary_by_id[run_id]
        expected_metadata = {
            "task_name": str(summary["task_name"]),
            "base_seed": int(summary["base_seed"]),
            "protocol_version": str(summary["protocol_version"]),
            "primary_scaling_mode": str(summary["primary_scaling_mode"]),
            "primary_allocation_rule": str(summary["primary_allocation_rule"]),
            "primary_reference_rule": str(summary["primary_reference_rule"]),
            "primary_metric": str(summary["primary_metric"]),
            "independent_unit": str(summary["independent_unit"]),
        }
        for key in metadata_keys:
            if aggregate_entry.get(key) != expected_metadata[key]:
                raise ValueError(
                    f"aggregate source metadata mismatch for {run_id}.{key}: "
                    f"recorded={aggregate_entry.get(key)!r}, "
                    f"expected={expected_metadata[key]!r}"
                )
        hashes = aggregate_entry.get("sha256")
        if not isinstance(hashes, dict) or set(hashes) != required_hashed_files:
            raise ValueError(
                f"aggregate source hashes for {run_id} must cover exactly the required raw files"
            )
        source_dir = release_dir / _safe_relative_path(
            str(release_entry["relative_path"]), "source-run path"
        )
        for relative_text, expected_digest in hashes.items():
            relative = _safe_relative_path(str(relative_text), "aggregate source hash path")
            path = source_dir / relative
            if not path.is_file():
                raise ValueError(f"aggregate-bound source file is missing: {run_id}/{relative_text}")
            if _sha256(path) != str(expected_digest):
                raise ValueError(f"aggregate source hash mismatch: {run_id}/{relative_text}")

    expected_roles = {
        "protocol_version": manifest["protocol_version"],
        "primary_scaling_mode": manifest["primary_scaling_mode"],
        "primary_allocation_rule": manifest["primary_allocation_rule"],
        "primary_reference_rule": manifest["primary_reference_rule"],
        "primary_metric": manifest["primary_metric"],
        "independent_unit": manifest["unit_of_inference"],
    }
    for key, expected in expected_roles.items():
        if aggregate_manifest.get(key) != expected:
            raise ValueError(
                f"aggregate manifest {key}={aggregate_manifest.get(key)!r}; expected {expected!r}"
            )
    if aggregate_manifest.get("scaling_conditions_analyzed_separately") is not True:
        raise ValueError("aggregate pooled distinct scaling conditions")
    if aggregate_manifest.get("exact_cost_primary_outputs") is not True:
        raise ValueError("aggregate does not certify exact-cost primary outputs")
    if set(aggregate_manifest.get("scaling_modes", [])) != EXPECTED_SCALING_MODES:
        raise ValueError("aggregate scaling modes do not match the publication design")

    expected_task_counts = Counter(str(item["task_name"]) for item in summaries)
    recorded_task_counts = aggregate_manifest.get("independent_runs_by_task")
    if not isinstance(recorded_task_counts, dict):
        raise ValueError("aggregate manifest lacks independent_runs_by_task")
    normalised_task_counts = {
        str(task): int(count) for task, count in recorded_task_counts.items()
    }
    if normalised_task_counts != dict(expected_task_counts):
        raise ValueError(
            "aggregate task/run counts do not match source runs; "
            f"recorded={normalised_task_counts}, expected={dict(expected_task_counts)}"
        )

    required_outputs = (
        "all_budget_results.csv",
        "run_level_budget_results.csv",
        "deltas_exact_cost.csv",
        "run_level_deltas_exact_cost.csv",
        "delta_summary_exact_cost.csv",
        "run_cluster_deltas_exact_cost.csv",
        "run_cluster_summary_exact_cost.csv",
    )
    for name in required_outputs:
        if not (aggregate_dir / name).is_file():
            raise ValueError(f"publication aggregate is missing required output: {name}")

    cluster = pd.read_csv(aggregate_dir / "run_cluster_summary_exact_cost.csv")
    required_columns = {
        "task_family",
        "scaling_mode",
        "allocation_rule",
        "is_primary_allocation_rule",
        "n_independent_runs",
        "mean_loss_delta",
        "cluster_bootstrap_ci_low",
        "cluster_bootstrap_ci_high",
        "exact_sign_flip_p_two_sided",
    }
    if not required_columns.issubset(cluster.columns):
        raise ValueError(
            "run-cluster summary is missing columns: "
            f"{sorted(required_columns - set(cluster.columns))}"
        )
    primary_flags = cluster["is_primary_allocation_rule"].astype(str).str.lower().eq("true")
    primary = cluster[
        cluster["scaling_mode"].astype(str).eq(manifest["primary_scaling_mode"])
        & cluster["allocation_rule"].astype(str).eq(manifest["primary_allocation_rule"])
        & primary_flags
    ]
    if len(primary) != len(expected_task_counts):
        raise ValueError(
            "run-cluster summary must contain exactly one pre-specified primary row per task"
        )
    if set(primary["task_family"].astype(str)) != set(expected_task_counts):
        raise ValueError("primary run-cluster rows do not cover every task family")
    for row in primary.itertuples(index=False):
        expected_n = int(expected_task_counts[str(row.task_family)])
        if int(row.n_independent_runs) != expected_n:
            raise ValueError(
                f"primary aggregate run count mismatch for {row.task_family}: "
                f"recorded={row.n_independent_runs}, expected={expected_n}"
            )

def _validate_publication_design(manifest: dict, summaries: list[dict], release_dir: Path) -> None:
    task_counts = Counter(str(item["task_name"]) for item in summaries)
    if len(task_counts) < PUBLICATION_MIN_TASKS:
        raise ValueError(
            f"publication release needs at least {PUBLICATION_MIN_TASKS} task families; "
            f"found {dict(task_counts)}"
        )
    underpowered = {
        task: count for task, count in task_counts.items() if count < PUBLICATION_MIN_RUNS_PER_TASK
    }
    if underpowered:
        raise ValueError(
            "publication release has too few independent runs per task; "
            f"minimum={PUBLICATION_MIN_RUNS_PER_TASK}, found={underpowered}"
        )
    pairs = [(str(item["task_name"]), int(item["base_seed"])) for item in summaries]
    if len(set(pairs)) != len(pairs):
        raise ValueError("publication release contains duplicate task/base-seed units")
    for summary in summaries:
        if set(summary["scaling_modes"]) != EXPECTED_SCALING_MODES:
            raise ValueError(
                f"source run {summary['run_id']} lacks the full pre-specified scaling design"
            )
        if summary["primary_scaling_mode"] != "fixed_update_scale":
            raise ValueError("publication primary scaling mode must be fixed_update_scale")
        if int(summary["null_bootstrap"]) < PUBLICATION_MIN_NULL_MAXIMA:
            raise ValueError(
                f"source run {summary['run_id']} has only {summary['null_bootstrap']} null maxima; "
                f"publication minimum is {PUBLICATION_MIN_NULL_MAXIMA}"
            )
        if abs(float(summary["null_quantile"]) - PUBLICATION_PRIMARY_NULL_QUANTILE) > 1e-15:
            raise ValueError("publication primary null quantile must be 0.995")
        if int(summary["null_uncertainty_resamples"]) < PUBLICATION_MIN_NULL_UNCERTAINTY_RESAMPLES:
            raise ValueError(
                f"source run {summary['run_id']} has too few null uncertainty resamples"
            )
        if summary["task_name"] == "modular" and not summary["exact_modular_evaluation"]:
            raise ValueError("publication modular runs must use exhaustive evaluation")
        if not summary["exact_cost_matched_rules"]:
            raise ValueError("publication runs must include pre-specified exact-cost comparators")
        if summary["primary_reference_rule"] != "uniform_exact_cost":
            raise ValueError("publication primary reference must be uniform_exact_cost")
        if summary["primary_allocation_rule"] not in summary["exact_cost_matched_rules"]:
            raise ValueError("primary allocation rule lacks an exact-cost comparator")

    aggregate = manifest.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError("publication release is missing its aggregate directory")
    aggregate_relative = _safe_relative_path(
        str(aggregate.get("relative_path", "")), "aggregate path"
    )
    if len(aggregate_relative.parts) != 2 or aggregate_relative.parts[0] != "aggregate":
        raise ValueError("aggregate must be a direct child of aggregate/")
    aggregate_dir = release_dir / aggregate_relative
    aggregate_manifest = _read_json(aggregate_dir / "manifest.json")
    aggregate_count = int(aggregate_manifest.get("n_independent_runs", -1))
    if aggregate_count != len(summaries):
        raise ValueError("aggregate independent-run count does not match source runs")
    if aggregate.get("n_independent_runs") != aggregate_count:
        raise ValueError("release manifest aggregate run count is stale")
    unit = str(aggregate_manifest.get("unit_of_inference", ""))
    if aggregate.get("unit_of_inference") != unit:
        raise ValueError("release manifest aggregate inference unit is stale")
    if "run" not in unit or "adaptation" not in unit:
        raise ValueError("aggregate manifest does not declare run-cluster inference")
    _validate_aggregate_source_binding(
        manifest, aggregate_manifest, aggregate_dir, release_dir, summaries
    )

    git_info = _read_json(release_dir / "provenance" / "git.json")
    if git_info.get("status_porcelain"):
        raise ValueError("publication release must be built from a clean committed worktree")


def validate_release(release_dir: Path, *, expected_kind: str | None = None) -> dict:
    release_dir = release_dir.resolve()
    if not release_dir.is_dir():
        raise ValueError(f"release directory does not exist: {release_dir}")
    if any(path.is_symlink() for path in release_dir.rglob("*")):
        raise ValueError("release must not contain symbolic links")

    checksum_path = release_dir / "SHA256SUMS.txt"
    recorded = _read_checksums(checksum_path)
    actual_files = {
        path.relative_to(release_dir).as_posix()
        for path in release_dir.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if set(recorded) != actual_files:
        unlisted = sorted(actual_files - set(recorded))
        missing = sorted(set(recorded) - actual_files)
        raise ValueError(f"checksum file set mismatch; unlisted={unlisted}, missing={missing}")
    for relative, expected_digest in recorded.items():
        actual_digest = _sha256(release_dir / _safe_relative_path(relative, "checksum path"))
        if actual_digest != expected_digest:
            raise ValueError(
                f"checksum mismatch for {relative}: expected {expected_digest}, got {actual_digest}"
            )

    manifest = _read_json(release_dir / "release_manifest.json")
    if manifest.get("release_id") != release_dir.name:
        raise ValueError("release_manifest release_id does not match directory name")
    release_kind = str(manifest.get("release_kind"))
    if release_kind not in {"smoke", "publication"}:
        raise ValueError(f"unsupported release kind: {release_kind!r}")
    if expected_kind is not None and release_kind != expected_kind:
        raise ValueError(f"release kind {release_kind!r} != expected {expected_kind!r}")
    if manifest.get("protocol_version") != SYNTHETIC_TRANSFORMER_PROTOCOL_VERSION:
        raise ValueError("release manifest protocol version is not current")
    if manifest.get("checksum_algorithm") != "sha256":
        raise ValueError("release manifest does not declare SHA-256")
    if manifest.get("primary_scaling_mode") != "fixed_update_scale":
        raise ValueError("release manifest primary scaling mode is not fixed_update_scale")

    source_entries = manifest.get("source_runs")
    if not isinstance(source_entries, list) or not source_entries:
        raise ValueError("release manifest has no source runs")
    summaries = []
    seen_paths: set[str] = set()
    for entry in source_entries:
        if not isinstance(entry, dict):
            raise ValueError("source-run manifest entry is not an object")
        relative_text = str(entry.get("relative_path", ""))
        if relative_text in seen_paths:
            raise ValueError(f"duplicate source-run manifest path: {relative_text}")
        seen_paths.add(relative_text)
        relative = _safe_relative_path(relative_text, "source-run path")
        if len(relative.parts) != 2 or relative.parts[0] != "source_runs":
            raise ValueError(
                f"source run must be a direct child of source_runs/: {relative_text}"
            )
        source_dir = release_dir / relative
        summary = validate_source_run(source_dir, fail_on_divergence=True)
        checks = {
            "protocol_version": str(summary["protocol_version"]),
            "base_seed": int(summary["base_seed"]),
            "task_name": str(summary["task_name"]),
            "scaling_modes": list(summary["scaling_modes"]),
            "primary_scaling_mode": str(summary["primary_scaling_mode"]),
            "scale_reference_rank": int(summary["scale_reference_rank"]),
            "primary_allocation_rule": str(summary["primary_allocation_rule"]),
            "primary_reference_rule": str(summary["primary_reference_rule"]),
            "primary_metric": str(summary["primary_metric"]),
            "independent_unit": str(summary["independent_unit"]),
            "n_sites": int(summary["n_sites"]),
            "n_sweep_rows": int(summary["n_sweep_rows"]),
            "n_budget_rows": int(summary["n_budget_rows"]),
            "n_allocation_rows": int(summary["n_allocation_rows"]),
            "adaptation_replicates": int(summary["adaptation_replicates"]),
            "null_bootstrap": int(summary["null_bootstrap"]),
            "null_quantile": float(summary["null_quantile"]),
            "null_uncertainty_resamples": int(summary["null_uncertainty_resamples"]),
            "exact_modular_evaluation": bool(summary["exact_modular_evaluation"]),
            "exact_cost_matched_rules": list(summary["exact_cost_matched_rules"]),
        }
        for key, expected in checks.items():
            recorded_value = entry.get(key)
            if recorded_value != expected:
                raise ValueError(
                    f"source-run manifest mismatch for {relative_text}.{key}: "
                    f"recorded={recorded_value!r}, validated={expected!r}"
                )
        summaries.append(summary)

    common_protocol = {
        key: _common_summary_value(summaries, key)
        for key in (
            "protocol_version",
            "scaling_modes",
            "primary_scaling_mode",
            "scale_reference_rank",
            "primary_allocation_rule",
            "primary_reference_rule",
            "primary_metric",
            "independent_unit",
            "exact_cost_matched_rules",
        )
    }
    expected_manifest = {
        "protocol_version": common_protocol["protocol_version"],
        "unit_of_inference": common_protocol["independent_unit"],
        "cluster_structure_note": (
            "adaptation replicates and budgets are nested within each independent "
            "task/base-model seed run"
        ),
        "primary_scaling_mode": common_protocol["primary_scaling_mode"],
        "sensitivity_scaling_modes": sorted(
            set(common_protocol["scaling_modes"])
            - {common_protocol["primary_scaling_mode"]}
        ),
        "scale_reference_rank": common_protocol["scale_reference_rank"],
        "primary_allocation_rule": common_protocol["primary_allocation_rule"],
        "primary_reference_rule": common_protocol["primary_reference_rule"],
        "primary_metric": common_protocol["primary_metric"],
        "exact_cost_matched_rules": common_protocol["exact_cost_matched_rules"],
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"release manifest mismatch for {key}: "
                f"recorded={manifest.get(key)!r}, validated={expected!r}"
            )

    required_snapshot = [
        "code_snapshot/strank/protocol.py",
        "code_snapshot/strank/scaling.py",
        "code_snapshot/strank/allocation.py",
        "code_snapshot/strank/calibrate.py",
        "code_snapshot/strank/spectral.py",
        "code_snapshot/scripts/run_synthetic_transformer.py",
        "code_snapshot/scripts/validate_rank_scaling_run.py",
        "code_snapshot/scripts/build_transformer_release.py",
        "code_snapshot/scripts/validate_transformer_release.py",
        "code_snapshot/requirements.txt",
        "code_snapshot/pyproject.toml",
        "code_snapshot/tests/test_core.py",
        "provenance/python_environment.txt",
        "provenance/system.json",
        "provenance/git.json",
    ]
    for relative in required_snapshot:
        if not (release_dir / relative).is_file():
            raise ValueError(f"release is missing required provenance file: {relative}")
    protocol_text = (release_dir / "code_snapshot" / "strank" / "protocol.py").read_text(
        encoding="utf-8"
    )
    if SYNTHETIC_TRANSFORMER_PROTOCOL_VERSION not in protocol_text:
        raise ValueError("code snapshot does not declare the release protocol version")

    if release_kind == "publication":
        _validate_publication_design(manifest, summaries, release_dir)

    summary = {
        "status": "PASS",
        "release_id": release_dir.name,
        "release_kind": release_kind,
        "protocol_version": SYNTHETIC_TRANSFORMER_PROTOCOL_VERSION,
        "n_source_runs": len(summaries),
        "tasks": sorted({item["task_name"] for item in summaries}),
        "seeds": sorted({int(item["base_seed"]) for item in summaries}),
        "scaling_modes": sorted({mode for item in summaries for mode in item["scaling_modes"]}),
        "n_files": len(recorded) + 1,
    }
    print("Transformer release validation: PASS")
    for key, value in summary.items():
        if key != "status":
            print(f"{key}: {value}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a self-contained transformer release.")
    parser.add_argument("release_dir")
    parser.add_argument("--expected-kind", choices=("smoke", "publication"), default=None)
    args = parser.parse_args()
    validate_release(Path(args.release_dir), expected_kind=args.expected_kind)


if __name__ == "__main__":
    main()
