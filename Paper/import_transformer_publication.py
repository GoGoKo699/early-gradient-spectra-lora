#!/usr/bin/env python3
"""Import a validated transformer publication release into paper-facing tables.

The importer rejects smoke, stale-plan, incomplete, checksum-inconsistent, or
non-reproducible releases.  It independently rebuilds the confirmatory
soft-dimension versus exact-cost-uniform run deltas from each raw source run
before writing any paper-facing artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
TRANSFORMER = ROOT / "Code" / "transformer"
RELEASE_ROOT = TRANSFORMER / "runs" / "transformer_releases"
TABLE_ROOT = PAPER / "tables" / "transformer_publication"

EXPECTED_PLAN_SHA256 = "89c74a8d3e773cd20950bf2a9854bcdf39cd6f0a38a8eed78ca0d3345c85be6d"
EXPECTED_PLAN_VERSION = "transformer_publication_plan_v1"
EXPECTED_PROTOCOL = "synthetic_transformer_publication_protocol_v4"
EXPECTED_TASKS = {"modular", "associative_recall"}
EXPECTED_SEEDS = {101, 103, 107, 109, 113}
PRIMARY_SCALING_MODE = "fixed_update_scale"
PRIMARY_ALLOCATION_RULE = "soft_dimension"
PRIMARY_REFERENCE_RULE = "uniform_exact_cost"
PRIMARY_METRIC = "final_val_loss"
SEED_COLUMNS = [
    "condition_seed",
    "comparison_seed",
    "adapter_seed",
    "train_data_seed",
    "eval_data_seed",
    "dropout_seed",
]
METRIC_COLUMNS = ["final_train_loss", "final_val_loss", "final_val_accuracy"]

sys.path.insert(0, str(TRANSFORMER))
from strank.inference import (  # noqa: E402
    TASK_STRATIFIED_BOOTSTRAP_SEED_SCHEME,
    task_stratified_bootstrap_interval,
    task_stratified_mean,
    task_stratified_sign_flip_p,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve_release(explicit: Path | None) -> Path:
    if explicit is not None:
        release = explicit.resolve()
    else:
        latest = RELEASE_ROOT / "LATEST"
        if not latest.is_file():
            raise FileNotFoundError(
                f"missing {latest}; run the transformer publication driver first"
            )
        release_id = latest.read_text(encoding="utf-8").strip()
        if not release_id:
            raise ValueError(f"empty release pointer: {latest}")
        release = (RELEASE_ROOT / release_id).resolve()
    if not release.is_dir():
        raise FileNotFoundError(f"transformer release directory not found: {release}")
    return release


def _verify_release_checksums(release: Path) -> int:
    if any(path.is_symlink() for path in release.rglob("*")):
        raise ValueError("transformer release must not contain symbolic links")
    manifest_path = release / "SHA256SUMS.txt"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing recursive checksum manifest: {manifest_path}")
    listed: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"malformed checksum line: {line!r}") from exc
        if relative in listed:
            raise ValueError(f"duplicate checksum entry: {relative}")
        candidate = (release / relative).resolve()
        try:
            candidate.relative_to(release.resolve())
        except ValueError as exc:
            raise ValueError(f"checksum path escapes release: {relative}") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"checksummed file is missing: {candidate}")
        observed = _sha256(candidate)
        if observed != digest:
            raise ValueError(
                f"checksum mismatch for {relative}: {observed} != {digest}"
            )
        listed[relative] = digest
    actual = {
        str(path.relative_to(release))
        for path in release.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    }
    if set(listed) != actual:
        raise ValueError(
            "release checksum manifest is not exhaustive; "
            f"unlisted={sorted(actual - set(listed))[:5]}, "
            f"missing={sorted(set(listed) - actual)[:5]}"
        )
    return len(listed)


def _verify_archive(release: Path) -> str:
    archive = release.parent / f"{release.name}.tar.gz"
    sidecar = Path(f"{archive}.sha256")
    if not archive.is_file() or not sidecar.is_file():
        raise FileNotFoundError(
            f"release archive or SHA-256 sidecar is missing: {archive}, {sidecar}"
        )
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) < 2:
        raise ValueError(f"malformed archive checksum sidecar: {sidecar}")
    expected = fields[0]
    if Path(fields[-1]).name != archive.name:
        raise ValueError(f"archive sidecar names {fields[-1]!r}, expected {archive.name!r}")
    observed = _sha256(archive)
    if observed != expected:
        raise ValueError(f"archive checksum mismatch: {observed} != {expected}")
    return observed


def _task_stratified_exact_p(values: Iterable[float], tasks: Iterable[str]) -> float:
    # Independent implementation used as a cross-check on the released helper.
    data = np.asarray(list(values), dtype=float)
    labels = np.asarray(list(tasks), dtype=str)
    task_names = sorted(set(labels.tolist()))
    weights = np.zeros(len(data), dtype=float)
    for task in task_names:
        mask = labels == task
        weights[mask] = 1.0 / (len(task_names) * int(mask.sum()))
    observed = abs(float(np.sum(weights * data)))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(data)):
        statistic = abs(float(np.sum(weights * data * np.asarray(signs))))
        exceed += statistic >= observed - 1e-15
        total += 1
    return float(exceed / total)


def _rebuild_primary_runs(release: Path, release_manifest: dict) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    exact_pairs = 0
    identical_groups = 0
    max_metric_spread = 0.0
    max_seed_disagreement = 0
    divergence_count = 0

    source_runs = release_manifest.get("source_runs")
    if not isinstance(source_runs, list) or len(source_runs) != 10:
        raise ValueError("publication release must contain exactly 10 source runs")

    observed_units = {
        (str(entry.get("task_name")), int(entry.get("base_seed")))
        for entry in source_runs
    }
    expected_units = {(task, seed) for task in EXPECTED_TASKS for seed in EXPECTED_SEEDS}
    if observed_units != expected_units:
        raise ValueError(
            f"source task/seed design differs from the frozen plan: {sorted(observed_units)}"
        )

    for entry in source_runs:
        run_dir = release / str(entry["relative_path"])
        raw = pd.read_csv(run_dir / "budget" / "budget_results.csv")
        divergence_count += int(raw["diverged"].astype(bool).sum())

        identity_keys = [
            "scaling_mode",
            "budget",
            "adaptation_replicate",
            "allocation_signature",
        ]
        for _, group in raw.groupby(identity_keys, dropna=False):
            if len(group) < 2:
                continue
            identical_groups += 1
            for column in METRIC_COLUMNS:
                spread = float(group[column].max() - group[column].min())
                max_metric_spread = max(max_metric_spread, spread)
            for column in ["condition_seed", *SEED_COLUMNS]:
                max_seed_disagreement = max(
                    max_seed_disagreement, int(group[column].nunique(dropna=False) - 1)
                )

        candidates = raw[
            raw["comparison_role"].astype(str).eq("candidate")
            & raw["exact_cost_match_required"].astype(bool)
        ].copy()
        references = raw[
            raw["comparison_role"].astype(str).eq("exact_cost_baseline")
        ][
            [
                "scaling_mode",
                "budget",
                "adaptation_replicate",
                "condition_id",
                "actual_cost",
                *SEED_COLUMNS,
                "final_val_loss",
                "final_val_accuracy",
            ]
        ].rename(
            columns={
                "condition_id": "matched_baseline_condition_id",
                "actual_cost": "reference_cost",
                **{column: f"reference_{column}" for column in SEED_COLUMNS},
                "final_val_loss": "reference_loss",
                "final_val_accuracy": "reference_accuracy",
            }
        )
        paired = candidates.merge(
            references,
            on=[
                "scaling_mode",
                "budget",
                "adaptation_replicate",
                "matched_baseline_condition_id",
            ],
            how="left",
            validate="many_to_one",
        )
        if len(paired) != len(candidates) or paired["reference_cost"].isna().any():
            raise ValueError(f"unmatched exact-cost candidate in {run_dir.name}")
        exact_pairs += len(paired)
        if not (paired["actual_cost"].astype(int) == paired["reference_cost"].astype(int)).all():
            raise ValueError(f"unequal exact-cost pairing in {run_dir.name}")
        for column in SEED_COLUMNS:
            if not (paired[column] == paired[f"reference_{column}"]).all():
                raise ValueError(f"CRN seed mismatch for {column} in {run_dir.name}")

        primary = paired[
            paired["scaling_mode"].astype(str).eq(PRIMARY_SCALING_MODE)
            & paired["rule"].astype(str).eq(PRIMARY_ALLOCATION_RULE)
            & paired["is_primary_allocation_rule"].astype(bool)
        ].copy()
        if primary.empty:
            raise ValueError(f"missing primary rows in {run_dir.name}")
        primary["loss_delta_vs_exact_uniform"] = (
            primary["final_val_loss"] - primary["reference_loss"]
        )
        primary["acc_delta_vs_exact_uniform"] = (
            primary["final_val_accuracy"] - primary["reference_accuracy"]
        )
        by_budget = primary.groupby("budget", as_index=False).agg(
            loss_delta_vs_exact_uniform=("loss_delta_vs_exact_uniform", "mean"),
            acc_delta_vs_exact_uniform=("acc_delta_vs_exact_uniform", "mean"),
            mean_actual_cost=("actual_cost", "mean"),
            n_adaptation_replicates=("adaptation_replicate", "nunique"),
        )
        if not by_budget["n_adaptation_replicates"].eq(3).all():
            raise ValueError(f"primary comparison lacks three replicates in {run_dir.name}")
        rows.append(
            {
                "task_family": str(entry["task_name"]),
                "run": run_dir.name,
                "seed": int(entry["base_seed"]),
                "scaling_mode": PRIMARY_SCALING_MODE,
                "condition_id": PRIMARY_ALLOCATION_RULE,
                "rule": PRIMARY_ALLOCATION_RULE,
                "allocation_rule": PRIMARY_ALLOCATION_RULE,
                "is_primary_allocation_rule": True,
                "n_budgets": int(len(by_budget)),
                "mean_adaptation_replicates": float(
                    by_budget["n_adaptation_replicates"].mean()
                ),
                "loss_delta_vs_exact_uniform": float(
                    by_budget["loss_delta_vs_exact_uniform"].mean()
                ),
                "acc_delta_vs_exact_uniform": float(
                    by_budget["acc_delta_vs_exact_uniform"].mean()
                ),
                "mean_actual_cost": float(by_budget["mean_actual_cost"].mean()),
            }
        )

    if divergence_count:
        raise ValueError(f"publication source rows contain {divergence_count} divergences")
    if max_metric_spread > 1e-12 or max_seed_disagreement:
        raise ValueError(
            "identical-allocation CRN invariant failed: "
            f"metric spread={max_metric_spread}, seed disagreements={max_seed_disagreement}"
        )

    frame = pd.DataFrame(rows).sort_values(["task_family", "seed", "run"]).reset_index(drop=True)
    diagnostics = {
        "n_source_runs": len(frame),
        "n_exact_candidate_pairs": exact_pairs,
        "n_identical_allocation_groups": identical_groups,
        "max_identical_metric_spread": max_metric_spread,
        "max_identical_seed_disagreement": max_seed_disagreement,
        "n_diverged_rows": divergence_count,
    }
    return frame, diagnostics


def _assert_frame_close(actual: pd.DataFrame, expected: pd.DataFrame, keys: list[str]) -> None:
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    if list(actual.columns) != list(expected.columns):
        raise ValueError(
            "paper recomputation and aggregate columns differ: "
            f"{list(actual.columns)} != {list(expected.columns)}"
        )
    if len(actual) != len(expected):
        raise ValueError("paper recomputation and aggregate row counts differ")
    for column in actual.columns:
        left = actual[column]
        right = expected[column]
        if pd.api.types.is_numeric_dtype(left) or pd.api.types.is_numeric_dtype(right):
            if not np.allclose(
                pd.to_numeric(left, errors="coerce"),
                pd.to_numeric(right, errors="coerce"),
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            ):
                raise ValueError(f"aggregate mismatch in column {column}")
        elif left.astype(str).tolist() != right.astype(str).tolist():
            raise ValueError(f"aggregate mismatch in column {column}")


def _summarize_scaling(run_cluster: pd.DataFrame) -> pd.DataFrame:
    selected = run_cluster[
        run_cluster["allocation_rule"].astype(str).eq(PRIMARY_ALLOCATION_RULE)
    ].copy()
    rows = []
    for mode, group in selected.groupby("scaling_mode"):
        values = group["loss_delta_vs_exact_uniform"].to_numpy(dtype=float)
        accuracy = group["acc_delta_vs_exact_uniform"].to_numpy(dtype=float)
        tasks = group["task_family"].astype(str).tolist()
        task_means = group.groupby("task_family")["loss_delta_vs_exact_uniform"].mean()
        rows.append(
            {
                "scaling_mode": str(mode),
                "analysis_role": (
                    "confirmatory_primary"
                    if str(mode) == PRIMARY_SCALING_MODE
                    else "exploratory_scaling_sensitivity"
                ),
                "n_independent_runs": int(len(group)),
                "mean_loss_delta": task_stratified_mean(values, tasks),
                "exact_sign_flip_p_two_sided": task_stratified_sign_flip_p(values, tasks),
                "run_wins_loss": int(np.sum(values < 0)),
                "task_wins_loss": int(np.sum(task_means.to_numpy() < 0)),
                "associative_recall_mean_loss_delta": float(
                    task_means.loc["associative_recall"]
                ),
                "modular_mean_loss_delta": float(task_means.loc["modular"]),
                "mean_accuracy_delta": task_stratified_mean(accuracy, tasks),
            }
        )
    order = {"fixed_update_scale": 0, "standard": 1, "rslora": 2}
    frame = pd.DataFrame(rows)
    frame["_order"] = frame["scaling_mode"].map(order)
    return frame.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def import_release(release: Path, *, write_outputs: bool = True) -> dict:
    n_checksums = _verify_release_checksums(release)
    archive_sha256 = _verify_archive(release)
    release_manifest = _read_json(release / "release_manifest.json")

    required_manifest = {
        "release_id": release.name,
        "release_kind": "publication",
        "protocol_version": EXPECTED_PROTOCOL,
        "primary_scaling_mode": PRIMARY_SCALING_MODE,
        "primary_allocation_rule": PRIMARY_ALLOCATION_RULE,
        "primary_reference_rule": PRIMARY_REFERENCE_RULE,
        "primary_metric": PRIMARY_METRIC,
    }
    for key, expected in required_manifest.items():
        if release_manifest.get(key) != expected:
            raise ValueError(
                f"release manifest {key}={release_manifest.get(key)!r}; expected {expected!r}"
            )
    aggregate_meta = release_manifest.get("aggregate")
    if not isinstance(aggregate_meta, dict):
        raise ValueError("publication release is missing aggregate metadata")
    if aggregate_meta.get("analysis_plan_sha256") != EXPECTED_PLAN_SHA256:
        raise ValueError("publication release does not use the frozen analysis plan")
    if aggregate_meta.get("analysis_plan_version") != EXPECTED_PLAN_VERSION:
        raise ValueError("publication release has a stale analysis-plan version")
    if int(aggregate_meta.get("n_independent_runs", -1)) != 10:
        raise ValueError("publication aggregate must contain 10 independent runs")

    aggregate = release / str(aggregate_meta["relative_path"])
    aggregate_manifest = _read_json(aggregate / "manifest.json")
    if aggregate_manifest.get("analysis_plan_sha256") != EXPECTED_PLAN_SHA256:
        raise ValueError("aggregate manifest is not bound to the frozen plan")
    if aggregate_manifest.get("publication_release_id") != release.name:
        raise ValueError("aggregate is not bound to this publication release")

    rebuilt, diagnostics = _rebuild_primary_runs(release, release_manifest)
    recorded_runs = pd.read_csv(aggregate / "primary_run_deltas.csv")
    _assert_frame_close(rebuilt[recorded_runs.columns], recorded_runs, ["task_family", "run"])

    overall = pd.read_csv(aggregate / "primary_analysis.csv")
    if len(overall) != 1:
        raise ValueError("primary_analysis.csv must contain exactly one row")
    row = overall.iloc[0]
    values = rebuilt["loss_delta_vs_exact_uniform"].to_numpy(dtype=float)
    accuracy = rebuilt["acc_delta_vs_exact_uniform"].to_numpy(dtype=float)
    tasks = rebuilt["task_family"].astype(str).tolist()
    point = task_stratified_mean(values, tasks)
    ci_low, ci_high = task_stratified_bootstrap_interval(
        values,
        tasks,
        confidence=float(row["confidence"]),
        n_resamples=int(row["cluster_bootstrap_resamples"]),
        seed_parts=("primary", PRIMARY_SCALING_MODE, PRIMARY_ALLOCATION_RULE),
    )
    p_value = task_stratified_sign_flip_p(values, tasks)
    independent_p = _task_stratified_exact_p(values, tasks)
    task_means = rebuilt.groupby("task_family")["loss_delta_vs_exact_uniform"].mean()
    expected = {
        "mean_loss_delta": point,
        "cluster_bootstrap_ci_low": ci_low,
        "cluster_bootstrap_ci_high": ci_high,
        "exact_sign_flip_p_two_sided": p_value,
        "run_wins_loss": int(np.sum(values < 0)),
        "task_wins_loss": int(np.sum(task_means.to_numpy() < 0)),
        "mean_accuracy_delta": task_stratified_mean(accuracy, tasks),
    }
    for key, expected_value in expected.items():
        observed = float(row[key])
        if not math.isclose(observed, float(expected_value), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"primary {key}={observed}; recomputed {expected_value}")
    if not math.isclose(p_value, independent_p, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("released and independent exact sign-flip calculations disagree")
    if row["cluster_bootstrap_seed_scheme"] != TASK_STRATIFIED_BOOTSTRAP_SEED_SCHEME:
        raise ValueError("primary bootstrap seed scheme is stale")

    provenance = {
        "source_release_id": release.name,
        "source_archive_sha256": archive_sha256,
        "analysis_plan_sha256": EXPECTED_PLAN_SHA256,
        "analysis_plan_version": EXPECTED_PLAN_VERSION,
        "protocol_version": EXPECTED_PROTOCOL,
    }
    output_records: dict[str, dict[str, object]] = {}
    if write_outputs:
        TABLE_ROOT.mkdir(parents=True, exist_ok=True)

    def write_csv(frame: pd.DataFrame, name: str) -> None:
        if not write_outputs:
            return
        output = frame.copy()
        for key, value in reversed(list(provenance.items())):
            output.insert(0, key, value)
        path = TABLE_ROOT / name
        output.to_csv(path, index=False)
        output_records[name] = {
            "relative_path": path.relative_to(PAPER).as_posix(),
            "sha256": _sha256(path),
            "rows": int(len(output)),
        }

    write_csv(overall, "primary_analysis.csv")
    write_csv(
        pd.read_csv(aggregate / "primary_analysis_by_task.csv"),
        "primary_analysis_by_task.csv",
    )
    write_csv(rebuilt, "primary_run_deltas.csv")
    run_cluster = pd.read_csv(aggregate / "run_cluster_deltas_exact_cost.csv")
    write_csv(_summarize_scaling(run_cluster), "scaling_sensitivity.csv")
    site = pd.read_csv(aggregate / "site_prediction_summary.csv")
    site = site[
        site["scaling_mode"].astype(str).eq(PRIMARY_SCALING_MODE)
        & site["target"].astype(str).eq("near_best_rank_gap_0.1")
        & site["predictor"].astype(str).isin(["effective_rank", "soft_dimension"])
    ].sort_values(["task_family", "predictor"])
    if len(site) != 4:
        raise ValueError("expected four fixed-scale site-prediction summary rows")
    write_csv(site, "site_prediction_selected.csv")

    source_record = {
        **provenance,
        **diagnostics,
        "n_recursive_checksums": n_checksums,
        "source_release_manifest_sha256": _sha256(release / "release_manifest.json"),
        "source_release_checksum_manifest_sha256": _sha256(
            release / "SHA256SUMS.txt"
        ),
        "source_aggregate_manifest_sha256": _sha256(aggregate / "manifest.json"),
        "source_primary_analysis_sha256": _sha256(aggregate / "primary_analysis.csv"),
        "primary_mean_loss_delta": point,
        "primary_cluster_bootstrap_ci": [ci_low, ci_high],
        "primary_exact_sign_flip_p_two_sided": p_value,
        "primary_run_wins": int(np.sum(values < 0)),
        "primary_task_wins": int(np.sum(task_means.to_numpy() < 0)),
        "outputs": output_records,
        "importer": Path(__file__).name,
    }
    if write_outputs:
        (TABLE_ROOT / "source.json").write_text(
            json.dumps(source_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return source_record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a validated transformer publication release into the paper."
    )
    parser.add_argument(
        "--release",
        type=Path,
        default=None,
        help="Release directory. Default: Code/transformer/runs/transformer_releases/LATEST.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify and recompute the release without modifying paper artifacts.",
    )
    args = parser.parse_args()
    release = _resolve_release(args.release)
    summary = import_release(release, write_outputs=not args.check_only)
    print(
        "Transformer manuscript import check: PASS"
        if args.check_only
        else "Transformer manuscript import: PASS"
    )
    print(f"source_release_id: {summary['source_release_id']}")
    print(f"source_archive_sha256: {summary['source_archive_sha256']}")
    print(f"primary_mean_loss_delta: {summary['primary_mean_loss_delta']:.12f}")
    lo, hi = summary["primary_cluster_bootstrap_ci"]
    print(f"primary_cluster_bootstrap_ci: [{lo:.12f}, {hi:.12f}]")
    print(
        "primary_exact_sign_flip_p_two_sided: "
        f"{summary['primary_exact_sign_flip_p_two_sided']:.12f}"
    )


if __name__ == "__main__":
    main()
