#!/usr/bin/env python3
"""Import the checksum-bound real-model LoRA publication release.

The importer accepts only the frozen 8+3 source-run GPT-2/Wikitext-2 release,
verifies the archive and recursive manifests, reconstructs all paired effects
from source-run CSVs, recomputes the pre-specified inference, and writes compact
paper-facing tables with immutable provenance columns.

Run from any directory:

    python Paper/import_real_lora_results.py
    python Paper/import_real_lora_results.py --check-only

The optional ``--release-dir`` and ``--archive`` arguments are intended for
independent audit copies.  The default paths point to the local project release.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
CODE = ROOT / "Code" / "real_lora_validation"
TABLE_ROOT = PAPER / "tables" / "real_lora_publication"

RELEASE_ID = "real_lora_publication_20260623T074520Z"
RELEASE_KIND = "publication"
RELEASE_SCHEMA = "real_lora_publication_release_v1"
PROTOCOL_VERSION = "real_lora_publication_protocol_v3"
PLAN_VERSION = "real_lora_publication_plan_v1"
PLAN_SHA256 = "e407ba9f34946cfaa6a1246ebd55247b4fdc208d22de5bc1bd279d0c63a7da7b"
ARCHIVE_SHA256 = "4368ca32afd32fde04c68e1b889bcba2add57a801d8360db4e99557953c991de"
SOURCE_GIT_COMMIT = "9d8547ca172ffcc8e059351b8d91a4fc92d36b7e"
ROOT_CHECKSUM_COUNT = 354
AGGREGATE_CHECKSUM_COUNT = 9
PRIMARY_SEEDS = (101, 103, 107, 109, 113, 127, 131, 137)
BOUNDARY_SEEDS = (101, 103, 107)
EXPECTED_UNITS = {
    *(("cattn_cfc", seed) for seed in PRIMARY_SEEDS),
    *(("attnproj", seed) for seed in BOUNDARY_SEEDS),
}
CANDIDATES = (
    "gradient_norm",
    "spectral_effective",
    "eva_activation",
    "fim_gradient_variance",
    "gora_sensitivity",
)
ALL_STRATEGIES = (
    "uniform_r4",
    *CANDIDATES,
    "uniform_identity_control",
)
PAPER_STRATEGIES = (
    "uniform_r4",
    "spectral_effective",
    "eva_activation",
    "gora_sensitivity",
    "gradient_norm",
    "fim_gradient_variance",
)
PROVENANCE = {
    "source_release_id": RELEASE_ID,
    "source_archive_sha256": ARCHIVE_SHA256,
    "analysis_plan_sha256": PLAN_SHA256,
    "analysis_plan_version": PLAN_VERSION,
    "protocol_version": PROTOCOL_VERSION,
    "source_git_commit": SOURCE_GIT_COMMIT,
}
OUTPUT_FILENAMES = (
    "primary_analysis.csv",
    "primary_seed_deltas.csv",
    "analysis_by_suite_strategy.csv",
    "strategy_summary.csv",
    "spectral_seed_deltas.csv",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def safe_relative(value: str, label: str) -> Path:
    pure = PurePosixPath(str(value))
    require(not pure.is_absolute(), f"unsafe absolute {label}: {value!r}")
    require(bool(pure.parts) and "." not in pure.parts and ".." not in pure.parts,
            f"unsafe {label}: {value!r}")
    return Path(*pure.parts)


def verify_checksum_file(directory: Path, filename: str, expected_count: int | None = None) -> int:
    manifest = directory / filename
    require(manifest.is_file(), f"missing checksum manifest: {manifest}")
    recorded: dict[str, str] = {}
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        require(len(parts) == 2, f"invalid {filename} line {number}")
        digest, relative = parts
        require(relative not in recorded, f"duplicate checksum path: {relative}")
        path = directory / safe_relative(relative, f"{filename} path")
        require(path.is_file(), f"checksummed file missing: {path}")
        require(sha256_file(path) == digest, f"checksum mismatch: {path}")
        recorded[relative] = digest
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path != manifest
    }
    require(set(recorded) == actual,
            f"{filename} coverage mismatch: missing={sorted(actual-set(recorded))}, "
            f"stale={sorted(set(recorded)-actual)}")
    if expected_count is not None:
        require(len(recorded) == expected_count,
                f"{filename} has {len(recorded)} entries; expected {expected_count}")
    return len(recorded)


def verify_archive(archive: Path, release_dir: Path) -> None:
    require(archive.is_file(), f"missing publication archive: {archive}")
    require(sha256_file(archive) == ARCHIVE_SHA256,
            f"archive SHA-256 does not match frozen release: {archive}")
    sidecar = Path(str(archive) + ".sha256")
    require(sidecar.is_file(), f"missing archive sidecar: {sidecar}")
    tokens = sidecar.read_text(encoding="utf-8").strip().split()
    require(len(tokens) >= 2 and tokens[0] == ARCHIVE_SHA256 and tokens[-1] == archive.name,
            f"invalid archive sidecar: {sidecar}")

    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        require(bool(members), "publication archive is empty")
        expected_prefix = f"{RELEASE_ID}/"
        names: set[str] = set()
        for member in members:
            pure = PurePosixPath(member.name)
            require(not pure.is_absolute() and ".." not in pure.parts,
                    f"unsafe archive member: {member.name}")
            require(not member.issym() and not member.islnk(),
                    f"archive links are forbidden: {member.name}")
            require(member.name == RELEASE_ID or member.name.startswith(expected_prefix),
                    f"unexpected archive top level: {member.name}")
            names.add(member.name)
        manifest_name = f"{RELEASE_ID}/SHA256SUMS.txt"
        require(manifest_name in names, "archive lacks SHA256SUMS.txt")
        archived = handle.extractfile(manifest_name)
        require(archived is not None, "cannot read archived SHA256SUMS.txt")
        require(archived.read() == (release_dir / "SHA256SUMS.txt").read_bytes(),
                "archive and release directory have different root manifests")


def all_boolean_checks_true(value: Any, prefix: str = "") -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            failures.extend(all_boolean_checks_true(child, path))
    elif isinstance(value, bool) and not value:
        failures.append(prefix)
    return failures


def exact_sign_flip_p(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    require(array.ndim == 1 and array.size > 0 and np.isfinite(array).all(),
            "invalid exact sign-flip values")
    nonzero = array[array != 0.0]
    if nonzero.size == 0:
        return 1.0
    observed = abs(float(nonzero.mean()))
    count = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=nonzero.size):
        statistic = abs(float(np.mean(nonzero * np.asarray(signs))))
        count += statistic >= observed - 1e-15
        total += 1
    return float(count / total)


def stable_seed(plan_sha256: str, label: str) -> int:
    payload = f"{plan_sha256}|{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def summarize(values: Sequence[float], *, confidence: float, n_resamples: int, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    require(array.ndim == 1 and array.size > 0 and np.isfinite(array).all(),
            "invalid paired values")
    mean = float(array.mean())
    if array.size > 1:
        sd = float(array.std(ddof=1))
        sem = sd / math.sqrt(array.size)
        critical = float(student_t.ppf((1.0 + confidence) / 2.0, df=array.size - 1))
        t_low, t_high = mean - critical * sem, mean + critical * sem
    else:
        sd = sem = t_low = t_high = float("nan")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(n_resamples), array.size))
    means = array[indices].mean(axis=1)
    alpha = 1.0 - confidence
    boot_low, boot_high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    n_nonzero = int(np.count_nonzero(array))
    return {
        "n_independent_runs": int(array.size),
        "n_nonzero_deltas": n_nonzero,
        "mean_loss_delta": mean,
        "median_loss_delta": float(np.median(array)),
        "sd_loss_delta": sd,
        "sem_loss_delta": sem,
        "t_ci_low": float(t_low),
        "t_ci_high": float(t_high),
        "bootstrap_ci_low": float(boot_low),
        "bootstrap_ci_high": float(boot_high),
        "exact_sign_flip_p_two_sided": exact_sign_flip_p(array),
        "wins": int(np.sum(array < 0)),
        "ties": int(np.sum(array == 0)),
        "losses": int(np.sum(array > 0)),
        "planned_no_tie_min_two_sided_p": float(2.0 / (2**array.size)),
        "attainable_min_two_sided_p": float(2.0 / (2**n_nonzero)) if n_nonzero else 1.0,
    }


def holm_adjust(values: Sequence[float]) -> list[float]:
    p_values = [float(value) for value in values]
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def compare_frames(actual: pd.DataFrame, expected: pd.DataFrame, *, keys: Sequence[str], label: str) -> None:
    actual = actual.sort_values(list(keys)).reset_index(drop=True)
    expected = expected.sort_values(list(keys)).reset_index(drop=True)
    require(actual.shape == expected.shape,
            f"{label} shape mismatch: {actual.shape} != {expected.shape}")
    require(list(actual.columns) == list(expected.columns),
            f"{label} column mismatch")
    for column in actual.columns:
        left, right = actual[column], expected[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            require(np.allclose(left.astype(float), right.astype(float),
                                rtol=0.0, atol=5e-12, equal_nan=True),
                    f"{label}.{column} mismatch")
        else:
            require(left.astype(str).tolist() == right.astype(str).tolist(),
                    f"{label}.{column} mismatch")


def add_provenance(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column, value in PROVENANCE.items():
        result[column] = value
    return result


def validate_and_build(release_dir: Path, archive: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    require(release_dir.is_dir(), f"missing release directory: {release_dir}")
    verify_archive(archive, release_dir)
    n_root = verify_checksum_file(release_dir, "SHA256SUMS.txt", ROOT_CHECKSUM_COUNT)
    n_aggregate = verify_checksum_file(
        release_dir / "aggregate", "AGGREGATE_SHA256SUMS.txt", AGGREGATE_CHECKSUM_COUNT
    )

    manifest = read_json(release_dir / "release_manifest.json")
    expected_manifest = {
        "release_id": RELEASE_ID,
        "release_kind": RELEASE_KIND,
        "schema_version": RELEASE_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "analysis_plan_version": PLAN_VERSION,
        "analysis_plan_sha256": PLAN_SHA256,
        "source_git_commit": SOURCE_GIT_COMMIT,
        "git_commit": SOURCE_GIT_COMMIT,
        "n_source_runs": 11,
    }
    for key, expected in expected_manifest.items():
        require(manifest.get(key) == expected,
                f"release_manifest {key}={manifest.get(key)!r}; expected {expected!r}")

    plan = read_json(release_dir / "analysis_plan.json")
    require(sha256_file(release_dir / "analysis_plan.json") == PLAN_SHA256,
            "analysis-plan file hash mismatch")
    require(plan.get("plan_version") == PLAN_VERSION, "analysis-plan version mismatch")
    require(plan.get("protocol_version") == PROTOCOL_VERSION, "analysis protocol mismatch")
    analysis = plan.get("analysis")
    require(isinstance(analysis, dict), "analysis plan lacks analysis object")
    require(analysis.get("primary_suite") == "cattn_cfc", "wrong primary suite")
    require(analysis.get("primary_candidate_strategy") == "spectral_effective",
            "wrong primary candidate")
    require(analysis.get("primary_reference_requested_strategy") == "uniform",
            "wrong primary reference")
    require(int(analysis.get("bootstrap_resamples", 0)) == 10000,
            "wrong bootstrap count")

    entries = manifest.get("source_runs")
    require(isinstance(entries, list) and len(entries) == 11,
            "release must contain exactly 11 source runs")

    units: set[tuple[str, int]] = set()
    pair_rows: list[dict[str, Any]] = []
    result_frames: list[pd.DataFrame] = []
    allocation_frames: list[pd.DataFrame] = []
    input_hashes: set[str] = set()
    run_checksum_counts: list[int] = []

    for entry in entries:
        require(isinstance(entry, dict), "source-run entry must be an object")
        run_dir = release_dir / safe_relative(str(entry["path"]), "source-run path")
        run_checksum_counts.append(verify_checksum_file(run_dir, "RUN_SHA256SUMS.txt"))
        config = read_json(run_dir / "config.json")
        checks = read_json(run_dir / "protocol_checks.json")
        results = pd.read_csv(run_dir / "results.csv")
        allocations = pd.read_csv(run_dir / "allocations.csv")
        provenance = read_json(run_dir / "input_provenance.json")
        input_hashes.add(sha256_file(run_dir / "input_provenance.json"))

        suite = str(config["suite_id"])
        seed = int(config["seed"])
        units.add((suite, seed))
        require(str(config["run_id"]) == str(entry["run_id"]), "run ID mismatch")
        require(config.get("suite_role") == ("primary" if suite == "cattn_cfc" else "boundary"),
                f"wrong suite role for {suite}")
        require(config.get("protocol_version") == PROTOCOL_VERSION, "stale run protocol")
        require(config.get("publication_plan_sha256") == PLAN_SHA256, "run-plan mismatch")
        require(config.get("source_git_commit") == SOURCE_GIT_COMMIT, "run Git mismatch")
        require(config.get("status") == "complete", "incomplete source run")
        require(config.get("run_kind") == "publication", "non-publication source run")
        require(config.get("deterministic_algorithms") is True, "determinism disabled")
        require(config.get("local_files_only") is True, "source run was not local-only")
        require(provenance.get("provenance_sha256") ==
                canonical_json_sha256({k: v for k, v in provenance.items() if k != "provenance_sha256"}),
                "input provenance self-hash mismatch")

        failures = all_boolean_checks_true(checks)
        require(not failures, f"false protocol checks in {run_dir.name}: {failures}")
        require(set(results["strategy"].astype(str)) == set(ALL_STRATEGIES),
                f"wrong strategy set in {run_dir.name}")
        require(len(results) == len(ALL_STRATEGIES), f"duplicate strategies in {run_dir.name}")
        require(np.isfinite(results.select_dtypes(include=np.number).to_numpy()).all(),
                f"nonfinite result in {run_dir.name}")
        require((pd.to_numeric(results["trainable_params"]) ==
                 int(config["target_parameter_budget"])).all(),
                f"parameter-cost mismatch in {run_dir.name}")
        for column in (
            "first_step_lora_b_gradient_l2",
            "adapter_parameter_delta_l2",
            "lora_b_parameter_delta_l2",
            "effective_update_frobenius_l2",
            "changed_parameter_count",
        ):
            require((pd.to_numeric(results[column]) > 0).all(),
                    f"inactive adapter field {column} in {run_dir.name}")
        for column in (
            "model_load_seed", "training_data_seed", "training_dropout_seed",
            "evaluation_data_seed", "evaluation_seed", "initial_val_loss",
        ):
            require(results[column].nunique(dropna=False) == 1,
                    f"paired-stream mismatch in {run_dir.name}: {column}")

        grouped = allocations.groupby("strategy", sort=False)
        require(set(grouped["params"].sum().astype(int)) == {int(config["target_parameter_budget"])},
                f"allocation sums are not exact in {run_dir.name}")
        require((pd.to_numeric(allocations["budget_residual"]) == 0).all(),
                f"nonzero allocation residual in {run_dir.name}")
        require(allocations["rank"].between(int(config["min_rank"]), int(config["max_rank"])).all(),
                f"rank outside frozen grid in {run_dir.name}")

        indexed = results.set_index("strategy", drop=False)
        uniform = indexed.loc["uniform_r4"]
        identity = indexed.loc["uniform_identity_control"]
        identity_columns = (
            "initial_val_loss", "final_val_loss", "val_loss_delta", "perplexity",
            "assignment_sha256", "train_trace_sha256", "n_train_steps",
            "model_load_seed", "training_data_seed", "training_dropout_seed",
            "evaluation_data_seed", "evaluation_seed", "initial_adapter_state_sha256",
            "final_adapter_state_sha256", "adapter_parameter_count",
            "changed_parameter_count", "first_step_adapter_gradient_l2",
            "first_step_lora_b_gradient_l2", "adapter_parameter_delta_l2",
            "lora_a_parameter_delta_l2", "lora_b_parameter_delta_l2",
            "effective_update_frobenius_l2", "adapter_activity_passed",
            "trainable_params", "budget_residual", "rank_min", "rank_mean", "rank_max",
        )
        for column in identity_columns:
            require(str(uniform[column]) == str(identity[column]),
                    f"uniform identity control differs in {run_dir.name}: {column}")

        for candidate in CANDIDATES:
            row = indexed.loc[candidate]
            require(str(row["assignment_sha256"]) != str(uniform["assignment_sha256"]),
                    f"candidate allocation equals uniform in publication run: {run_dir.name}/{candidate}")
            pair_rows.append({
                "run_id": str(config["run_id"]),
                "suite_id": suite,
                "suite_role": str(config["suite_role"]),
                "seed": seed,
                "candidate_strategy": candidate,
                "reference_strategy": "uniform_r4",
                "candidate_minus_reference_final_val_loss": float(row["final_val_loss"]) - float(uniform["final_val_loss"]),
                "candidate_minus_reference_val_loss_delta": float(row["val_loss_delta"]) - float(uniform["val_loss_delta"]),
                "candidate_minus_reference_perplexity": float(row["perplexity"]) - float(uniform["perplexity"]),
                "candidate_trainable_params": int(row["trainable_params"]),
                "reference_trainable_params": int(uniform["trainable_params"]),
                "parameter_difference": int(row["trainable_params"]) - int(uniform["trainable_params"]),
                "candidate_assignment_sha256": str(row["assignment_sha256"]),
                "reference_assignment_sha256": str(uniform["assignment_sha256"]),
                "assignments_identical": False,
                "candidate_adapter_parameter_delta_l2": float(row["adapter_parameter_delta_l2"]),
                "reference_adapter_parameter_delta_l2": float(uniform["adapter_parameter_delta_l2"]),
                "candidate_effective_update_frobenius_l2": float(row["effective_update_frobenius_l2"]),
                "reference_effective_update_frobenius_l2": float(uniform["effective_update_frobenius_l2"]),
                "candidate_final_adapter_state_sha256": str(row["final_adapter_state_sha256"]),
                "reference_final_adapter_state_sha256": str(uniform["final_adapter_state_sha256"]),
            })

        result_copy = results.copy()
        result_copy.insert(0, "reference_strategy", "uniform_r4")
        result_copy.insert(0, "seed", seed)
        result_copy.insert(0, "suite_role", str(config["suite_role"]))
        result_copy.insert(0, "suite_id", suite)
        result_copy.insert(0, "run_id", str(config["run_id"]))
        result_frames.append(result_copy)

        allocation_copy = allocations.copy()
        allocation_copy.insert(0, "seed", seed)
        allocation_copy.insert(0, "suite_role", str(config["suite_role"]))
        allocation_copy.insert(0, "suite_id", suite)
        allocation_copy.insert(0, "run_id", str(config["run_id"]))
        allocation_frames.append(allocation_copy)

    require(units == EXPECTED_UNITS,
            f"source-unit set differs from frozen design: {sorted(units ^ EXPECTED_UNITS)}")
    require(len(input_hashes) == 1,
            f"source runs use mixed input provenance: {sorted(input_hashes)}")

    pairs = pd.DataFrame(pair_rows).sort_values(
        ["suite_id", "candidate_strategy", "seed"]
    ).reset_index(drop=True)
    released_pairs = pd.read_csv(release_dir / "aggregate" / "paired_deltas.csv")
    compare_frames(pairs, released_pairs,
                   keys=("suite_id", "candidate_strategy", "seed"),
                   label="paired_deltas")

    all_results = pd.concat(result_frames, ignore_index=True)
    released_results = pd.read_csv(release_dir / "aggregate" / "all_results.csv")
    compare_frames(all_results, released_results,
                   keys=("run_id", "strategy"), label="all_results")
    all_allocations = pd.concat(allocation_frames, ignore_index=True)
    released_allocations = pd.read_csv(release_dir / "aggregate" / "all_allocations.csv")
    compare_frames(all_allocations, released_allocations,
                   keys=("run_id", "strategy", "module"), label="all_allocations")

    confidence = float(analysis["confidence"])
    n_resamples = int(analysis["bootstrap_resamples"])
    summary_rows: list[dict[str, Any]] = []
    for (suite, candidate), group in pairs.groupby(
        ["suite_id", "candidate_strategy"], sort=True
    ):
        values = group["candidate_minus_reference_final_val_loss"].to_numpy(float)
        seed = stable_seed(PLAN_SHA256, f"suite={suite}|candidate={candidate}")
        stats = summarize(values, confidence=confidence, n_resamples=n_resamples, seed=seed)
        summary_rows.append({
            "suite_id": suite,
            "suite_role": str(group["suite_role"].iloc[0]),
            "candidate_strategy": candidate,
            "reference_strategy": "uniform_r4",
            **stats,
            "exact_test_resolution_sufficient": bool(
                float(stats["attainable_min_two_sided_p"]) <= float(analysis["two_sided_alpha"])
            ),
            "distinct_assignment_runs": int((~group["assignments_identical"].astype(bool)).sum()),
            "identical_assignment_runs": int(group["assignments_identical"].astype(bool).sum()),
            "confidence": confidence,
            "bootstrap_resamples": n_resamples,
            "bootstrap_seed": seed,
            "bootstrap_seed_scheme": "sha256(plan_sha256|label)_first8_mod_2^63_v1",
        })
    summary = pd.DataFrame(summary_rows)
    summary["holm_p_within_suite"] = np.nan
    for suite, indices in summary.groupby("suite_id", sort=False).groups.items():
        index_list = list(indices)
        adjusted = holm_adjust(summary.loc[index_list, "exact_sign_flip_p_two_sided"])
        summary.loc[index_list, "holm_p_within_suite"] = adjusted
    released_summary = pd.read_csv(
        release_dir / "aggregate" / "analysis_by_suite_strategy.csv"
    )
    compare_frames(summary[released_summary.columns], released_summary,
                   keys=("suite_id", "candidate_strategy"),
                   label="analysis_by_suite_strategy")

    primary = summary[
        summary["suite_id"].eq("cattn_cfc")
        & summary["candidate_strategy"].eq("spectral_effective")
    ].copy()
    require(len(primary) == 1, "primary summary is not unique")
    released_primary = pd.read_csv(release_dir / "aggregate" / "primary_analysis.csv")
    for column in released_primary.columns:
        if column in primary.columns:
            left = primary.iloc[0][column]
            right = released_primary.iloc[0][column]
            if isinstance(left, (float, np.floating, int, np.integer)):
                require(
                    (pd.isna(left) and pd.isna(right))
                    or math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=5e-12),
                    f"primary_analysis.{column} mismatch",
                )
            else:
                require(str(left) == str(right), f"primary_analysis.{column} mismatch")

    primary_seeds = pairs[
        pairs["suite_id"].eq("cattn_cfc")
        & pairs["candidate_strategy"].eq("spectral_effective")
    ].copy()
    released_primary_seeds = pd.read_csv(
        release_dir / "aggregate" / "primary_seed_deltas.csv"
    )
    compare_frames(primary_seeds[released_primary_seeds.columns], released_primary_seeds,
                   keys=("seed",), label="primary_seed_deltas")

    main_results = all_results[all_results["strategy"].isin(PAPER_STRATEGIES)].copy()
    winner = main_results.loc[
        main_results.groupby(["suite_id", "seed"])["final_val_loss"].idxmin(),
        ["suite_id", "seed", "strategy"],
    ].rename(columns={"strategy": "winner_strategy"})
    main_results = main_results.merge(winner, on=["suite_id", "seed"], how="left")
    strategy_summary = (
        main_results.groupby(["suite_id", "suite_role", "strategy"], sort=True)
        .agg(
            n=("seed", "size"),
            mean_initial_val_loss=("initial_val_loss", "mean"),
            mean_final_val_loss=("final_val_loss", "mean"),
            sd_final_val_loss=("final_val_loss", "std"),
            mean_val_loss_delta=("val_loss_delta", "mean"),
            sd_val_loss_delta=("val_loss_delta", "std"),
            mean_perplexity=("perplexity", "mean"),
            sd_perplexity=("perplexity", "std"),
            mean_trainable_params=("trainable_params", "mean"),
            min_rank=("rank_min", "min"),
            mean_rank=("rank_mean", "mean"),
            max_rank=("rank_max", "max"),
            best_strategy_wins=("winner_strategy", lambda s: int((s == s.name).sum())),
        )
        .reset_index()
    )
    # GroupBy's lambda does not expose the strategy key portably; compute wins explicitly.
    wins = (
        winner.groupby(["suite_id", "winner_strategy"]).size()
        .rename("best_strategy_wins").reset_index()
        .rename(columns={"winner_strategy": "strategy"})
    )
    strategy_summary = strategy_summary.drop(columns=["best_strategy_wins"]).merge(
        wins, on=["suite_id", "strategy"], how="left"
    )
    strategy_summary["best_strategy_wins"] = (
        strategy_summary["best_strategy_wins"].fillna(0).astype(int)
    )
    for base in ("final_val_loss", "val_loss_delta", "perplexity"):
        strategy_summary[f"sem_{base}"] = (
            strategy_summary[f"sd_{base}"] / np.sqrt(strategy_summary["n"])
        )
    strategy_summary = strategy_summary[
        [
            "suite_id", "suite_role", "strategy", "n",
            "mean_initial_val_loss", "mean_final_val_loss", "sem_final_val_loss",
            "mean_val_loss_delta", "sem_val_loss_delta",
            "mean_perplexity", "sem_perplexity", "mean_trainable_params",
            "min_rank", "mean_rank", "max_rank", "best_strategy_wins",
        ]
    ]

    spectral_seed_deltas = pairs[pairs["candidate_strategy"].eq("spectral_effective")].copy()

    frames = {
        "primary_analysis.csv": add_provenance(released_primary),
        "primary_seed_deltas.csv": add_provenance(released_primary_seeds),
        "analysis_by_suite_strategy.csv": add_provenance(released_summary),
        "strategy_summary.csv": add_provenance(strategy_summary),
        "spectral_seed_deltas.csv": add_provenance(spectral_seed_deltas),
    }

    primary_row = released_primary.iloc[0]
    source = {
        "source_release_id": RELEASE_ID,
        "source_archive_sha256": ARCHIVE_SHA256,
        "analysis_plan_sha256": PLAN_SHA256,
        "analysis_plan_version": PLAN_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "source_git_commit": SOURCE_GIT_COMMIT,
        "release_schema_version": RELEASE_SCHEMA,
        "n_recursive_checksums": n_root,
        "n_aggregate_checksums": n_aggregate,
        "n_source_runs": len(entries),
        "n_run_checksum_entries": int(sum(run_checksum_counts)),
        "n_primary_runs": len(PRIMARY_SEEDS),
        "n_boundary_runs": len(BOUNDARY_SEEDS),
        "n_paired_candidate_rows": len(pairs),
        "n_result_rows": len(all_results),
        "n_allocation_rows": len(all_allocations),
        "identity_control_runs": len(entries),
        "exact_cost_runs": len(entries),
        "activity_gate_runs": len(entries),
        "primary_mean_loss_delta": float(primary_row["mean_loss_delta"]),
        "primary_bootstrap_ci": [
            float(primary_row["bootstrap_ci_low"]),
            float(primary_row["bootstrap_ci_high"]),
        ],
        "primary_exact_sign_flip_p_two_sided": float(
            primary_row["exact_sign_flip_p_two_sided"]
        ),
        "primary_holm_p_within_suite": float(primary_row["holm_p_within_suite"]),
        "primary_wins": int(primary_row["wins"]),
        "primary_ties": int(primary_row["ties"]),
        "primary_losses": int(primary_row["losses"]),
        "boundary_spectral_mean_loss_delta": float(
            released_summary[
                released_summary["suite_id"].eq("attnproj")
                & released_summary["candidate_strategy"].eq("spectral_effective")
            ].iloc[0]["mean_loss_delta"]
        ),
        "importer": "import_real_lora_results.py",
    }
    return frames, source


def frame_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def source_with_outputs(source: dict[str, Any], frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    result = dict(source)
    outputs: dict[str, Any] = {}
    for filename in OUTPUT_FILENAMES:
        payload = frame_bytes(frames[filename])
        outputs[filename] = {
            "relative_path": (TABLE_ROOT / filename).relative_to(PAPER).as_posix(),
            "rows": int(len(frames[filename])),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    result["outputs"] = outputs
    return result


def write_outputs(frames: dict[str, pd.DataFrame], source: dict[str, Any]) -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    expected = set(OUTPUT_FILENAMES) | {"source.json"}
    for path in TABLE_ROOT.iterdir() if TABLE_ROOT.exists() else []:
        if path.is_file() and path.name not in expected:
            raise ValueError(f"unexpected file in real-model paper table directory: {path}")
    for filename in OUTPUT_FILENAMES:
        (TABLE_ROOT / filename).write_bytes(frame_bytes(frames[filename]))
    final_source = source_with_outputs(source, frames)
    (TABLE_ROOT / "source.json").write_text(
        json.dumps(final_source, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def check_outputs(frames: dict[str, pd.DataFrame], source: dict[str, Any]) -> None:
    expected_source = source_with_outputs(source, frames)
    source_path = TABLE_ROOT / "source.json"
    require(source_path.is_file(), f"missing paper provenance: {source_path}")
    require(read_json(source_path) == expected_source,
            "paper real-model source.json is stale or inconsistent")
    expected_names = set(OUTPUT_FILENAMES) | {"source.json"}
    observed_names = {path.name for path in TABLE_ROOT.iterdir() if path.is_file()}
    require(observed_names == expected_names,
            f"paper real-model table set mismatch: {observed_names ^ expected_names}")
    for filename in OUTPUT_FILENAMES:
        path = TABLE_ROOT / filename
        require(path.is_file(), f"missing paper real-model table: {path}")
        require(path.read_bytes() == frame_bytes(frames[filename]),
                f"paper real-model table differs from verified release: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_release = CODE / "runs" / "real_lora_releases" / RELEASE_ID
    parser.add_argument("--release-dir", type=Path, default=default_release)
    parser.add_argument("--archive", type=Path, default=Path(str(default_release) + ".tar.gz"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    frames, source = validate_and_build(args.release_dir.resolve(), args.archive.resolve())
    if args.check_only:
        check_outputs(frames, source)
        print("Real-model manuscript import check: PASS")
    else:
        write_outputs(frames, source)
        print("Real-model manuscript import: PASS")
        print(f"release_id: {RELEASE_ID}")
        print(f"archive_sha256: {ARCHIVE_SHA256}")
        print(f"paper_tables: {TABLE_ROOT}")


if __name__ == "__main__":
    main()
