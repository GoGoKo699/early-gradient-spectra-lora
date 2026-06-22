#!/usr/bin/env python3
"""Validate one controlled real-model LoRA source run."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_protocol import (  # noqa: E402
    ADAPTER_INIT_PROTOCOL_VERSION,
    ALLOCATION_PROTOCOL_VERSION,
    REAL_PROTOCOL_VERSION,
    RNG_PROTOCOL_VERSION,
    RUN_MANIFEST_VERSION,
    assignment_sha256,
    bank_slice,
    canonical_json_sha256,
    float_sequence_sha256,
    seed_manifest,
    sha256_file,
    tensor_sha256,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_number(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {value!r}")
    return number


def verify_checksum_manifest(run_dir: Path) -> int:
    manifest = run_dir / "RUN_SHA256SUMS.txt"
    require(manifest.is_file(), "missing RUN_SHA256SUMS.txt")
    seen: set[str] = set()
    count = 0
    for line_number, raw in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        parts = raw.split("  ", 1)
        require(len(parts) == 2, f"invalid checksum line {line_number}")
        expected, relative = parts
        require(relative not in seen, f"duplicate checksum path {relative}")
        seen.add(relative)
        path = run_dir / relative
        require(path.is_file(), f"checksummed file is missing: {relative}")
        actual = sha256_file(path)
        require(
            actual == expected,
            f"checksum mismatch for {relative}: {actual}; expected {expected}",
        )
        count += 1
    actual_files = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path != manifest
    }
    require(
        seen == actual_files,
        "run checksum manifest does not exactly cover run files: "
        f"missing={sorted(actual_files - seen)}, stale={sorted(seen - actual_files)}",
    )
    return count


def load_bank(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch.
        payload = torch.load(path, map_location="cpu")
    require(isinstance(payload, dict), "adapter bank payload must be a dictionary")
    require(
        payload.get("protocol_version") == ADAPTER_INIT_PROTOCOL_VERSION,
        "adapter bank protocol version mismatch",
    )
    tensors = payload.get("tensors")
    require(isinstance(tensors, dict), "adapter bank tensors must be a dictionary")
    require(
        all(isinstance(value, torch.Tensor) for value in tensors.values()),
        "adapter bank contains a non-tensor value",
    )
    return tensors


def validate_run(run_dir: Path, expected_kind: str | None = None) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    require(run_dir.is_dir(), f"run directory does not exist: {run_dir}")
    checksum_count = verify_checksum_manifest(run_dir)

    config = read_json(run_dir / "config.json")
    require(config.get("schema_version") == RUN_MANIFEST_VERSION, "run schema mismatch")
    require(config.get("protocol_version") == REAL_PROTOCOL_VERSION, "protocol mismatch")
    require(config.get("rng_protocol_version") == RNG_PROTOCOL_VERSION, "RNG protocol mismatch")
    require(
        config.get("adapter_init_protocol_version") == ADAPTER_INIT_PROTOCOL_VERSION,
        "adapter initialization protocol mismatch",
    )
    require(
        config.get("allocation_protocol_version") == ALLOCATION_PROTOCOL_VERSION,
        "allocation protocol mismatch",
    )
    if expected_kind is not None:
        require(config.get("run_kind") == expected_kind, "run kind mismatch")
    require(config.get("calibration_model_mode") == "eval_with_gradients", "calibration was not eval-mode")
    require(bool(config.get("matched_strategy_batch_order")), "batch-order CRN flag is false")
    require(bool(config.get("nested_adapter_initialization")), "nested-init flag is false")
    require(bool(config.get("exact_parameter_cost")), "exact-cost flag is false")

    seeds = read_json(run_dir / "seed_manifest.json")
    expected_seeds = seed_manifest(int(config["seed"]))
    require(seeds == expected_seeds, "named seed manifest does not recompute")
    require(
        config.get("seed_manifest_sha256") == canonical_json_sha256(seeds),
        "seed manifest hash mismatch",
    )

    bank = load_bank(run_dir / "adapter_init_bank.pt")
    init_manifest = read_json(run_dir / "adapter_init_manifest.json")
    require(
        init_manifest.get("protocol_version") == ADAPTER_INIT_PROTOCOL_VERSION,
        "adapter init manifest protocol mismatch",
    )
    init_rows = init_manifest.get("modules")
    require(isinstance(init_rows, list) and init_rows, "adapter init manifest has no modules")
    require(
        init_manifest.get("content_sha256") == canonical_json_sha256(init_rows),
        "adapter init manifest content hash mismatch",
    )
    init_names = {str(row["module"]) for row in init_rows}
    require(init_names == set(bank), "adapter bank module set mismatch")
    for row in init_rows:
        name = str(row["module"])
        tensor = bank[name]
        require(list(tensor.shape) == row["shape"], f"adapter shape mismatch for {name}")
        require(tensor_sha256(tensor) == row["sha256"], f"adapter tensor hash mismatch for {name}")

    metric_rows = read_csv(run_dir / "calibration_metrics.csv")
    require(metric_rows, "calibration_metrics.csv is empty")
    metric_by_module = {row["name"]: row for row in metric_rows}
    require(set(metric_by_module) == set(bank), "calibration module set mismatch")
    for module, row in metric_by_module.items():
        require(int(row["calibration_batches"]) >= 1, f"no calibration batches for {module}")
        for field in [
            "raw_frobenius",
            "raw_effective_rank",
            "gradient_sensitivity",
            "fim_b_mean_square",
            "whitened_effective_rank",
            "activation_effective_rank",
            "cov_trace",
            "cov_condition_est",
        ]:
            value = finite_number(row[field], f"{module}.{field}")
            require(value >= 0, f"{module}.{field} is negative")

    batch_rows = read_csv(run_dir / "calibration_batch_metrics.csv")
    require(batch_rows, "calibration_batch_metrics.csv is empty")
    require(
        all(row["calibration_model_mode"] == "eval" for row in batch_rows),
        "calibration batch was recorded outside eval mode",
    )
    require({row["module"] for row in batch_rows} == set(bank), "calibration batch module set mismatch")

    component_rows = read_csv(run_dir / "activation_components.csv")
    require(component_rows, "activation_components.csv is empty")
    require({row["module"] for row in component_rows} == set(bank), "activation component module set mismatch")
    for row in component_rows:
        value = finite_number(row["explained_variance"], "explained_variance")
        require(0 <= value <= 1 + 1e-12, "activation explained variance outside [0, 1]")

    allocation_rows = read_csv(run_dir / "allocations.csv")
    require(allocation_rows, "allocations.csv is empty")
    expected_strategies = list(config["output_strategies"])
    allocation_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in allocation_rows:
        allocation_groups[row["strategy"]].append(row)
    require(set(allocation_groups) == set(expected_strategies), "allocation strategy set mismatch")
    target_budget = int(config["target_parameter_budget"])
    ranks_by_strategy: dict[str, dict[str, int]] = {}
    assignment_by_strategy: dict[str, str] = {}
    for strategy in expected_strategies:
        rows = allocation_groups[strategy]
        require({row["module"] for row in rows} == set(bank), f"allocation module set mismatch for {strategy}")
        ranks: dict[str, int] = {}
        realized = 0
        row_assignment_hashes = set()
        for row in rows:
            module = row["module"]
            rank = int(row["rank"])
            require(int(config["min_rank"]) <= rank <= int(config["max_rank"]), f"rank out of bounds for {strategy}/{module}")
            cost = int(row["cost_per_rank"])
            require(int(row["params"]) == rank * cost, f"parameter arithmetic mismatch for {strategy}/{module}")
            require(int(row["budget_residual"]) == 0, f"nonzero budget residual for {strategy}/{module}")
            require(
                row["init_slice_sha256"] == tensor_sha256(bank_slice(bank, module, rank)),
                f"initialization slice hash mismatch for {strategy}/{module}",
            )
            ranks[module] = rank
            realized += rank * cost
            row_assignment_hashes.add(row["assignment_sha256"])
        require(realized == target_budget, f"strategy {strategy} realized {realized}, expected {target_budget}")
        require(len(row_assignment_hashes) == 1, f"multiple assignment hashes for {strategy}")
        expected_hash = assignment_sha256(ranks, bank)
        require(row_assignment_hashes == {expected_hash}, f"assignment hash mismatch for {strategy}")
        require(config["assignment_sha256"][strategy] == expected_hash, f"config assignment hash mismatch for {strategy}")
        ranks_by_strategy[strategy] = ranks
        assignment_by_strategy[strategy] = expected_hash

    status = str(config.get("status"))
    checks = read_json(run_dir / "protocol_checks.json")
    require(bool(checks.get("exact_parameter_cost")), "protocol exact-cost check is false")
    require(bool(checks.get("calibration_model_mode_eval")), "protocol eval-calibration check is false")
    require(bool(checks.get("named_rng_streams")), "protocol named-RNG check is false")
    require(bool(checks.get("nested_adapter_initialization")), "protocol nested-init check is false")

    n_results = 0
    if status == "complete":
        result_rows = read_csv(run_dir / "results.csv")
        n_results = len(result_rows)
        require(n_results == len(expected_strategies), "result row count mismatch")
        result_by_strategy = {row["strategy"]: row for row in result_rows}
        require(set(result_by_strategy) == set(expected_strategies), "result strategy set mismatch")
        initial_losses = []
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for strategy, row in result_by_strategy.items():
            require(int(row["trainable_params"]) == target_budget, f"result budget mismatch for {strategy}")
            require(int(row["budget_residual"]) == 0, f"result budget residual for {strategy}")
            require(row["assignment_sha256"] == assignment_by_strategy[strategy], f"result assignment hash mismatch for {strategy}")
            for field in ["initial_val_loss", "final_val_loss", "val_loss_delta", "perplexity"]:
                finite_number(row[field], f"{strategy}.{field}")
            initial_losses.append(float(row["initial_val_loss"]))
            history = read_csv(run_dir / f"train_history_{strategy}.csv")
            require(len(history) == int(config["steps"]), f"history length mismatch for {strategy}")
            trace = float_sequence_sha256([float(item["train_loss"]) for item in history])
            require(trace == row["train_trace_sha256"], f"training trace hash mismatch for {strategy}")
            groups[row["assignment_sha256"]].append(row)
        tolerance = float(config["identity_tolerance"])
        require(max(initial_losses) - min(initial_losses) <= tolerance, "initial-loss identity check failed")
        for assignment_hash, rows in groups.items():
            if len(rows) < 2:
                continue
            reference = rows[0]
            for row in rows[1:]:
                require(row["train_trace_sha256"] == reference["train_trace_sha256"], f"identical assignment trace mismatch for {assignment_hash}")
                for field in ["initial_val_loss", "final_val_loss", "val_loss_delta"]:
                    require(abs(float(row[field]) - float(reference[field])) <= tolerance, f"identical assignment metric mismatch for {assignment_hash}/{field}")
        require(bool(checks.get("training_completed")), "training completion check is false")
        require(bool(checks.get("initial_loss_identity")), "initial-loss check is false")
        require(bool(checks.get("identical_assignment_identity")), "identical-assignment check is false")
    elif status == "complete_calibration_only":
        require(not (run_dir / "results.csv").exists(), "calibration-only run unexpectedly has results.csv")
        require(checks.get("training_completed") is False, "calibration-only training flag mismatch")
    else:
        raise ValueError(f"run status is not complete: {status!r}")

    summary = {
        "run_id": config["run_id"],
        "run_kind": config["run_kind"],
        "status": status,
        "protocol_version": config["protocol_version"],
        "n_targets": len(bank),
        "n_strategies": len(expected_strategies),
        "n_results": n_results,
        "target_parameter_budget": target_budget,
        "n_checksummed_files": checksum_count,
    }
    print("Real LoRA run validation: PASS")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--expected-kind", choices=["exploratory", "smoke", "publication"], default=None
    )
    args = parser.parse_args()
    validate_run(args.run_dir, expected_kind=args.expected_kind)


if __name__ == "__main__":
    main()
