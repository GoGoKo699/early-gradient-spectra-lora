"""Deterministic protocol helpers for the real-model LoRA validation."""
from __future__ import annotations

import hashlib
import json
import math
import random
from functools import reduce
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

REAL_PROTOCOL_VERSION = "real_lora_publication_protocol_v3"
RNG_PROTOCOL_VERSION = "named_sha256_streams_v1"
ADAPTER_INIT_PROTOCOL_VERSION = "nested_max_rank_lora_a_v1"
# Backward-compatible alias for any local notebooks created during development.
INIT_PROTOCOL_VERSION = ADAPTER_INIT_PROTOCOL_VERSION
ALLOCATION_PROTOCOL_VERSION = "exact_parameter_cost_dp_v1"
ADAPTER_STATE_PROTOCOL_VERSION = "real_lora_adapter_state_v1"
RUN_MANIFEST_VERSION = "real_lora_run_manifest_v2"
RELEASE_MANIFEST_VERSION = "real_lora_release_manifest_v2"

STREAM_NAMES = (
    "model_load",
    "adapter_bank",
    "calibration_data",
    "training_data",
    "evaluation_data",
    "training_dropout",
    "evaluation",
)


def derive_seed(base_seed: int, stream: str, *parts: object) -> int:
    """Derive a portable positive 31-bit seed from a named stream."""
    payload = "|".join([str(int(base_seed)), str(stream), *(str(p) for p in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    # Fits Python, NumPy, CPU torch, CUDA and ROCm seed APIs.
    return int.from_bytes(digest[:8], "big") % 2_147_483_647


def seed_manifest(base_seed: int) -> dict[str, object]:
    return {
        "schema_version": RNG_PROTOCOL_VERSION,
        "base_seed": int(base_seed),
        "streams": {name: derive_seed(base_seed, name) for name in STREAM_NAMES},
    }


def set_global_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256_bytes(payload)


def float_sequence_sha256(values: Sequence[float]) -> str:
    array = np.asarray([float(value) for value in values], dtype="<f8")
    return sha256_bytes(array.tobytes(order="C"))


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def tensor_mapping_sha256(values: Mapping[str, torch.Tensor]) -> str:
    """Hash a named tensor mapping independently of serialization format."""
    digest = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"tensor mapping value for {name!r} is not a tensor")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor_sha256(tensor).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_initial_adapter_state(
    *,
    ranks: Mapping[str, int],
    bank: Mapping[str, torch.Tensor],
    module_classes: Mapping[str, str],
    output_dims: Mapping[str, int],
) -> dict[str, torch.Tensor]:
    """Reconstruct the exact zero-B adapter state used before optimization."""
    names = sorted(ranks)
    if set(names) != set(module_classes) or set(names) != set(output_dims):
        raise ValueError("rank, class and output-dimension keys must match")
    state: dict[str, torch.Tensor] = {}
    for name in names:
        rank = int(ranks[name])
        canonical_a = bank_slice(bank, name, rank).to(dtype=torch.float32)
        out_dim = int(output_dims[name])
        if out_dim < 1:
            raise ValueError(f"output dimension for {name} must be >= 1")
        module_class = str(module_classes[name])
        if module_class == "Linear":
            state[f"{name}.lora_A"] = canonical_a.clone()
            state[f"{name}.lora_B"] = torch.zeros(
                (out_dim, rank), dtype=torch.float32
            )
        elif module_class == "Conv1D":
            state[f"{name}.lora_A"] = canonical_a.t().contiguous()
            state[f"{name}.lora_B"] = torch.zeros(
                (rank, out_dim), dtype=torch.float32
            )
        else:
            raise ValueError(f"unsupported adapter module class {module_class!r}")
    return state


def adapter_activity_metrics(
    *,
    initial_state: Mapping[str, torch.Tensor],
    final_state: Mapping[str, torch.Tensor],
    module_classes: Mapping[str, str],
    alpha_scale: float,
) -> dict[str, object]:
    """Compute auditable adapter movement and effective-update diagnostics."""
    if set(initial_state) != set(final_state):
        raise ValueError("initial and final adapter state keys must match")
    if not math.isfinite(float(alpha_scale)):
        raise ValueError("alpha_scale must be finite")

    total_sq = 0.0
    a_sq = 0.0
    b_sq = 0.0
    changed = 0
    parameter_count = 0
    for name in sorted(initial_state):
        initial = initial_state[name].detach().to(device="cpu", dtype=torch.float64)
        final = final_state[name].detach().to(device="cpu", dtype=torch.float64)
        if initial.shape != final.shape:
            raise ValueError(f"adapter state shape mismatch for {name}")
        delta = final - initial
        value = float(delta.square().sum().item())
        total_sq += value
        if name.endswith(".lora_A"):
            a_sq += value
        elif name.endswith(".lora_B"):
            b_sq += value
        else:
            raise ValueError(f"unexpected adapter parameter name {name!r}")
        changed += int(torch.count_nonzero(delta).item())
        parameter_count += int(final.numel())

    effective_sq = 0.0
    per_module: list[dict[str, object]] = []
    for module in sorted(module_classes):
        a_name = f"{module}.lora_A"
        b_name = f"{module}.lora_B"
        if a_name not in final_state or b_name not in final_state:
            raise ValueError(f"adapter state is missing tensors for {module}")
        a = final_state[a_name].detach().to(device="cpu", dtype=torch.float64)
        b = final_state[b_name].detach().to(device="cpu", dtype=torch.float64)
        module_class = str(module_classes[module])
        if module_class == "Linear":
            update = b @ a
        elif module_class == "Conv1D":
            update = a @ b
        else:
            raise ValueError(f"unsupported adapter module class {module_class!r}")
        update = update * float(alpha_scale)
        frobenius = float(torch.linalg.norm(update).item())
        effective_sq += frobenius * frobenius
        per_module.append(
            {
                "module": module,
                "module_class": module_class,
                "effective_update_frobenius": frobenius,
            }
        )

    return {
        "initial_adapter_state_sha256": tensor_mapping_sha256(initial_state),
        "final_adapter_state_sha256": tensor_mapping_sha256(final_state),
        "adapter_parameter_count": parameter_count,
        "changed_parameter_count": changed,
        "adapter_parameter_delta_l2": math.sqrt(total_sq),
        "lora_a_parameter_delta_l2": math.sqrt(a_sq),
        "lora_b_parameter_delta_l2": math.sqrt(b_sq),
        "effective_update_frobenius_l2": math.sqrt(effective_sq),
        "per_module": per_module,
    }


def build_nested_init_bank(
    input_dims: Mapping[str, int],
    *,
    max_rank: int,
    init_std: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Build one canonical ``(max_rank, in_dim)`` LoRA-A matrix per module.

    Each module receives its own seed derived from the adapter-bank stream and
    module name. Consequently, changing target enumeration order or another
    module's assigned rank cannot change a module's initialization.
    """
    if max_rank < 1:
        raise ValueError("max_rank must be >= 1")
    if not math.isfinite(init_std) or init_std <= 0:
        raise ValueError("init_std must be finite and > 0")
    bank: dict[str, torch.Tensor] = {}
    for name in sorted(input_dims):
        in_dim = int(input_dims[name])
        if in_dim < 1:
            raise ValueError(f"input dimension for {name} must be >= 1")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(derive_seed(seed, "module", name))
        bank[name] = torch.randn(
            (int(max_rank), in_dim), generator=generator, dtype=torch.float32
        ) * float(init_std)
    return bank


def bank_slice(bank: Mapping[str, torch.Tensor], name: str, rank: int) -> torch.Tensor:
    if name not in bank:
        raise KeyError(f"adapter initialization bank has no module {name!r}")
    value = bank[name]
    if value.ndim != 2:
        raise ValueError(f"adapter bank entry for {name} must be 2D")
    if rank < 1 or rank > int(value.shape[0]):
        raise ValueError(
            f"rank {rank} outside [1, {int(value.shape[0])}] for adapter bank module {name}"
        )
    return value[: int(rank)].clone()


def assignment_sha256(
    ranks: Mapping[str, int], bank: Mapping[str, torch.Tensor]
) -> str:
    digest = hashlib.sha256()
    for name in sorted(ranks):
        rank = int(ranks[name])
        digest.update(f"{name}\0{rank}\0".encode("utf-8"))
        digest.update(tensor_sha256(bank_slice(bank, name, rank)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def scalar_marginal_utilities(
    scores: Mapping[str, float], max_rank: int
) -> dict[str, list[float]]:
    """Convert one module score into diminishing per-rank utilities."""
    values = {
        name: max(float(score), 0.0) if math.isfinite(float(score)) else 0.0
        for name, score in scores.items()
    }
    scale = max(values.values(), default=0.0)
    if scale <= 0:
        return {name: [0.0] * int(max_rank) for name in values}
    return {
        name: [
            (values[name] / scale) / math.sqrt(float(rank))
            for rank in range(1, int(max_rank) + 1)
        ]
        for name in values
    }


def component_marginal_utilities(
    components: Mapping[str, Sequence[float]], max_rank: int
) -> dict[str, list[float]]:
    """Normalize module-specific component utilities for global allocation."""
    rows: dict[str, list[float]] = {}
    scale = 0.0
    for name, sequence in components.items():
        row = []
        for value in list(sequence)[: int(max_rank)]:
            number = float(value)
            row.append(max(number, 0.0) if math.isfinite(number) else 0.0)
        row.extend([0.0] * (int(max_rank) - len(row)))
        rows[name] = row
        scale = max(scale, max(row, default=0.0))
    if scale > 0:
        rows = {name: [value / scale for value in row] for name, row in rows.items()}
    return rows


def _gcd(values: Sequence[int]) -> int:
    positives = [abs(int(value)) for value in values if int(value) != 0]
    return reduce(math.gcd, positives) if positives else 1


def allocate_exact_budget(
    *,
    costs: Mapping[str, int],
    marginal_utilities: Mapping[str, Sequence[float]],
    total_budget: int,
    min_rank: int,
    max_rank: int,
    reference_rank: int,
) -> dict[str, int]:
    """Deterministic exact-cost bounded multiple-choice knapsack allocator.

    The target budget must be exactly reachable. Ties first prefer allocations
    closest to the uniform reference rank and then the lexicographically
    smallest rank tuple in sorted module-name order.
    """
    names = sorted(costs)
    if not names:
        raise ValueError("at least one target module is required")
    if min_rank < 1:
        raise ValueError("min_rank must be >= 1")
    if max_rank < min_rank:
        raise ValueError("max_rank must be >= min_rank")
    if not min_rank <= reference_rank <= max_rank:
        raise ValueError("reference_rank must be inside [min_rank, max_rank]")
    if set(marginal_utilities) != set(names):
        raise ValueError("utility keys must exactly match cost keys")

    checked_costs = {name: int(costs[name]) for name in names}
    if any(cost <= 0 for cost in checked_costs.values()):
        raise ValueError("all per-rank costs must be > 0")
    minimum = sum(checked_costs[name] * min_rank for name in names)
    maximum = sum(checked_costs[name] * max_rank for name in names)
    total_budget = int(total_budget)
    if total_budget < minimum:
        raise ValueError(
            f"total_budget={total_budget} is infeasible: minimum-rank floor costs {minimum}"
        )
    if total_budget > maximum:
        raise ValueError(
            f"total_budget={total_budget} is infeasible: maximum ranks cost {maximum}"
        )

    gcd = _gcd(list(checked_costs.values()))
    if total_budget % gcd:
        raise ValueError(
            f"total_budget={total_budget} is not divisible by per-rank cost gcd={gcd}"
        )
    target = total_budget // gcd
    unit_costs = {name: checked_costs[name] // gcd for name in names}

    cumulative: dict[str, list[float]] = {}
    for name in names:
        row = [float(value) for value in marginal_utilities[name]]
        if len(row) < max_rank:
            raise ValueError(f"utilities for {name} have length {len(row)} < {max_rank}")
        if any(not math.isfinite(value) or value < 0 for value in row[:max_rank]):
            raise ValueError(f"utilities for {name} must be finite and nonnegative")
        prefix = [0.0]
        for value in row[:max_rank]:
            prefix.append(prefix[-1] + value)
        cumulative[name] = prefix

    # cost -> (objective, distance from uniform reference, rank tuple)
    states: dict[int, tuple[float, int, tuple[int, ...]]] = {0: (0.0, 0, ())}
    epsilon = 1e-12
    for name in names:
        new_states: dict[int, tuple[float, int, tuple[int, ...]]] = {}
        for prior_cost, (prior_value, prior_distance, prior_tuple) in states.items():
            for rank in range(min_rank, max_rank + 1):
                cost = prior_cost + unit_costs[name] * rank
                if cost > target:
                    continue
                candidate = (
                    prior_value + cumulative[name][rank],
                    prior_distance + abs(rank - reference_rank),
                    prior_tuple + (rank,),
                )
                incumbent = new_states.get(cost)
                if incumbent is None:
                    new_states[cost] = candidate
                    continue
                better_value = candidate[0] > incumbent[0] + epsilon
                tied_value = abs(candidate[0] - incumbent[0]) <= epsilon
                better_tie = tied_value and (
                    candidate[1] < incumbent[1]
                    or (candidate[1] == incumbent[1] and candidate[2] < incumbent[2])
                )
                if better_value or better_tie:
                    new_states[cost] = candidate
        states = new_states
        if not states:
            raise ValueError(f"no feasible allocation remains after processing {name}")

    if target not in states:
        raise ValueError(
            f"no exact rank allocation realizes total_budget={total_budget} under these costs"
        )
    rank_tuple = states[target][2]
    ranks = {name: int(rank_tuple[index]) for index, name in enumerate(names)}
    realized = sum(checked_costs[name] * ranks[name] for name in names)
    if realized != total_budget:
        raise AssertionError(f"allocator realized {realized}, expected {total_budget}")
    return ranks
