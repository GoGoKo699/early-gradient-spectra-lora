#!/usr/bin/env python3
"""Controlled real-model LoRA rank-allocation validation.

The protocol compares exact-parameter-cost rank allocations on one frozen
pretrained causal language model. Calibration runs in evaluation mode with
gradients enabled, every strategy receives the same named random streams and
nested maximum-rank LoRA initialization, and identical allocations must produce
identical metrics within a declared numerical tolerance.

The default strategy set preserves the historical three-way invocation. The
publication smoke script explicitly adds allocation-only activation-spectrum,
FIM-gradient, and GoRA-sensitivity controls. Those controls use a common random
LoRA initialization so that this experiment tests rank allocation rather than
each external method's full initialization/training recipe.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from real_protocol import (
    ADAPTER_INIT_PROTOCOL_VERSION,
    ADAPTER_STATE_PROTOCOL_VERSION,
    ALLOCATION_PROTOCOL_VERSION,
    REAL_PROTOCOL_VERSION,
    RNG_PROTOCOL_VERSION,
    RUN_MANIFEST_VERSION,
    adapter_activity_metrics,
    allocate_exact_budget,
    assignment_sha256,
    bank_slice,
    build_nested_init_bank,
    canonical_json_sha256,
    component_marginal_utilities,
    float_sequence_sha256,
    scalar_marginal_utilities,
    seed_manifest,
    set_global_seed,
    sha256_file,
    tensor_mapping_sha256,
    tensor_sha256,
)
from spectral_metrics import (
    EFFECTIVE_RANK_DEFINITION,
    entropy_effective_rank_from_energy,
    singular_effective_rank,
)

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - only used when datasets is unavailable.
    load_dataset = None


BUILTIN_TEXT = """
Low-rank adaptation changes a pretrained model through a small number of trainable
matrix directions. A layer with a large gradient norm may want to change, but the
singular spectrum of the early gradient can reveal how many independent directions
of change are present. Activation whitening removes geometry induced by common
input directions, leaving a spectrum closer to the useful task operator.

This small built-in corpus is only a smoke-test fallback. It is not publication
evidence. Use pinned local model and dataset inputs for a publication run.
""".strip()

ALLOWED_STRATEGIES = (
    "uniform",
    "gradient_norm",
    "spectral_effective",
    "eva_activation",
    "fim_gradient_variance",
    "gora_sensitivity",
)

STRATEGY_DESCRIPTIONS = {
    "uniform": "exact-cost uniform-rank reference",
    "gradient_norm": "allocation from early full-weight gradient Frobenius norm",
    "spectral_effective": "allocation from activation-whitened gradient effective rank",
    "eva_activation": (
        "allocation-only activation-spectrum control with common nested random LoRA "
        "initialization; not the full EVA initialization recipe"
    ),
    "fim_gradient_variance": (
        "allocation-only LoRA-B gradient-square control with common nested random "
        "LoRA initialization; not the full FIM-LoRA recipe"
    ),
    "gora_sensitivity": (
        "allocation-only mean-absolute weight-times-gradient sensitivity control with "
        "common nested random LoRA initialization; not the full GoRA recipe"
    ),
    "uniform_identity_control": "duplicate uniform allocation used as a CRN identity control",
}


@dataclass(frozen=True)
class TargetSpec:
    name: str
    module_class: str
    in_dim: int
    out_dim: int

    @property
    def cost_per_rank(self) -> int:
        return int(self.in_dim + self.out_dim)


@dataclass(frozen=True)
class SpectralMetrics:
    name: str
    module_class: str
    in_dim: int
    out_dim: int
    cost_per_rank: int
    tokens: int
    calibration_batches: int
    raw_frobenius: float
    raw_nuclear_norm: float
    raw_stable_rank: float
    raw_effective_rank: float
    gradient_sensitivity: float
    fim_b_mean_square: float
    whitened_frobenius: float
    whitened_stable_rank: float
    whitened_effective_rank: float
    activation_effective_rank: float
    activation_top_variance_ratio: float
    cov_trace: float
    cov_condition_est: float


class CovAccumulator:
    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self.cov_sum = torch.zeros((dim, dim), dtype=torch.float64, device="cpu")
        self.n = 0

    def add(self, x: torch.Tensor) -> None:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"activation dim mismatch: expected {self.dim}, got {x.shape[-1]}"
            )
        x_cpu = x.reshape(-1, self.dim).detach().to(device="cpu", dtype=torch.float64)
        if x_cpu.numel() == 0:
            return
        self.cov_sum.add_(x_cpu.T @ x_cpu)
        self.n += int(x_cpu.shape[0])

    def covariance(self) -> torch.Tensor:
        if self.n <= 0:
            raise RuntimeError("no activations were accumulated")
        return self.cov_sum / float(self.n)


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha_scale: float,
        canonical_a: torch.Tensor,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be >= 1")
        if tuple(canonical_a.shape) != (rank, base.in_features):
            raise ValueError(
                f"canonical A shape mismatch: expected {(rank, base.in_features)}, "
                f"got {tuple(canonical_a.shape)}"
            )
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.scaling = float(alpha_scale)
        self.lora_A = nn.Parameter(
            canonical_a.to(device=base.weight.device, dtype=torch.float32).clone()
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base.out_features,
                rank,
                dtype=torch.float32,
                device=base.weight.device,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        z = torch.matmul(x.to(torch.float32), self.lora_A.t())
        z = torch.matmul(z, self.lora_B.t()) * self.scaling
        return base_out + z.to(base_out.dtype)


class LoRAConv1D(nn.Module):
    """LoRA wrapper for HuggingFace GPT-style Conv1D.

    HF Conv1D stores weight as ``(in_dim, out_dim)``. The initialization bank
    always uses canonical ``(rank, in_dim)`` orientation.
    """

    def __init__(
        self,
        base: nn.Module,
        rank: int,
        alpha_scale: float,
        canonical_a: torch.Tensor,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be >= 1")
        if not hasattr(base, "weight") or base.weight.ndim != 2:
            raise TypeError("LoRAConv1D expects a module with 2D weight")
        in_dim = int(base.weight.shape[0])
        out_dim = int(base.weight.shape[1])
        if tuple(canonical_a.shape) != (rank, in_dim):
            raise ValueError(
                f"canonical A shape mismatch: expected {(rank, in_dim)}, "
                f"got {tuple(canonical_a.shape)}"
            )
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.scaling = float(alpha_scale)
        self.lora_A = nn.Parameter(
            canonical_a.t().to(device=base.weight.device, dtype=torch.float32).clone()
        )
        self.lora_B = nn.Parameter(
            torch.zeros(rank, out_dim, dtype=torch.float32, device=base.weight.device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        z = torch.matmul(x.to(torch.float32), self.lora_A)
        z = torch.matmul(z, self.lora_B) * self.scaling
        return base_out + z.to(base_out.dtype)


def set_seed(seed: int) -> None:
    """Backward-compatible local alias."""
    set_global_seed(seed)


def torch_dtype_from_name(name: str) -> torch.dtype:
    key = name.lower()
    if key in {"float32", "fp32", "32"}:
        return torch.float32
    if key in {"float16", "fp16", "16"}:
        return torch.float16
    if key in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"unknown dtype {name!r}")


def timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment_record(device: torch.device) -> dict[str, Any]:
    packages = [
        "torch",
        "numpy",
        "transformers",
        "datasets",
        "accelerate",
        "tokenizers",
        "safetensors",
        "huggingface_hub",
    ]
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in packages},
        "torch_version": torch.__version__,
        "torch_cuda_runtime": getattr(torch.version, "cuda", None),
        "torch_hip_runtime": getattr(torch.version, "hip", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if torch.cuda.is_available() and device.type == "cuda"
            else None
        ),
    }


def configure_determinism(enabled: bool) -> dict[str, Any]:
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(bool(enabled), warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(enabled)
        torch.backends.cudnn.benchmark = False
    return {
        "enabled": bool(enabled),
        "torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": (
            bool(torch.backends.cudnn.deterministic)
            if hasattr(torch.backends, "cudnn")
            else None
        ),
        "cudnn_benchmark": (
            bool(torch.backends.cudnn.benchmark)
            if hasattr(torch.backends, "cudnn")
            else None
        ),
    }


def read_local_text_file(path: str | None, limit: int) -> list[str]:
    if not path:
        raise ValueError(
            "local dataset mode requires --train_text_file and --val_text_file"
        )
    value = Path(path).expanduser().resolve()
    if not value.exists():
        raise FileNotFoundError(value)
    text = value.read_text(encoding="utf-8", errors="replace")
    parts = [line.strip() for line in text.splitlines() if line.strip()]
    if not parts and text.strip():
        parts = [text.strip()]
    if not parts:
        raise RuntimeError(f"no non-empty text found in {value}")
    return parts[:limit]


def load_text_corpus(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    if args.dataset_mode == "local":
        return (
            read_local_text_file(args.train_text_file, args.max_train_texts),
            read_local_text_file(args.val_text_file, args.max_val_texts),
        )

    if args.dataset_mode == "builtin":
        repeated = [
            BUILTIN_TEXT
            for _ in range(max(args.max_train_texts, args.max_val_texts, 8))
        ]
        return repeated[: args.max_train_texts], repeated[: args.max_val_texts]

    if load_dataset is None:
        raise RuntimeError(
            "datasets is not installed; use --dataset_mode builtin/local or install datasets"
        )

    kwargs: dict[str, Any] = {}
    if args.dataset_config:
        kwargs["name"] = args.dataset_config
    if args.dataset_revision:
        kwargs["revision"] = args.dataset_revision

    train_ds = load_dataset(args.dataset_name, **kwargs, split=args.train_split)
    val_ds = load_dataset(args.dataset_name, **kwargs, split=args.val_split)

    text_col = args.text_column
    if text_col is None:
        for candidate in ["text", "content", "sentence"]:
            if candidate in train_ds.column_names:
                text_col = candidate
                break
    if text_col is None:
        raise ValueError(
            f"could not infer text column from columns={train_ds.column_names}"
        )

    train_texts = [str(x) for x in train_ds[text_col] if str(x).strip()]
    val_texts = [str(x) for x in val_ds[text_col] if str(x).strip()]
    return train_texts[: args.max_train_texts], val_texts[: args.max_val_texts]


def encode_blocks(
    tokenizer: Any,
    texts: Iterable[str],
    block_size: int,
    max_blocks: int,
) -> torch.Tensor:
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.pad_token_id
    if eos_id is None:
        raise ValueError("tokenizer has neither eos_token_id nor pad_token_id")

    ids: list[int] = []
    target_tokens = max_blocks * block_size
    for text in texts:
        value = str(text).strip()
        if not value:
            continue
        ids.extend(tokenizer.encode(value, add_special_tokens=False))
        ids.append(int(eos_id))
        if len(ids) >= target_tokens + block_size:
            break

    total = (len(ids) // block_size) * block_size
    total = min(total, target_tokens)
    if total < block_size:
        raise RuntimeError(
            f"not enough tokens for one block: have {len(ids)}, need at least {block_size}; "
            "use more texts or a smaller --block_size"
        )
    return torch.tensor(ids[:total], dtype=torch.long).view(-1, block_size)


def make_loader(
    blocks: torch.Tensor, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return DataLoader(
        TensorDataset(blocks),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=generator,
    )


def infinite_loader(loader: DataLoader) -> Iterator[torch.Tensor]:
    while True:
        for (value,) in loader:
            yield value


def is_conv1d_module(module: nn.Module) -> bool:
    return (
        module.__class__.__name__ == "Conv1D"
        and hasattr(module, "weight")
        and module.weight.ndim == 2
    )


def is_targetable_module(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) or is_conv1d_module(module)


def module_dims(module: nn.Module) -> tuple[int, int]:
    if isinstance(module, nn.Linear):
        return int(module.in_features), int(module.out_features)
    if is_conv1d_module(module):
        return int(module.weight.shape[0]), int(module.weight.shape[1])
    raise TypeError(f"unsupported target module type {type(module)}")


def find_targets(
    model: nn.Module,
    suffixes: list[str],
    max_targets: int | None = None,
) -> list[TargetSpec]:
    suffixes = [suffix.strip() for suffix in suffixes if suffix.strip()]
    targets: list[TargetSpec] = []
    for name, module in model.named_modules():
        if not name:
            continue
        leaf = name.split(".")[-1]
        if suffixes and not (
            leaf in suffixes or any(name.endswith(suffix) for suffix in suffixes)
        ):
            continue
        if not is_targetable_module(module):
            continue
        in_dim, out_dim = module_dims(module)
        targets.append(
            TargetSpec(
                name=name,
                module_class=module.__class__.__name__,
                in_dim=in_dim,
                out_dim=out_dim,
            )
        )
    if max_targets is not None and max_targets > 0:
        targets = targets[:max_targets]
    if not targets:
        raise RuntimeError(f"no target modules found for suffixes={suffixes}")
    return targets


def get_submodule(model: nn.Module, name: str) -> nn.Module:
    current: nn.Module = model
    for part in name.split("."):
        current = current._modules[part]
    return current


def set_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    parent: nn.Module = model
    for part in parts[:-1]:
        parent = parent._modules[part]
    parent._modules[parts[-1]] = module


def oriented_weight_grad(module: nn.Module) -> torch.Tensor:
    if module.weight.grad is None:
        raise RuntimeError("target module has no weight gradient")
    gradient = module.weight.grad.detach()
    if isinstance(module, nn.Linear):
        return gradient.to(device="cpu", dtype=torch.float64)
    if is_conv1d_module(module):
        return gradient.t().to(device="cpu", dtype=torch.float64)
    raise TypeError(f"unsupported module type {type(module)}")


def oriented_weight(module: nn.Module) -> torch.Tensor:
    weight = module.weight.detach()
    if isinstance(module, nn.Linear):
        return weight.to(device="cpu", dtype=torch.float64)
    if is_conv1d_module(module):
        return weight.t().to(device="cpu", dtype=torch.float64)
    raise TypeError(f"unsupported module type {type(module)}")


def singular_stable_rank(singular_values: torch.Tensor, eps: float = 1e-12) -> float:
    values = singular_values.detach().to(dtype=torch.float64, device="cpu")
    if values.numel() == 0 or float(values[0]) <= eps:
        return 0.0
    return float((values.square().sum() / (values[0].square() + eps)).item())


def invsqrt_from_cov(
    covariance: torch.Tensor,
    ridge_scale: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, float]:
    covariance = covariance.to(dtype=torch.float64, device="cpu")
    dimension = covariance.shape[0]
    ridge = (
        float(ridge_scale)
        * float(torch.trace(covariance).item())
        / max(dimension, 1)
    )
    values, vectors = torch.linalg.eigh(
        covariance + ridge * torch.eye(dimension, dtype=torch.float64)
    )
    values = values.clamp_min(eps)
    condition = float((values.max() / values.min()).item())
    inverse_sqrt = (vectors * torch.rsqrt(values).unsqueeze(0)) @ vectors.t()
    return inverse_sqrt, condition


def calibrate_spectra(
    model: nn.Module,
    targets: list[TargetSpec],
    calib_loader: DataLoader,
    device: torch.device,
    calib_batches: int,
    ridge_scale: float,
    init_bank: Mapping[str, torch.Tensor],
    fisher_reference_rank: int,
    alpha_scale: float,
    max_rank: int,
) -> tuple[list[SpectralMetrics], dict[str, list[float]], list[dict[str, Any]]]:
    """Measure calibration statistics with dropout disabled and autograd enabled."""
    if calib_batches < 1:
        raise ValueError("calib_batches must be >= 1")
    if fisher_reference_rank < 1 or fisher_reference_rank > max_rank:
        raise ValueError("fisher_reference_rank must be inside [1, max_rank]")

    was_training = bool(model.training)
    original_requires_grad = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    original_use_cache = getattr(getattr(model, "config", None), "use_cache", None)
    model.eval()  # keep autograd; disable dropout and other training-time randomness.
    if hasattr(model, "config"):
        model.config.use_cache = False

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    modules = {target.name: get_submodule(model, target.name) for target in targets}
    for module in modules.values():
        module.weight.requires_grad_(True)

    covariances = {target.name: CovAccumulator(target.in_dim) for target in targets}
    gradient_sums = {
        target.name: torch.zeros(
            (target.out_dim, target.in_dim), dtype=torch.float64, device="cpu"
        )
        for target in targets
    }
    fisher_sums = {target.name: 0.0 for target in targets}
    batch_rows: list[dict[str, Any]] = []

    handles = []
    for target in targets:

        def hook(
            _module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            _output: torch.Tensor,
            name: str = target.name,
        ) -> None:
            if inputs:
                covariances[name].add(inputs[0])

        handles.append(modules[target.name].register_forward_hook(hook))

    actual_batches = 0
    try:
        model.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(calib_loader):
            if batch_index >= calib_batches:
                break
            values = batch[0].to(device, non_blocking=True)
            output = model(input_ids=values, labels=values)
            if not bool(torch.isfinite(output.loss.detach())):
                raise RuntimeError(f"non-finite calibration loss at batch {batch_index}")
            output.loss.backward()
            actual_batches += 1
            for target in targets:
                gradient = oriented_weight_grad(modules[target.name])
                gradient_sums[target.name].add_(gradient)
                reference_a = bank_slice(
                    init_bank, target.name, fisher_reference_rank
                ).to(dtype=torch.float64)
                # B is zero at initialization. Under delta W = scale * B A,
                # grad_B = scale * grad_W @ A^T.
                b_gradient = float(alpha_scale) * (gradient @ reference_a.t())
                fisher_value = float(b_gradient.square().mean().item())
                fisher_sums[target.name] += fisher_value
                batch_rows.append(
                    {
                        "batch_index": batch_index,
                        "module": target.name,
                        "loss": float(output.loss.detach().cpu().item()),
                        "raw_gradient_frobenius": float(
                            torch.linalg.norm(gradient).item()
                        ),
                        "fim_b_mean_square": fisher_value,
                        "calibration_model_mode": "eval",
                    }
                )
            model.zero_grad(set_to_none=True)
    finally:
        for handle in handles:
            handle.remove()
        named_parameters = dict(model.named_parameters())
        for name, requires_grad in original_requires_grad.items():
            if name in named_parameters:
                named_parameters[name].requires_grad_(requires_grad)
        if hasattr(model, "config") and original_use_cache is not None:
            model.config.use_cache = original_use_cache
        model.train(was_training)

    if actual_batches == 0:
        raise RuntimeError("calibration loader yielded no batches")

    metrics: list[SpectralMetrics] = []
    activation_components: dict[str, list[float]] = {}
    for target in targets:
        gradient = gradient_sums[target.name] / float(actual_batches)
        raw_singular = torch.linalg.svdvals(gradient)
        covariance = covariances[target.name].covariance()
        covariance_eigenvalues = (
            torch.linalg.eigvalsh(covariance).clamp_min(0.0).flip(0)
        )
        covariance_total = float(covariance_eigenvalues.sum().item())
        if covariance_total > 0:
            explained = covariance_eigenvalues / covariance_total
        else:
            explained = torch.zeros_like(covariance_eigenvalues)
        activation_components[target.name] = [
            float(value) for value in explained[: int(max_rank)].tolist()
        ]

        inverse_sqrt, condition = invsqrt_from_cov(
            covariance, ridge_scale=ridge_scale
        )
        whitened_gradient = gradient @ inverse_sqrt
        whitened_singular = torch.linalg.svdvals(whitened_gradient)
        base_weight = oriented_weight(modules[target.name])

        metrics.append(
            SpectralMetrics(
                name=target.name,
                module_class=target.module_class,
                in_dim=target.in_dim,
                out_dim=target.out_dim,
                cost_per_rank=target.cost_per_rank,
                tokens=covariances[target.name].n,
                calibration_batches=actual_batches,
                raw_frobenius=float(torch.linalg.norm(gradient).item()),
                raw_nuclear_norm=float(raw_singular.sum().item()),
                raw_stable_rank=singular_stable_rank(raw_singular),
                raw_effective_rank=singular_effective_rank(raw_singular),
                gradient_sensitivity=float(
                    (base_weight * gradient).abs().mean().item()
                ),
                fim_b_mean_square=float(
                    fisher_sums[target.name] / float(actual_batches)
                ),
                whitened_frobenius=float(
                    torch.linalg.norm(whitened_gradient).item()
                ),
                whitened_stable_rank=singular_stable_rank(whitened_singular),
                whitened_effective_rank=singular_effective_rank(
                    whitened_singular
                ),
                activation_effective_rank=entropy_effective_rank_from_energy(
                    covariance_eigenvalues
                ),
                activation_top_variance_ratio=(
                    float(explained[0].item()) if explained.numel() else 0.0
                ),
                cov_trace=float(torch.trace(covariance).item()),
                cov_condition_est=condition,
            )
        )
    return metrics, activation_components, batch_rows


def total_lora_params(
    ranks: Mapping[str, int], specs: Mapping[str, TargetSpec]
) -> int:
    return int(
        sum(int(ranks[name]) * specs[name].cost_per_rank for name in ranks)
    )


def allocate_uniform(
    metrics: list[SpectralMetrics], rank: int
) -> dict[str, int]:
    return {metric.name: int(rank) for metric in metrics}


def allocate_by_score(
    metrics: list[SpectralMetrics],
    score_key: str,
    total_budget: int,
    min_rank: int,
    max_rank: int,
    reference_rank: int,
) -> dict[str, int]:
    costs = {metric.name: int(metric.cost_per_rank) for metric in metrics}
    scores = {metric.name: float(getattr(metric, score_key)) for metric in metrics}
    return allocate_exact_budget(
        costs=costs,
        marginal_utilities=scalar_marginal_utilities(scores, max_rank=max_rank),
        total_budget=total_budget,
        min_rank=min_rank,
        max_rank=max_rank,
        reference_rank=reference_rank,
    )


def allocate_by_components(
    metrics: list[SpectralMetrics],
    components: Mapping[str, list[float]],
    total_budget: int,
    min_rank: int,
    max_rank: int,
    reference_rank: int,
) -> dict[str, int]:
    costs = {metric.name: int(metric.cost_per_rank) for metric in metrics}
    return allocate_exact_budget(
        costs=costs,
        marginal_utilities=component_marginal_utilities(
            components, max_rank=max_rank
        ),
        total_budget=total_budget,
        min_rank=min_rank,
        max_rank=max_rank,
        reference_rank=reference_rank,
    )


def parse_strategy_list(raw: str) -> list[str]:
    strategies: list[str] = []
    for value in raw.split(","):
        name = value.strip()
        if not name:
            continue
        if name not in ALLOWED_STRATEGIES:
            raise ValueError(
                f"unknown strategy {name!r}; allowed={list(ALLOWED_STRATEGIES)}"
            )
        if name not in strategies:
            strategies.append(name)
    if not strategies:
        raise ValueError("at least one allocation strategy is required")
    return strategies


def build_allocations(
    metrics: list[SpectralMetrics],
    activation_components: Mapping[str, list[float]],
    requested: list[str],
    uniform_rank: int,
    min_rank: int,
    max_rank: int,
    include_identity_control: bool,
) -> tuple[
    dict[str, dict[str, int]],
    dict[str, dict[str, str]],
    int,
]:
    specs = {
        metric.name: TargetSpec(
            name=metric.name,
            module_class=metric.module_class,
            in_dim=metric.in_dim,
            out_dim=metric.out_dim,
        )
        for metric in metrics
    }
    uniform = allocate_uniform(metrics, uniform_rank)
    total_budget = total_lora_params(uniform, specs)
    allocations: dict[str, dict[str, int]] = {}
    strategy_metadata: dict[str, dict[str, str]] = {}

    for requested_name in requested:
        if requested_name == "uniform":
            output_name = f"uniform_r{uniform_rank}"
            ranks = dict(uniform)
            score = "constant uniform rank"
        elif requested_name == "gradient_norm":
            output_name = requested_name
            ranks = allocate_by_score(
                metrics,
                "raw_frobenius",
                total_budget,
                min_rank,
                max_rank,
                uniform_rank,
            )
            score = "raw_frobenius"
        elif requested_name == "spectral_effective":
            output_name = requested_name
            ranks = allocate_by_score(
                metrics,
                "whitened_effective_rank",
                total_budget,
                min_rank,
                max_rank,
                uniform_rank,
            )
            score = "whitened_effective_rank"
        elif requested_name == "eva_activation":
            output_name = requested_name
            ranks = allocate_by_components(
                metrics,
                activation_components,
                total_budget,
                min_rank,
                max_rank,
                uniform_rank,
            )
            score = "activation_explained_variance_components"
        elif requested_name == "fim_gradient_variance":
            output_name = requested_name
            ranks = allocate_by_score(
                metrics,
                "fim_b_mean_square",
                total_budget,
                min_rank,
                max_rank,
                uniform_rank,
            )
            score = "fim_b_mean_square"
        elif requested_name == "gora_sensitivity":
            output_name = requested_name
            ranks = allocate_by_score(
                metrics,
                "gradient_sensitivity",
                total_budget,
                min_rank,
                max_rank,
                uniform_rank,
            )
            score = "mean_abs_weight_times_average_gradient"
        else:  # pragma: no cover - guarded by parse_strategy_list.
            raise AssertionError(requested_name)

        realized = total_lora_params(ranks, specs)
        if realized != total_budget:
            raise AssertionError(
                f"strategy {output_name} realized {realized}, expected {total_budget}"
            )
        allocations[output_name] = ranks
        strategy_metadata[output_name] = {
            "requested_strategy": requested_name,
            "allocation_score": score,
            "scope": STRATEGY_DESCRIPTIONS[requested_name],
        }

    if include_identity_control:
        name = "uniform_identity_control"
        allocations[name] = dict(uniform)
        strategy_metadata[name] = {
            "requested_strategy": name,
            "allocation_score": "constant uniform rank",
            "scope": STRATEGY_DESCRIPTIONS[name],
        }

    return allocations, strategy_metadata, total_budget


def load_model(
    model_name: str,
    dtype: torch.dtype,
    device: torch.device,
    trust_remote_code: bool,
    revision: str | None = None,
    local_files_only: bool = False,
) -> nn.Module:
    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype,
        "local_files_only": bool(local_files_only),
    }
    if revision:
        kwargs["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.config.use_cache = False
    return model.to(device)


def apply_lora(
    model: nn.Module,
    ranks: Mapping[str, int],
    alpha_scale: float,
    init_bank: Mapping[str, torch.Tensor],
) -> None:
    for name in sorted(ranks):
        rank = int(ranks[name])
        module = get_submodule(model, name)
        canonical_a = bank_slice(init_bank, name, rank)
        if isinstance(module, nn.Linear):
            wrapped = LoRALinear(
                module,
                rank=rank,
                alpha_scale=alpha_scale,
                canonical_a=canonical_a,
            )
        elif is_conv1d_module(module):
            wrapped = LoRAConv1D(
                module,
                rank=rank,
                alpha_scale=alpha_scale,
                canonical_a=canonical_a,
            )
        else:
            raise TypeError(f"unsupported LoRA module {name}: {type(module)}")
        set_submodule(model, name, wrapped)


def named_adapter_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if (name.endswith(".lora_A") or name.endswith(".lora_B"))
        and parameter.requires_grad
    }


def adapter_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Backward-compatible list view used by older local notebooks."""
    return list(named_adapter_parameters(model).values())


def clone_adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().to(device="cpu", dtype=torch.float32).clone()
        for name, parameter in named_adapter_parameters(model).items()
    }


def gradient_l2(
    parameters: Mapping[str, nn.Parameter], *, suffix: str | None = None
) -> float:
    total = 0.0
    found = False
    for name, parameter in parameters.items():
        if suffix is not None and not name.endswith(suffix):
            continue
        gradient = parameter.grad
        if gradient is None:
            continue
        found = True
        value = gradient.detach().to(device="cpu", dtype=torch.float64)
        total += float(value.square().sum().item())
    return math.sqrt(total) if found else 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eval_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= eval_batches:
            break
        values = batch[0].to(device, non_blocking=True)
        output = model(input_ids=values, labels=values)
        loss = float(output.loss.detach().cpu().item())
        if not math.isfinite(loss):
            raise RuntimeError(f"non-finite evaluation loss at batch {batch_index}")
        losses.append(loss)
    if not losses:
        raise RuntimeError("evaluation loader yielded no batches")
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)
    return float(np.mean(losses))


def train_strategy(
    args: argparse.Namespace,
    strategy_name: str,
    ranks: Mapping[str, int],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    run_dir: Path,
    init_bank: Mapping[str, torch.Tensor],
    streams: Mapping[str, int],
) -> dict[str, Any]:
    # Model-loading randomness is identical across strategies.
    set_global_seed(int(streams["model_load"]))
    model = load_model(
        args.model,
        dtype=dtype,
        device=device,
        trust_remote_code=args.trust_remote_code,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    apply_lora(
        model,
        ranks,
        alpha_scale=args.lora_alpha_scale,
        init_bank=init_bank,
    )
    named_parameters = named_adapter_parameters(model)
    if not named_parameters:
        raise RuntimeError("no adapter parameters found")
    parameters = list(named_parameters.values())
    initial_state = clone_adapter_state(model)
    initial_state_hash = tensor_mapping_sha256(initial_state)
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay
    )

    # Evaluation and training RNG are reset only after model, adapter and
    # optimizer construction, preventing allocation-dependent constructor draws
    # from shifting dropout masks.
    set_global_seed(int(streams["evaluation"]))
    initial_loss = evaluate(
        model, val_loader, device=device, eval_batches=args.eval_batches
    )

    set_global_seed(int(streams["training_dropout"]))
    model.train()
    stream = infinite_loader(train_loader)
    history_path = run_dir / f"train_history_{strategy_name}.csv"
    train_losses: list[float] = []
    first_step_adapter_gradient_l2: float | None = None
    first_step_lora_b_gradient_l2: float | None = None
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["step", "train_loss", "elapsed_sec"]
        )
        writer.writeheader()
        started = time.time()
        for step in range(1, args.steps + 1):
            values = next(stream).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(input_ids=values, labels=values)
            loss = output.loss
            if not bool(torch.isfinite(loss.detach())):
                raise RuntimeError(
                    f"non-finite training loss for {strategy_name} at step {step}"
                )
            loss.backward()
            if step == 1:
                first_step_adapter_gradient_l2 = gradient_l2(named_parameters)
                first_step_lora_b_gradient_l2 = gradient_l2(
                    named_parameters, suffix=".lora_B"
                )
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            train_losses.append(loss_value)
            writer.writerow(
                {
                    "step": step,
                    "train_loss": loss_value,
                    "elapsed_sec": time.time() - started,
                }
            )
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                handle.flush()
                print(
                    f"[{strategy_name}] step {step:5d}/{args.steps}: "
                    f"train_loss={loss_value:.6f}",
                    flush=True,
                )

    if first_step_adapter_gradient_l2 is None or first_step_lora_b_gradient_l2 is None:
        raise RuntimeError("first-step adapter gradients were not recorded")

    final_state = clone_adapter_state(model)
    module_classes: dict[str, str] = {}
    for module_name in ranks:
        wrapped = get_submodule(model, module_name)
        if isinstance(wrapped, LoRALinear):
            module_classes[module_name] = "Linear"
        elif isinstance(wrapped, LoRAConv1D):
            module_classes[module_name] = "Conv1D"
        else:  # pragma: no cover - apply_lora guarantees this.
            raise TypeError(f"unexpected wrapped adapter module {type(wrapped)}")
    activity = adapter_activity_metrics(
        initial_state=initial_state,
        final_state=final_state,
        module_classes=module_classes,
        alpha_scale=args.lora_alpha_scale,
    )
    if activity["initial_adapter_state_sha256"] != initial_state_hash:
        raise AssertionError("initial adapter state hash changed during diagnostics")

    activity_failures: list[str] = []
    threshold_checks = [
        (
            "first_step_adapter_gradient_l2",
            first_step_adapter_gradient_l2,
            args.minimum_first_step_gradient_l2,
        ),
        (
            "first_step_lora_b_gradient_l2",
            first_step_lora_b_gradient_l2,
            args.minimum_first_step_gradient_l2,
        ),
        (
            "adapter_parameter_delta_l2",
            float(activity["adapter_parameter_delta_l2"]),
            args.minimum_adapter_delta_l2,
        ),
        (
            "lora_b_parameter_delta_l2",
            float(activity["lora_b_parameter_delta_l2"]),
            args.minimum_adapter_delta_l2,
        ),
        (
            "effective_update_frobenius_l2",
            float(activity["effective_update_frobenius_l2"]),
            args.minimum_effective_update_l2,
        ),
    ]
    for label, value, threshold in threshold_checks:
        if not math.isfinite(float(value)) or float(value) <= float(threshold):
            activity_failures.append(
                f"{label}={value} is not greater than threshold={threshold}"
            )
    if int(activity["changed_parameter_count"]) <= 0:
        activity_failures.append("changed_parameter_count is zero")
    if activity_failures:
        raise RuntimeError(
            f"inactive LoRA adapter detected for {strategy_name}: "
            + "; ".join(activity_failures)
        )

    assignment_hash = assignment_sha256(ranks, init_bank)
    state_filename = f"adapter_state_{strategy_name}.pt"
    state_path = run_dir / state_filename
    torch.save(
        {
            "schema_version": ADAPTER_STATE_PROTOCOL_VERSION,
            "strategy": strategy_name,
            "assignment_sha256": assignment_hash,
            "alpha_scale": float(args.lora_alpha_scale),
            "parameters": final_state,
        },
        state_path,
    )

    set_global_seed(int(streams["evaluation"]))
    final_loss = evaluate(
        model, val_loader, device=device, eval_batches=args.eval_batches
    )

    del model, optimizer, parameters, named_parameters
    gc.collect()
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "strategy": strategy_name,
        "initial_val_loss": initial_loss,
        "final_val_loss": final_loss,
        "val_loss_delta": final_loss - initial_loss,
        "perplexity": math.exp(final_loss) if final_loss < 20 else float("inf"),
        "assignment_sha256": assignment_hash,
        "train_trace_sha256": float_sequence_sha256(train_losses),
        "n_train_steps": len(train_losses),
        "model_load_seed": int(streams["model_load"]),
        "training_data_seed": int(streams["training_data"]),
        "training_dropout_seed": int(streams["training_dropout"]),
        "evaluation_data_seed": int(streams["evaluation_data"]),
        "evaluation_seed": int(streams["evaluation"]),
        "adapter_state_file": state_filename,
        "adapter_state_file_sha256": sha256_file(state_path),
        "initial_adapter_state_sha256": activity[
            "initial_adapter_state_sha256"
        ],
        "final_adapter_state_sha256": activity["final_adapter_state_sha256"],
        "adapter_parameter_count": activity["adapter_parameter_count"],
        "changed_parameter_count": activity["changed_parameter_count"],
        "first_step_adapter_gradient_l2": first_step_adapter_gradient_l2,
        "first_step_lora_b_gradient_l2": first_step_lora_b_gradient_l2,
        "adapter_parameter_delta_l2": activity["adapter_parameter_delta_l2"],
        "lora_a_parameter_delta_l2": activity["lora_a_parameter_delta_l2"],
        "lora_b_parameter_delta_l2": activity["lora_b_parameter_delta_l2"],
        "effective_update_frobenius_l2": activity[
            "effective_update_frobenius_l2"
        ],
        "adapter_activity_passed": True,
    }



def assert_protocol_invariants(
    *,
    results: list[dict[str, Any]],
    allocations: Mapping[str, Mapping[str, int]],
    specs: Mapping[str, TargetSpec],
    target_budget: int,
    tolerance: float,
    require_identity_control: bool,
    minimum_first_step_gradient_l2: float = 0.0,
    minimum_adapter_delta_l2: float = 0.0,
    minimum_effective_update_l2: float = 0.0,
) -> dict[str, Any]:
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("identity tolerance must be finite and nonnegative")
    thresholds = {
        "minimum_first_step_gradient_l2": minimum_first_step_gradient_l2,
        "minimum_adapter_delta_l2": minimum_adapter_delta_l2,
        "minimum_effective_update_l2": minimum_effective_update_l2,
    }
    for label, value in thresholds.items():
        if float(value) < 0 or not math.isfinite(float(value)):
            raise ValueError(f"{label} must be finite and nonnegative")
    budget_rows = {
        strategy: total_lora_params(ranks, specs)
        for strategy, ranks in allocations.items()
    }
    bad_budgets = {
        strategy: budget
        for strategy, budget in budget_rows.items()
        if budget != target_budget
    }
    if bad_budgets:
        raise RuntimeError(
            f"exact-cost invariant failed: target={target_budget}, actual={bad_budgets}"
        )

    checks: dict[str, Any] = {
        "schema_version": RUN_MANIFEST_VERSION,
        "exact_parameter_cost": True,
        "target_budget": int(target_budget),
        "strategy_budgets": budget_rows,
        "initial_loss_identity": None,
        "identical_assignment_identity": None,
        "identity_control_present": "uniform_identity_control" in allocations,
        "identity_tolerance": float(tolerance),
        "identical_assignment_groups": [],
        "adapter_activity_nonzero": None,
        **{key: float(value) for key, value in thresholds.items()},
    }
    if require_identity_control and not checks["identity_control_present"]:
        raise RuntimeError("uniform identity control was requested but is absent")

    if not results:
        return checks
    by_strategy = {str(row["strategy"]): row for row in results}
    if set(by_strategy) != set(allocations):
        raise RuntimeError(
            "result strategies do not match allocations: "
            f"results={sorted(by_strategy)}, allocations={sorted(allocations)}"
        )

    activity_fields = {
        "first_step_adapter_gradient_l2": minimum_first_step_gradient_l2,
        "first_step_lora_b_gradient_l2": minimum_first_step_gradient_l2,
        "adapter_parameter_delta_l2": minimum_adapter_delta_l2,
        "lora_b_parameter_delta_l2": minimum_adapter_delta_l2,
        "effective_update_frobenius_l2": minimum_effective_update_l2,
    }
    for row in results:
        strategy = str(row["strategy"])
        if row.get("adapter_activity_passed") is not True:
            raise RuntimeError(f"adapter activity flag is false for {strategy}")
        if int(row.get("changed_parameter_count", 0)) <= 0:
            raise RuntimeError(f"no adapter parameters changed for {strategy}")
        for field, threshold in activity_fields.items():
            value = float(row[field])
            if not math.isfinite(value) or value <= float(threshold):
                raise RuntimeError(
                    f"inactive adapter diagnostic for {strategy}: "
                    f"{field}={value}, threshold={threshold}"
                )
        if row["initial_adapter_state_sha256"] == row["final_adapter_state_sha256"]:
            raise RuntimeError(f"adapter state hash did not change for {strategy}")
    checks["adapter_activity_nonzero"] = True

    initial_values = [float(row["initial_val_loss"]) for row in results]
    initial_range = max(initial_values) - min(initial_values)
    checks["initial_loss_range"] = initial_range
    checks["initial_loss_identity"] = initial_range <= tolerance
    if initial_range > tolerance:
        raise RuntimeError(
            f"zero-B initial-loss identity failed: range={initial_range}, "
            f"tolerance={tolerance}"
        )

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        groups.setdefault(str(row["assignment_sha256"]), []).append(row)
    for assignment_hash, rows in sorted(groups.items()):
        if len(rows) < 2:
            continue
        reference = rows[0]
        strategy_names = [str(row["strategy"]) for row in rows]
        maximum_difference = 0.0
        for row in rows[1:]:
            for key in ["initial_val_loss", "final_val_loss", "val_loss_delta"]:
                maximum_difference = max(
                    maximum_difference,
                    abs(float(row[key]) - float(reference[key])),
                )
            if row["train_trace_sha256"] != reference["train_trace_sha256"]:
                raise RuntimeError(
                    "identical allocation produced different training traces: "
                    f"{strategy_names}"
                )
            if (
                row["final_adapter_state_sha256"]
                != reference["final_adapter_state_sha256"]
            ):
                raise RuntimeError(
                    "identical allocation produced different final adapter states: "
                    f"{strategy_names}"
                )
        if maximum_difference > tolerance:
            raise RuntimeError(
                "identical allocation produced different evaluation metrics: "
                f"strategies={strategy_names}, max_difference={maximum_difference}, "
                f"tolerance={tolerance}"
            )
        checks["identical_assignment_groups"].append(
            {
                "assignment_sha256": assignment_hash,
                "strategies": strategy_names,
                "maximum_metric_difference": maximum_difference,
                "trace_sha256": reference["train_trace_sha256"],
                "final_adapter_state_sha256": reference[
                    "final_adapter_state_sha256"
                ],
            }
        )
    checks["identical_assignment_identity"] = True
    return checks


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_run_checksums(run_dir: Path) -> None:
    manifest = run_dir / "RUN_SHA256SUMS.txt"
    rows = []
    for path in sorted(
        value
        for value in run_dir.rglob("*")
        if value.is_file() and value != manifest
    ):
        rows.append(f"{sha256_file(path)}  {path.relative_to(run_dir).as_posix()}")
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")


def local_input_provenance(args: argparse.Namespace) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "model_argument": args.model,
        "model_revision_requested": args.model_revision,
        "dataset_mode": args.dataset_mode,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_revision_requested": args.dataset_revision,
    }
    model_path = Path(args.model).expanduser()
    if model_path.exists():
        model_path = model_path.resolve()
        provenance["local_model_path"] = str(model_path)
        if model_path.is_file():
            provenance["local_model_files"] = [
                {
                    "path": model_path.name,
                    "size": model_path.stat().st_size,
                    "sha256": sha256_file(model_path),
                }
            ]
        else:
            provenance["local_model_files"] = [
                {
                    "path": value.relative_to(model_path).as_posix(),
                    "size": value.stat().st_size,
                    "sha256": sha256_file(value),
                }
                for value in sorted(item for item in model_path.rglob("*") if item.is_file())
            ]
    if args.dataset_mode == "local":
        provenance["local_text_files"] = {
            key: {
                "path": str(Path(path).expanduser().resolve()),
                "size": Path(path).expanduser().resolve().stat().st_size,
                "sha256": sha256_file(Path(path).expanduser().resolve()),
            }
            for key, path in {
                "train": args.train_text_file,
                "validation": args.val_text_file,
            }.items()
        }
    provenance["provenance_sha256"] = canonical_json_sha256(provenance)
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--model_revision", "--model-revision", dest="model_revision", default=None)
    parser.add_argument("--local_files_only", "--local-files-only", dest="local_files_only", action="store_true")
    parser.add_argument(
        "--dataset_mode", choices=["hf", "builtin", "local"], default="builtin"
    )
    parser.add_argument("--dataset_name", default="wikitext")
    parser.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset_revision", "--dataset-revision", dest="dataset_revision", default=None)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="validation")
    parser.add_argument("--text_column", default=None)
    parser.add_argument("--train_text_file", default=None)
    parser.add_argument("--val_text_file", default=None)
    parser.add_argument("--max_train_texts", type=int, default=2000)
    parser.add_argument("--max_val_texts", type=int, default=500)
    parser.add_argument("--max_train_blocks", type=int, default=512)
    parser.add_argument("--max_val_blocks", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--calib_batches", type=int, default=4)
    parser.add_argument("--eval_batches", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--uniform_rank", type=int, default=2)
    parser.add_argument("--min_rank", type=int, default=1)
    parser.add_argument("--max_rank", type=int, default=8)
    parser.add_argument("--fisher_reference_rank", "--fisher-reference-rank", dest="fisher_reference_rank", type=int, default=0)
    parser.add_argument("--target_suffixes", default="c_attn,c_proj,c_fc")
    parser.add_argument("--max_targets", type=int, default=0, help="0 means no cap")
    parser.add_argument("--ridge_scale", type=float, default=1e-3)
    parser.add_argument("--lora_alpha_scale", type=float, default=2.0)
    parser.add_argument("--lora_init_std", type=float, default=0.01)
    parser.add_argument(
        "--strategies",
        default="uniform,gradient_norm,spectral_effective",
        help="comma-separated allocation strategies",
    )
    parser.add_argument("--include_identity_control", "--include-identity-control", dest="include_identity_control", action="store_true")
    parser.add_argument("--identity_tolerance", "--identity-tolerance", dest="identity_tolerance", type=float, default=1e-7)
    parser.add_argument(
        "--minimum_first_step_gradient_l2",
        "--minimum-first-step-gradient-l2",
        dest="minimum_first_step_gradient_l2",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--minimum_adapter_delta_l2",
        "--minimum-adapter-delta-l2",
        dest="minimum_adapter_delta_l2",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--minimum_effective_update_l2",
        "--minimum-effective-update-l2",
        dest="minimum_effective_update_l2",
        type=float,
        default=0.0,
    )
    parser.add_argument("--deterministic_algorithms", "--deterministic-algorithms", dest="deterministic_algorithms", action="store_true")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out_dir", "--out-dir", dest="out_dir", default="real_lora_runs")
    parser.add_argument("--run_id", "--run-id", dest="run_id", default=None)
    parser.add_argument(
        "--run_kind",
        "--run-kind",
        dest="run_kind",
        choices=["exploratory", "smoke", "publication"],
        default="exploratory",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--calibrate_only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> list[str]:
    strategies = parse_strategy_list(args.strategies)
    if args.min_rank < 1:
        raise ValueError("min_rank must be >= 1")
    if args.max_rank < args.min_rank:
        raise ValueError("max_rank must be >= min_rank")
    if not args.min_rank <= args.uniform_rank <= args.max_rank:
        raise ValueError("uniform_rank must be inside [min_rank, max_rank]")
    if args.fisher_reference_rank == 0:
        args.fisher_reference_rank = args.uniform_rank
    if not 1 <= args.fisher_reference_rank <= args.max_rank:
        raise ValueError("fisher_reference_rank must be inside [1, max_rank]")
    for name in [
        "block_size",
        "batch_size",
        "calib_batches",
        "eval_batches",
        "steps",
        "max_train_blocks",
        "max_val_blocks",
    ]:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be >= 1")
    if args.identity_tolerance < 0 or not math.isfinite(args.identity_tolerance):
        raise ValueError("identity_tolerance must be finite and nonnegative")
    for name in [
        "minimum_first_step_gradient_l2",
        "minimum_adapter_delta_l2",
        "minimum_effective_update_l2",
    ]:
        value = float(getattr(args, name))
        if value < 0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and nonnegative")
    if args.run_kind == "publication":
        if args.dataset_mode != "local":
            raise ValueError("publication runs require --dataset_mode local")
        if not Path(args.model).expanduser().exists():
            raise ValueError("publication runs require a pinned local --model path")
        for name in ["train_text_file", "val_text_file"]:
            value = getattr(args, name)
            if not value or not Path(value).expanduser().is_file():
                raise ValueError(f"publication runs require an existing --{name}")
        if not args.local_files_only:
            raise ValueError("publication runs require --local_files_only")
        if not args.deterministic_algorithms:
            raise ValueError("publication runs require --deterministic_algorithms")
        if not args.include_identity_control:
            raise ValueError("publication runs require --include_identity_control")
    return strategies


def main() -> None:
    args = parse_args()
    requested_strategies = validate_args(args)
    determinism = configure_determinism(args.deterministic_algorithms)
    seeds = seed_manifest(args.seed)
    streams = {key: int(value) for key, value in seeds["streams"].items()}
    set_global_seed(streams["model_load"])

    device = torch.device(args.device)
    dtype = torch_dtype_from_name(args.dtype)
    suffixes = [
        suffix.strip() for suffix in args.target_suffixes.split(",") if suffix.strip()
    ]

    run_id = args.run_id or f"real_lora_{timestamp()}_{sanitize_name(args.model)}"
    run_dir = Path(args.out_dir).expanduser().resolve() / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory already exists and is non-empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print("run_id:", run_id, flush=True)
    print("run_dir:", run_dir, flush=True)
    print("device:", device, "dtype:", dtype, flush=True)
    if torch.cuda.is_available() and device.type == "cuda":
        print(
            "torch:",
            torch.__version__,
            "hip:",
            getattr(torch.version, "hip", None),
            flush=True,
        )
        print("gpu:", torch.cuda.get_device_name(device), flush=True)

    tokenizer_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "use_fast": False,
        "local_files_only": bool(args.local_files_only),
    }
    if args.model_revision:
        tokenizer_kwargs["revision"] = args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_texts, val_texts = load_text_corpus(args)
    train_blocks = encode_blocks(
        tokenizer, train_texts, args.block_size, args.max_train_blocks
    )
    val_blocks = encode_blocks(
        tokenizer, val_texts, args.block_size, args.max_val_blocks
    )
    calib_loader = make_loader(
        train_blocks,
        args.batch_size,
        shuffle=False,
        seed=streams["calibration_data"],
    )

    write_json(run_dir / "seed_manifest.json", seeds)
    write_json(run_dir / "environment.json", environment_record(device))
    write_json(run_dir / "input_provenance.json", local_input_provenance(args))

    config: dict[str, Any] = vars(args).copy()
    config.update(
        {
            "schema_version": RUN_MANIFEST_VERSION,
            "protocol_version": REAL_PROTOCOL_VERSION,
            "rng_protocol_version": RNG_PROTOCOL_VERSION,
            "adapter_init_protocol_version": ADAPTER_INIT_PROTOCOL_VERSION,
            "adapter_state_protocol_version": ADAPTER_STATE_PROTOCOL_VERSION,
            "allocation_protocol_version": ALLOCATION_PROTOCOL_VERSION,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "status": "initializing",
            "requested_strategies": requested_strategies,
            "train_blocks": int(train_blocks.shape[0]),
            "val_blocks": int(val_blocks.shape[0]),
            "effective_rank_definition": EFFECTIVE_RANK_DEFINITION,
            "calibration_model_mode": "eval_with_gradients",
            "matched_strategy_batch_order": True,
            "matched_strategy_evaluation_data": True,
            "training_rng_reset_after_adapter_and_optimizer_construction": True,
            "nested_adapter_initialization": True,
            "exact_parameter_cost": True,
            "determinism": determinism,
            "seed_manifest_sha256": canonical_json_sha256(seeds),
        }
    )
    write_json(run_dir / "config.json", config)

    print("loading base model for target discovery/calibration", flush=True)
    set_global_seed(streams["model_load"])
    base = load_model(
        args.model,
        dtype=dtype,
        device=device,
        trust_remote_code=args.trust_remote_code,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
    )
    resolved_commit = getattr(getattr(base, "config", None), "_commit_hash", None)
    targets = find_targets(
        base,
        suffixes=suffixes,
        max_targets=args.max_targets if args.max_targets > 0 else None,
    )
    print(f"found {len(targets)} target modules", flush=True)
    for target in targets[:20]:
        print(
            f"  {target.name}: {target.module_class} in={target.in_dim} "
            f"out={target.out_dim} cost/rank={target.cost_per_rank}",
            flush=True,
        )
    if len(targets) > 20:
        print(f"  ... {len(targets) - 20} more", flush=True)

    init_bank = build_nested_init_bank(
        {target.name: target.in_dim for target in targets},
        max_rank=args.max_rank,
        init_std=args.lora_init_std,
        seed=streams["adapter_bank"],
    )
    torch.save(
        {
            "protocol_version": ADAPTER_INIT_PROTOCOL_VERSION,
            "max_rank": int(args.max_rank),
            "tensors": dict(init_bank),
        },
        run_dir / "adapter_init_bank.pt",
    )
    init_rows = [
        {
            "module": name,
            "shape": list(init_bank[name].shape),
            "dtype": str(init_bank[name].dtype),
            "sha256": tensor_sha256(init_bank[name]),
        }
        for name in sorted(init_bank)
    ]
    init_manifest = {
        "protocol_version": ADAPTER_INIT_PROTOCOL_VERSION,
        "adapter_bank_seed": streams["adapter_bank"],
        "max_rank": int(args.max_rank),
        "init_std": float(args.lora_init_std),
        "modules": init_rows,
        "content_sha256": canonical_json_sha256(init_rows),
    }
    write_json(run_dir / "adapter_init_manifest.json", init_manifest)

    set_global_seed(streams["calibration_data"])
    metrics, activation_components, calibration_batch_rows = calibrate_spectra(
        base,
        targets=targets,
        calib_loader=calib_loader,
        device=device,
        calib_batches=args.calib_batches,
        ridge_scale=args.ridge_scale,
        init_bank=init_bank,
        fisher_reference_rank=args.fisher_reference_rank,
        alpha_scale=args.lora_alpha_scale,
        max_rank=args.max_rank,
    )
    del base
    gc.collect()
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()

    metric_rows = [dataclasses.asdict(metric) for metric in metrics]
    write_csv(run_dir / "calibration_metrics.csv", metric_rows)
    write_csv(run_dir / "calibration_batch_metrics.csv", calibration_batch_rows)
    component_rows = [
        {
            "module": module,
            "component_index": index,
            "explained_variance": value,
        }
        for module in sorted(activation_components)
        for index, value in enumerate(activation_components[module], start=1)
    ]
    write_csv(run_dir / "activation_components.csv", component_rows)

    allocations, strategy_metadata, total_budget = build_allocations(
        metrics,
        activation_components,
        requested=requested_strategies,
        uniform_rank=args.uniform_rank,
        min_rank=args.min_rank,
        max_rank=args.max_rank,
        include_identity_control=args.include_identity_control,
    )
    specs = {
        metric.name: TargetSpec(
            name=metric.name,
            module_class=metric.module_class,
            in_dim=metric.in_dim,
            out_dim=metric.out_dim,
        )
        for metric in metrics
    }

    allocation_rows: list[dict[str, Any]] = []
    assignment_hashes: dict[str, str] = {}
    for strategy, ranks in allocations.items():
        realized = total_lora_params(ranks, specs)
        assignment_hashes[strategy] = assignment_sha256(ranks, init_bank)
        for metric in metrics:
            rank = int(ranks[metric.name])
            allocation_rows.append(
                {
                    "strategy": strategy,
                    "module": metric.name,
                    "rank": rank,
                    "params": rank * metric.cost_per_rank,
                    "cost_per_rank": metric.cost_per_rank,
                    "target_budget": total_budget,
                    "realized_budget": realized,
                    "budget_residual": realized - total_budget,
                    "assignment_sha256": assignment_hashes[strategy],
                    "init_slice_sha256": tensor_sha256(
                        bank_slice(init_bank, metric.name, rank)
                    ),
                    "requested_strategy": strategy_metadata[strategy][
                        "requested_strategy"
                    ],
                    "allocation_score": strategy_metadata[strategy][
                        "allocation_score"
                    ],
                    "baseline_scope": strategy_metadata[strategy]["scope"],
                    "raw_frobenius": metric.raw_frobenius,
                    "whitened_effective_rank": metric.whitened_effective_rank,
                    "activation_effective_rank": metric.activation_effective_rank,
                    "fim_b_mean_square": metric.fim_b_mean_square,
                    "gradient_sensitivity": metric.gradient_sensitivity,
                }
            )
    write_csv(run_dir / "allocations.csv", allocation_rows)

    for strategy, ranks in allocations.items():
        print(
            f"allocation {strategy}: params={total_lora_params(ranks, specs)} "
            f"ranks min/mean/max={min(ranks.values())}/"
            f"{np.mean(list(ranks.values())):.2f}/{max(ranks.values())}",
            flush=True,
        )

    config.update(
        {
            "resolved_model_commit": resolved_commit,
            "n_targets": len(targets),
            "target_modules": [dataclasses.asdict(target) for target in targets],
            "actual_calibration_batches": metrics[0].calibration_batches,
            "target_parameter_budget": total_budget,
            "output_strategies": list(allocations),
            "strategy_metadata": strategy_metadata,
            "assignment_sha256": assignment_hashes,
            "adapter_init_manifest_sha256": canonical_json_sha256(init_manifest),
            "status": "calibration_complete" if args.calibrate_only else "training",
        }
    )
    write_json(run_dir / "config.json", config)

    if args.calibrate_only:
        checks = assert_protocol_invariants(
            results=[],
            allocations=allocations,
            specs=specs,
            target_budget=total_budget,
            tolerance=args.identity_tolerance,
            require_identity_control=args.include_identity_control,
            minimum_first_step_gradient_l2=args.minimum_first_step_gradient_l2,
            minimum_adapter_delta_l2=args.minimum_adapter_delta_l2,
            minimum_effective_update_l2=args.minimum_effective_update_l2,
        )
        checks.update(
            {
                "calibration_model_mode_eval": True,
                "named_rng_streams": True,
                "nested_adapter_initialization": True,
                "training_completed": False,
            }
        )
        write_json(run_dir / "protocol_checks.json", checks)
        config["status"] = "complete_calibration_only"
        write_json(run_dir / "config.json", config)
        write_run_checksums(run_dir)
        print("Real LoRA calibration protocol: PASS", flush=True)
        print("run_dir:", run_dir, flush=True)
        return

    results: list[dict[str, Any]] = []
    for strategy, ranks in allocations.items():
        print(f"\n=== training {strategy} ===", flush=True)
        strategy_train_loader = make_loader(
            train_blocks,
            args.batch_size,
            shuffle=True,
            seed=streams["training_data"],
        )
        strategy_val_loader = make_loader(
            val_blocks,
            args.batch_size,
            shuffle=False,
            seed=streams["evaluation_data"],
        )
        result = train_strategy(
            args=args,
            strategy_name=strategy,
            ranks=ranks,
            train_loader=strategy_train_loader,
            val_loader=strategy_val_loader,
            device=device,
            dtype=dtype,
            run_dir=run_dir,
            init_bank=init_bank,
            streams=streams,
        )
        result["trainable_params"] = total_lora_params(ranks, specs)
        result["budget_residual"] = result["trainable_params"] - total_budget
        result["rank_min"] = min(ranks.values())
        result["rank_mean"] = float(np.mean(list(ranks.values())))
        result["rank_max"] = max(ranks.values())
        result["baseline_scope"] = strategy_metadata[strategy]["scope"]
        results.append(result)
        write_csv(run_dir / "results.csv", results)
        print(
            f"{strategy}: final_val_loss={result['final_val_loss']:.6f}",
            flush=True,
        )

    checks = assert_protocol_invariants(
        results=results,
        allocations=allocations,
        specs=specs,
        target_budget=total_budget,
        tolerance=args.identity_tolerance,
        require_identity_control=args.include_identity_control,
        minimum_first_step_gradient_l2=args.minimum_first_step_gradient_l2,
        minimum_adapter_delta_l2=args.minimum_adapter_delta_l2,
        minimum_effective_update_l2=args.minimum_effective_update_l2,
    )
    checks.update(
        {
            "calibration_model_mode_eval": True,
            "named_rng_streams": True,
            "nested_adapter_initialization": True,
            "training_rng_reset_after_construction": True,
            "training_completed": True,
        }
    )
    write_json(run_dir / "protocol_checks.json", checks)

    best = min(results, key=lambda row: float(row["final_val_loss"]))
    summary = [
        f"run_id: {run_id}",
        f"run_kind: {args.run_kind}",
        f"model: {args.model}",
        f"dataset_mode: {args.dataset_mode}",
        f"protocol_version: {REAL_PROTOCOL_VERSION}",
        f"target_parameter_budget: {total_budget}",
        f"best_strategy: {best['strategy']}",
        "",
        "results:",
    ]
    for row in results:
        summary.append(
            f"  {row['strategy']}: initial={row['initial_val_loss']:.6f} "
            f"final={row['final_val_loss']:.6f} "
            f"delta={row['val_loss_delta']:.6f} "
            f"ppl={row['perplexity']:.4f} params={row['trainable_params']} "
            f"grad={row['first_step_lora_b_gradient_l2']:.3e} "
            f"update={row['effective_update_frobenius_l2']:.3e}"
        )
    (run_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    config["status"] = "complete"
    config["protocol_checks_sha256"] = canonical_json_sha256(checks)
    write_json(run_dir / "config.json", config)
    write_run_checksums(run_dir)

    print("\n" + "\n".join(summary), flush=True)
    print("Real LoRA controlled run: PASS", flush=True)
    print("run_dir:", run_dir, flush=True)


if __name__ == "__main__":
    main()
