#!/usr/bin/env python3
"""Real pretrained-transformer LoRA allocation validation.

This script compares three approximately budget-matched LoRA allocation rules on a real
pretrained causal language model:

  1. uniform rank per target module,
  2. rank allocation proportional to early gradient norm,
  3. rank allocation proportional to activation-whitened early-gradient
     spectral effective rank.

The code is intentionally self-contained. It does not require PEFT; it wraps
Linear and HuggingFace GPT-style Conv1D modules directly so per-module rank
allocation is exact and auditable.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import json
import math
import os
import random
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from spectral_metrics import EFFECTIVE_RANK_DEFINITION, singular_effective_rank

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - only used when datasets is unavailable.
    load_dataset = None


BUILTIN_TEXT = """
Low-rank adaptation changes a pretrained model through a small number of trainable
matrix directions.  A layer with a large gradient norm may want to change, but the
singular spectrum of the early gradient can reveal how many independent directions
of change are present.  Activation whitening removes geometry induced by common
input directions, leaving a spectrum closer to the useful task operator.

This small built-in corpus is only a smoke-test fallback.  It is not meant to be a
publication dataset.  Use --dataset_mode hf with Wikitext or another real corpus
for validation evidence.
""".strip()


@dataclass
class TargetSpec:
    name: str
    module_class: str
    in_dim: int
    out_dim: int

    @property
    def cost_per_rank(self) -> int:
        return int(self.in_dim + self.out_dim)


@dataclass
class SpectralMetrics:
    name: str
    module_class: str
    in_dim: int
    out_dim: int
    cost_per_rank: int
    tokens: int
    raw_frobenius: float
    raw_stable_rank: float
    raw_effective_rank: float
    whitened_frobenius: float
    whitened_stable_rank: float
    whitened_effective_rank: float
    cov_trace: float
    cov_condition_est: float


class CovAccumulator:
    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self.cov_sum = torch.zeros((dim, dim), dtype=torch.float64, device="cpu")
        self.n = 0

    def add(self, x: torch.Tensor) -> None:
        # x: (..., dim)
        if x.shape[-1] != self.dim:
            raise ValueError(f"activation dim mismatch: expected {self.dim}, got {x.shape[-1]}")
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
    def __init__(self, base: nn.Linear, rank: int, alpha_scale: float, init_std: float) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be >= 1")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.scaling = float(alpha_scale)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=base.weight.device))
        nn.init.normal_(self.lora_A, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        z = torch.matmul(x.to(torch.float32), self.lora_A.t())
        z = torch.matmul(z, self.lora_B.t()) * self.scaling
        return base_out + z.to(base_out.dtype)


class LoRAConv1D(nn.Module):
    """LoRA wrapper for HuggingFace GPT-style Conv1D.

    HF Conv1D stores weight as (in_dim, out_dim) and computes x @ weight + bias.
    """

    def __init__(self, base: nn.Module, rank: int, alpha_scale: float, init_std: float) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be >= 1")
        if not hasattr(base, "weight") or base.weight.ndim != 2:
            raise TypeError("LoRAConv1D expects a module with 2D weight")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.scaling = float(alpha_scale)
        in_dim = int(base.weight.shape[0])
        out_dim = int(base.weight.shape[1])
        self.lora_A = nn.Parameter(torch.empty(in_dim, rank, dtype=torch.float32, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim, dtype=torch.float32, device=base.weight.device))
        nn.init.normal_(self.lora_A, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        z = torch.matmul(x.to(torch.float32), self.lora_A)
        z = torch.matmul(z, self.lora_B) * self.scaling
        return base_out + z.to(base_out.dtype)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    return time.strftime("%Y%m%d_%H%M%S")


def sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def read_local_text_file(path: str | None, limit: int) -> list[str]:
    if not path:
        raise ValueError("local dataset mode requires --train_text_file and --val_text_file")
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(p)
    text = p.read_text(encoding="utf-8", errors="replace")
    parts = [line.strip() for line in text.splitlines() if line.strip()]
    if not parts and text.strip():
        parts = [text.strip()]
    if not parts:
        raise RuntimeError(f"no non-empty text found in {p}")
    return parts[:limit]


def load_text_corpus(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    if args.dataset_mode == "local":
        return (
            read_local_text_file(args.train_text_file, args.max_train_texts),
            read_local_text_file(args.val_text_file, args.max_val_texts),
        )

    if args.dataset_mode == "builtin":
        repeated = [BUILTIN_TEXT for _ in range(max(args.max_train_texts, args.max_val_texts, 8))]
        return repeated[: args.max_train_texts], repeated[: args.max_val_texts]

    if load_dataset is None:
        raise RuntimeError("datasets is not installed; use --dataset_mode builtin or install datasets")

    kwargs: dict[str, Any] = {}
    if args.dataset_config:
        kwargs["name"] = args.dataset_config

    train_ds = load_dataset(args.dataset_name, **kwargs, split=args.train_split)
    val_ds = load_dataset(args.dataset_name, **kwargs, split=args.val_split)

    text_col = args.text_column
    if text_col is None:
        for candidate in ["text", "content", "sentence"]:
            if candidate in train_ds.column_names:
                text_col = candidate
                break
    if text_col is None:
        raise ValueError(f"could not infer text column from columns={train_ds.column_names}")

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
        text = str(text).strip()
        if not text:
            continue
        ids.extend(tokenizer.encode(text, add_special_tokens=False))
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
    arr = torch.tensor(ids[:total], dtype=torch.long).view(-1, block_size)
    return arr


def make_loader(blocks: torch.Tensor, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(blocks),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=generator,
    )


def infinite_loader(loader: DataLoader) -> Iterator[torch.Tensor]:
    while True:
        for (x,) in loader:
            yield x


def is_conv1d_module(module: nn.Module) -> bool:
    return module.__class__.__name__ == "Conv1D" and hasattr(module, "weight") and module.weight.ndim == 2


def is_targetable_module(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) or is_conv1d_module(module)


def module_dims(module: nn.Module) -> tuple[int, int]:
    if isinstance(module, nn.Linear):
        return int(module.in_features), int(module.out_features)
    if is_conv1d_module(module):
        return int(module.weight.shape[0]), int(module.weight.shape[1])
    raise TypeError(f"unsupported target module type {type(module)}")


def find_targets(model: nn.Module, suffixes: list[str], max_targets: int | None = None) -> list[TargetSpec]:
    suffixes = [s.strip() for s in suffixes if s.strip()]
    out: list[TargetSpec] = []
    for name, module in model.named_modules():
        if not name:
            continue
        leaf = name.split(".")[-1]
        if suffixes and not (leaf in suffixes or any(name.endswith(s) for s in suffixes)):
            continue
        if not is_targetable_module(module):
            continue
        in_dim, out_dim = module_dims(module)
        out.append(TargetSpec(name=name, module_class=module.__class__.__name__, in_dim=in_dim, out_dim=out_dim))
    if max_targets is not None and max_targets > 0:
        out = out[:max_targets]
    if not out:
        raise RuntimeError(f"no target modules found for suffixes={suffixes}")
    return out


def get_submodule(model: nn.Module, name: str) -> nn.Module:
    cur: nn.Module = model
    for part in name.split("."):
        cur = cur._modules[part]
    return cur


def set_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    parent: nn.Module = model
    for part in parts[:-1]:
        parent = parent._modules[part]
    parent._modules[parts[-1]] = module


def oriented_weight_grad(module: nn.Module) -> torch.Tensor:
    if module.weight.grad is None:
        raise RuntimeError("target module has no weight gradient")
    grad = module.weight.grad.detach()
    if isinstance(module, nn.Linear):
        # Linear weight is (out_dim, in_dim).
        return grad.to(device="cpu", dtype=torch.float64)
    if is_conv1d_module(module):
        # HF Conv1D weight is (in_dim, out_dim); transpose to (out_dim, in_dim).
        return grad.t().to(device="cpu", dtype=torch.float64)
    raise TypeError(f"unsupported module type {type(module)}")


def singular_stable_rank(s: torch.Tensor, eps: float = 1e-12) -> float:
    s = s.detach().to(dtype=torch.float64, device="cpu")
    if s.numel() == 0 or float(s[0]) <= eps:
        return 0.0
    return float((s.square().sum() / (s[0].square() + eps)).item())


def invsqrt_from_cov(cov: torch.Tensor, ridge_scale: float, eps: float = 1e-12) -> tuple[torch.Tensor, float]:
    cov = cov.to(dtype=torch.float64, device="cpu")
    d = cov.shape[0]
    ridge = float(ridge_scale) * float(torch.trace(cov).item()) / max(d, 1)
    vals, vecs = torch.linalg.eigh(cov + ridge * torch.eye(d, dtype=torch.float64))
    vals = vals.clamp_min(eps)
    cond = float((vals.max() / vals.min()).item())
    invsqrt = (vecs * torch.rsqrt(vals).unsqueeze(0)) @ vecs.t()
    return invsqrt, cond


def calibrate_spectra(
    model: nn.Module,
    targets: list[TargetSpec],
    calib_loader: DataLoader,
    device: torch.device,
    calib_batches: int,
    ridge_scale: float,
) -> list[SpectralMetrics]:
    model.train()
    model.config.use_cache = False

    # Freeze everything except the target full weights during calibration.
    for p in model.parameters():
        p.requires_grad_(False)

    modules = {t.name: get_submodule(model, t.name) for t in targets}
    for module in modules.values():
        module.weight.requires_grad_(True)

    covs = {t.name: CovAccumulator(t.in_dim) for t in targets}
    grad_sums = {
        t.name: torch.zeros((t.out_dim, t.in_dim), dtype=torch.float64, device="cpu") for t in targets
    }

    handles = []
    for t in targets:
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor, name: str = t.name) -> None:
            if not inputs:
                return
            covs[name].add(inputs[0])

        handles.append(modules[t.name].register_forward_hook(hook))

    try:
        model.zero_grad(set_to_none=True)
        for bi, batch in enumerate(calib_loader):
            if bi >= calib_batches:
                break
            x = batch[0].to(device, non_blocking=True)
            out = model(input_ids=x, labels=x)
            loss = out.loss
            loss.backward()
            for t in targets:
                grad_sums[t.name].add_(oriented_weight_grad(modules[t.name]))
            model.zero_grad(set_to_none=True)
    finally:
        for handle in handles:
            handle.remove()

    metrics: list[SpectralMetrics] = []
    for t in targets:
        g = grad_sums[t.name] / max(1, calib_batches)
        raw_s = torch.linalg.svdvals(g)
        cov = covs[t.name].covariance()
        invsqrt, cond = invsqrt_from_cov(cov, ridge_scale=ridge_scale)
        w = g @ invsqrt
        w_s = torch.linalg.svdvals(w)
        metrics.append(
            SpectralMetrics(
                name=t.name,
                module_class=t.module_class,
                in_dim=t.in_dim,
                out_dim=t.out_dim,
                cost_per_rank=t.cost_per_rank,
                tokens=covs[t.name].n,
                raw_frobenius=float(torch.linalg.norm(g).item()),
                raw_stable_rank=singular_stable_rank(raw_s),
                raw_effective_rank=singular_effective_rank(raw_s),
                whitened_frobenius=float(torch.linalg.norm(w).item()),
                whitened_stable_rank=singular_stable_rank(w_s),
                whitened_effective_rank=singular_effective_rank(w_s),
                cov_trace=float(torch.trace(cov).item()),
                cov_condition_est=cond,
            )
        )
    return metrics


def total_lora_params(ranks: dict[str, int], specs: dict[str, TargetSpec]) -> int:
    return int(sum(int(ranks[name]) * specs[name].cost_per_rank for name in ranks))


def allocate_uniform(metrics: list[SpectralMetrics], rank: int) -> dict[str, int]:
    return {m.name: int(rank) for m in metrics}


def allocate_by_score(
    metrics: list[SpectralMetrics],
    score_key: str,
    total_budget: int,
    min_rank: int,
    max_rank: int,
) -> dict[str, int]:
    if min_rank < 1:
        raise ValueError("min_rank must be >= 1")
    if max_rank < min_rank:
        raise ValueError("max_rank must be >= min_rank")

    ranks = {m.name: int(min_rank) for m in metrics}
    costs = {m.name: int(m.cost_per_rank) for m in metrics}
    scores = {m.name: max(float(getattr(m, score_key)), 0.0) for m in metrics}
    if sum(scores.values()) <= 0:
        return ranks

    spent = sum(costs[n] * ranks[n] for n in ranks)
    remaining = int(total_budget - spent)
    if remaining < min(costs.values()):
        return ranks

    while True:
        best_name = None
        best_gain = -1.0
        for m in metrics:
            name = m.name
            if ranks[name] >= max_rank:
                continue
            c = costs[name]
            if c > remaining:
                continue
            # Diminishing returns avoid giving all rank to one high-score layer.
            gain = scores[name] / (c * math.sqrt(ranks[name] + 1.0))
            if gain > best_gain:
                best_gain = gain
                best_name = name
        if best_name is None:
            break
        ranks[best_name] += 1
        remaining -= costs[best_name]
    return ranks


def load_model(model_name: str, dtype: torch.dtype, device: torch.device, trust_remote_code: bool) -> nn.Module:
    kwargs = {"trust_remote_code": trust_remote_code}
    # HF versions differ: torch_dtype is accepted by v4 and current v5 compatibility.
    kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.config.use_cache = False
    return model.to(device)


def apply_lora(
    model: nn.Module,
    ranks: dict[str, int],
    alpha_scale: float,
    init_std: float,
) -> None:
    for name, rank in ranks.items():
        module = get_submodule(model, name)
        if isinstance(module, nn.Linear):
            wrapped = LoRALinear(module, rank=rank, alpha_scale=alpha_scale, init_std=init_std)
        elif is_conv1d_module(module):
            wrapped = LoRAConv1D(module, rank=rank, alpha_scale=alpha_scale, init_std=init_std)
        else:
            raise TypeError(f"unsupported LoRA module {name}: {type(module)}")
        set_submodule(model, name, wrapped)


def adapter_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if ("lora_A" in n or "lora_B" in n) and p.requires_grad]


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, eval_batches: int) -> float:
    model.eval()
    losses: list[float] = []
    for bi, batch in enumerate(loader):
        if bi >= eval_batches:
            break
        x = batch[0].to(device, non_blocking=True)
        out = model(input_ids=x, labels=x)
        losses.append(float(out.loss.detach().cpu().item()))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return float(np.mean(losses))


def train_strategy(
    args: argparse.Namespace,
    strategy_name: str,
    ranks: dict[str, int],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    run_dir: Path,
) -> dict[str, Any]:
    set_seed(args.seed)
    model = load_model(args.model, dtype=dtype, device=device, trust_remote_code=args.trust_remote_code)
    for p in model.parameters():
        p.requires_grad_(False)
    apply_lora(model, ranks, alpha_scale=args.lora_alpha_scale, init_std=args.lora_init_std)
    params = adapter_parameters(model)
    if not params:
        raise RuntimeError("no adapter parameters found")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    initial_loss = evaluate(model, val_loader, device=device, eval_batches=args.eval_batches)
    model.train()
    stream = infinite_loader(train_loader)
    history_path = run_dir / f"train_history_{strategy_name}.csv"
    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "elapsed_sec"])
        writer.writeheader()
        t0 = time.time()
        for step in range(1, args.steps + 1):
            x = next(stream).to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(input_ids=x, labels=x)
            loss = out.loss
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                writer.writerow({"step": step, "train_loss": float(loss.detach().cpu().item()), "elapsed_sec": time.time() - t0})
                f.flush()
                print(f"[{strategy_name}] step {step:5d}/{args.steps}: train_loss={float(loss.detach().cpu().item()):.4f}", flush=True)

    final_loss = evaluate(model, val_loader, device=device, eval_batches=args.eval_batches)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del model, opt, params
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "strategy": strategy_name,
        "initial_val_loss": initial_loss,
        "final_val_loss": final_loss,
        "val_loss_delta": final_loss - initial_loss,
        "perplexity": math.exp(final_loss) if final_loss < 20 else float("inf"),
        "trainable_params": int(sum(int(r) * 0 for r in [])),  # filled below
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="sshleifer/tiny-gpt2")
    ap.add_argument("--dataset_mode", choices=["hf", "builtin", "local"], default="builtin")
    ap.add_argument("--dataset_name", default="wikitext")
    ap.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default="validation")
    ap.add_argument("--text_column", default=None)
    ap.add_argument("--train_text_file", default=None)
    ap.add_argument("--val_text_file", default=None)
    ap.add_argument("--max_train_texts", type=int, default=2000)
    ap.add_argument("--max_val_texts", type=int, default=500)
    ap.add_argument("--max_train_blocks", type=int, default=512)
    ap.add_argument("--max_val_blocks", type=int, default=128)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--calib_batches", type=int, default=4)
    ap.add_argument("--eval_batches", type=int, default=16)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--uniform_rank", type=int, default=2)
    ap.add_argument("--min_rank", type=int, default=1)
    ap.add_argument("--max_rank", type=int, default=8)
    ap.add_argument("--target_suffixes", default="c_attn,c_proj,c_fc")
    ap.add_argument("--max_targets", type=int, default=0, help="0 means no cap")
    ap.add_argument("--ridge_scale", type=float, default=1e-3)
    ap.add_argument("--lora_alpha_scale", type=float, default=2.0)
    ap.add_argument("--lora_init_std", type=float, default=0.01)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="real_lora_runs")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--calibrate_only", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch_dtype_from_name(args.dtype)
    suffixes = [s.strip() for s in args.target_suffixes.split(",") if s.strip()]

    run_dir = Path(args.out_dir) / f"{timestamp()}_{sanitize_name(args.model)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("run_dir:", run_dir, flush=True)
    print("device:", device, "dtype:", dtype, flush=True)
    if torch.cuda.is_available():
        print("torch:", torch.__version__, "hip:", getattr(torch.version, "hip", None), flush=True)
        print("gpu:", torch.cuda.get_device_name(0), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_texts, val_texts = load_text_corpus(args)
    train_blocks = encode_blocks(tokenizer, train_texts, args.block_size, args.max_train_blocks)
    val_blocks = encode_blocks(tokenizer, val_texts, args.block_size, args.max_val_blocks)
    # Calibration has its own fixed, unshuffled loader.  Each training strategy
    # receives fresh train/validation loaders below so all strategies see the
    # same mini-batch order for a given seed.
    calib_loader = make_loader(train_blocks, args.batch_size, shuffle=False, seed=args.seed + 2)

    config = vars(args).copy()
    config.update({
        "run_dir": str(run_dir),
        "torch_version": torch.__version__,
        "torch_hip": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_blocks": int(train_blocks.shape[0]),
        "val_blocks": int(val_blocks.shape[0]),
        "effective_rank_definition": EFFECTIVE_RANK_DEFINITION,
        "matched_strategy_batch_order": True,
    })
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    print("loading base model for target discovery/calibration", flush=True)
    base = load_model(args.model, dtype=dtype, device=device, trust_remote_code=args.trust_remote_code)
    targets = find_targets(base, suffixes=suffixes, max_targets=args.max_targets if args.max_targets > 0 else None)
    print(f"found {len(targets)} target modules", flush=True)
    for t in targets[:20]:
        print(f"  {t.name}: {t.module_class} in={t.in_dim} out={t.out_dim} cost/rank={t.cost_per_rank}", flush=True)
    if len(targets) > 20:
        print(f"  ... {len(targets) - 20} more", flush=True)

    metrics = calibrate_spectra(
        base,
        targets=targets,
        calib_loader=calib_loader,
        device=device,
        calib_batches=args.calib_batches,
        ridge_scale=args.ridge_scale,
    )
    del base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metric_rows = [dataclasses.asdict(m) for m in metrics]
    write_csv(run_dir / "calibration_metrics.csv", metric_rows)
    print("wrote", run_dir / "calibration_metrics.csv", flush=True)

    specs = {TargetSpec(**{k: getattr(m, k) for k in ["name", "module_class", "in_dim", "out_dim"]}).name: TargetSpec(
        name=m.name, module_class=m.module_class, in_dim=m.in_dim, out_dim=m.out_dim
    ) for m in metrics}
    uniform = allocate_uniform(metrics, args.uniform_rank)
    total_budget = total_lora_params(uniform, specs)
    allocations = {
        f"uniform_r{args.uniform_rank}": uniform,
        "gradient_norm": allocate_by_score(metrics, "raw_frobenius", total_budget, args.min_rank, args.max_rank),
        "spectral_effective": allocate_by_score(metrics, "whitened_effective_rank", total_budget, args.min_rank, args.max_rank),
    }

    alloc_rows: list[dict[str, Any]] = []
    for strategy, ranks in allocations.items():
        for m in metrics:
            alloc_rows.append({
                "strategy": strategy,
                "module": m.name,
                "rank": ranks[m.name],
                "params": ranks[m.name] * m.cost_per_rank,
                "cost_per_rank": m.cost_per_rank,
                "raw_frobenius": m.raw_frobenius,
                "whitened_effective_rank": m.whitened_effective_rank,
            })
    write_csv(run_dir / "allocations.csv", alloc_rows)
    print("wrote", run_dir / "allocations.csv", flush=True)

    for strategy, ranks in allocations.items():
        print(f"allocation {strategy}: params={total_lora_params(ranks, specs)} ranks min/mean/max="
              f"{min(ranks.values())}/{np.mean(list(ranks.values())):.2f}/{max(ranks.values())}", flush=True)

    if args.calibrate_only:
        print("calibrate_only requested; stopping before training", flush=True)
        return

    results: list[dict[str, Any]] = []
    for strategy, ranks in allocations.items():
        print("\n=== training", strategy, "===", flush=True)
        strategy_train_loader = make_loader(
            train_blocks, args.batch_size, shuffle=True, seed=args.seed
        )
        strategy_val_loader = make_loader(
            val_blocks, args.batch_size, shuffle=False, seed=args.seed + 1
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
        )
        result["trainable_params"] = total_lora_params(ranks, specs)
        result["rank_min"] = min(ranks.values())
        result["rank_mean"] = float(np.mean(list(ranks.values())))
        result["rank_max"] = max(ranks.values())
        results.append(result)
        write_csv(run_dir / "results.csv", results)
        print(f"{strategy}: final_val_loss={result['final_val_loss']:.4f}", flush=True)

    best = min(results, key=lambda r: r["final_val_loss"])
    summary = [
        f"run_dir: {run_dir}",
        f"model: {args.model}",
        f"dataset_mode: {args.dataset_mode}",
        f"best_strategy: {best['strategy']}",
        "",
        "results:",
    ]
    for r in results:
        summary.append(
            f"  {r['strategy']}: initial={r['initial_val_loss']:.4f} "
            f"final={r['final_val_loss']:.4f} delta={r['val_loss_delta']:.4f} "
            f"ppl={r['perplexity']:.2f} params={r['trainable_params']}"
        )
    (run_dir / "summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary), flush=True)


if __name__ == "__main__":
    main()
