from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

IGNORE_INDEX = -100


@dataclass
class TaskSpec:
    name: str
    vocab_size: int
    seq_len: int
    label_pos: int
    params: Dict


def make_task_spec(cfg: Dict, split: str) -> TaskSpec:
    name = cfg["name"]
    if name == "modular":
        modulus = int(cfg["modulus"])
        seq_len = int(cfg.get("seq_len", 4))
        if seq_len != 4:
            raise ValueError("modular task currently requires seq_len=4")
        # number tokens 0..modulus-1, plus '+', '='
        return TaskSpec(
            name=name,
            vocab_size=modulus + 2,
            seq_len=seq_len,
            label_pos=seq_len - 1,
            params={**cfg, "offset": int(cfg["base_offset"] if split == "base" else cfg["adapt_offset"])},
        )
    if name == "associative_recall":
        k = int(cfg["num_keys"])
        num_pairs = int(cfg.get("num_pairs", 4))
        # keys 0..k-1, values k..2k-1. Sequence: k1,v1,...,kp,vp,query.
        return TaskSpec(
            name=name,
            vocab_size=2 * k,
            seq_len=2 * num_pairs + 1,
            label_pos=2 * num_pairs,
            params={**cfg, "shift": int(cfg["base_shift"] if split == "base" else cfg["adapt_shift"])},
        )
    raise ValueError(f"unknown task name: {name}")


def batch_modular(
    spec: TaskSpec,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    p = int(spec.params["modulus"])
    offset = int(spec.params["offset"])
    plus = p
    eq = p + 1
    a = torch.randint(0, p, (batch_size,), device=device, generator=generator)
    b = torch.randint(0, p, (batch_size,), device=device, generator=generator)
    x = torch.empty((batch_size, spec.seq_len), dtype=torch.long, device=device)
    x[:, 0] = a
    x[:, 1] = plus
    x[:, 2] = b
    x[:, 3] = eq
    y = torch.full_like(x, IGNORE_INDEX)
    y[:, spec.label_pos] = (a + b + offset) % p
    return x, y


def batch_associative(
    spec: TaskSpec,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    k = int(spec.params["num_keys"])
    num_pairs = int(spec.params.get("num_pairs", 4))
    shift = int(spec.params["shift"])
    x = torch.empty((batch_size, spec.seq_len), dtype=torch.long, device=device)
    y = torch.full_like(x, IGNORE_INDEX)
    for b in range(batch_size):
        keys = torch.randperm(k, device=device, generator=generator)[:num_pairs]
        values = k + ((keys + shift) % k)
        query_idx = torch.randint(0, num_pairs, (1,), device=device, generator=generator).item()
        query_key = keys[query_idx]
        for i in range(num_pairs):
            x[b, 2 * i] = keys[i]
            x[b, 2 * i + 1] = values[i]
        x[b, 2 * num_pairs] = query_key
        y[b, 2 * num_pairs] = values[query_idx]
    return x, y


def make_batch(
    spec: TaskSpec,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Draw one task batch, optionally from an explicit RNG stream.

    Passing a generator is required for publication comparisons so that data
    randomness is independent of model/adaptor/dropout randomness.
    """
    if spec.name == "modular":
        return batch_modular(spec, batch_size, device, generator=generator)
    if spec.name == "associative_recall":
        return batch_associative(spec, batch_size, device, generator=generator)
    raise ValueError(spec.name)


def exact_modular_batches(
    spec: TaskSpec,
    batch_size: int,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Enumerate all modular-arithmetic input pairs exactly, in fixed order."""
    if spec.name != "modular":
        raise ValueError("exact enumeration is currently defined only for the modular task")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    p = int(spec.params["modulus"])
    offset = int(spec.params["offset"])
    plus = p
    eq = p + 1
    a = torch.arange(p, device=device).repeat_interleave(p)
    b = torch.arange(p, device=device).repeat(p)
    batches = []
    for start in range(0, p * p, batch_size):
        aa = a[start : start + batch_size]
        bb = b[start : start + batch_size]
        x = torch.empty((len(aa), spec.seq_len), dtype=torch.long, device=device)
        x[:, 0] = aa
        x[:, 1] = plus
        x[:, 2] = bb
        x[:, 3] = eq
        y = torch.full_like(x, IGNORE_INDEX)
        y[:, spec.label_pos] = (aa + bb + offset) % p
        batches.append((x, y))
    return tuple(batches)
