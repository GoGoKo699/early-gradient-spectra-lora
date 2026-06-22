from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

from .tasks import IGNORE_INDEX, TaskSpec, exact_modular_batches, make_batch
from .model import freeze_base_enable_lora, lora_parameters, lora_parameter_count

Batch = Tuple[torch.Tensor, torch.Tensor]


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=IGNORE_INDEX)


def _torch_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    return generator


def _set_torch_rng(seed: int) -> None:
    """Reset only the model stochasticity stream (for example dropout)."""
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def materialize_batches(
    task: TaskSpec,
    batch_size: int,
    batches: int,
    device: torch.device,
    seed: int,
) -> Tuple[Batch, ...]:
    """Pre-generate an immutable sequence of batches from a dedicated stream."""
    generator = _torch_generator(device, seed)
    return tuple(
        make_batch(task, int(batch_size), device, generator=generator)
        for _ in range(int(batches))
    )


def materialize_evaluation_batches(
    task: TaskSpec,
    batch_size: int,
    batches: int,
    device: torch.device,
    seed: int,
    *,
    exact_modular: bool = False,
) -> Tuple[Batch, ...]:
    """Create a fixed evaluation set, optionally enumerating modular inputs."""
    if exact_modular and task.name == "modular":
        return exact_modular_batches(task, batch_size, device)
    return materialize_batches(task, batch_size, batches, device, seed)


@torch.no_grad()
def evaluate(
    model,
    task: TaskSpec,
    batch_size: int,
    batches: int,
    device: torch.device,
    *,
    fixed_batches: Sequence[Batch] | None = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    if fixed_batches is None:
        batch_iter = (make_batch(task, batch_size, device) for _ in range(int(batches)))
        n_batches = int(batches)
    else:
        batch_iter = iter(fixed_batches)
        n_batches = len(fixed_batches)
        if n_batches == 0:
            raise ValueError("fixed_batches must contain at least one batch")
    for x, y in batch_iter:
        logits = model(x)
        loss = loss_fn(logits, y)
        mask = y != IGNORE_INDEX
        count = int(mask.sum().item())
        total_loss += float(loss.item()) * count
        pred = logits.argmax(dim=-1)
        total_correct += int((pred[mask] == y[mask]).sum().item())
        total_count += count
    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
        "n_examples": int(total_count),
        "n_batches": int(n_batches),
    }


def train_base(
    model,
    task: TaskSpec,
    cfg: Dict,
    device: torch.device,
    eval_task: TaskSpec | None = None,
    *,
    train_batches: Sequence[Batch] | None = None,
    dropout_seed: int | None = None,
) -> Dict[str, float]:
    if dropout_seed is not None:
        _set_torch_rng(dropout_seed)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 0.003)), weight_decay=float(cfg.get("weight_decay", 0.0)))
    steps = int(cfg.get("steps", 100))
    bs = int(task.params.get("train_batch_size", cfg.get("batch_size", 64)))
    log_every = int(cfg.get("log_every", max(steps, 1)))
    if train_batches is not None and len(train_batches) < steps:
        raise ValueError(f"need at least {steps} fixed training batches, got {len(train_batches)}")
    last_loss = None
    for step in range(1, steps + 1):
        if train_batches is None:
            x, y = make_batch(task, bs, device)
        else:
            x, y = train_batches[step - 1]
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
        if log_every and step % log_every == 0:
            print(f"base step {step:5d}/{steps}: loss={last_loss:.4f}", flush=True)
    return {"train_loss_last": float(last_loss if last_loss is not None else 0.0)}


def train_lora(
    model,
    task: TaskSpec,
    cfg: Dict,
    device: torch.device,
    eval_task: TaskSpec | None = None,
    *,
    train_batches: Sequence[Batch] | None = None,
    validation_batches: Sequence[Batch] | None = None,
    dropout_seed: int | None = None,
) -> Tuple[Dict[str, float], bool]:
    freeze_base_enable_lora(model)
    params = list(lora_parameters(model))
    bs = int(task.params.get("train_batch_size", cfg.get("batch_size", 64)))
    eval_count = int(task.params.get("eval_batches", 4))
    if not params:
        metrics = evaluate(
            model,
            eval_task or task,
            bs,
            eval_count,
            device,
            fixed_batches=validation_batches,
        )
        return {
            "train_loss_last": metrics["loss"],
            "val_loss": metrics["loss"],
            "val_accuracy": metrics["accuracy"],
            "val_n_examples": metrics["n_examples"],
            "val_n_batches": metrics["n_batches"],
            "trainable_params": 0,
        }, False
    if dropout_seed is not None:
        _set_torch_rng(dropout_seed)
    opt = torch.optim.AdamW(params, lr=float(cfg.get("lr", 0.01)), weight_decay=float(cfg.get("weight_decay", 0.0)))
    steps = int(cfg.get("steps", 100))
    if train_batches is not None and len(train_batches) < steps:
        raise ValueError(f"need at least {steps} fixed training batches, got {len(train_batches)}")
    log_every = int(cfg.get("log_every", max(steps, 1)))
    grad_clip = cfg.get("grad_clip", None)
    diverged = False
    last_loss = None
    model.train()
    for step in range(1, steps + 1):
        if train_batches is None:
            x, y = make_batch(task, bs, device)
        else:
            x, y = train_batches[step - 1]
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        if not torch.isfinite(loss):
            diverged = True
            break
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(params, float(grad_clip))
        opt.step()
        last_loss = float(loss.item())
        if log_every and step % log_every == 0:
            print(f"lora step {step:5d}/{steps}: loss={last_loss:.4f}", flush=True)
    val = evaluate(
        model,
        eval_task or task,
        bs,
        eval_count,
        device,
        fixed_batches=validation_batches,
    )
    return {
        "train_loss_last": float(last_loss if last_loss is not None else val["loss"]),
        "val_loss": val["loss"],
        "val_accuracy": val["accuracy"],
        "val_n_examples": val["n_examples"],
        "val_n_batches": val["n_batches"],
        "trainable_params": lora_parameter_count(model),
    }, diverged
