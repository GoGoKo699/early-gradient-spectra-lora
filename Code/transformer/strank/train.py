from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .tasks import IGNORE_INDEX, TaskSpec, make_batch
from .model import freeze_base_enable_lora, lora_parameters, lora_parameter_count


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=IGNORE_INDEX)


@torch.no_grad()
def evaluate(model, task: TaskSpec, batch_size: int, batches: int, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for _ in range(int(batches)):
        x, y = make_batch(task, batch_size, device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += float(loss.item())
        mask = y != IGNORE_INDEX
        pred = logits.argmax(dim=-1)
        total_correct += int((pred[mask] == y[mask]).sum().item())
        total_count += int(mask.sum().item())
    return {
        "loss": total_loss / max(int(batches), 1),
        "accuracy": total_correct / max(total_count, 1),
    }


def train_base(model, task: TaskSpec, cfg: Dict, device: torch.device, eval_task: TaskSpec | None = None) -> Dict[str, float]:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 0.003)), weight_decay=float(cfg.get("weight_decay", 0.0)))
    steps = int(cfg.get("steps", 100))
    bs = int(task.params.get("train_batch_size", cfg.get("batch_size", 64)))
    log_every = int(cfg.get("log_every", max(steps, 1)))
    last_loss = None
    for step in range(1, steps + 1):
        x, y = make_batch(task, bs, device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
        if log_every and step % log_every == 0:
            print(f"base step {step:5d}/{steps}: loss={last_loss:.4f}", flush=True)
    out = {"train_loss_last": float(last_loss if last_loss is not None else 0.0)}
    return out


def train_lora(model, task: TaskSpec, cfg: Dict, device: torch.device, eval_task: TaskSpec | None = None) -> Tuple[Dict[str, float], bool]:
    freeze_base_enable_lora(model)
    params = list(lora_parameters(model))
    if not params:
        metrics = evaluate(model, eval_task or task, int(task.params.get("train_batch_size", 64)), int(task.params.get("eval_batches", 4)), device)
        return {"train_loss_last": metrics["loss"], "val_loss": metrics["loss"], "val_accuracy": metrics["accuracy"], "trainable_params": 0}, False
    opt = torch.optim.AdamW(params, lr=float(cfg.get("lr", 0.01)), weight_decay=float(cfg.get("weight_decay", 0.0)))
    steps = int(cfg.get("steps", 100))
    bs = int(task.params.get("train_batch_size", cfg.get("batch_size", 64)))
    log_every = int(cfg.get("log_every", max(steps, 1)))
    grad_clip = cfg.get("grad_clip", None)
    diverged = False
    last_loss = None
    model.train()
    for step in range(1, steps + 1):
        x, y = make_batch(task, bs, device)
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
    val = evaluate(model, eval_task or task, bs, int(task.params.get("eval_batches", 4)), device)
    return {"train_loss_last": float(last_loss if last_loss is not None else val["loss"]), "val_loss": val["loss"], "val_accuracy": val["accuracy"], "trainable_params": lora_parameter_count(model)}, diverged
