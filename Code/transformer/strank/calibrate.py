from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from .model import LoRALinear, iter_lora_modules
from .tasks import TaskSpec, make_batch
from .train import loss_fn
from .spectral import permutation_edge_diagnostics, summarize_matrix
from .utils import stable_seed


@dataclass
class Accum:
    c: torch.Tensor
    g: torch.Tensor
    n_tokens: int = 0
    n_batches: int = 0


def select_sites(model, include: List[str] | None) -> Dict[str, LoRALinear]:
    mods = dict(iter_lora_modules(model))
    if include is None or len(include) == 0:
        return mods
    missing = [x for x in include if x not in mods]
    if missing:
        raise ValueError(f"unknown module sites: {missing}; available={list(mods)}")
    return {name: mods[name] for name in include}


def inv_sqrt(mat: np.ndarray, whitening: str, lambda_scale: float) -> np.ndarray:
    d = mat.shape[0]
    lam = float(lambda_scale) * float(np.trace(mat) / max(d, 1))
    if whitening == "none":
        return np.eye(d, dtype=np.float64)
    if whitening == "diag":
        diag = np.diag(mat).copy()
        return np.diag(1.0 / np.sqrt(diag + lam + 1e-12))
    if whitening == "full":
        vals, vecs = np.linalg.eigh(mat + lam * np.eye(d))
        vals = np.clip(vals, 1e-12, None)
        return (vecs * (1.0 / np.sqrt(vals))[None, :]) @ vecs.T
    raise ValueError(f"unknown whitening={whitening}")


def calibrate_module_spectra(
    model,
    task: TaskSpec,
    cfg: Dict,
    site_names: List[str],
    device: torch.device,
    seed: int = 0,
    *,
    data_seed: int | None = None,
    return_null_maxima: bool = False,
):
    # Evaluation mode disables dropout while retaining gradients. Preserve the
    # caller's mode so calibration has no hidden stateful side effect.
    was_training = bool(model.training)
    model.eval()
    sites = select_sites(model, site_names)
    max_tokens_per_site = int(cfg.get("max_tokens_per_site", 0) or 0)
    accums: Dict[str, Accum] = {}
    current_inputs: Dict[str, torch.Tensor] = {}
    handles = []

    for name, module in sites.items():
        d_in, d_out = module.in_features, module.out_features
        accums[name] = Accum(
            c=torch.zeros(d_in, d_in, device=device, dtype=torch.float64),
            g=torch.zeros(d_out, d_in, device=device, dtype=torch.float64),
        )

        def make_fwd(nm):
            def hook(mod, inp, out):
                current_inputs[nm] = inp[0].detach()
            return hook

        def make_bwd(nm):
            def hook(mod, grad_in, grad_out):
                x = current_inputs[nm]
                gamma = grad_out[0].detach()
                X = x.reshape(-1, x.shape[-1]).to(torch.float64)
                Gm = gamma.reshape(-1, gamma.shape[-1]).to(torch.float64)
                # Ignore exactly-zero output-gradient rows to reduce padding/noncontributing tokens.
                mask = torch.linalg.norm(Gm, dim=1) > 0
                X = X[mask]
                Gm = Gm[mask]
                if X.numel() == 0:
                    return
                a = accums[nm]
                if max_tokens_per_site > 0:
                    remaining = max_tokens_per_site - a.n_tokens
                    if remaining <= 0:
                        return
                    if X.shape[0] > remaining:
                        X = X[:remaining]
                        Gm = Gm[:remaining]
                a.c += X.T @ X
                a.g += Gm.T @ X
                a.n_tokens += X.shape[0]
                a.n_batches += 1
            return hook

        handles.append(module.register_forward_hook(make_fwd(name)))
        handles.append(module.register_full_backward_hook(make_bwd(name)))

    batches = int(cfg.get("batches", 4))
    batch_size = int(cfg.get("batch_size", task.params.get("train_batch_size", 64)))
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    data_generator = torch.Generator(device=generator_device)
    data_generator.manual_seed(int(seed if data_seed is None else data_seed))
    try:
        for _ in range(batches):
            x, y = make_batch(task, batch_size, device, generator=data_generator)
            model.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            if max_tokens_per_site > 0 and all(
                a.n_tokens >= max_tokens_per_site for a in accums.values()
            ):
                break
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    rows = []
    sv_dict = {}
    null_dict = {}
    whitening = str(cfg.get("whitening", "diag"))
    lambda_scale = float(cfg.get("lambda_scale", 0.01))
    n_boot = int(cfg.get("null_bootstrap", 8))
    quantile = float(cfg.get("null_quantile", 0.995))
    uncertainty_resamples = int(cfg.get("null_uncertainty_resamples", 1000))
    uncertainty_confidence = float(cfg.get("null_uncertainty_confidence", 0.95))

    for name, a in accums.items():
        n = max(a.n_tokens, 1)
        C = (a.c / n).detach().cpu().numpy()
        # ``a.g`` accumulates autograd gradients of the mean loss.  Within each
        # backward call, PyTorch has already averaged token contributions through
        # the cross-entropy reduction, so dividing by backward calls estimates the
        # expected full-gradient matrix for the calibration distribution.
        G = (a.g / max(a.n_batches, 1)).detach().cpu().numpy()
        W = inv_sqrt(C, whitening=whitening, lambda_scale=lambda_scale)
        M = -G @ W
        site_null_seed = stable_seed(int(seed), "permutation_null", name)
        edge_diagnostics, null_maxima = permutation_edge_diagnostics(
            M,
            n_boot=n_boot,
            quantile=quantile,
            seed=site_null_seed,
            uncertainty_resamples=uncertainty_resamples,
            uncertainty_confidence=uncertainty_confidence,
        )
        summary = summarize_matrix(M, edge=float(edge_diagnostics["edge"]))
        mod = sites[name]
        sv_dict[name] = summary.pop("singular_values")
        null_dict[name] = null_maxima
        rows.append({
            "site_name": name,
            "d_in": mod.in_features,
            "d_out": mod.out_features,
            "n_tokens": a.n_tokens,
            "n_batches": a.n_batches,
            "whitening": whitening,
            "lambda_scale": lambda_scale,
            "calibration_data_seed": int(seed if data_seed is None else data_seed),
            "permutation_seed": site_null_seed,
            "dropout_disabled_during_calibration": True,
            **edge_diagnostics,
            "gradient_norm": float(np.linalg.norm(G)),
            "cov_trace": float(np.trace(C)),
            "cov_effective_rank": float((np.trace(C) ** 2) / (np.sum(C * C) + 1e-12)),
            **summary,
        })
    if return_null_maxima:
        return rows, sv_dict, null_dict
    return rows, sv_dict
