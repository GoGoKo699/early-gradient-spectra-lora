"""Spectral summary helpers for the real-model validation."""
from __future__ import annotations

import math
import torch

EFFECTIVE_RANK_DEFINITION = "p_i = sigma_i^2 / sum_j sigma_j^2"



def entropy_effective_rank_from_energy(energy: torch.Tensor) -> float:
    """Entropy effective rank for nonnegative component energies."""
    values = energy.detach().to(dtype=torch.float64, device="cpu")
    if values.numel() == 0:
        return 0.0
    if not bool(torch.isfinite(values).all()):
        raise ValueError("component energies must all be finite")
    if bool((values < 0).any()):
        raise ValueError("component energies must be non-negative")
    scale = float(values.max().item())
    if scale <= 0.0:
        return 0.0
    probabilities = (values / scale) / (values / scale).sum()
    probabilities = probabilities[probabilities > 0]
    entropy = float(-(probabilities * torch.log(probabilities)).sum().item())
    return float(math.exp(entropy))

def singular_effective_rank(singular_values: torch.Tensor) -> float:
    """Entropy effective rank from normalized squared singular values."""
    values = singular_values.detach().to(dtype=torch.float64, device="cpu")
    if values.numel() == 0:
        return 0.0
    if not bool(torch.isfinite(values).all()):
        raise ValueError("singular values must all be finite")
    if bool((values < 0).any()):
        raise ValueError("singular values must be non-negative")
    scale = float(values.max().item())
    if scale <= 0.0:
        return 0.0
    # Scaling before squaring prevents avoidable overflow/underflow while
    # leaving the normalized spectral-energy probabilities unchanged.
    energy = (values / scale).square()
    probabilities = energy / energy.sum()
    probabilities = probabilities[probabilities > 0]
    entropy = float(-(probabilities * torch.log(probabilities)).sum().item())
    return float(math.exp(entropy))
