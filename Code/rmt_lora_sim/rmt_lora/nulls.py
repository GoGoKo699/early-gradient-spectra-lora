from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .spectra import svdvals


Array = np.ndarray


@dataclass(frozen=True)
class NullEdgeResult:
    edge: float
    max_singular_values: Array
    method: str
    quantile: float


def permuted_matrix(matrix: Array, rng: np.random.Generator) -> Array:
    """Entry permutation null preserving the empirical entry distribution."""
    flat = np.asarray(matrix).ravel().copy()
    rng.shuffle(flat)
    return flat.reshape(matrix.shape)


def signflip_matrix(matrix: Array, rng: np.random.Generator) -> Array:
    """Random sign-flip null preserving entry magnitudes."""
    signs = rng.choice(np.array([-1.0, 1.0]), size=matrix.shape)
    return np.asarray(matrix) * signs


def gaussian_like_matrix(matrix: Array, rng: np.random.Generator, scale_by_n: bool = False) -> Array:
    """Gaussian null matching the empirical standard deviation of the entries.

    If scale_by_n=True, entries are N(0, sigma^2 / n) where sigma is estimated
    from matrix * sqrt(n). For generic empirical matrices, leave False.
    """
    m, n = matrix.shape
    if scale_by_n:
        sigma = float(np.std(matrix) * np.sqrt(n))
        return rng.normal(0.0, sigma / np.sqrt(n), size=(m, n))
    sigma = float(np.std(matrix))
    return rng.normal(0.0, sigma, size=(m, n))


def bootstrap_edge(
    matrix: Array,
    rng: np.random.Generator,
    method: Literal["permute", "signflip", "gaussian"] = "permute",
    n_boot: int = 100,
    quantile: float = 0.995,
) -> NullEdgeResult:
    """Estimate a bulk edge by repeatedly destroying structure in the matrix."""
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    max_svs = []
    for _ in range(n_boot):
        if method == "permute":
            z = permuted_matrix(matrix, rng)
        elif method == "signflip":
            z = signflip_matrix(matrix, rng)
        elif method == "gaussian":
            z = gaussian_like_matrix(matrix, rng)
        else:
            raise ValueError(f"unknown null method: {method}")
        max_svs.append(float(svdvals(z)[0]))
    max_svs_arr = np.asarray(max_svs)
    edge = float(np.quantile(max_svs_arr, quantile))
    return NullEdgeResult(edge=edge, max_singular_values=max_svs_arr, method=method, quantile=quantile)
