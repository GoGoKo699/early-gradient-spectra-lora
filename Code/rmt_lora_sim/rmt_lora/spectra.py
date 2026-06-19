from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class SpectrumSummary:
    singular_values: Array
    mp_edge: float
    detectable_rank: int
    stable_rank: float
    effective_rank: float
    frobenius_norm: float
    nuclear_norm: float


def svdvals(matrix: Array) -> Array:
    """Return singular values in descending order."""
    return np.linalg.svd(matrix, compute_uv=False)


def mp_singular_edge(m: int, n: int, noise_sigma: float = 1.0) -> float:
    """Marchenko-Pastur upper singular-value edge.

    Assumes a noise matrix Z_{m x n} with entries N(0, noise_sigma^2 / n).
    The asymptotic top singular value is noise_sigma * (1 + sqrt(m/n)).
    """
    if n <= 0 or m <= 0:
        raise ValueError("m and n must be positive")
    return float(noise_sigma * (1.0 + np.sqrt(m / n)))


def stable_rank(s: Array) -> float:
    """Stable rank ||M||_F^2 / ||M||_2^2 from singular values."""
    if len(s) == 0 or s[0] <= 0:
        return 0.0
    return float(np.sum(s**2) / (s[0] ** 2))


def effective_rank(s: Array, eps: float = 1e-12) -> float:
    """Entropy effective rank exp(H(p)), where p_i proportional to sigma_i^2."""
    energy = s**2
    total = float(np.sum(energy))
    if total <= eps:
        return 0.0
    p = energy / total
    h = -float(np.sum(p * np.log(p + eps)))
    return float(np.exp(h))


def entropy_from_singular_values(s: Array, normalize_by_log_rank: bool = False, eps: float = 1e-12) -> float:
    """Von-Neumann-style entropy from squared normalized singular values."""
    energy = s**2
    total = float(np.sum(energy))
    if total <= eps:
        return 0.0
    p = energy / total
    h = -float(np.sum(p * np.log(p + eps)))
    if normalize_by_log_rank and len(s) > 1:
        h /= np.log(len(s))
    return h


def detectable_rank_from_singular_values(
    s: Array,
    edge: float,
    margin: float = 1.0,
) -> int:
    """Count singular values above a bulk edge."""
    return int(np.sum(s > margin * edge))


def summarize_spectrum(
    matrix: Array,
    edge: float | None = None,
    noise_sigma: float = 1.0,
    margin: float = 1.0,
) -> SpectrumSummary:
    """Compute basic spectral diagnostics."""
    m, n = matrix.shape
    s = svdvals(matrix)
    if edge is None:
        edge = mp_singular_edge(m, n, noise_sigma=noise_sigma)
    return SpectrumSummary(
        singular_values=s,
        mp_edge=float(edge),
        detectable_rank=detectable_rank_from_singular_values(s, edge, margin=margin),
        stable_rank=stable_rank(s),
        effective_rank=effective_rank(s),
        frobenius_norm=float(np.sqrt(np.sum(s**2))),
        nuclear_norm=float(np.sum(s)),
    )


def tensor_entropy_profile(
    matrix: Array,
    dims: list[int] | tuple[int, ...],
    log_base: Literal["e", "2"] = "e",
    normalize: bool = False,
    eps: float = 1e-12,
) -> Array:
    """Compute an MPS-like entropy profile for a tensorized matrix.

    Parameters
    ----------
    matrix:
        Matrix whose total number of elements must equal prod(dims).
    dims:
        Tensorization dimensions. The matrix is reshaped in C order.
    log_base:
        Natural log or log2.
    normalize:
        If True, divide each entropy by log(min(left_dim, right_dim)).

    Notes
    -----
    This is a diagnostic only. Different tensor index orderings can change the
    profile, so store `dims` and the reshape convention in every experiment.
    """
    arr = np.asarray(matrix)
    total = int(np.prod(dims))
    if arr.size != total:
        raise ValueError(f"matrix has {arr.size} entries but dims multiply to {total}")
    tensor = arr.reshape(tuple(dims), order="C")
    entropies: list[float] = []
    for cut in range(1, len(dims)):
        left_dim = int(np.prod(dims[:cut]))
        right_dim = int(np.prod(dims[cut:]))
        flat = tensor.reshape((left_dim, right_dim), order="C")
        s = svdvals(flat)
        h = entropy_from_singular_values(s, normalize_by_log_rank=False, eps=eps)
        if log_base == "2":
            h /= np.log(2.0)
        if normalize and min(left_dim, right_dim) > 1:
            denom = np.log(min(left_dim, right_dim))
            if log_base == "2":
                denom /= np.log(2.0)
            h /= denom
        entropies.append(float(h))
    return np.asarray(entropies)


def principal_subspace_overlap(u_hat: Array, u_true: Array, k: int | None = None) -> float:
    """Return normalized squared Frobenius overlap between two subspaces.

    If U and V have orthonormal columns, this is ||U^T V||_F^2 / min(k_U, k_V).
    It is 1 for identical subspaces and approximately 0 for orthogonal ones.
    """
    if u_hat.size == 0 or u_true.size == 0:
        return 0.0
    if k is not None:
        u_hat = u_hat[:, :k]
        u_true = u_true[:, :k]
    denom = min(u_hat.shape[1], u_true.shape[1])
    if denom <= 0:
        return 0.0
    return float(np.linalg.norm(u_hat.T @ u_true, ord="fro") ** 2 / denom)


def vector_alignment(a: Array, b: Array, eps: float = 1e-12) -> float:
    """Squared cosine alignment between vectors."""
    aa = float(np.linalg.norm(a))
    bb = float(np.linalg.norm(b))
    if aa <= eps or bb <= eps:
        return 0.0
    return float((np.dot(a.ravel(), b.ravel()) / (aa * bb)) ** 2)
