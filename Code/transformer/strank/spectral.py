from __future__ import annotations

import numpy as np


def singular_values(mat: np.ndarray) -> np.ndarray:
    return np.linalg.svd(mat, compute_uv=False)


def spectral_entropy_from_s(s: np.ndarray, eps: float = 1e-12) -> float:
    e = s.astype(float) ** 2
    total = float(e.sum())
    if total <= eps:
        return 0.0
    p = e / total
    return float(-(p * np.log(p + eps)).sum())


def effective_rank_from_s(s: np.ndarray) -> float:
    return float(np.exp(spectral_entropy_from_s(s)))


def stable_rank_from_s(s: np.ndarray, eps: float = 1e-12) -> float:
    if len(s) == 0 or s[0] <= eps:
        return 0.0
    return float(np.sum(s**2) / (s[0] ** 2))


def soft_dimension_from_s(s: np.ndarray, edge: float, eps: float = 1e-12) -> float:
    if edge <= eps:
        return float(np.count_nonzero(s > eps))
    return float(np.sum((s**2) / (s**2 + edge**2 + eps)))


def hard_rank_from_s(s: np.ndarray, edge: float) -> int:
    return int(np.sum(s > edge))


def permutation_edge(mat: np.ndarray, n_boot: int = 8, quantile: float = 0.995, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    if n_boot <= 0:
        return 0.0
    flat = mat.ravel().copy()
    tops = []
    for _ in range(int(n_boot)):
        rng.shuffle(flat)
        tops.append(singular_values(flat.reshape(mat.shape))[0])
    return float(np.quantile(tops, quantile))


def summarize_matrix(mat: np.ndarray, edge: float) -> dict:
    s = singular_values(mat)
    return {
        "top_sv": float(s[0]) if len(s) else 0.0,
        "frobenius": float(np.linalg.norm(s)),
        "nuclear": float(np.sum(s)),
        "effective_rank": effective_rank_from_s(s),
        "stable_rank": stable_rank_from_s(s),
        "hard_detectable_rank": hard_rank_from_s(s, edge),
        "soft_dimension": soft_dimension_from_s(s, edge),
        "edge": float(edge),
        "singular_values": s,
    }
