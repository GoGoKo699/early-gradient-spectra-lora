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


def permutation_null_maxima(mat: np.ndarray, n_boot: int = 8, seed: int = 0) -> np.ndarray:
    """Return top singular values from entry-permuted null matrices."""
    n_boot = int(n_boot)
    if n_boot < 0:
        raise ValueError("n_boot must be non-negative")
    if n_boot == 0:
        return np.empty(0, dtype=float)
    rng = np.random.default_rng(int(seed))
    flat = np.asarray(mat, dtype=float).ravel().copy()
    tops = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        rng.shuffle(flat)
        tops[index] = singular_values(flat.reshape(mat.shape))[0]
    return tops


def quantile_bootstrap_interval(
    values: np.ndarray,
    quantile: float,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0, 0.0
    if n_resamples <= 0:
        q = float(np.quantile(values, quantile, method="linear"))
        return q, q
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(n_resamples), dtype=float)
    for index in range(int(n_resamples)):
        sample = rng.choice(values, size=len(values), replace=True)
        estimates[index] = np.quantile(sample, quantile, method="linear")
    tail = (1.0 - float(confidence)) / 2.0
    return (
        float(np.quantile(estimates, tail, method="linear")),
        float(np.quantile(estimates, 1.0 - tail, method="linear")),
    )


def permutation_edge_diagnostics(
    mat: np.ndarray,
    n_boot: int = 8,
    quantile: float = 0.995,
    seed: int = 0,
    *,
    uncertainty_resamples: int = 1000,
    uncertainty_confidence: float = 0.95,
) -> tuple[dict[str, float | int | str], np.ndarray]:
    if not 0.0 < float(quantile) <= 1.0:
        raise ValueError("quantile must lie in (0, 1]")
    if not 0.0 < float(uncertainty_confidence) < 1.0:
        raise ValueError("uncertainty_confidence must lie in (0, 1)")
    maxima = permutation_null_maxima(mat, n_boot=n_boot, seed=seed)
    if len(maxima) == 0:
        diagnostics: dict[str, float | int | str] = {
            "edge": 0.0,
            "null_bootstrap": 0,
            "null_quantile": float(quantile),
            "null_quantile_method": "linear",
            "null_uncertainty_resamples": int(uncertainty_resamples),
            "edge_mc_confidence": float(uncertainty_confidence),
            "edge_mc_ci_low": 0.0,
            "edge_mc_ci_high": 0.0,
            "edge_q_0_99": 0.0,
            "edge_q_0_995": 0.0,
            "edge_q_0_999": 0.0,
        }
        return diagnostics, maxima
    edge = float(np.quantile(maxima, quantile, method="linear"))
    ci_low, ci_high = quantile_bootstrap_interval(
        maxima,
        quantile,
        n_resamples=uncertainty_resamples,
        seed=int(seed) + 1_000_003,
        confidence=float(uncertainty_confidence),
    )
    diagnostics = {
        "edge": edge,
        "null_bootstrap": int(len(maxima)),
        "null_quantile": float(quantile),
        "null_quantile_method": "linear",
        "null_uncertainty_resamples": int(uncertainty_resamples),
        "edge_mc_confidence": float(uncertainty_confidence),
        "edge_mc_ci_low": ci_low,
        "edge_mc_ci_high": ci_high,
        "edge_q_0_99": float(np.quantile(maxima, 0.99, method="linear")),
        "edge_q_0_995": float(np.quantile(maxima, 0.995, method="linear")),
        "edge_q_0_999": float(np.quantile(maxima, 0.999, method="linear")),
    }
    return diagnostics, maxima


def permutation_edge(mat: np.ndarray, n_boot: int = 8, quantile: float = 0.995, seed: int = 0) -> float:
    diagnostics, _ = permutation_edge_diagnostics(
        mat,
        n_boot=n_boot,
        quantile=quantile,
        seed=seed,
        uncertainty_resamples=0,
    )
    return float(diagnostics["edge"])


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
