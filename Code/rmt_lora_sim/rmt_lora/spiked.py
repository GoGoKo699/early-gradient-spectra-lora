from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectra import principal_subspace_overlap, svdvals


Array = np.ndarray


@dataclass
class SpikedMatrix:
    matrix: Array
    signal: Array
    noise: Array
    u_true: Array
    v_true: Array
    spikes: Array
    noise_sigma: float


def random_orthonormal(n: int, k: int, rng: np.random.Generator) -> Array:
    """Generate an n x k random orthonormal matrix."""
    if k > n:
        raise ValueError(f"k={k} cannot exceed n={n}")
    a = rng.normal(size=(n, k))
    q, r = np.linalg.qr(a, mode="reduced")
    # Fix signs for reproducibility across BLAS choices.
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    q = q * signs
    return q[:, :k]


def make_spiked_matrix(
    m: int,
    n: int,
    spikes: list[float] | Array,
    noise_sigma: float,
    rng: np.random.Generator,
    heavy_tail_df: float | None = None,
) -> SpikedMatrix:
    """Generate G = U diag(spikes) V^T + Z.

    Noise entries use the RMT scaling noise_sigma / sqrt(n). If heavy_tail_df is
    given, Z is Student-t noise rescaled to have approximately the same variance.
    """
    spikes_arr = np.asarray(spikes, dtype=float)
    rank = int(len(spikes_arr))
    u = random_orthonormal(m, rank, rng)
    v = random_orthonormal(n, rank, rng)
    signal = u @ np.diag(spikes_arr) @ v.T
    if heavy_tail_df is None:
        noise = rng.normal(0.0, noise_sigma / np.sqrt(n), size=(m, n))
    else:
        if heavy_tail_df <= 2:
            raise ValueError("heavy_tail_df must be > 2 for finite variance")
        raw = rng.standard_t(df=heavy_tail_df, size=(m, n))
        # Standard t variance = df/(df-2). Normalize to variance 1.
        raw = raw / np.sqrt(heavy_tail_df / (heavy_tail_df - 2.0))
        noise = (noise_sigma / np.sqrt(n)) * raw
    return SpikedMatrix(
        matrix=signal + noise,
        signal=signal,
        noise=noise,
        u_true=u,
        v_true=v,
        spikes=spikes_arr,
        noise_sigma=noise_sigma,
    )


def singular_vector_alignment(matrix: Array, u_true: Array, v_true: Array, k: int | None = None) -> dict[str, float]:
    """Alignment of empirical top singular subspaces with planted subspaces."""
    u_hat, _, vh_hat = np.linalg.svd(matrix, full_matrices=False)
    v_hat = vh_hat.T
    if k is None:
        k = min(u_true.shape[1], v_true.shape[1])
    return {
        "left_overlap": principal_subspace_overlap(u_hat[:, :k], u_true[:, :k]),
        "right_overlap": principal_subspace_overlap(v_hat[:, :k], v_true[:, :k]),
        "mean_overlap": 0.5
        * (
            principal_subspace_overlap(u_hat[:, :k], u_true[:, :k])
            + principal_subspace_overlap(v_hat[:, :k], v_true[:, :k])
        ),
    }


def make_overlapping_spikes(
    m: int,
    n: int,
    rank: int,
    overlap: float,
    rng: np.random.Generator,
) -> tuple[Array, Array, Array, Array]:
    """Make two pairs of singular subspaces with controllable overlap.

    overlap=1 gives identical subspaces; overlap=0 gives approximately orthogonal
    components mixed with fresh random directions. The construction is simple and
    intended for controlled simulations, not for exact Haar conditioning.
    """
    if not (0.0 <= overlap <= 1.0):
        raise ValueError("overlap must be in [0, 1]")
    u1 = random_orthonormal(m, rank, rng)
    v1 = random_orthonormal(n, rank, rng)
    u_rand = random_orthonormal(m, rank, rng)
    v_rand = random_orthonormal(n, rank, rng)
    # Remove components parallel to first subspace, then re-orthonormalize.
    u_res = u_rand - u1 @ (u1.T @ u_rand)
    v_res = v_rand - v1 @ (v1.T @ v_rand)
    u_res, _ = np.linalg.qr(u_res, mode="reduced")
    v_res, _ = np.linalg.qr(v_res, mode="reduced")
    c = float(np.sqrt(overlap))
    s = float(np.sqrt(max(0.0, 1.0 - overlap)))
    u2, _ = np.linalg.qr(c * u1 + s * u_res, mode="reduced")
    v2, _ = np.linalg.qr(c * v1 + s * v_res, mode="reduced")
    return u1, v1, u2[:, :rank], v2[:, :rank]


def matrix_from_svd(u: Array, spikes: Array, v: Array) -> Array:
    return u @ np.diag(np.asarray(spikes, dtype=float)) @ v.T


def spectral_interference_score(delta_a: Array, delta_b: Array, edge_a: float, edge_b: float) -> float:
    """Overlap among detectable singular directions of two task updates."""
    ua, sa, vha = np.linalg.svd(delta_a, full_matrices=False)
    ub, sb, vhb = np.linalg.svd(delta_b, full_matrices=False)
    ka = int(np.sum(sa > edge_a))
    kb = int(np.sum(sb > edge_b))
    if ka == 0 or kb == 0:
        return 0.0
    va = vha.T
    vb = vhb.T
    left = np.abs(ua[:, :ka].T @ ub[:, :kb]) ** 2
    right = np.abs(va[:, :ka].T @ vb[:, :kb]) ** 2
    weights = np.outer(sa[:ka], sb[:kb])
    return float(np.sum(weights * left * right) / (np.sum(weights) + 1e-12))
