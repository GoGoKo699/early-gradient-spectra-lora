from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .spiked import random_orthonormal
from .spectra import effective_rank, stable_rank, summarize_spectrum, svdvals
from .nulls import bootstrap_edge


Array = np.ndarray


@dataclass
class LinearTask:
    x_train: Array
    y_train: Array
    x_val: Array
    y_val: Array
    w_base: Array
    delta_true: Array
    u_true: Array
    v_true: Array
    spikes: Array
    noise_std: float
    x_eigs: Array


@dataclass
class LoRAResult:
    rank: int
    alpha: float
    delta: Array
    train_loss_trace: list[float]
    val_loss_trace: list[float]
    final_train_loss: float
    final_val_loss: float
    forgetting_loss: float
    diverged: bool
    steps_completed: int
    diagnostics: dict[str, Any]


def make_activation_matrix(n: int, eigs: Array, rng: np.random.Generator) -> Array:
    z = rng.normal(size=(n, len(eigs)))
    return z * np.sqrt(eigs)[None, :]


def make_x_eigs(d_in: int, spectrum: str = "isotropic", decay: float = 1.0) -> Array:
    if spectrum == "isotropic":
        eigs = np.ones(d_in)
    elif spectrum == "powerlaw":
        idx = np.arange(1, d_in + 1, dtype=float)
        eigs = idx ** (-decay)
        eigs = eigs / np.mean(eigs)
    elif spectrum == "two_scale":
        eigs = np.ones(d_in)
        eigs[: max(1, d_in // 8)] = 5.0
        eigs = eigs / np.mean(eigs)
    else:
        raise ValueError(f"unknown activation spectrum: {spectrum}")
    return eigs.astype(float)


def make_linear_task(
    d_in: int,
    d_out: int,
    n_train: int,
    n_val: int,
    task_rank: int,
    task_spikes: list[float] | None,
    rng: np.random.Generator,
    label_noise_std: float = 0.05,
    base_scale: float = 0.2,
    activation_spectrum: str = "isotropic",
    activation_decay: float = 1.0,
) -> LinearTask:
    """Create a synthetic linear adaptation task.

    The base model is y = W0 x. The task model is y = (W0 + Delta*) x + eps,
    where Delta* is low-rank. This gives a controlled analogue of layerwise
    fine-tuning with generated data only.
    """
    if task_spikes is None:
        # Mildly decaying default spikes.
        task_spikes = [2.5 / (1.0 + 0.25 * i) for i in range(task_rank)]
    spikes = np.asarray(task_spikes, dtype=float)
    if len(spikes) != task_rank:
        raise ValueError("len(task_spikes) must equal task_rank")
    u = random_orthonormal(d_out, task_rank, rng)
    v = random_orthonormal(d_in, task_rank, rng)
    delta_true = u @ np.diag(spikes) @ v.T
    w_base = rng.normal(0.0, base_scale / np.sqrt(d_in), size=(d_out, d_in))
    x_eigs = make_x_eigs(d_in, spectrum=activation_spectrum, decay=activation_decay)
    x_train = make_activation_matrix(n_train, x_eigs, rng)
    x_val = make_activation_matrix(n_val, x_eigs, rng)
    w_task = w_base + delta_true
    y_train = x_train @ w_task.T + rng.normal(0.0, label_noise_std, size=(n_train, d_out))
    y_val = x_val @ w_task.T + rng.normal(0.0, label_noise_std, size=(n_val, d_out))
    return LinearTask(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        w_base=w_base,
        delta_true=delta_true,
        u_true=u,
        v_true=v,
        spikes=spikes,
        noise_std=label_noise_std,
        x_eigs=x_eigs,
    )


def mse_loss(x: Array, y: Array, w_base: Array, delta: Array) -> float:
    pred = x @ (w_base + delta).T
    return float(np.mean((pred - y) ** 2))


def forgetting_loss(x: Array, delta: Array) -> float:
    """Output drift on the base task caused by the adapter."""
    drift = x @ delta.T
    return float(np.mean(drift**2))


def full_gradient_delta(x: Array, y: Array, w_base: Array, delta: Array | None = None) -> Array:
    """Gradient d MSE / d Delta for the linear adaptation problem."""
    if delta is None:
        delta = np.zeros_like(w_base)
    pred = x @ (w_base + delta).T
    err = pred - y
    n, d_out = err.shape
    return (2.0 / (n * d_out)) * (err.T @ x)


def descent_direction_at_init(task: LinearTask) -> Array:
    """Negative full gradient at Delta=0.

    In the noiseless linear model this raw descent direction is proportional to
    Delta_* C, where C = E[x x^T].  The theory in the paper uses the
    right-whitened object Delta_* C^{1/2}; use ``early_gradient_matrix`` with
    ``matrix_mode="activation_whitened"`` to obtain that estimator.
    """
    return -full_gradient_delta(task.x_train, task.y_train, task.w_base)


def activation_second_moment(x: Array) -> Array:
    """Empirical uncentered activation second moment X^T X / n."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] <= 0:
        raise ValueError("x must be a non-empty 2D activation matrix")
    return (x.T @ x) / float(x.shape[0])


def _symmetric_matrix_power(matrix: Array, power: float, ridge: float = 0.0, eps: float = 1e-12) -> tuple[Array, dict[str, float]]:
    """Return (matrix + ridge I)^power and basic eigen diagnostics."""
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("matrix must be square")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    sym = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals, 0.0)
    shifted = np.maximum(eigvals + ridge, eps)
    mat_power = (eigvecs * (shifted ** power)[None, :]) @ eigvecs.T
    positive = eigvals[eigvals > 0]
    min_pos = float(np.min(positive)) if len(positive) else 0.0
    max_eig = float(np.max(eigvals)) if len(eigvals) else 0.0
    condition = float(max_eig / min_pos) if min_pos > 0 else float("inf")
    return mat_power, {
        "activation_cov_min_eig": float(np.min(eigvals)) if len(eigvals) else 0.0,
        "activation_cov_min_positive_eig": min_pos,
        "activation_cov_max_eig": max_eig,
        "activation_cov_condition": condition,
        "activation_whitening_ridge": float(ridge),
    }


def _ridge_from_scale(cov: Array, ridge_scale: float) -> float:
    d = cov.shape[0]
    return float(ridge_scale) * float(np.trace(cov)) / float(d)


def normalize_gradient_matrix_mode(matrix_mode: str) -> str:
    """Normalize aliases for the early-gradient matrix used in diagnostics."""
    mode = str(matrix_mode).strip().lower().replace("-", "_")
    aliases = {
        "none": "raw",
        "unwhitened": "raw",
        "raw_gradient": "raw",
        "full_gradient": "raw",
        "covariance_whitened": "activation_whitened",
        "empirical_whitened": "activation_whitened",
        "right_whitened": "activation_whitened",
        "whitened": "activation_whitened",
        "activation_whitened_gradient": "activation_whitened",
        "oracle_whitened": "oracle_activation_whitened",
        "population_whitened": "oracle_activation_whitened",
    }
    mode = aliases.get(mode, mode)
    allowed = {"raw", "activation_whitened", "oracle_activation_whitened"}
    if mode not in allowed:
        raise ValueError(f"unknown gradient matrix mode {matrix_mode!r}; use one of {sorted(allowed)}")
    return mode


def useful_task_operator(task: LinearTask, covariance: Array | None = None) -> Array:
    """Return Delta_* C^{1/2}, the object whose spectrum sets useful rank."""
    cov = activation_second_moment(task.x_train) if covariance is None else covariance
    sqrt_cov, _ = _symmetric_matrix_power(cov, 0.5, ridge=0.0)
    return task.delta_true @ sqrt_cov


def early_gradient_matrix(
    task: LinearTask,
    matrix_mode: str = "activation_whitened",
    ridge_scale: float = 1e-6,
) -> tuple[Array, dict[str, float | str]]:
    """Return the early-gradient matrix for spectral diagnostics.

    ``raw`` returns the negative full gradient, proportional to Delta_* C.
    ``activation_whitened`` right-multiplies by empirical C^{-1/2}, producing
    an estimator proportional to Delta_* C^{1/2}.
    ``oracle_activation_whitened`` uses the synthetic population eigenvalues and
    is intended only as a diagnostic.
    """
    mode = normalize_gradient_matrix_mode(matrix_mode)
    raw = descent_direction_at_init(task)
    if mode == "raw":
        return raw, {"gradient_matrix_mode": mode, "activation_whitening_ridge": 0.0}
    if mode == "activation_whitened":
        cov = activation_second_moment(task.x_train)
        ridge = _ridge_from_scale(cov, ridge_scale)
        inv_sqrt, info = _symmetric_matrix_power(cov, -0.5, ridge=ridge)
        return raw @ inv_sqrt, {"gradient_matrix_mode": mode, "gradient_ridge_scale": float(ridge_scale), **info}
    eigs = np.asarray(task.x_eigs, dtype=float)
    cov = np.diag(eigs)
    ridge = _ridge_from_scale(cov, ridge_scale)
    inv_sqrt, info = _symmetric_matrix_power(cov, -0.5, ridge=ridge)
    return raw @ inv_sqrt, {"gradient_matrix_mode": mode, "gradient_ridge_scale": float(ridge_scale), **info}


def train_lora_fullbatch(
    task: LinearTask,
    rank: int,
    alpha: float,
    lr: float,
    steps: int,
    rng: np.random.Generator,
    weight_decay: float = 0.0,
    init_scale: float = 0.01,
    eval_every: int = 25,
    batch_size: int | None = None,
    grad_clip: float | None = None,
    stop_on_diverge: bool = True,
) -> LoRAResult:
    """Train a LoRA-factorized adapter with NumPy SGD.

    Delta = (alpha / rank) * B A, with B shape d_out x rank and A shape
    rank x d_in. A is initialized randomly and B at zero, matching the usual
    LoRA convention that the initial adapter is exactly zero.
    """
    if rank <= 0:
        raise ValueError("rank must be positive")
    x_train, y_train = task.x_train, task.y_train
    n_train, d_in = x_train.shape
    d_out = y_train.shape[1]
    scale = alpha / rank
    a = rng.normal(0.0, init_scale / np.sqrt(d_in), size=(rank, d_in))
    b = np.zeros((d_out, rank))
    train_trace: list[float] = []
    val_trace: list[float] = []
    diverged = False
    steps_completed = 0

    for step in range(1, steps + 1):
        if batch_size is None or batch_size >= n_train:
            xb, yb = x_train, y_train
        else:
            idx = rng.choice(n_train, size=batch_size, replace=False)
            xb, yb = x_train[idx], y_train[idx]
        delta = scale * (b @ a)
        pred = xb @ (task.w_base + delta).T
        err = pred - yb
        n_batch = xb.shape[0]
        grad_delta = (2.0 / (n_batch * d_out)) * (err.T @ xb)
        grad_b = scale * (grad_delta @ a.T) + weight_decay * b
        grad_a = scale * (b.T @ grad_delta) + weight_decay * a
        if grad_clip is not None:
            grad_norm = float(np.sqrt(np.sum(grad_b**2) + np.sum(grad_a**2)))
            if grad_norm > grad_clip:
                factor = grad_clip / (grad_norm + 1e-12)
                grad_b *= factor
                grad_a *= factor
        b -= lr * grad_b
        a -= lr * grad_a
        steps_completed = step

        if step % eval_every == 0 or step == 1 or step == steps:
            delta_eval = scale * (b @ a)
            tr_loss = mse_loss(x_train, y_train, task.w_base, delta_eval)
            va_loss = mse_loss(task.x_val, task.y_val, task.w_base, delta_eval)
            train_trace.append(tr_loss)
            val_trace.append(va_loss)
            if not np.isfinite(tr_loss) or not np.isfinite(va_loss) or max(tr_loss, va_loss) > 1e8:
                diverged = True
                if stop_on_diverge:
                    break

    delta_final = scale * (b @ a)
    final_train = mse_loss(x_train, y_train, task.w_base, delta_final)
    final_val = mse_loss(task.x_val, task.y_val, task.w_base, delta_final)
    f_loss = forgetting_loss(task.x_val, delta_final)
    diagnostics = adapter_diagnostics(task, delta_final)
    return LoRAResult(
        rank=rank,
        alpha=float(alpha),
        delta=delta_final,
        train_loss_trace=train_trace,
        val_loss_trace=val_trace,
        final_train_loss=final_train,
        final_val_loss=final_val,
        forgetting_loss=f_loss,
        diverged=diverged,
        steps_completed=steps_completed,
        diagnostics=diagnostics,
    )


def _subspace_overlap_from_svd(matrix: Array, reference: Array, k: int) -> tuple[float, float, float]:
    if k <= 0:
        return 0.0, 0.0, 0.0
    u_hat, _, vh_hat = np.linalg.svd(matrix, full_matrices=False)
    u_ref, _, vh_ref = np.linalg.svd(reference, full_matrices=False)
    k_eff = min(k, u_hat.shape[1], u_ref.shape[1], vh_hat.shape[0], vh_ref.shape[0])
    if k_eff <= 0:
        return 0.0, 0.0, 0.0
    v_hat = vh_hat.T
    v_ref = vh_ref.T
    left = float(np.linalg.norm(u_hat[:, :k_eff].T @ u_ref[:, :k_eff], ord="fro") ** 2 / k_eff)
    right = float(np.linalg.norm(v_hat[:, :k_eff].T @ v_ref[:, :k_eff], ord="fro") ** 2 / k_eff)
    return left, right, 0.5 * (left + right)


def adapter_diagnostics(task: LinearTask, delta: Array, edge: float | None = None) -> dict[str, Any]:
    s = svdvals(delta)
    if edge is None:
        # A conservative default: compare to an entry-permuted null externally when possible.
        edge = 0.0
    true_rank = task.u_true.shape[1]
    left_overlap, right_overlap, mean_overlap = _subspace_overlap_from_svd(delta, task.delta_true, true_rank)
    return {
        "adapter_frobenius": float(np.sqrt(np.sum(s**2))),
        "adapter_nuclear": float(np.sum(s)),
        "adapter_stable_rank": stable_rank(s),
        "adapter_effective_rank": effective_rank(s),
        "adapter_top_sv": float(s[0]) if len(s) else 0.0,
        "adapter_left_overlap_true": left_overlap,
        "adapter_right_overlap_true": right_overlap,
        "adapter_mean_overlap_true": mean_overlap,
    }


def _prefixed_gradient_summary(prefix: str, matrix: Array, rng: np.random.Generator, null_method: str, n_boot: int, quantile: float) -> dict[str, Any]:
    null = bootstrap_edge(matrix, rng, method=null_method, n_boot=n_boot, quantile=quantile)
    summary = summarize_spectrum(matrix, edge=null.edge)
    return {
        f"{prefix}edge": null.edge,
        f"{prefix}detectable_rank": summary.detectable_rank,
        f"{prefix}stable_rank": summary.stable_rank,
        f"{prefix}effective_rank": summary.effective_rank,
        f"{prefix}frobenius": summary.frobenius_norm,
        f"{prefix}nuclear": summary.nuclear_norm,
        f"{prefix}top_sv": float(summary.singular_values[0]) if len(summary.singular_values) else 0.0,
        f"{prefix}null_method": null.method,
        f"{prefix}null_quantile": null.quantile,
    }


def estimate_gradient_detectable_rank(
    task: LinearTask,
    rng: np.random.Generator,
    null_method: str = "permute",
    n_boot: int = 50,
    quantile: float = 0.995,
    matrix_mode: str = "activation_whitened",
    whitening: str | None = None,
    ridge_scale: float = 1e-6,
    include_raw: bool = True,
) -> dict[str, Any]:
    """Compute early-gradient spectral features and a bootstrap null edge.

    Public ``gradient_*`` fields use ``matrix_mode``.  The default is the
    activation-whitened estimator ``(-dL/dDelta) C^{-1/2}``, matching the
    theorem's ``Delta_* C^{1/2}`` spectrum.  Set ``matrix_mode='raw'`` to
    reproduce older unwhitened diagnostics.  With ``include_raw=True``, the raw
    diagnostics are also emitted under ``raw_gradient_*`` for ablations.
    """
    if whitening is not None:
        matrix_mode = whitening
    mode = normalize_gradient_matrix_mode(matrix_mode)
    g, info = early_gradient_matrix(task, matrix_mode=mode, ridge_scale=ridge_scale)
    selected = _prefixed_gradient_summary("gradient_", g, rng, null_method, n_boot, quantile)

    cov = activation_second_moment(task.x_train)
    if mode == "raw":
        reference = task.delta_true @ cov
    elif mode == "oracle_activation_whitened":
        reference = task.delta_true @ np.diag(np.sqrt(np.asarray(task.x_eigs, dtype=float)))
    else:
        reference = useful_task_operator(task, covariance=cov)
    k = min(task.u_true.shape[1], min(g.shape), min(reference.shape))
    left_overlap, right_overlap, mean_overlap = _subspace_overlap_from_svd(g, reference, k)

    result = {
        **selected,
        "gradient_left_overlap_true": left_overlap,
        "gradient_right_overlap_true": right_overlap,
        "gradient_mean_overlap_true": mean_overlap,
        "gradient_left_overlap_useful_operator": left_overlap,
        "gradient_right_overlap_useful_operator": right_overlap,
        "gradient_mean_overlap_useful_operator": mean_overlap,
        "gradient_matrix_mode": mode,
        "gradient_whitening": mode,
        "gradient_cov_trace": float(np.trace(cov)),
        "gradient_cov_effective_rank": float((np.trace(cov) ** 2) / (np.sum(cov * cov) + 1e-12)),
        **info,
    }

    if mode == "activation_whitened":
        for suffix in [
            "edge",
            "detectable_rank",
            "stable_rank",
            "effective_rank",
            "frobenius",
            "nuclear",
            "top_sv",
            "null_method",
            "null_quantile",
            "left_overlap_true",
            "right_overlap_true",
            "mean_overlap_true",
        ]:
            src = f"gradient_{suffix}"
            if src in result:
                result[f"whitened_gradient_{suffix}"] = result[src]
        result["whitened_gradient_matrix_mode"] = mode

    if include_raw:
        raw_g, _ = early_gradient_matrix(task, matrix_mode="raw", ridge_scale=0.0)
        raw_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        raw = _prefixed_gradient_summary("raw_gradient_", raw_g, raw_rng, null_method, n_boot, quantile)
        raw_left, raw_right, raw_mean = _subspace_overlap_from_svd(raw_g, task.delta_true @ cov, k)
        result.update({
            **raw,
            "raw_gradient_left_overlap_true": raw_left,
            "raw_gradient_right_overlap_true": raw_right,
            "raw_gradient_mean_overlap_true": raw_mean,
        })
    return result


def intruder_diagnostics(
    task: LinearTask,
    delta: Array,
    edge: float,
    overlap_threshold: float = 0.05,
) -> dict[str, Any]:
    """Count large adapter singular directions poorly aligned with true task subspaces."""
    u, s, vh = np.linalg.svd(delta, full_matrices=False)
    v = vh.T
    detectable = s > edge
    intruder_count = 0
    intruder_energy = 0.0
    signal_energy = 0.0
    for i, is_det in enumerate(detectable):
        if not is_det:
            continue
        left = float(np.linalg.norm(task.u_true.T @ u[:, i]) ** 2)
        right = float(np.linalg.norm(task.v_true.T @ v[:, i]) ** 2)
        overlap = left * right
        if overlap < overlap_threshold:
            intruder_count += 1
            intruder_energy += float(s[i] ** 2)
        else:
            signal_energy += float(s[i] ** 2)
    return {
        "intruder_count": intruder_count,
        "intruder_energy": intruder_energy,
        "detectable_signal_energy": signal_energy,
        "detectable_adapter_rank": int(np.sum(detectable)),
    }
