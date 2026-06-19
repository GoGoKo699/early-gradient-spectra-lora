from __future__ import annotations

import numpy as np

from rmt_lora.lora_sim import (
    LinearTask,
    adapter_diagnostics,
    activation_second_moment,
    early_gradient_matrix,
    estimate_gradient_detectable_rank,
    full_gradient_delta,
    make_linear_task,
    mse_loss,
    useful_task_operator,
)
from rmt_lora.spiked import random_orthonormal
from rmt_lora.spectra import effective_rank, mp_singular_edge, stable_rank


def _exact_covariance_task() -> LinearTask:
    rng = np.random.default_rng(123)
    d_in, d_out, rank = 5, 4, 2
    eigs = np.array([5.0, 2.0, 1.0, 0.5, 0.25])
    n = d_in
    x = np.sqrt(n) * np.diag(np.sqrt(eigs))
    u = random_orthonormal(d_out, rank, rng)
    v = random_orthonormal(d_in, rank, rng)
    spikes = np.array([3.0, 1.25])
    delta = u @ np.diag(spikes) @ v.T
    w_base = np.zeros((d_out, d_in))
    y = x @ delta.T
    return LinearTask(
        x_train=x,
        y_train=y,
        x_val=x.copy(),
        y_val=y.copy(),
        w_base=w_base,
        delta_true=delta,
        u_true=u,
        v_true=v,
        spikes=spikes,
        noise_std=0.0,
        x_eigs=eigs,
    )


def test_effective_rank_zero_matrix_is_zero() -> None:
    assert effective_rank(np.zeros(4)) == 0.0
    assert stable_rank(np.zeros(4)) == 0.0
    task = _exact_covariance_task()
    diag = adapter_diagnostics(task, np.zeros_like(task.delta_true))
    assert diag["adapter_effective_rank"] == 0.0
    assert diag["adapter_stable_rank"] == 0.0


def test_mp_edge_uses_standard_rmt_scaling() -> None:
    assert np.isclose(mp_singular_edge(64, 256, noise_sigma=2.0), 3.0)


def test_full_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(7)
    task = make_linear_task(
        d_in=4,
        d_out=3,
        n_train=20,
        n_val=8,
        task_rank=2,
        task_spikes=[1.5, 0.7],
        rng=rng,
        label_noise_std=0.0,
    )
    delta = rng.normal(scale=0.05, size=task.w_base.shape)
    grad = full_gradient_delta(task.x_train, task.y_train, task.w_base, delta)
    direction = rng.normal(size=delta.shape)
    direction /= np.linalg.norm(direction)
    eps = 1e-6
    plus = mse_loss(task.x_train, task.y_train, task.w_base, delta + eps * direction)
    minus = mse_loss(task.x_train, task.y_train, task.w_base, delta - eps * direction)
    finite_diff = (plus - minus) / (2.0 * eps)
    assert np.isclose(finite_diff, float(np.sum(grad * direction)), rtol=1e-5, atol=1e-7)


def test_activation_whitened_gradient_matches_useful_operator_spectrum() -> None:
    task = _exact_covariance_task()
    cov = activation_second_moment(task.x_train)
    assert np.allclose(cov, np.diag(task.x_eigs))

    whitened, info = early_gradient_matrix(task, matrix_mode="activation_whitened", ridge_scale=0.0)
    useful = useful_task_operator(task, covariance=cov)
    scale = 2.0 / task.y_train.shape[1]

    assert info["gradient_matrix_mode"] == "activation_whitened"
    assert np.allclose(
        np.linalg.svd(whitened, compute_uv=False),
        scale * np.linalg.svd(useful, compute_uv=False),
        rtol=1e-10,
        atol=1e-10,
    )


def test_gradient_diagnostics_default_to_whitened_and_keep_raw_ablation() -> None:
    task = _exact_covariance_task()
    diag = estimate_gradient_detectable_rank(
        task,
        np.random.default_rng(5),
        n_boot=3,
        quantile=0.8,
        ridge_scale=0.0,
    )
    assert diag["gradient_matrix_mode"] == "activation_whitened"
    assert diag["gradient_whitening"] == "activation_whitened"
    assert "raw_gradient_effective_rank" in diag
    assert diag["gradient_effective_rank"] > 0.0
    assert diag["raw_gradient_effective_rank"] > 0.0
