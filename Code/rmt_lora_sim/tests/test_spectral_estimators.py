from __future__ import annotations

import numpy as np

from rmt_lora.lora_sim import (
    activation_second_moment,
    adapter_diagnostics,
    early_gradient_matrix,
    estimate_gradient_detectable_rank,
    full_gradient_delta,
    make_linear_task,
    mse_loss,
    useful_task_operator,
)
from rmt_lora.spectra import effective_rank


def test_full_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(11)
    task = make_linear_task(
        d_in=5,
        d_out=4,
        n_train=23,
        n_val=7,
        task_rank=2,
        task_spikes=[1.4, 0.7],
        rng=rng,
        label_noise_std=0.03,
        activation_spectrum="powerlaw",
        activation_decay=0.5,
    )
    delta = rng.normal(scale=0.05, size=task.w_base.shape)
    grad = full_gradient_delta(task.x_train, task.y_train, task.w_base, delta)

    eps = 1e-6
    for i, j in [(0, 0), (1, 3), (3, 4)]:
        direction = np.zeros_like(delta)
        direction[i, j] = 1.0
        plus = mse_loss(task.x_train, task.y_train, task.w_base, delta + eps * direction)
        minus = mse_loss(task.x_train, task.y_train, task.w_base, delta - eps * direction)
        finite_difference = (plus - minus) / (2.0 * eps)
        assert np.isclose(grad[i, j], finite_difference, rtol=2e-6, atol=2e-8)


def test_activation_whitened_gradient_matches_useful_operator_noiseless() -> None:
    rng = np.random.default_rng(22)
    task = make_linear_task(
        d_in=9,
        d_out=7,
        n_train=128,
        n_val=32,
        task_rank=3,
        task_spikes=[2.0, 1.2, 0.6],
        rng=rng,
        label_noise_std=0.0,
        activation_spectrum="powerlaw",
        activation_decay=0.8,
    )
    raw, raw_info = early_gradient_matrix(task, matrix_mode="raw", ridge_scale=0.0)
    whitened, info = early_gradient_matrix(task, matrix_mode="activation_whitened", ridge_scale=0.0)

    cov = activation_second_moment(task.x_train)
    expected_raw = (2.0 / task.y_train.shape[1]) * (task.delta_true @ cov)
    expected_whitened = (2.0 / task.y_train.shape[1]) * useful_task_operator(task, covariance=cov)

    assert raw_info["gradient_matrix_mode"] == "raw"
    assert info["gradient_matrix_mode"] == "activation_whitened"
    assert np.allclose(raw, expected_raw, rtol=1e-9, atol=1e-9)
    assert np.allclose(whitened, expected_whitened, rtol=1e-8, atol=1e-8)


def test_gradient_diagnostics_emit_whitened_alias_and_raw_ablation() -> None:
    rng = np.random.default_rng(33)
    task = make_linear_task(
        d_in=12,
        d_out=10,
        n_train=96,
        n_val=32,
        task_rank=3,
        task_spikes=[1.8, 1.1, 0.4],
        rng=rng,
        label_noise_std=0.0,
        activation_spectrum="two_scale",
    )
    diag = estimate_gradient_detectable_rank(
        task,
        np.random.default_rng(44),
        n_boot=2,
        quantile=0.9,
        matrix_mode="activation_whitened",
        ridge_scale=0.0,
        include_raw=True,
    )
    assert diag["gradient_matrix_mode"] == "activation_whitened"
    assert diag["gradient_effective_rank"] == diag["whitened_gradient_effective_rank"]
    assert "raw_gradient_effective_rank" in diag
    assert diag["gradient_effective_rank"] > 0.0
    assert diag["raw_gradient_effective_rank"] > 0.0


def test_effective_rank_and_adapter_diagnostics_handle_zero_matrix() -> None:
    s = np.zeros(5)
    assert effective_rank(s) == 0.0

    rng = np.random.default_rng(55)
    task = make_linear_task(
        d_in=6,
        d_out=5,
        n_train=24,
        n_val=12,
        task_rank=2,
        task_spikes=[1.0, 0.5],
        rng=rng,
        label_noise_std=0.0,
    )
    diag = adapter_diagnostics(task, np.zeros_like(task.w_base))
    assert diag["adapter_stable_rank"] == 0.0
    assert diag["adapter_effective_rank"] == 0.0
    assert diag["adapter_frobenius"] == 0.0
