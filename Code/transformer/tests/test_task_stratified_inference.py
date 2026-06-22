from __future__ import annotations

import numpy as np
import pytest

from strank.inference import (
    task_stratified_bootstrap_interval,
    task_stratified_mean,
    task_stratified_sign_flip_p,
)


def test_task_stratified_mean_equal_weights_tasks_not_runs() -> None:
    values = [1.0, 3.0, 10.0, 20.0, 30.0]
    tasks = ["a", "a", "b", "b", "b"]
    assert task_stratified_mean(values, tasks) == pytest.approx(11.0)


def test_task_stratified_sign_flip_is_exact_and_two_sided() -> None:
    values = [-1.0, -1.0, -2.0, -2.0]
    tasks = ["a", "a", "b", "b"]
    assert task_stratified_sign_flip_p(values, tasks) == pytest.approx(2 / 16)


def test_task_stratified_bootstrap_is_deterministic() -> None:
    values = [-0.4, -0.2, 0.1, 0.3]
    tasks = ["a", "a", "b", "b"]
    first = task_stratified_bootstrap_interval(
        values, tasks, n_resamples=500, seed_parts=("test",)
    )
    second = task_stratified_bootstrap_interval(
        values, tasks, n_resamples=500, seed_parts=("test",)
    )
    assert first == second
    assert first[0] <= task_stratified_mean(values, tasks) <= first[1]


def test_task_stratified_inference_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        task_stratified_mean([0.0, float("nan")], ["a", "b"])


def test_task_stratified_bootstrap_is_permutation_invariant() -> None:
    values = [-0.4, -0.2, 0.1, 0.3, 0.7]
    tasks = ["a", "a", "b", "b", "b"]
    expected = task_stratified_bootstrap_interval(
        values, tasks, n_resamples=1000, seed_parts=("permutation",)
    )
    order = [4, 1, 3, 0, 2]
    observed = task_stratified_bootstrap_interval(
        [values[index] for index in order],
        [tasks[index] for index in order],
        n_resamples=1000,
        seed_parts=("permutation",),
    )
    assert observed == expected


def test_task_stratified_bootstrap_is_stable_to_float_roundtrip_noise() -> None:
    values = np.asarray(
        [-0.01, -0.005, 0.002, -0.003, -0.008, -0.02, 0.01, -0.004, -0.006, 0.003]
    )
    tasks = ["a"] * 5 + ["b"] * 5
    expected = task_stratified_bootstrap_interval(
        values, tasks, n_resamples=10000, seed_parts=("roundtrip",)
    )
    perturbed = values.copy()
    perturbed[0] = np.nextafter(perturbed[0], np.inf)
    observed = task_stratified_bootstrap_interval(
        perturbed, tasks, n_resamples=10000, seed_parts=("roundtrip",)
    )
    assert observed == pytest.approx(expected, abs=1e-14)
