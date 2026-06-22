from __future__ import annotations

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
