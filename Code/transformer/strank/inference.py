"""Deterministic task-stratified inference for paired transformer runs."""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Iterable, Sequence

import numpy as np


def _stable_int_seed(*parts: object) -> int:
    payload = json.dumps(
        parts, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _validated_arrays(
    values: Iterable[float], strata: Sequence[object] | Iterable[object]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = np.asarray(list(values), dtype=float)
    labels = np.asarray([str(value) for value in strata], dtype=object)
    if data.ndim != 1 or labels.ndim != 1 or len(data) != len(labels):
        raise ValueError("values and strata must be one-dimensional and equally sized")
    if len(data) == 0:
        raise ValueError("task-stratified inference requires at least one run")
    if not np.isfinite(data).all():
        raise ValueError("task-stratified inference values must be finite")
    tasks = sorted(set(labels.tolist()))
    if not tasks or any(not task for task in tasks):
        raise ValueError("task-stratified inference requires non-empty task labels")
    if any(int(np.sum(labels == task)) == 0 for task in tasks):  # pragma: no cover
        raise ValueError("every task stratum must contain at least one run")
    return data, labels, tasks


def task_stratified_mean(
    values: Iterable[float], strata: Sequence[object] | Iterable[object]
) -> float:
    """Return the equal-task-weighted mean of within-task run means."""

    data, labels, tasks = _validated_arrays(values, strata)
    task_means = [float(np.mean(data[labels == task])) for task in tasks]
    return float(np.mean(task_means))


def task_stratified_bootstrap_interval(
    values: Iterable[float],
    strata: Sequence[object] | Iterable[object],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed_parts: tuple[object, ...] = (),
) -> tuple[float, float]:
    """Percentile interval from resampling independent runs within each task."""

    data, labels, tasks = _validated_arrays(values, strata)
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if int(n_resamples) <= 0:
        raise ValueError("n_resamples must be positive")
    rng = np.random.default_rng(
        _stable_int_seed(
            "task_stratified_bootstrap",
            data.tolist(),
            labels.tolist(),
            float(confidence),
            int(n_resamples),
            *seed_parts,
        )
    )
    estimates = np.zeros(int(n_resamples), dtype=float)
    for task in tasks:
        task_values = data[labels == task]
        indices = rng.integers(
            0, len(task_values), size=(int(n_resamples), len(task_values))
        )
        estimates += task_values[indices].mean(axis=1) / len(tasks)
    tail = (1.0 - float(confidence)) / 2.0
    return (
        float(np.quantile(estimates, tail, method="linear")),
        float(np.quantile(estimates, 1.0 - tail, method="linear")),
    )


def task_stratified_sign_flip_p(
    values: Iterable[float],
    strata: Sequence[object] | Iterable[object],
    *,
    max_exact_n: int = 20,
    monte_carlo_draws: int = 100_000,
) -> float:
    """Two-sided sign-flip p-value for the equal-task-weighted estimator."""

    data, labels, tasks = _validated_arrays(values, strata)
    weights = np.zeros(len(data), dtype=float)
    for task in tasks:
        mask = labels == task
        weights[mask] = 1.0 / (len(tasks) * int(np.sum(mask)))
    observed = abs(float(np.sum(weights * data)))
    tolerance = 1e-15
    if len(data) <= int(max_exact_n):
        exceed = 0
        total = 0
        for signs in itertools.product((-1.0, 1.0), repeat=len(data)):
            statistic = abs(float(np.sum(weights * data * np.asarray(signs))))
            exceed += statistic >= observed - tolerance
            total += 1
        return float(exceed / total)
    if int(monte_carlo_draws) <= 0:
        raise ValueError("monte_carlo_draws must be positive")
    rng = np.random.default_rng(
        _stable_int_seed(
            "task_stratified_sign_flip",
            data.tolist(),
            labels.tolist(),
            int(monte_carlo_draws),
        )
    )
    signs = rng.choice(
        np.array([-1.0, 1.0]), size=(int(monte_carlo_draws), len(data))
    )
    statistics = np.abs(np.sum(signs * data[None, :] * weights[None, :], axis=1))
    return float(
        (1 + np.sum(statistics >= observed - tolerance))
        / (int(monte_carlo_draws) + 1)
    )
