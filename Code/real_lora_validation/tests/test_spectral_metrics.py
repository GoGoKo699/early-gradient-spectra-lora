import math
import pytest
import torch
from spectral_metrics import (
    EFFECTIVE_RANK_DEFINITION,
    entropy_effective_rank_from_energy,
    singular_effective_rank,
)


def test_definition():
    assert EFFECTIVE_RANK_DEFINITION == "p_i = sigma_i^2 / sum_j sigma_j^2"


def test_squared_singular_values():
    p = torch.tensor([0.8, 0.2], dtype=torch.float64)
    expected = math.exp(float(-(p * p.log()).sum()))
    assert singular_effective_rank(torch.tensor([2.0, 1.0])) == pytest.approx(expected)


def test_equal_values_and_scale_invariance():
    assert singular_effective_rank(torch.ones(5)) == pytest.approx(5.0)
    s = torch.tensor([4.0, 2.0, 1.0])
    assert singular_effective_rank(7.5 * s) == pytest.approx(singular_effective_rank(s))
    assert singular_effective_rank(1e-20 * s) == pytest.approx(singular_effective_rank(s))


def test_zero_empty_and_nonfinite():
    assert singular_effective_rank(torch.zeros(4)) == 0.0
    assert singular_effective_rank(torch.tensor([])) == 0.0
    with pytest.raises(ValueError):
        singular_effective_rank(torch.tensor([1.0, float('nan')]))
    with pytest.raises(ValueError):
        singular_effective_rank(torch.tensor([1.0, -0.5]))


def test_extreme_scale_stability():
    baseline = singular_effective_rank(torch.tensor([4.0, 2.0, 1.0], dtype=torch.float64))
    assert singular_effective_rank(torch.tensor([4e200, 2e200, 1e200], dtype=torch.float64)) == pytest.approx(baseline)
    assert singular_effective_rank(torch.tensor([4e-200, 2e-200, 1e-200], dtype=torch.float64)) == pytest.approx(baseline)


def test_entropy_effective_rank_from_energy():
    assert entropy_effective_rank_from_energy(torch.tensor([1.0, 1.0, 0.0])) == pytest.approx(2.0)
    assert entropy_effective_rank_from_energy(torch.zeros(3)) == 0.0
    with pytest.raises(ValueError):
        entropy_effective_rank_from_energy(torch.tensor([1.0, -1.0]))
