import pytest
import torch

from real_protocol import (
    allocate_exact_budget,
    assignment_sha256,
    bank_slice,
    build_nested_init_bank,
    component_marginal_utilities,
    derive_seed,
    scalar_marginal_utilities,
    seed_manifest,
    tensor_sha256,
)


def test_named_seed_streams_are_stable_and_distinct():
    first = seed_manifest(101)
    second = seed_manifest(101)
    assert first == second
    values = list(first["streams"].values())
    assert len(values) == len(set(values))
    assert derive_seed(101, "training_data") != derive_seed(103, "training_data")


def test_nested_initialization_is_order_independent_and_prefix_nested():
    first = build_nested_init_bank(
        {"b": 3, "a": 4}, max_rank=5, init_std=0.01, seed=17
    )
    second = build_nested_init_bank(
        {"a": 4, "b": 3}, max_rank=5, init_std=0.01, seed=17
    )
    assert torch.equal(first["a"], second["a"])
    assert torch.equal(first["b"], second["b"])
    assert torch.equal(bank_slice(first, "a", 2), bank_slice(first, "a", 5)[:2])
    assert tensor_sha256(first["a"]) == tensor_sha256(second["a"])


def test_assignment_hash_changes_with_rank_but_not_mapping_order():
    bank = build_nested_init_bank(
        {"a": 3, "b": 3}, max_rank=4, init_std=0.01, seed=19
    )
    assert assignment_sha256({"a": 2, "b": 3}, bank) == assignment_sha256(
        {"b": 3, "a": 2}, bank
    )
    assert assignment_sha256({"a": 2, "b": 3}, bank) != assignment_sha256(
        {"a": 3, "b": 2}, bank
    )


def test_exact_allocator_hits_budget_and_uses_utility():
    ranks = allocate_exact_budget(
        costs={"a": 1, "b": 2},
        marginal_utilities={
            "a": [10.0, 9.0, 8.0, 7.0],
            "b": [1.0, 0.5, 0.25, 0.1],
        },
        total_budget=6,
        min_rank=1,
        max_rank=4,
        reference_rank=2,
    )
    assert ranks == {"a": 4, "b": 1}
    assert ranks["a"] + 2 * ranks["b"] == 6


def test_exact_allocator_ties_prefer_uniform_reference():
    ranks = allocate_exact_budget(
        costs={"a": 1, "b": 2},
        marginal_utilities={"a": [0.0] * 4, "b": [0.0] * 4},
        total_budget=6,
        min_rank=1,
        max_rank=4,
        reference_rank=2,
    )
    assert ranks == {"a": 2, "b": 2}


def test_exact_allocator_rejects_infeasible_budget():
    with pytest.raises(ValueError, match="minimum-rank floor"):
        allocate_exact_budget(
            costs={"a": 2, "b": 4},
            marginal_utilities={"a": [1.0] * 3, "b": [1.0] * 3},
            total_budget=5,
            min_rank=1,
            max_rank=3,
            reference_rank=2,
        )
    with pytest.raises(ValueError, match="divisible"):
        allocate_exact_budget(
            costs={"a": 2, "b": 4},
            marginal_utilities={"a": [1.0] * 3, "b": [1.0] * 3},
            total_budget=7,
            min_rank=1,
            max_rank=3,
            reference_rank=2,
        )


def test_utility_helpers_are_finite_nonnegative_and_padded():
    scalar = scalar_marginal_utilities({"a": 4.0, "b": -2.0}, max_rank=3)
    assert scalar["a"][0] > scalar["a"][1] > scalar["a"][2]
    assert scalar["b"] == [0.0, 0.0, 0.0]
    component = component_marginal_utilities(
        {"a": [0.8, 0.2], "b": [0.5]}, max_rank=3
    )
    assert component["a"] == pytest.approx([1.0, 0.25, 0.0])
    assert component["b"] == pytest.approx([0.625, 0.0, 0.0])
