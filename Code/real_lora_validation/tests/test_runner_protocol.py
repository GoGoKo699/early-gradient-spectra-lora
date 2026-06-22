import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# The protocol tests only need torch. Supply import stubs when the packaging/test
# host does not have the full Hugging Face stack installed.
try:
    import transformers  # noqa: F401
except ModuleNotFoundError:
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoModelForCausalLM = object
    transformers_stub.AutoTokenizer = object
    sys.modules["transformers"] = transformers_stub

from real_protocol import build_nested_init_bank, float_sequence_sha256
from run_real_lora_validation import (
    LoRALinear,
    TargetSpec,
    apply_lora,
    assert_protocol_invariants,
    calibrate_spectra,
    make_loader,
)


def flatten_batches(loader):
    return torch.cat([batch[0].reshape(-1) for batch in loader])


def test_fresh_loaders_reproduce_identical_batch_order():
    blocks = torch.arange(80, dtype=torch.long).reshape(20, 4)
    first = make_loader(blocks, batch_size=3, shuffle=True, seed=107)
    second = make_loader(blocks, batch_size=3, shuffle=True, seed=107)
    assert torch.equal(flatten_batches(first), flatten_batches(second))


def test_consumed_loader_does_not_change_new_loader_order():
    blocks = torch.arange(80, dtype=torch.long).reshape(20, 4)
    consumed = make_loader(blocks, batch_size=3, shuffle=True, seed=107)
    _ = list(consumed)
    fresh_a = make_loader(blocks, batch_size=3, shuffle=True, seed=107)
    fresh_b = make_loader(blocks, batch_size=3, shuffle=True, seed=107)
    assert torch.equal(flatten_batches(fresh_a), flatten_batches(fresh_b))


def test_lora_linear_is_zero_update_and_uses_canonical_prefix():
    torch.manual_seed(7)
    base = nn.Linear(4, 3)
    x = torch.randn(2, 4)
    expected = base(x).detach()
    bank = build_nested_init_bank(
        {"proj": 4}, max_rank=3, init_std=0.01, seed=11
    )
    wrapped = LoRALinear(
        base, rank=2, alpha_scale=2.0, canonical_a=bank["proj"][:2]
    )
    assert torch.equal(wrapped(x), expected)
    assert torch.equal(wrapped.lora_A.detach().cpu(), bank["proj"][:2])
    assert torch.count_nonzero(wrapped.lora_B) == 0


def test_apply_lora_uses_same_nested_bank_for_different_ranks():
    first = nn.Sequential()
    first.add_module("proj", nn.Linear(4, 3))
    second = nn.Sequential()
    second.add_module("proj", nn.Linear(4, 3))
    bank = build_nested_init_bank(
        {"proj": 4}, max_rank=3, init_std=0.01, seed=13
    )
    apply_lora(first, {"proj": 1}, alpha_scale=2.0, init_bank=bank)
    apply_lora(second, {"proj": 3}, alpha_scale=2.0, init_bank=bank)
    assert torch.equal(first.proj.lora_A, second.proj.lora_A[:1])


class ToyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=0.9)
        self.proj = nn.Linear(4, 4)
        self.config = SimpleNamespace(use_cache=True)

    def forward(self, input_ids, labels):
        x = F.one_hot(input_ids % 4, num_classes=4).to(torch.float32)
        x = self.dropout(x)
        prediction = self.proj(x)
        target = F.one_hot(labels % 4, num_classes=4).to(torch.float32)
        return SimpleNamespace(loss=(prediction - target).square().mean())


def test_calibration_disables_dropout_and_restores_model_state():
    model = ToyLM()
    model.train()
    target = TargetSpec(
        name="proj", module_class="Linear", in_dim=4, out_dim=4
    )
    blocks = torch.tensor(
        [[0, 1, 2, 3], [3, 2, 1, 0], [0, 2, 0, 2], [1, 3, 1, 3]],
        dtype=torch.long,
    )
    loader_a = make_loader(blocks, batch_size=2, shuffle=False, seed=1)
    loader_b = make_loader(blocks, batch_size=2, shuffle=False, seed=999)
    bank = build_nested_init_bank(
        {"proj": 4}, max_rank=3, init_std=0.01, seed=17
    )

    torch.manual_seed(1)
    first, first_components, first_batches = calibrate_spectra(
        model,
        [target],
        loader_a,
        torch.device("cpu"),
        calib_batches=2,
        ridge_scale=1e-3,
        init_bank=bank,
        fisher_reference_rank=2,
        alpha_scale=2.0,
        max_rank=3,
    )
    torch.manual_seed(999)
    second, second_components, second_batches = calibrate_spectra(
        model,
        [target],
        loader_b,
        torch.device("cpu"),
        calib_batches=2,
        ridge_scale=1e-3,
        init_bank=bank,
        fisher_reference_rank=2,
        alpha_scale=2.0,
        max_rank=3,
    )

    assert first == second
    assert first_components == second_components
    assert [row["loss"] for row in first_batches] == [
        row["loss"] for row in second_batches
    ]
    assert all(row["calibration_model_mode"] == "eval" for row in first_batches)
    assert model.training is True
    assert model.config.use_cache is True
    assert model.proj.weight.requires_grad is True


def test_protocol_invariants_accept_identical_assignment_control():
    specs = {
        "a": TargetSpec("a", "Linear", 2, 2),
        "b": TargetSpec("b", "Linear", 3, 1),
    }
    allocations = {
        "uniform_r2": {"a": 2, "b": 2},
        "uniform_identity_control": {"a": 2, "b": 2},
    }
    trace = float_sequence_sha256([1.0, 0.5])
    results = [
        {
            "strategy": name,
            "initial_val_loss": 2.0,
            "final_val_loss": 1.5,
            "val_loss_delta": -0.5,
            "assignment_sha256": "same",
            "train_trace_sha256": trace,
        }
        for name in allocations
    ]
    checks = assert_protocol_invariants(
        results=results,
        allocations=allocations,
        specs=specs,
        target_budget=16,
        tolerance=0.0,
        require_identity_control=True,
    )
    assert checks["initial_loss_identity"] is True
    assert checks["identical_assignment_identity"] is True
    assert len(checks["identical_assignment_groups"]) == 1


def test_protocol_invariants_reject_identical_assignment_divergence():
    specs = {"a": TargetSpec("a", "Linear", 2, 2)}
    allocations = {
        "uniform_r2": {"a": 2},
        "uniform_identity_control": {"a": 2},
    }
    rows = [
        {
            "strategy": "uniform_r2",
            "initial_val_loss": 2.0,
            "final_val_loss": 1.5,
            "val_loss_delta": -0.5,
            "assignment_sha256": "same",
            "train_trace_sha256": "trace-a",
        },
        {
            "strategy": "uniform_identity_control",
            "initial_val_loss": 2.0,
            "final_val_loss": 1.6,
            "val_loss_delta": -0.4,
            "assignment_sha256": "same",
            "train_trace_sha256": "trace-b",
        },
    ]
    with pytest.raises(RuntimeError, match="different training traces"):
        assert_protocol_invariants(
            results=rows,
            allocations=allocations,
            specs=specs,
            target_budget=8,
            tolerance=1e-7,
            require_identity_control=True,
        )
