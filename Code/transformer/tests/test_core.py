import pandas as pd
import pytest
import torch

from strank.allocation import allocation_cost, uniform_fill_allocation
from strank.model import build_model
from strank.tasks import make_batch, make_task_spec


def test_modular_task_rejects_non_four_seq_len():
    cfg = {
        "name": "modular",
        "modulus": 17,
        "base_offset": 0,
        "adapt_offset": 3,
        "seq_len": 5,
    }
    with pytest.raises(ValueError):
        make_task_spec(cfg, split="base")


def test_model_forward_shape():
    task = make_task_spec(
        {
            "name": "modular",
            "modulus": 17,
            "base_offset": 0,
            "adapt_offset": 3,
            "seq_len": 4,
        },
        split="base",
    )
    model = build_model(
        task.vocab_size,
        task.seq_len,
        {
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 2,
            "d_mlp": 32,
            "dropout": 0.0,
        },
    )
    x, _ = make_batch(task, 8, torch.device("cpu"))
    y = model(x)
    assert y.shape == (8, task.seq_len, task.vocab_size)


def test_uniform_fill_respects_budget():
    stats = pd.DataFrame(
        [
            {"site_name": "a", "d_in": 4, "d_out": 4},
            {"site_name": "b", "d_in": 8, "d_out": 8},
        ]
    )
    alloc = uniform_fill_allocation(stats, [0, 1, 2, 4], budget=24)
    assert allocation_cost(stats, alloc) <= 24
    assert any(rank > 0 for rank in alloc.values())


def test_calibration_respects_max_tokens_per_site():
    from strank.calibrate import calibrate_module_spectra

    task = make_task_spec(
        {
            "name": "modular",
            "modulus": 17,
            "base_offset": 0,
            "adapt_offset": 3,
            "seq_len": 4,
            "train_batch_size": 16,
        },
        split="adapt",
    )
    model = build_model(
        task.vocab_size,
        task.seq_len,
        {
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 2,
            "d_mlp": 32,
            "dropout": 0.0,
        },
    )
    rows, _ = calibrate_module_spectra(
        model,
        task,
        {
            "batches": 4,
            "batch_size": 16,
            "whitening": "diag",
            "lambda_scale": 0.01,
            "null_bootstrap": 1,
            "null_quantile": 0.995,
            "max_tokens_per_site": 10,
        },
        ["blocks.0.attn.q_proj", "blocks.0.mlp.fc2"],
        torch.device("cpu"),
        seed=0,
    )

    assert rows
    assert all(row["n_tokens"] <= 10 for row in rows)


def test_stable_seed_is_deterministic_and_supports_separate_streams():
    from strank.utils import stable_seed

    comparison_a = stable_seed(101, "modular", "budget", 512, 0)
    comparison_b = stable_seed(101, "modular", "budget", 512, 0)
    adapter = stable_seed(comparison_a, "adapter_init")
    train_data = stable_seed(comparison_a, "train_data")
    evaluation = stable_seed(comparison_a, "eval_data")
    dropout = stable_seed(comparison_a, "dropout")
    assert comparison_a == comparison_b
    assert len({adapter, train_data, evaluation, dropout}) == 4
    assert 0 <= comparison_a < 2**31 - 1


def test_explicit_batch_generator_is_reproducible_and_isolated():
    task = make_task_spec(
        {
            "name": "modular",
            "modulus": 17,
            "base_offset": 0,
            "adapt_offset": 3,
            "seq_len": 4,
        },
        split="adapt",
    )
    device = torch.device("cpu")
    g1 = torch.Generator(device="cpu").manual_seed(991)
    x1, y1 = make_batch(task, 16, device, generator=g1)
    _ = torch.rand(100)  # perturb the global RNG; explicit stream must not change
    g2 = torch.Generator(device="cpu").manual_seed(991)
    x2, y2 = make_batch(task, 16, device, generator=g2)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_nested_lora_initialization_uses_prefix_slices():
    from strank.model import make_lora_init_bank, set_lora_ranks

    task = make_task_spec(
        {
            "name": "modular",
            "modulus": 17,
            "base_offset": 0,
            "adapt_offset": 3,
            "seq_len": 4,
        },
        split="adapt",
    )
    cfg = {"d_model": 16, "n_layers": 1, "n_heads": 2, "d_mlp": 32, "dropout": 0.0}
    model = build_model(task.vocab_size, task.seq_len, cfg)
    site = "blocks.0.attn.q_proj"
    bank = make_lora_init_bank(model, max_rank=4, seed=1234)
    set_lora_ranks(model, {site: 2}, alpha=4.0, init_bank=bank)
    a_rank2 = dict(model.named_modules())[site].lora_A.detach().clone()
    set_lora_ranks(model, {site: 4}, alpha=4.0, init_bank=bank)
    a_rank4 = dict(model.named_modules())[site].lora_A.detach().clone()
    assert torch.equal(a_rank2, a_rank4[:2])


def test_identical_allocation_with_fixed_streams_has_identical_metrics():
    torch.set_num_threads(1)
    from strank.model import make_lora_init_bank, set_lora_ranks
    from strank.train import materialize_batches, train_lora
    from strank.utils import set_seed

    task = make_task_spec(
        {
            "name": "modular",
            "modulus": 17,
            "base_offset": 0,
            "adapt_offset": 3,
            "seq_len": 4,
            "train_batch_size": 16,
            "eval_batches": 2,
        },
        split="adapt",
    )
    cfg = {"d_model": 16, "n_layers": 1, "n_heads": 2, "d_mlp": 32, "dropout": 0.1}
    set_seed(7)
    base = build_model(task.vocab_size, task.seq_len, cfg)
    state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    allocation = {"blocks.0.attn.q_proj": 2, "blocks.0.mlp.fc2": 2}
    bank = make_lora_init_bank(base, max_rank=4, seed=111)
    train_batches = materialize_batches(task, 16, 5, torch.device("cpu"), seed=222)
    val_batches = materialize_batches(task, 16, 2, torch.device("cpu"), seed=333)
    results = []
    for global_perturbation_seed in (10, 999):
        set_seed(global_perturbation_seed)
        model = build_model(task.vocab_size, task.seq_len, cfg)
        model.load_state_dict(state)
        set_lora_ranks(model, allocation, alpha=4.0, init_bank=bank)
        metrics, diverged = train_lora(
            model,
            task,
            {"steps": 5, "lr": 0.01, "weight_decay": 0.0, "log_every": 0},
            torch.device("cpu"),
            train_batches=train_batches,
            validation_batches=val_batches,
            dropout_seed=444,
        )
        assert not diverged
        results.append(metrics)
    assert results[0] == results[1]
