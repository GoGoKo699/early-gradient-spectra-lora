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


def test_stable_seed_is_deterministic():
    from strank.utils import stable_seed

    a = stable_seed(101, "budget", 512, "soft_dimension")
    b = stable_seed(101, "budget", 512, "soft_dimension")
    c = stable_seed(101, "budget", 512, "gradient_norm")
    assert a == b
    assert a != c
    assert 0 <= a < 2**31 - 1
