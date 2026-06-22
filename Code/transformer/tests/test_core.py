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


def test_reference_matched_rank_scaling_formulas():
    from strank.scaling import resolve_lora_scaling

    standard = resolve_lora_scaling(2, reference_alpha=16.0, scaling_mode="standard", reference_rank=8)
    fixed = resolve_lora_scaling(2, reference_alpha=16.0, scaling_mode="fixed", reference_rank=8)
    rslora = resolve_lora_scaling(2, reference_alpha=16.0, scaling_mode="rslora", reference_rank=8)
    assert standard.lora_scale == pytest.approx(8.0)
    assert fixed.lora_scale == pytest.approx(2.0)
    assert rslora.lora_scale == pytest.approx(4.0)

    for mode in ("standard", "fixed", "rslora"):
        at_reference = resolve_lora_scaling(8, 16.0, mode, reference_rank=8)
        assert at_reference.lora_scale == pytest.approx(2.0)
        assert at_reference.effective_alpha == pytest.approx(16.0)


def test_nonstandard_scaling_requires_declared_reference_rank():
    from strank.scaling import resolve_lora_scaling, scaling_protocol

    with pytest.raises(ValueError, match="reference_rank"):
        resolve_lora_scaling(2, 16.0, "fixed")
    with pytest.raises(ValueError, match="scale_reference_rank"):
        scaling_protocol({"ranks": [0, 1, 2], "alpha": 16.0, "scaling_modes": ["fixed"]})
    with pytest.raises(ValueError, match="unknown"):
        resolve_lora_scaling(2, 16.0, "not-a-mode", reference_rank=8)


def test_set_lora_ranks_records_resolved_scale():
    from strank.model import set_lora_ranks

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
    model = build_model(
        task.vocab_size,
        task.seq_len,
        {"d_model": 16, "n_layers": 1, "n_heads": 2, "d_mlp": 32, "dropout": 0.0},
    )
    site = "blocks.0.attn.q_proj"
    set_lora_ranks(
        model,
        {site: 2},
        alpha=16.0,
        scaling_mode="fixed",
        scale_reference_rank=8,
    )
    module = dict(model.named_modules())[site]
    assert module.scaling_mode == "fixed_update_scale"
    assert module.reference_alpha == pytest.approx(16.0)
    assert module.alpha == pytest.approx(4.0)
    assert module.lora_scale == pytest.approx(2.0)


def test_useful_rank_targets_do_not_pool_scaling_conditions():
    from strank.targets import compute_targets

    rows = []
    losses = {
        "fixed": {0: 10.0, 1: 5.0, 2: 1.0},
        "standard": {0: 10.0, 1: 1.0, 2: 4.0},
    }
    for mode, by_rank in losses.items():
        for rank, loss in by_rank.items():
            rows.append({
                "scaling_mode": mode,
                "is_primary_scaling_mode": mode == "fixed",
                "reference_alpha": 16.0,
                "scale_reference_rank": 8,
                "scale_at_reference_rank": 2.0,
                "site_name": "site.a",
                "rank": rank,
                "final_val_loss": loss,
            })
    targets = compute_targets(
        pd.DataFrame(rows),
        near_gaps=[0.1],
        recovery_fracs=[0.8],
        lambdas=[0.2],
    ).set_index("scaling_mode")
    assert targets.loc["fixed", "best_rank"] == 2
    assert targets.loc["standard", "best_rank"] == 1
    assert len(targets) == 2


def test_prediction_fit_reports_each_scaling_condition():
    from strank.targets import prediction_fit

    targets = pd.DataFrame([
        {"scaling_mode": mode, "site_name": site, "recovery_rank_0.8": rank}
        for mode in ("fixed", "standard")
        for site, rank in (("a", 1), ("b", 2), ("c", 4))
    ])
    stats = pd.DataFrame([
        {"site_name": "a", "effective_rank": 1.0},
        {"site_name": "b", "effective_rank": 2.0},
        {"site_name": "c", "effective_rank": 3.0},
    ])
    fit = prediction_fit(targets, stats)
    rows = fit[(fit["target"] == "recovery_rank_0.8") & (fit["predictor"] == "effective_rank")]
    assert set(rows["scaling_mode"]) == {"fixed", "standard"}
    assert rows["spearman"].tolist() == pytest.approx([1.0, 1.0])


def test_exact_modular_evaluation_enumerates_every_pair_once():
    from strank.tasks import IGNORE_INDEX, exact_modular_batches

    task = make_task_spec(
        {
            "name": "modular",
            "modulus": 7,
            "base_offset": 0,
            "adapt_offset": 3,
            "seq_len": 4,
        },
        split="adapt",
    )
    batches = exact_modular_batches(task, batch_size=11, device=torch.device("cpu"))
    x = torch.cat([batch_x for batch_x, _ in batches], dim=0)
    y = torch.cat([batch_y for _, batch_y in batches], dim=0)
    pairs = {(int(a), int(b)) for a, b in zip(x[:, 0], x[:, 2])}
    assert len(x) == 49
    assert len(pairs) == 49
    assert torch.all(y[:, :3] == IGNORE_INDEX)
    assert torch.equal(y[:, 3], (x[:, 0] + x[:, 2] + 3) % 7)


def test_uniform_exact_cost_baseline_hits_candidate_cost_and_is_most_uniform():
    import itertools

    from strank.allocation import uniform_exact_cost_allocation

    stats = pd.DataFrame(
        [
            {"site_name": "a", "d_in": 4, "d_out": 4},
            {"site_name": "b", "d_in": 8, "d_out": 8},
            {"site_name": "c", "d_in": 4, "d_out": 4},
        ]
    )
    grid = [0, 1, 2, 4]
    target = 48
    alloc = uniform_exact_cost_allocation(stats, grid, target)
    assert allocation_cost(stats, alloc) == target
    ranks = tuple(alloc[name] for name in ("a", "b", "c"))
    objective = 3 * sum(r * r for r in ranks) - sum(ranks) ** 2
    feasible = []
    costs = (8, 16, 8)
    for candidate in itertools.product(grid, repeat=3):
        if sum(c * r for c, r in zip(costs, candidate)) == target:
            feasible.append(3 * sum(r * r for r in candidate) - sum(candidate) ** 2)
    assert objective == min(feasible)


def test_permutation_edge_diagnostics_are_reproducible_and_report_uncertainty():
    import numpy as np

    from strank.spectral import permutation_edge_diagnostics

    mat = np.arange(24, dtype=float).reshape(4, 6)
    first, first_maxima = permutation_edge_diagnostics(
        mat, n_boot=32, quantile=0.95, seed=17, uncertainty_resamples=100
    )
    second, second_maxima = permutation_edge_diagnostics(
        mat, n_boot=32, quantile=0.95, seed=17, uncertainty_resamples=100
    )
    assert np.array_equal(first_maxima, second_maxima)
    assert first == second
    assert len(first_maxima) == 32
    assert first["edge_mc_ci_low"] <= first["edge"] <= first["edge_mc_ci_high"]
    assert first["edge_q_0_99"] <= first["edge_q_0_995"] <= first["edge_q_0_999"]


def test_protocol_identifier_is_shared_by_runner_validator_and_builder():
    from scripts.build_transformer_release import PROTOCOL_VERSION as builder_protocol
    from scripts.run_synthetic_transformer import PROTOCOL_VERSION as runner_protocol
    from scripts.validate_rank_scaling_run import EXPECTED_PROTOCOL as validator_protocol
    from strank.protocol import SYNTHETIC_TRANSFORMER_PROTOCOL_VERSION

    assert {
        builder_protocol,
        runner_protocol,
        validator_protocol,
    } == {SYNTHETIC_TRANSFORMER_PROTOCOL_VERSION}


def test_exact_cost_pairing_uses_explicit_candidate_links_and_primary_flag():
    from scripts.validate_rank_scaling_run import _validate_exact_cost_pairing

    common = {
        "scaling_mode": "fixed_update_scale",
        "budget": 128,
        "adaptation_replicate": 0,
    }
    rows = [
        {
            **common,
            "condition_id": "effective_rank",
            "rule": "effective_rank",
            "comparison_role": "candidate",
            "matched_cost": 48,
            "matched_to_rules": "[]",
            "exact_cost_match_required": True,
            "matched_baseline_condition_id": "uniform_exact_cost__48",
            "is_primary_allocation_rule": False,
            "actual_cost": 48,
        },
        {
            **common,
            "condition_id": "soft_dimension",
            "rule": "soft_dimension",
            "comparison_role": "candidate",
            "matched_cost": 64,
            "matched_to_rules": "[]",
            "exact_cost_match_required": True,
            "matched_baseline_condition_id": "uniform_exact_cost__64",
            "is_primary_allocation_rule": True,
            "actual_cost": 64,
        },
        {
            **common,
            "condition_id": "uniform_exact_cost__48",
            "rule": "uniform_exact_cost",
            "comparison_role": "exact_cost_baseline",
            "matched_cost": 48,
            "matched_to_rules": '["effective_rank"]',
            "exact_cost_match_required": False,
            "matched_baseline_condition_id": "",
            "is_primary_allocation_rule": False,
            "actual_cost": 48,
        },
        {
            **common,
            "condition_id": "uniform_exact_cost__64",
            "rule": "uniform_exact_cost",
            "comparison_role": "exact_cost_baseline",
            "matched_cost": 64,
            "matched_to_rules": '["soft_dimension"]',
            "exact_cost_match_required": False,
            "matched_baseline_condition_id": "",
            "is_primary_allocation_rule": False,
            "actual_cost": 64,
        },
    ]
    frame = pd.DataFrame(rows)
    _validate_exact_cost_pairing(
        frame,
        ["effective_rank", "soft_dimension"],
        primary_rule="soft_dimension",
        primary_reference_rule="uniform_exact_cost",
    )

    broken = frame.copy()
    broken.loc[broken["rule"] == "soft_dimension", "matched_baseline_condition_id"] = (
        "uniform_exact_cost__48"
    )
    with pytest.raises(ValueError, match="points to"):
        _validate_exact_cost_pairing(
            broken,
            ["effective_rank", "soft_dimension"],
            primary_rule="soft_dimension",
            primary_reference_rule="uniform_exact_cost",
        )


def test_uniform_exact_cost_allocations_support_multiple_candidate_costs():
    from strank.allocation import uniform_exact_cost_allocations

    stats = pd.DataFrame(
        [
            {"site_name": "a", "d_in": 4, "d_out": 4},
            {"site_name": "b", "d_in": 8, "d_out": 8},
            {"site_name": "c", "d_in": 4, "d_out": 4},
        ]
    )
    allocations = uniform_exact_cost_allocations(stats, [0, 1, 2, 4], [48, 64])
    assert set(allocations) == {48, 64}
    for cost, allocation in allocations.items():
        assert allocation_cost(stats, allocation) == cost


def test_legacy_aggregator_keeps_multiple_exact_cost_baselines_distinct():
    from scripts.aggregate_step3_replicates import _normalise_protocol_columns

    frame = pd.DataFrame(
        [
            {
                "run": "r1",
                "scaling_mode": "fixed_update_scale",
                "budget": 128,
                "adaptation_replicate": 0,
                "condition_id": "uniform_exact_cost__48",
                "rule": "uniform_exact_cost",
            },
            {
                "run": "r1",
                "scaling_mode": "fixed_update_scale",
                "budget": 128,
                "adaptation_replicate": 0,
                "condition_id": "uniform_exact_cost__64",
                "rule": "uniform_exact_cost",
            },
        ]
    )
    normalised = _normalise_protocol_columns(frame)
    assert set(normalised["rule"]) == {
        "uniform_exact_cost__48",
        "uniform_exact_cost__64",
    }
    assert set(normalised["allocation_rule"]) == {"uniform_exact_cost"}


def test_calibration_disables_dropout_restores_mode_and_is_reproducible():
    import numpy as np

    from strank.calibrate import calibrate_module_spectra

    task = make_task_spec(
        {
            "name": "modular",
            "modulus": 7,
            "base_offset": 0,
            "adapt_offset": 2,
            "seq_len": 4,
            "train_batch_size": 8,
        },
        split="adapt",
    )
    model = build_model(
        task.vocab_size,
        task.seq_len,
        {
            "d_model": 12,
            "n_layers": 1,
            "n_heads": 2,
            "d_mlp": 24,
            "dropout": 0.4,
        },
    )
    model.train()
    config = {
        "batches": 2,
        "batch_size": 8,
        "whitening": "diag",
        "lambda_scale": 0.01,
        "null_bootstrap": 8,
        "null_quantile": 0.95,
        "null_uncertainty_resamples": 20,
        "null_uncertainty_confidence": 0.9,
        "max_tokens_per_site": 32,
    }
    sites = ["blocks.0.attn.q_proj", "blocks.0.mlp.fc2"]
    first_rows, first_singular, first_null = calibrate_module_spectra(
        model,
        task,
        config,
        sites,
        torch.device("cpu"),
        seed=123,
        data_seed=456,
        return_null_maxima=True,
    )
    assert model.training is True
    _ = torch.rand(1000)
    second_rows, second_singular, second_null = calibrate_module_spectra(
        model,
        task,
        config,
        sites,
        torch.device("cpu"),
        seed=123,
        data_seed=456,
        return_null_maxima=True,
    )
    assert model.training is True
    assert first_rows == second_rows
    for site in sites:
        assert np.array_equal(first_singular[site], second_singular[site])
        assert np.array_equal(first_null[site], second_null[site])
    assert {row["calibration_data_seed"] for row in first_rows} == {456}
    assert all(row["dropout_disabled_during_calibration"] for row in first_rows)
    assert len({row["permutation_seed"] for row in first_rows}) == len(sites)


def test_corrected_config_rejects_unmatched_primary_and_missing_reference_rank():
    from copy import deepcopy

    from strank.protocol import validate_synthetic_transformer_config

    config = {
        "run": {"seed": 1},
        "task": {
            "name": "modular",
            "modulus": 7,
            "base_offset": 0,
            "adapt_offset": 1,
            "seq_len": 4,
        },
        "model": {
            "d_model": 8,
            "n_layers": 1,
            "n_heads": 1,
            "d_mlp": 16,
        },
        "base_train": {"steps": 1},
        "calibration": {
            "null_bootstrap": 8,
            "null_quantile": 0.95,
            "null_uncertainty_resamples": 10,
            "null_uncertainty_confidence": 0.95,
        },
        "sites": {"include": ["blocks.0.attn.q_proj"]},
        "lora": {
            "ranks": [0, 1, 2, 4],
            "alpha": 8.0,
            "scaling_modes": ["fixed_update_scale", "standard", "rslora"],
            "scale_reference_rank": 4,
            "steps": 1,
        },
        "budgets": {
            "param_budgets": [16],
            "rules": ["uniform_fill", "soft_dimension"],
            "cost_matched_rules": ["soft_dimension"],
        },
        "protocol": {
            "primary_scaling_mode": "fixed_update_scale",
            "primary_allocation_rule": "soft_dimension",
            "primary_reference_rule": "uniform_exact_cost",
            "primary_metric": "final_val_loss",
            "independent_unit": "task_seed_run",
        },
    }
    validated = validate_synthetic_transformer_config(config)
    assert validated["primary_allocation_rule"] == "soft_dimension"

    broken = deepcopy(config)
    broken["budgets"]["cost_matched_rules"] = []
    with pytest.raises(ValueError, match="exact-cost comparator"):
        validate_synthetic_transformer_config(broken)

    broken = deepcopy(config)
    broken["lora"]["scale_reference_rank"] = 8
    with pytest.raises(ValueError, match="present in lora.ranks"):
        validate_synthetic_transformer_config(broken)
