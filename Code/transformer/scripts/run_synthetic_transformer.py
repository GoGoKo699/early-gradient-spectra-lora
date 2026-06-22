#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from strank.allocation import (
    allocation_cost,
    gradient_norm_allocation,
    marginal_gain_allocation,
    score_allocation,
    uniform_allocation,
    uniform_exact_cost_allocations,
    uniform_fill_allocation,
)
from strank.calibrate import calibrate_module_spectra
from strank.model import build_model, clear_all_lora, make_lora_init_bank, set_lora_ranks
from strank.plotting import plot_budget_curve, plot_prediction_scatter, plot_rank_sweeps, plot_spectra
from strank.protocol import (
    SYNTHETIC_TRANSFORMER_PROTOCOL_VERSION,
    validate_synthetic_transformer_config,
)
from strank.scaling import resolve_lora_scaling, scaling_protocol
from strank.targets import compute_targets, prediction_fit
from strank.tasks import make_task_spec
from strank.train import (
    evaluate,
    materialize_batches,
    materialize_evaluation_batches,
    train_base,
    train_lora,
)
from strank.utils import get_device, load_yaml, make_run_dir, save_yaml, set_seed, stable_seed, write_json

PROTOCOL_VERSION = SYNTHETIC_TRANSFORMER_PROTOCOL_VERSION


def _task_key(task) -> str:
    return json.dumps(
        {"name": task.name, "params": task.params},
        sort_keys=True,
        separators=(",", ":"),
    )


def clone_model_from_base(base_state, vocab_size, seq_len, model_cfg, device):
    model = build_model(vocab_size, seq_len, model_cfg).to(device)
    model.load_state_dict(base_state, strict=True)
    clear_all_lora(model)
    return model


def _seed_bundle(comparison_seed: int) -> dict[str, int]:
    return {
        "comparison_seed": int(comparison_seed),
        "adapter_seed": int(stable_seed(comparison_seed, "adapter_init")),
        "train_data_seed": int(stable_seed(comparison_seed, "train_data")),
        "eval_data_seed": int(stable_seed(comparison_seed, "eval_data")),
        "dropout_seed": int(stable_seed(comparison_seed, "dropout")),
    }


def _exact_modular_evaluation(cfg: dict, task) -> bool:
    return bool(cfg.get("protocol", {}).get("exact_modular_evaluation", False)) and task.name == "modular"


def _adaptation_batches(adapt_task, cfg, device, seeds: dict[str, int]):
    lora_cfg = cfg["lora"]
    steps = int(lora_cfg.get("steps", 100))
    batch_size = int(adapt_task.params.get("train_batch_size", lora_cfg.get("batch_size", 64)))
    eval_batches = int(adapt_task.params.get("eval_batches", 4))
    train_batches = materialize_batches(
        adapt_task,
        batch_size,
        steps,
        device,
        seed=seeds["train_data_seed"],
    )
    validation_batches = materialize_evaluation_batches(
        adapt_task,
        batch_size,
        eval_batches,
        device,
        seed=seeds["eval_data_seed"],
        exact_modular=_exact_modular_evaluation(cfg, adapt_task),
    )
    return train_batches, validation_batches


def _allocation_signature(alloc: dict[str, int]) -> str:
    return json.dumps(
        sorted((str(site), int(rank)) for site, rank in alloc.items()),
        separators=(",", ":"),
    )


def _allocation_scaling_metadata(alloc: dict[str, int], scaling: dict, scaling_mode: str) -> dict:
    resolved = [
        resolve_lora_scaling(
            rank,
            reference_alpha=scaling["reference_alpha"],
            scaling_mode=scaling_mode,
            reference_rank=scaling["reference_rank"],
        )
        for rank in alloc.values()
        if int(rank) > 0
    ]
    scales = [item.lora_scale for item in resolved]
    return {
        "scaling_mode": scaling_mode,
        "reference_alpha": scaling["reference_alpha"],
        "scale_reference_rank": scaling["reference_rank"],
        "scale_at_reference_rank": scaling["scale_at_reference_rank"],
        "active_sites": len(resolved),
        "min_lora_scale": min(scales) if scales else 0.0,
        "max_lora_scale": max(scales) if scales else 0.0,
        "mean_lora_scale": float(np.mean(scales)) if scales else 0.0,
    }


def _assert_common_random_number_invariant(df: pd.DataFrame, device: torch.device) -> None:
    """Identical allocations in one comparison block must produce identical metrics."""
    if df.empty:
        return
    atol = 1e-6 if device.type == "cuda" else 1e-10
    metrics = ["final_train_loss", "final_val_loss", "final_val_accuracy"]
    failures = []
    group_columns = [
        "scaling_mode",
        "budget",
        "adaptation_replicate",
        "allocation_signature",
    ]
    for keys, group in df.groupby(group_columns, dropna=False):
        if len(group) < 2:
            continue
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) and float(values.max() - values.min()) > atol:
                failures.append(
                    (keys, metric, float(values.max() - values.min()), group["condition_id"].tolist())
                )
    if failures:
        raise RuntimeError(
            "common-random-number invariant failed for identical allocations: "
            + "; ".join(str(item) for item in failures[:5])
        )


def _assert_zero_rank_scaling_invariant(df: pd.DataFrame, device: torch.device) -> None:
    if df.empty or "scaling_mode" not in df:
        return
    zero = df[df["rank"] == 0]
    if zero.empty:
        return
    atol = 1e-6 if device.type == "cuda" else 1e-10
    failures = []
    for keys, group in zero.groupby(["site_name", "adaptation_replicate"], dropna=False):
        if group["scaling_mode"].nunique() < 2:
            continue
        for metric in ["final_train_loss", "final_val_loss", "final_val_accuracy"]:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) and float(values.max() - values.min()) > atol:
                failures.append((keys, metric, float(values.max() - values.min())))
    if failures:
        raise RuntimeError(
            "rank-zero scaling invariant failed: "
            + "; ".join(str(item) for item in failures[:5])
        )


def site_sweeps(
    base_state,
    vocab_size,
    seq_len,
    cfg,
    adapt_task,
    device,
    sites,
    ranks,
    run_dir,
    base_seed: int,
):
    rows = []
    lora_cfg = cfg["lora"]
    scaling = scaling_protocol(lora_cfg, cfg.get("protocol", {}))
    sweep_replicates = int(cfg.get("protocol", {}).get("sweep_replicates", 1))
    if sweep_replicates < 1:
        raise ValueError("protocol.sweep_replicates must be at least 1")
    max_rank = max([int(rank) for rank in ranks] or [0])
    init_scale = float(lora_cfg.get("init_scale", 0.01))
    for site in sites:
        for replicate in range(sweep_replicates):
            comparison_seed = stable_seed(
                base_seed,
                _task_key(adapt_task),
                "site_sweep",
                site,
                replicate,
            )
            seeds = _seed_bundle(comparison_seed)
            template = clone_model_from_base(base_state, vocab_size, seq_len, cfg["model"], device)
            init_bank = make_lora_init_bank(
                template,
                max_rank,
                seeds["adapter_seed"],
                init_scale=init_scale,
            )
            train_batches, validation_batches = _adaptation_batches(adapt_task, cfg, device, seeds)
            for scaling_mode in scaling["scaling_modes"]:
                for rank in ranks:
                    print(
                        f"site sweep mode={scaling_mode} site={site} rank={rank} replicate={replicate}",
                        flush=True,
                    )
                    model = clone_model_from_base(base_state, vocab_size, seq_len, cfg["model"], device)
                    set_lora_ranks(
                        model,
                        {site: int(rank)},
                        alpha=scaling["reference_alpha"],
                        scaling_mode=scaling_mode,
                        scale_reference_rank=scaling["reference_rank"],
                        init_scale=init_scale,
                        init_bank=init_bank,
                    )
                    resolved = resolve_lora_scaling(
                        int(rank),
                        reference_alpha=scaling["reference_alpha"],
                        scaling_mode=scaling_mode,
                        reference_rank=scaling["reference_rank"],
                    )
                    metrics, diverged = train_lora(
                        model,
                        adapt_task,
                        lora_cfg,
                        device,
                        eval_task=adapt_task,
                        train_batches=train_batches,
                        validation_batches=validation_batches,
                        dropout_seed=seeds["dropout_seed"],
                    )
                    rows.append(
                        {
                            "protocol_version": PROTOCOL_VERSION,
                            "condition_seed": int(comparison_seed),
                            "adaptation_replicate": int(replicate),
                            **seeds,
                            "scaling_mode": scaling_mode,
                            "is_primary_scaling_mode": scaling_mode == scaling["primary_scaling_mode"],
                            "reference_alpha": resolved.reference_alpha,
                            "scale_reference_rank": resolved.reference_rank,
                            "scale_at_reference_rank": resolved.scale_at_reference_rank,
                            "effective_alpha": resolved.effective_alpha,
                            "lora_scale": resolved.lora_scale,
                            "site_name": site,
                            "rank": int(rank),
                            "diverged": bool(diverged),
                            "final_train_loss": metrics["train_loss_last"],
                            "final_val_loss": metrics["val_loss"],
                            "final_val_accuracy": metrics["val_accuracy"],
                            "trainable_params": metrics["trainable_params"],
                            "evaluation_examples": int(metrics["val_n_examples"]),
                            "evaluation_batches": int(metrics["val_n_batches"]),
                            "exact_modular_evaluation": _exact_modular_evaluation(cfg, adapt_task),
                        }
                    )
    df = pd.DataFrame(rows)
    _assert_zero_rank_scaling_invariant(df, device)
    out = run_dir / "sweeps" / "site_rank_sweep_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def _allocation_for_rule(rule, module_stats, sv_dict, rank_grid, budget):
    if rule == "uniform":
        return uniform_allocation(module_stats, rank_grid, budget)
    if rule == "uniform_fill":
        return uniform_fill_allocation(module_stats, rank_grid, budget)
    if rule == "effective_rank":
        return score_allocation(module_stats, rank_grid, budget, "effective_rank")
    if rule == "soft_dimension":
        return score_allocation(module_stats, rank_grid, budget, "soft_dimension")
    if rule == "gradient_norm":
        return gradient_norm_allocation(module_stats, rank_grid, budget)
    if rule == "marginal_gain_raw":
        return marginal_gain_allocation(module_stats, sv_dict, rank_grid, budget, mode="raw")
    if rule == "marginal_gain_edge":
        return marginal_gain_allocation(module_stats, sv_dict, rank_grid, budget, mode="edge")
    if rule == "marginal_gain_soft":
        return marginal_gain_allocation(module_stats, sv_dict, rank_grid, budget, mode="soft")
    raise ValueError(f"unknown allocation rule: {rule}")


def _budget_conditions(cfg, module_stats, sv_dict, rank_grid, budget):
    rules = list(cfg.get("budgets", {}).get("rules", []))
    allocations: dict[str, dict[str, int]] = {}
    for rule in rules:
        allocations[rule] = _allocation_for_rule(rule, module_stats, sv_dict, rank_grid, budget)

    matched_rules = list(cfg.get("budgets", {}).get("cost_matched_rules", []))
    missing = [rule for rule in matched_rules if rule not in allocations]
    if missing:
        raise ValueError(f"cost_matched_rules are not present in budgets.rules: {missing}")
    matched_cost_to_rules: dict[int, list[str]] = {}
    for rule in matched_rules:
        cost = allocation_cost(module_stats, allocations[rule])
        matched_cost_to_rules.setdefault(cost, []).append(rule)
    exact_baselines = uniform_exact_cost_allocations(
        module_stats,
        rank_grid,
        list(matched_cost_to_rules),
    )

    conditions = []
    for rule, allocation in allocations.items():
        realized_cost = allocation_cost(module_stats, allocation)
        exact_match_required = rule in matched_rules
        conditions.append(
            {
                "condition_id": rule,
                "rule": rule,
                "comparison_role": "cap_baseline" if rule in {"uniform", "uniform_fill"} else "candidate",
                "matched_cost": realized_cost,
                "matched_to_rules": "[]",
                "exact_cost_match_required": bool(exact_match_required),
                "matched_baseline_condition_id": (
                    f"uniform_exact_cost__{realized_cost}" if exact_match_required else ""
                ),
                "allocation": allocation,
            }
        )
    for cost, allocation in exact_baselines.items():
        matched_to = sorted(matched_cost_to_rules[cost])
        conditions.append(
            {
                "condition_id": f"uniform_exact_cost__{cost}",
                "rule": "uniform_exact_cost",
                "comparison_role": "exact_cost_baseline",
                "matched_cost": int(cost),
                "matched_to_rules": json.dumps(matched_to, separators=(",", ":")),
                "exact_cost_match_required": False,
                "matched_baseline_condition_id": "",
                "allocation": allocation,
            }
        )
    return conditions


def train_budget_allocations(
    base_state,
    vocab_size,
    seq_len,
    cfg,
    adapt_task,
    device,
    module_stats,
    sv_dict,
    run_dir,
    base_seed: int,
):
    budgets = [int(value) for value in cfg.get("budgets", {}).get("param_budgets", [])]
    rank_grid = [int(value) for value in cfg["lora"]["ranks"]]
    adaptation_replicates = int(cfg.get("protocol", {}).get("adaptation_replicates", 1))
    if adaptation_replicates < 1:
        raise ValueError("protocol.adaptation_replicates must be at least 1")
    lora_cfg = cfg["lora"]
    scaling = scaling_protocol(lora_cfg, cfg.get("protocol", {}))
    init_scale = float(lora_cfg.get("init_scale", 0.01))
    max_rank = max(rank_grid or [0])
    rows = []
    alloc_rows = []
    for budget in budgets:
        conditions = _budget_conditions(cfg, module_stats, sv_dict, rank_grid, budget)
        for replicate in range(adaptation_replicates):
            # Neither allocation rule nor scaling mode is part of this seed.
            comparison_seed = stable_seed(
                base_seed,
                _task_key(adapt_task),
                "budget",
                int(budget),
                replicate,
            )
            seeds = _seed_bundle(comparison_seed)
            template = clone_model_from_base(base_state, vocab_size, seq_len, cfg["model"], device)
            init_bank = make_lora_init_bank(
                template,
                max_rank,
                seeds["adapter_seed"],
                init_scale=init_scale,
            )
            train_batches, validation_batches = _adaptation_batches(adapt_task, cfg, device, seeds)
            for scaling_mode in scaling["scaling_modes"]:
                for condition in conditions:
                    rule = condition["rule"]
                    allocation = condition["allocation"]
                    cost = allocation_cost(module_stats, allocation)
                    if condition["comparison_role"] == "exact_cost_baseline" and cost != condition["matched_cost"]:
                        raise RuntimeError(
                            f"exact-cost baseline requested {condition['matched_cost']} but realized {cost}"
                        )
                    signature = _allocation_signature(allocation)
                    print(
                        f"budget allocation mode={scaling_mode} condition={condition['condition_id']} "
                        f"budget={budget} replicate={replicate} cost={cost} alloc={allocation}",
                        flush=True,
                    )
                    model = clone_model_from_base(base_state, vocab_size, seq_len, cfg["model"], device)
                    set_lora_ranks(
                        model,
                        allocation,
                        alpha=scaling["reference_alpha"],
                        scaling_mode=scaling_mode,
                        scale_reference_rank=scaling["reference_rank"],
                        init_scale=init_scale,
                        init_bank=init_bank,
                    )
                    scaling_meta = _allocation_scaling_metadata(allocation, scaling, scaling_mode)
                    metrics, diverged = train_lora(
                        model,
                        adapt_task,
                        lora_cfg,
                        device,
                        eval_task=adapt_task,
                        train_batches=train_batches,
                        validation_batches=validation_batches,
                        dropout_seed=seeds["dropout_seed"],
                    )
                    common = {
                        "protocol_version": PROTOCOL_VERSION,
                        "condition_seed": int(comparison_seed),
                        "adaptation_replicate": int(replicate),
                        **seeds,
                        **scaling_meta,
                        "is_primary_scaling_mode": scaling_mode == scaling["primary_scaling_mode"],
                        "condition_id": condition["condition_id"],
                        "rule": rule,
                        "comparison_role": condition["comparison_role"],
                        "matched_cost": int(condition["matched_cost"]),
                        "matched_to_rules": condition["matched_to_rules"],
                        "exact_cost_match_required": bool(condition["exact_cost_match_required"]),
                        "matched_baseline_condition_id": condition["matched_baseline_condition_id"],
                        "is_primary_allocation_rule": (
                            rule == str(cfg.get("protocol", {}).get("primary_allocation_rule", ""))
                        ),
                        "budget": int(budget),
                        "requested_budget": int(budget),
                        "actual_cost": int(cost),
                        "cost_utilization": float(cost / budget) if budget > 0 else 0.0,
                        "allocation_signature": signature,
                        "exact_modular_evaluation": _exact_modular_evaluation(cfg, adapt_task),
                        "evaluation_examples": int(metrics["val_n_examples"]),
                        "evaluation_batches": int(metrics["val_n_batches"]),
                    }
                    rows.append(
                        {
                            **common,
                            "diverged": bool(diverged),
                            "final_train_loss": metrics["train_loss_last"],
                            "final_val_loss": metrics["val_loss"],
                            "final_val_accuracy": metrics["val_accuracy"],
                            "trainable_params": metrics["trainable_params"],
                        }
                    )
                    for site, rank in allocation.items():
                        resolved = resolve_lora_scaling(
                            int(rank),
                            reference_alpha=scaling["reference_alpha"],
                            scaling_mode=scaling_mode,
                            reference_rank=scaling["reference_rank"],
                        )
                        alloc_rows.append(
                            {
                                **common,
                                "site_name": site,
                                "rank": int(rank),
                                "effective_alpha": resolved.effective_alpha,
                                "lora_scale": resolved.lora_scale,
                            }
                        )
    df = pd.DataFrame(rows)
    allocation_df = pd.DataFrame(alloc_rows)
    _assert_common_random_number_invariant(df, device)
    out_dir = run_dir / "budget"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "budget_results.csv", index=False)
    allocation_df.to_csv(out_dir / "allocation_comparison.csv", index=False)
    return df, allocation_df


def _make_run_directory(cfg: dict, explicit: str | None) -> Path:
    if explicit is None:
        return make_run_dir(
            cfg["run"].get("out_dir", "runs"),
            cfg["run"].get("name", "synthetic_transformer"),
        )
    path = Path(explicit)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None, help="Write to this exact new directory")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    validated_config = validate_synthetic_transformer_config(cfg)
    scaling = {
        key: validated_config[key]
        for key in (
            "scaling_modes",
            "primary_scaling_mode",
            "reference_alpha",
            "reference_rank",
            "scale_at_reference_rank",
        )
    }
    torch.set_num_threads(int(cfg.get("run", {}).get("torch_threads", 1)))
    base_seed = int(cfg["run"].get("seed", 0))
    device = get_device(str(cfg["run"].get("device", "cpu")))
    run_dir = _make_run_directory(cfg, args.run_dir)
    save_yaml(cfg, run_dir / "config.yaml")
    print(f"run_dir={run_dir}")
    print(f"device={device}")
    print(f"protocol={PROTOCOL_VERSION}")
    print(
        "scaling="
        f"modes={scaling['scaling_modes']} primary={scaling['primary_scaling_mode']} "
        f"reference_alpha={scaling['reference_alpha']} reference_rank={scaling['reference_rank']} "
        f"scale_at_reference_rank={scaling['scale_at_reference_rank']}"
    )

    base_task = make_task_spec(cfg["task"], split="base")
    adapt_task = make_task_spec(cfg["task"], split="adapt")
    vocab_size = max(base_task.vocab_size, adapt_task.vocab_size)
    seq_len = max(base_task.seq_len, adapt_task.seq_len)

    base_model_seed = stable_seed(base_seed, _task_key(base_task), "base_model")
    base_train_data_seed = stable_seed(base_seed, _task_key(base_task), "base_train_data")
    base_dropout_seed = stable_seed(base_seed, _task_key(base_task), "base_dropout")
    base_eval_seed = stable_seed(base_seed, _task_key(base_task), "base_eval")
    adapt_eval_seed = stable_seed(base_seed, _task_key(adapt_task), "pre_adapt_eval")
    set_seed(base_model_seed)
    model = build_model(vocab_size, seq_len, cfg["model"]).to(device)
    print("training base model", flush=True)
    base_steps = int(cfg["base_train"].get("steps", 100))
    base_batch_size = int(
        base_task.params.get("train_batch_size", cfg["base_train"].get("batch_size", 64))
    )
    base_train_batches = materialize_batches(
        base_task,
        base_batch_size,
        base_steps,
        device,
        base_train_data_seed,
    )
    base_train_metrics = train_base(
        model,
        base_task,
        cfg["base_train"],
        device,
        eval_task=base_task,
        train_batches=base_train_batches,
        dropout_seed=base_dropout_seed,
    )
    eval_batch_size = int(cfg["task"].get("train_batch_size", 64))
    eval_count = int(cfg["task"].get("eval_batches", 4))
    base_eval_batches = materialize_evaluation_batches(
        base_task,
        eval_batch_size,
        eval_count,
        device,
        base_eval_seed,
        exact_modular=_exact_modular_evaluation(cfg, base_task),
    )
    adapt_eval_batches = materialize_evaluation_batches(
        adapt_task,
        eval_batch_size,
        eval_count,
        device,
        adapt_eval_seed,
        exact_modular=_exact_modular_evaluation(cfg, adapt_task),
    )
    base_eval_base = evaluate(
        model,
        base_task,
        eval_batch_size,
        eval_count,
        device,
        fixed_batches=base_eval_batches,
    )
    base_eval_adapt = evaluate(
        model,
        adapt_task,
        eval_batch_size,
        eval_count,
        device,
        fixed_batches=adapt_eval_batches,
    )
    base_dir = run_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), base_dir / "base_model.pt")
    base_metrics = pd.DataFrame(
        [
            {
                "protocol_version": PROTOCOL_VERSION,
                "base_model_seed": int(base_model_seed),
                "base_train_data_seed": int(base_train_data_seed),
                "base_dropout_seed": int(base_dropout_seed),
                "base_eval_seed": int(base_eval_seed),
                "adapt_eval_seed": int(adapt_eval_seed),
                "exact_modular_evaluation": _exact_modular_evaluation(cfg, adapt_task),
                "base_eval_examples": int(base_eval_base["n_examples"]),
                "adapt_eval_examples": int(base_eval_adapt["n_examples"]),
                **base_train_metrics,
                "base_task_loss": base_eval_base["loss"],
                "base_task_accuracy": base_eval_base["accuracy"],
                "adapt_task_loss_before_lora": base_eval_adapt["loss"],
                "adapt_task_accuracy_before_lora": base_eval_adapt["accuracy"],
            }
        ]
    )
    base_metrics.to_csv(base_dir / "base_metrics.csv", index=False)
    print(base_metrics.to_string(index=False), flush=True)
    base_state_full = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    sites = list(cfg.get("sites", {}).get("include", []))
    print("calibrating module spectra", flush=True)
    calibration_data_seed = stable_seed(base_seed, _task_key(adapt_task), "calibration_data")
    permutation_seed = stable_seed(base_seed, _task_key(adapt_task), "calibration_null")
    calibration_result = calibrate_module_spectra(
        model,
        adapt_task,
        cfg["calibration"],
        sites,
        device,
        seed=permutation_seed,
        data_seed=calibration_data_seed,
        return_null_maxima=True,
    )
    calibration_rows, singular_value_dict, null_maxima_dict = calibration_result
    calibration_dir = run_dir / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    module_stats = pd.DataFrame(calibration_rows)
    module_stats.insert(0, "protocol_version", PROTOCOL_VERSION)
    module_stats.to_csv(calibration_dir / "module_stats.csv", index=False)
    np.savez(
        calibration_dir / "singular_values.npz",
        **{key.replace(".", "__"): value for key, value in singular_value_dict.items()},
    )
    np.savez(
        calibration_dir / "null_maxima.npz",
        **{key.replace(".", "__"): value for key, value in null_maxima_dict.items()},
    )
    print(
        module_stats[
            [
                "site_name",
                "effective_rank",
                "stable_rank",
                "hard_detectable_rank",
                "soft_dimension",
                "gradient_norm",
                "edge",
                "edge_mc_ci_low",
                "edge_mc_ci_high",
            ]
        ].to_string(index=False),
        flush=True,
    )

    ranks = [int(value) for value in cfg["lora"]["ranks"]]
    sweep_df = site_sweeps(
        base_state_full,
        vocab_size,
        seq_len,
        cfg,
        adapt_task,
        device,
        sites,
        ranks,
        run_dir,
        base_seed=base_seed,
    )
    targets = compute_targets(
        sweep_df,
        near_gaps=[float(value) for value in cfg.get("targets", {}).get("near_best_gaps", [0.1, 0.2])],
        recovery_fracs=[float(value) for value in cfg.get("targets", {}).get("recovery_fracs", [0.7, 0.8])],
        lambdas=[float(value) for value in cfg.get("targets", {}).get("penalty_lambdas", [0.2, 0.3])],
    )
    targets.to_csv(run_dir / "sweeps" / "site_target_summary.csv", index=False)
    fit = prediction_fit(targets, module_stats)
    fit.to_csv(run_dir / "sweeps" / "site_prediction_fit.csv", index=False)
    print("prediction fits")
    print(fit.to_string(index=False), flush=True)

    budget_df, _allocation_df = train_budget_allocations(
        base_state_full,
        vocab_size,
        seq_len,
        cfg,
        adapt_task,
        device,
        module_stats,
        singular_value_dict,
        run_dir,
        base_seed=base_seed,
    )

    figure_dir = run_dir / "figures"
    scatter_target = (
        "recovery_rank_0.7"
        if "recovery_rank_0.7" in targets.columns
        else next(column for column in targets.columns if column.startswith("near_best"))
    )
    for scaling_mode in scaling["scaling_modes"]:
        mode_sweeps = sweep_df[sweep_df["scaling_mode"] == scaling_mode]
        mode_targets = targets[targets["scaling_mode"] == scaling_mode]
        mode_budget = budget_df[
            (budget_df["scaling_mode"] == scaling_mode)
            & (budget_df["comparison_role"] != "exact_cost_baseline")
        ]
        plot_rank_sweeps(mode_sweeps, figure_dir / f"site_rank_sweeps_{scaling_mode}.png")
        plot_prediction_scatter(
            mode_targets,
            module_stats,
            figure_dir / f"prediction_scatter_{scaling_mode}.png",
            target_col=scatter_target,
        )
        if len(mode_budget):
            plot_budget_curve(mode_budget, figure_dir / f"budget_curve_{scaling_mode}.png")
    primary = scaling["primary_scaling_mode"]
    plot_rank_sweeps(
        sweep_df[sweep_df["scaling_mode"] == primary],
        figure_dir / "site_rank_sweeps.png",
    )
    plot_prediction_scatter(
        targets[targets["scaling_mode"] == primary],
        module_stats,
        figure_dir / "prediction_scatter.png",
        target_col=scatter_target,
    )
    primary_budget = budget_df[
        (budget_df["scaling_mode"] == primary)
        & (budget_df["comparison_role"] != "exact_cost_baseline")
    ]
    if len(primary_budget):
        plot_budget_curve(primary_budget, figure_dir / "budget_curve.png")
    plot_spectra(module_stats, singular_value_dict, figure_dir / "spectral_profiles.png")

    run_info = {
        "run_id": run_dir.name,
        "run_dir": ".",
        "protocol_version": PROTOCOL_VERSION,
        "base_seed": base_seed,
        "task_name": adapt_task.name,
        "adaptation_replicates": int(cfg.get("protocol", {}).get("adaptation_replicates", 1)),
        "sweep_replicates": int(cfg.get("protocol", {}).get("sweep_replicates", 1)),
        "common_random_numbers": True,
        "rule_in_comparison_seed": False,
        "scaling_mode_in_comparison_seed": False,
        "scaling_modes": scaling["scaling_modes"],
        "primary_scaling_mode": scaling["primary_scaling_mode"],
        "reference_alpha": scaling["reference_alpha"],
        "scale_reference_rank": scaling["reference_rank"],
        "scale_at_reference_rank": scaling["scale_at_reference_rank"],
        "fixed_training_batches": True,
        "fixed_validation_batches": True,
        "exact_modular_evaluation": _exact_modular_evaluation(cfg, adapt_task),
        "nested_max_rank_initialization": True,
        "exact_cost_baselines": bool(cfg.get("budgets", {}).get("cost_matched_rules", [])),
        "cost_matched_rules": list(cfg.get("budgets", {}).get("cost_matched_rules", [])),
        "primary_allocation_rule": str(cfg.get("protocol", {}).get("primary_allocation_rule", "")),
        "primary_reference_rule": str(cfg.get("protocol", {}).get("primary_reference_rule", "uniform_exact_cost")),
        "primary_metric": str(cfg.get("protocol", {}).get("primary_metric", "final_val_loss")),
        "independent_unit": str(cfg.get("protocol", {}).get("independent_unit", "task_seed_run")),
        "calibration_dropout_disabled": True,
        "null_bootstrap": int(cfg.get("calibration", {}).get("null_bootstrap", 8)),
        "null_quantile": float(cfg.get("calibration", {}).get("null_quantile", 0.995)),
        "null_quantile_method": "linear",
        "null_uncertainty_resamples": int(
            cfg.get("calibration", {}).get("null_uncertainty_resamples", 1000)
        ),
        "null_uncertainty_confidence": float(
            cfg.get("calibration", {}).get("null_uncertainty_confidence", 0.95)
        ),
    }
    write_json(run_info, run_dir / "run_info.json")
    write_json(
        {
            "status": "PASS",
            "protocol_version": PROTOCOL_VERSION,
            "base_seed": base_seed,
            "task_name": adapt_task.name,
        },
        run_dir / "_SUCCESS.json",
    )
    print(f"wrote {run_dir}")
    print("key outputs:")
    for path in [
        base_dir / "base_metrics.csv",
        calibration_dir / "module_stats.csv",
        calibration_dir / "singular_values.npz",
        calibration_dir / "null_maxima.npz",
        run_dir / "sweeps" / "site_rank_sweep_metrics.csv",
        run_dir / "sweeps" / "site_target_summary.csv",
        run_dir / "sweeps" / "site_prediction_fit.csv",
        run_dir / "budget" / "budget_results.csv",
        run_dir / "budget" / "allocation_comparison.csv",
        run_dir / "_SUCCESS.json",
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
