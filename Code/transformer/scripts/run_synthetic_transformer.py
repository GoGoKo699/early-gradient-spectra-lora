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

from strank.allocation import allocation_cost, gradient_norm_allocation, marginal_gain_allocation, score_allocation, uniform_allocation, uniform_fill_allocation
from strank.calibrate import calibrate_module_spectra
from strank.model import build_model, clear_all_lora, make_lora_init_bank, set_lora_ranks
from strank.plotting import plot_budget_curve, plot_prediction_scatter, plot_rank_sweeps, plot_spectra
from strank.targets import compute_targets, prediction_fit
from strank.tasks import make_task_spec
from strank.train import evaluate, materialize_batches, train_base, train_lora
from strank.utils import get_device, load_yaml, make_run_dir, save_yaml, set_seed, stable_seed, write_json

PROTOCOL_VERSION = "common_random_numbers_v1"


def _task_key(task) -> str:
    return json.dumps({"name": task.name, "params": task.params}, sort_keys=True, separators=(",", ":"))


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


def _adaptation_batches(adapt_task, lora_cfg, device, seeds: dict[str, int]):
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
    validation_batches = materialize_batches(
        adapt_task,
        batch_size,
        eval_batches,
        device,
        seed=seeds["eval_data_seed"],
    )
    return train_batches, validation_batches


def _allocation_signature(alloc: dict[str, int]) -> str:
    return json.dumps(sorted((str(site), int(rank)) for site, rank in alloc.items()), separators=(",", ":"))


def _assert_common_random_number_invariant(df: pd.DataFrame, device: torch.device) -> None:
    """Identical allocations in one comparison block must produce identical metrics."""
    if df.empty:
        return
    atol = 1e-6 if device.type == "cuda" else 1e-10
    metrics = ["final_train_loss", "final_val_loss", "final_val_accuracy"]
    failures = []
    for keys, group in df.groupby(["budget", "adaptation_replicate", "allocation_signature"], dropna=False):
        if len(group) < 2:
            continue
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) and float(values.max() - values.min()) > atol:
                failures.append((keys, metric, float(values.max() - values.min()), group["rule"].tolist()))
    if failures:
        raise RuntimeError(
            "common-random-number invariant failed for identical allocations: "
            + "; ".join(str(x) for x in failures[:5])
        )


def site_sweeps(base_state, vocab_size, seq_len, cfg, adapt_task, device, sites, ranks, run_dir, base_seed: int):
    rows = []
    lora_cfg = cfg["lora"]
    sweep_replicates = int(cfg.get("protocol", {}).get("sweep_replicates", 1))
    max_rank = max([int(r) for r in ranks] or [0])
    init_scale = float(lora_cfg.get("init_scale", 0.01))
    for site in sites:
        for replicate in range(sweep_replicates):
            comparison_seed = stable_seed(base_seed, _task_key(adapt_task), "site_sweep", site, replicate)
            seeds = _seed_bundle(comparison_seed)
            template = clone_model_from_base(base_state, vocab_size, seq_len, cfg["model"], device)
            init_bank = make_lora_init_bank(template, max_rank, seeds["adapter_seed"], init_scale=init_scale)
            train_batches, validation_batches = _adaptation_batches(adapt_task, lora_cfg, device, seeds)
            for rank in ranks:
                print(f"site sweep site={site} rank={rank} replicate={replicate}", flush=True)
                model = clone_model_from_base(base_state, vocab_size, seq_len, cfg["model"], device)
                set_lora_ranks(
                    model,
                    {site: int(rank)},
                    alpha=float(lora_cfg.get("alpha", 8.0)),
                    init_scale=init_scale,
                    init_bank=init_bank,
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
                rows.append({
                    "protocol_version": PROTOCOL_VERSION,
                    "condition_seed": int(comparison_seed),
                    "adaptation_replicate": int(replicate),
                    **seeds,
                    "site_name": site,
                    "rank": int(rank),
                    "alpha": float(lora_cfg.get("alpha", 8.0)),
                    "diverged": bool(diverged),
                    "final_train_loss": metrics["train_loss_last"],
                    "final_val_loss": metrics["val_loss"],
                    "final_val_accuracy": metrics["val_accuracy"],
                    "trainable_params": metrics["trainable_params"],
                })
    df = pd.DataFrame(rows)
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


def train_budget_allocations(base_state, vocab_size, seq_len, cfg, adapt_task, device, module_stats, sv_dict, run_dir, base_seed: int):
    budgets = [int(x) for x in cfg.get("budgets", {}).get("param_budgets", [])]
    rules = list(cfg.get("budgets", {}).get("rules", []))
    rank_grid = [int(x) for x in cfg["lora"]["ranks"]]
    adaptation_replicates = int(cfg.get("protocol", {}).get("adaptation_replicates", 1))
    if adaptation_replicates < 1:
        raise ValueError("protocol.adaptation_replicates must be at least 1")
    lora_cfg = cfg["lora"]
    init_scale = float(lora_cfg.get("init_scale", 0.01))
    max_rank = max(rank_grid or [0])
    rows = []
    alloc_rows = []
    for budget in budgets:
        allocations = {}
        for rule in rules:
            try:
                allocations[rule] = _allocation_for_rule(rule, module_stats, sv_dict, rank_grid, budget)
            except ValueError:
                print(f"skipping unknown allocation rule: {rule}")
        for replicate in range(adaptation_replicates):
            # Crucially, the rule name is not part of this seed.
            comparison_seed = stable_seed(base_seed, _task_key(adapt_task), "budget", int(budget), replicate)
            seeds = _seed_bundle(comparison_seed)
            template = clone_model_from_base(base_state, vocab_size, seq_len, cfg["model"], device)
            init_bank = make_lora_init_bank(template, max_rank, seeds["adapter_seed"], init_scale=init_scale)
            train_batches, validation_batches = _adaptation_batches(adapt_task, lora_cfg, device, seeds)
            for rule, alloc in allocations.items():
                cost = allocation_cost(module_stats, alloc)
                signature = _allocation_signature(alloc)
                print(
                    f"budget allocation rule={rule} budget={budget} replicate={replicate} cost={cost} alloc={alloc}",
                    flush=True,
                )
                model = clone_model_from_base(base_state, vocab_size, seq_len, cfg["model"], device)
                set_lora_ranks(
                    model,
                    alloc,
                    alpha=float(lora_cfg.get("alpha", 8.0)),
                    init_scale=init_scale,
                    init_bank=init_bank,
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
                rows.append({
                    "protocol_version": PROTOCOL_VERSION,
                    "condition_seed": int(comparison_seed),
                    "adaptation_replicate": int(replicate),
                    **seeds,
                    "rule": rule,
                    "budget": budget,
                    "actual_cost": cost,
                    "allocation_signature": signature,
                    "diverged": bool(diverged),
                    "final_train_loss": metrics["train_loss_last"],
                    "final_val_loss": metrics["val_loss"],
                    "final_val_accuracy": metrics["val_accuracy"],
                    "trainable_params": metrics["trainable_params"],
                })
                for site, rank in alloc.items():
                    alloc_rows.append({
                        "protocol_version": PROTOCOL_VERSION,
                        "rule": rule,
                        "budget": budget,
                        "adaptation_replicate": int(replicate),
                        "site_name": site,
                        "rank": int(rank),
                        "actual_cost": cost,
                        "allocation_signature": signature,
                        "condition_seed": int(comparison_seed),
                        **seeds,
                    })
    df = pd.DataFrame(rows)
    ar = pd.DataFrame(alloc_rows)
    _assert_common_random_number_invariant(df, device)
    out_dir = run_dir / "budget"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "budget_results.csv", index=False)
    ar.to_csv(out_dir / "allocation_comparison.csv", index=False)
    return df, ar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    torch.set_num_threads(int(cfg.get("run", {}).get("torch_threads", 1)))
    base_seed = int(cfg["run"].get("seed", 0))
    device = get_device(str(cfg["run"].get("device", "cpu")))
    run_dir = make_run_dir(cfg["run"].get("out_dir", "runs"), cfg["run"].get("name", "synthetic_transformer"))
    save_yaml(cfg, run_dir / "config.yaml")
    print(f"run_dir={run_dir}")
    print(f"device={device}")
    print(f"protocol={PROTOCOL_VERSION}")

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
    base_bs = int(base_task.params.get("train_batch_size", cfg["base_train"].get("batch_size", 64)))
    base_train_batches = materialize_batches(base_task, base_bs, base_steps, device, base_train_data_seed)
    base_train_metrics = train_base(
        model,
        base_task,
        cfg["base_train"],
        device,
        eval_task=base_task,
        train_batches=base_train_batches,
        dropout_seed=base_dropout_seed,
    )
    eval_bs = int(cfg["task"].get("train_batch_size", 64))
    eval_count = int(cfg["task"].get("eval_batches", 4))
    base_eval_batches = materialize_batches(base_task, eval_bs, eval_count, device, base_eval_seed)
    adapt_eval_batches = materialize_batches(adapt_task, eval_bs, eval_count, device, adapt_eval_seed)
    base_eval_base = evaluate(model, base_task, eval_bs, eval_count, device, fixed_batches=base_eval_batches)
    base_eval_adapt = evaluate(model, adapt_task, eval_bs, eval_count, device, fixed_batches=adapt_eval_batches)
    base_dir = run_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), base_dir / "base_model.pt")
    base_metrics = pd.DataFrame([{
        "protocol_version": PROTOCOL_VERSION,
        "base_model_seed": int(base_model_seed),
        "base_train_data_seed": int(base_train_data_seed),
        "base_dropout_seed": int(base_dropout_seed),
        "base_eval_seed": int(base_eval_seed),
        "adapt_eval_seed": int(adapt_eval_seed),
        **base_train_metrics,
        "base_task_loss": base_eval_base["loss"],
        "base_task_accuracy": base_eval_base["accuracy"],
        "adapt_task_loss_before_lora": base_eval_adapt["loss"],
        "adapt_task_accuracy_before_lora": base_eval_adapt["accuracy"],
    }])
    base_metrics.to_csv(base_dir / "base_metrics.csv", index=False)
    print(base_metrics.to_string(index=False), flush=True)
    base_state_full = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    sites = list(cfg.get("sites", {}).get("include", []))
    print("calibrating module spectra", flush=True)
    calibration_data_seed = stable_seed(base_seed, _task_key(adapt_task), "calibration_data")
    permutation_seed = stable_seed(base_seed, _task_key(adapt_task), "calibration_null")
    set_seed(calibration_data_seed)
    cal_rows, sv_dict = calibrate_module_spectra(model, adapt_task, cfg["calibration"], sites, device, seed=permutation_seed)
    cal_dir = run_dir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    module_stats = pd.DataFrame(cal_rows)
    module_stats.insert(0, "calibration_data_seed", int(calibration_data_seed))
    module_stats.insert(1, "permutation_seed", int(permutation_seed))
    module_stats.to_csv(cal_dir / "module_stats.csv", index=False)
    np.savez(cal_dir / "singular_values.npz", **{k.replace(".", "__"): v for k, v in sv_dict.items()})
    print(module_stats[["site_name", "effective_rank", "stable_rank", "hard_detectable_rank", "soft_dimension", "gradient_norm"]].to_string(index=False), flush=True)

    ranks = [int(x) for x in cfg["lora"]["ranks"]]
    sweep_df = site_sweeps(base_state_full, vocab_size, seq_len, cfg, adapt_task, device, sites, ranks, run_dir, base_seed=base_seed)
    targets = compute_targets(
        sweep_df,
        near_gaps=[float(x) for x in cfg.get("targets", {}).get("near_best_gaps", [0.1, 0.2])],
        recovery_fracs=[float(x) for x in cfg.get("targets", {}).get("recovery_fracs", [0.7, 0.8])],
        lambdas=[float(x) for x in cfg.get("targets", {}).get("penalty_lambdas", [0.2, 0.3])],
    )
    targets.to_csv(run_dir / "sweeps" / "site_target_summary.csv", index=False)
    fit = prediction_fit(targets, module_stats)
    fit.to_csv(run_dir / "sweeps" / "site_prediction_fit.csv", index=False)
    print("prediction fits")
    print(fit.to_string(index=False), flush=True)

    budget_df, alloc_df = train_budget_allocations(base_state_full, vocab_size, seq_len, cfg, adapt_task, device, module_stats, sv_dict, run_dir, base_seed=base_seed)

    fig_dir = run_dir / "figures"
    plot_rank_sweeps(sweep_df, fig_dir / "site_rank_sweeps.png")
    scatter_target = "recovery_rank_0.7" if "recovery_rank_0.7" in targets.columns else targets.columns[-1]
    plot_prediction_scatter(targets, module_stats, fig_dir / "prediction_scatter.png", target_col=scatter_target)
    if len(budget_df):
        plot_budget_curve(budget_df, fig_dir / "budget_curve.png")
    plot_spectra(module_stats, sv_dict, fig_dir / "spectral_profiles.png")

    write_json({
        "run_dir": str(run_dir),
        "protocol_version": PROTOCOL_VERSION,
        "base_seed": base_seed,
        "adaptation_replicates": int(cfg.get("protocol", {}).get("adaptation_replicates", 1)),
        "sweep_replicates": int(cfg.get("protocol", {}).get("sweep_replicates", 1)),
        "common_random_numbers": True,
        "rule_in_comparison_seed": False,
        "fixed_training_batches": True,
        "fixed_validation_batches": True,
        "nested_max_rank_initialization": True,
    }, run_dir / "run_info.json")
    print(f"wrote {run_dir}")
    print("key outputs:")
    for p in [
        base_dir / "base_metrics.csv",
        cal_dir / "module_stats.csv",
        run_dir / "sweeps" / "site_rank_sweep_metrics.csv",
        run_dir / "sweeps" / "site_target_summary.csv",
        run_dir / "sweeps" / "site_prediction_fit.csv",
        run_dir / "budget" / "budget_results.csv",
    ]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
