#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from strank.model import build_model, clear_all_lora, set_lora_ranks
from strank.plotting import plot_budget_curve, plot_prediction_scatter, plot_rank_sweeps, plot_spectra
from strank.targets import compute_targets, prediction_fit
from strank.tasks import make_batch, make_task_spec
from strank.train import evaluate, train_base, train_lora
from strank.utils import get_device, load_yaml, make_run_dir, save_yaml, set_seed, stable_seed, write_json


def clone_model_from_base(base_state, vocab_size, seq_len, model_cfg, device):
    model = build_model(vocab_size, seq_len, model_cfg).to(device)
    model.load_state_dict(base_state, strict=True)
    clear_all_lora(model)
    return model


def site_sweeps(base_state, vocab_size, seq_len, cfg, adapt_task, device, sites, ranks, run_dir, base_seed: int):
    rows = []
    lora_cfg = cfg["lora"]
    for site in sites:
        for rank in ranks:
            print(f"site sweep site={site} rank={rank}", flush=True)
            model = clone_model_from_base(base_state, vocab_size, seq_len, cfg["model"], device)
            condition_seed = stable_seed(base_seed, "site_sweep", site, int(rank))
            set_seed(condition_seed)
            set_lora_ranks(model, {site: int(rank)}, alpha=float(lora_cfg.get("alpha", 8.0)))
            metrics, diverged = train_lora(model, adapt_task, lora_cfg, device, eval_task=adapt_task)
            rows.append({
                "condition_seed": int(condition_seed),
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


def train_budget_allocations(base_state, vocab_size, seq_len, cfg, adapt_task, device, module_stats, sv_dict, run_dir, base_seed: int):
    budgets = [int(x) for x in cfg.get("budgets", {}).get("param_budgets", [])]
    rules = list(cfg.get("budgets", {}).get("rules", []))
    rank_grid = [int(x) for x in cfg["lora"]["ranks"]]
    rows = []
    alloc_rows = []
    for budget in budgets:
        for rule in rules:
            if rule == "uniform":
                alloc = uniform_allocation(module_stats, rank_grid, budget)
            elif rule == "uniform_fill":
                alloc = uniform_fill_allocation(module_stats, rank_grid, budget)
            elif rule == "effective_rank":
                alloc = score_allocation(module_stats, rank_grid, budget, "effective_rank")
            elif rule == "soft_dimension":
                alloc = score_allocation(module_stats, rank_grid, budget, "soft_dimension")
            elif rule == "gradient_norm":
                alloc = gradient_norm_allocation(module_stats, rank_grid, budget)
            elif rule == "marginal_gain_raw":
                alloc = marginal_gain_allocation(module_stats, sv_dict, rank_grid, budget, mode="raw")
            elif rule == "marginal_gain_edge":
                alloc = marginal_gain_allocation(module_stats, sv_dict, rank_grid, budget, mode="edge")
            elif rule == "marginal_gain_soft":
                alloc = marginal_gain_allocation(module_stats, sv_dict, rank_grid, budget, mode="soft")
            else:
                print(f"skipping unknown allocation rule: {rule}")
                continue
            cost = allocation_cost(module_stats, alloc)
            print(f"budget allocation rule={rule} budget={budget} cost={cost} alloc={alloc}", flush=True)
            model = clone_model_from_base(base_state, vocab_size, seq_len, cfg["model"], device)
            condition_seed = stable_seed(base_seed, "budget", int(budget), rule)
            set_seed(condition_seed)
            set_lora_ranks(model, alloc, alpha=float(cfg["lora"].get("alpha", 8.0)))
            metrics, diverged = train_lora(model, adapt_task, cfg["lora"], device, eval_task=adapt_task)
            rows.append({
                "condition_seed": int(condition_seed),
                "rule": rule,
                "budget": budget,
                "actual_cost": cost,
                "diverged": bool(diverged),
                "final_train_loss": metrics["train_loss_last"],
                "final_val_loss": metrics["val_loss"],
                "final_val_accuracy": metrics["val_accuracy"],
                "trainable_params": metrics["trainable_params"],
            })
            for site, rank in alloc.items():
                alloc_rows.append({"rule": rule, "budget": budget, "site_name": site, "rank": int(rank), "actual_cost": cost, "condition_seed": int(condition_seed)})
    df = pd.DataFrame(rows)
    ar = pd.DataFrame(alloc_rows)
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
    set_seed(int(cfg["run"].get("seed", 0)))
    device = get_device(str(cfg["run"].get("device", "cpu")))
    run_dir = make_run_dir(cfg["run"].get("out_dir", "runs"), cfg["run"].get("name", "synthetic_transformer"))
    save_yaml(cfg, run_dir / "config.yaml")
    print(f"run_dir={run_dir}")
    print(f"device={device}")

    base_task = make_task_spec(cfg["task"], split="base")
    adapt_task = make_task_spec(cfg["task"], split="adapt")
    vocab_size = max(base_task.vocab_size, adapt_task.vocab_size)
    seq_len = max(base_task.seq_len, adapt_task.seq_len)

    model = build_model(vocab_size, seq_len, cfg["model"]).to(device)
    print("training base model", flush=True)
    base_train_metrics = train_base(model, base_task, cfg["base_train"], device, eval_task=base_task)
    base_eval_base = evaluate(model, base_task, int(cfg["task"].get("train_batch_size", 64)), int(cfg["task"].get("eval_batches", 4)), device)
    base_eval_adapt = evaluate(model, adapt_task, int(cfg["task"].get("train_batch_size", 64)), int(cfg["task"].get("eval_batches", 4)), device)
    base_dir = run_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), base_dir / "base_model.pt")
    base_metrics = pd.DataFrame([{
        **base_train_metrics,
        "base_task_loss": base_eval_base["loss"],
        "base_task_accuracy": base_eval_base["accuracy"],
        "adapt_task_loss_before_lora": base_eval_adapt["loss"],
        "adapt_task_accuracy_before_lora": base_eval_adapt["accuracy"],
    }])
    base_metrics.to_csv(base_dir / "base_metrics.csv", index=False)
    print(base_metrics.to_string(index=False), flush=True)
    base_state = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" not in k}
    # Strict state dict needs no LoRA params because clear model has none.
    base_state_full = model.state_dict()

    sites = list(cfg.get("sites", {}).get("include", []))
    print("calibrating module spectra", flush=True)
    cal_rows, sv_dict = calibrate_module_spectra(model, adapt_task, cfg["calibration"], sites, device, seed=int(cfg["run"].get("seed", 0)))
    cal_dir = run_dir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    module_stats = pd.DataFrame(cal_rows)
    module_stats.to_csv(cal_dir / "module_stats.csv", index=False)
    np.savez(cal_dir / "singular_values.npz", **{k.replace(".", "__"): v for k, v in sv_dict.items()})
    print(module_stats[["site_name", "effective_rank", "stable_rank", "hard_detectable_rank", "soft_dimension", "gradient_norm"]].to_string(index=False), flush=True)

    ranks = [int(x) for x in cfg["lora"]["ranks"]]
    base_seed = int(cfg["run"].get("seed", 0))
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
    # Prefer 70% recovery scatter if present.
    scatter_target = "recovery_rank_0.7" if "recovery_rank_0.7" in targets.columns else targets.columns[-1]
    plot_prediction_scatter(targets, module_stats, fig_dir / "prediction_scatter.png", target_col=scatter_target)
    if len(budget_df):
        plot_budget_curve(budget_df, fig_dir / "budget_curve.png")
    plot_spectra(module_stats, sv_dict, fig_dir / "spectral_profiles.png")

    write_json({"run_dir": str(run_dir)}, run_dir / "run_info.json")
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
