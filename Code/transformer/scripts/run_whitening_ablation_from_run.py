#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from typing import Dict, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from strank.allocation import (
    allocation_cost,
    gradient_norm_allocation,
    marginal_gain_allocation,
    score_allocation,
    uniform_fill_allocation,
)
from strank.calibrate import calibrate_module_spectra
from strank.model import build_model, clear_all_lora, set_lora_ranks
from strank.tasks import make_task_spec
from strank.train import train_lora
from strank.utils import get_device, load_yaml, make_run_dir, save_yaml, set_seed, stable_seed



def _clone_base(base_state: Mapping[str, torch.Tensor], vocab_size: int, seq_len: int, model_cfg: Dict, device: torch.device):
    model = build_model(vocab_size, seq_len, model_cfg).to(device)
    model.load_state_dict(base_state, strict=True)
    clear_all_lora(model)
    return model


def _save_singular_values(path: Path, sv_dict: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k.replace(".", "__"): v for k, v in sv_dict.items()})


def _load_state_dict(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_base_run(base_run: Path, device: torch.device):
    cfg = load_yaml(base_run / "config.yaml")
    base_task = make_task_spec(cfg["task"], split="base")
    adapt_task = make_task_spec(cfg["task"], split="adapt")
    vocab_size = max(base_task.vocab_size, adapt_task.vocab_size)
    seq_len = max(base_task.seq_len, adapt_task.seq_len)
    base_state_path = base_run / "base" / "base_model.pt"
    if not base_state_path.exists():
        raise FileNotFoundError(f"missing base model: {base_state_path}")
    base_state = _load_state_dict(base_state_path, device)
    return cfg, adapt_task, vocab_size, seq_len, base_state


def _calibrate_all(
    cfg: Dict,
    adapt_task,
    vocab_size: int,
    seq_len: int,
    base_state: Mapping[str, torch.Tensor],
    device: torch.device,
    sites: list[str],
    whitenings: list[str],
    run_dir: Path,
    seed: int,
):
    stats_by_w: Dict[str, pd.DataFrame] = {}
    sv_by_w: Dict[str, Dict[str, np.ndarray]] = {}
    all_rows = []
    for wi, whitening in enumerate(whitenings):
        print(f"calibration whitening={whitening}", flush=True)
        model = _clone_base(base_state, vocab_size, seq_len, cfg["model"], device)
        cal_cfg = copy.deepcopy(cfg["calibration"])
        cal_cfg["whitening"] = whitening
        # Use identical calibration batches for every whitening condition.
        set_seed(seed + 424242)
        rows, svs = calibrate_module_spectra(model, adapt_task, cal_cfg, sites, device, seed=seed + 1000 * (wi + 1))
        for r in rows:
            r["whitening_ablation"] = whitening
        df = pd.DataFrame(rows)
        stats_by_w[whitening] = df
        sv_by_w[whitening] = svs
        all_rows.extend(rows)
        out = run_dir / "calibration" / whitening
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "module_stats.csv", index=False)
        _save_singular_values(out / "singular_values.npz", svs)
    pd.DataFrame(all_rows).to_csv(run_dir / "calibration" / "module_stats_all_whitenings.csv", index=False)
    return stats_by_w, sv_by_w


def _alloc_for_rule(rule: str, stats: pd.DataFrame, svs: Dict[str, np.ndarray], rank_grid: list[int], budget: int):
    if rule == "uniform_fill":
        return uniform_fill_allocation(stats, rank_grid, budget)
    if rule == "gradient_norm":
        return gradient_norm_allocation(stats, rank_grid, budget)
    if rule == "effective_rank":
        return score_allocation(stats, rank_grid, budget, "effective_rank")
    if rule == "soft_dimension":
        return score_allocation(stats, rank_grid, budget, "soft_dimension")
    if rule == "hard_detectable_rank":
        return score_allocation(stats, rank_grid, budget, "hard_detectable_rank")
    if rule == "marginal_gain_raw":
        return marginal_gain_allocation(stats, svs, rank_grid, budget, mode="raw")
    if rule == "marginal_gain_edge":
        return marginal_gain_allocation(stats, svs, rank_grid, budget, mode="edge")
    if rule == "marginal_gain_soft":
        return marginal_gain_allocation(stats, svs, rank_grid, budget, mode="soft")
    raise ValueError(f"unknown rule: {rule}")


def _budget_ablation(
    cfg: Dict,
    adapt_task,
    vocab_size: int,
    seq_len: int,
    base_state: Mapping[str, torch.Tensor],
    device: torch.device,
    run_dir: Path,
    stats_by_w: Dict[str, pd.DataFrame],
    sv_by_w: Dict[str, Dict[str, np.ndarray]],
    whitenings: list[str],
    spectral_rules: list[str],
    seed: int,
):
    rank_grid = [int(x) for x in cfg["lora"]["ranks"]]
    budgets = [int(x) for x in cfg.get("budgets", {}).get("param_budgets", [])]
    baseline_stats = stats_by_w["none"] if "none" in stats_by_w else next(iter(stats_by_w.values()))
    baseline_svs = sv_by_w["none"] if "none" in sv_by_w else next(iter(sv_by_w.values()))
    rows = []
    alloc_rows = []

    # Baselines independent of whitening.
    rule_specs = [("baseline", "uniform_fill", baseline_stats, baseline_svs, "uniform_fill")]
    rule_specs.append(("baseline", "gradient_norm", baseline_stats, baseline_svs, "gradient_norm"))
    for w in whitenings:
        for rule in spectral_rules:
            rule_specs.append((w, rule, stats_by_w[w], sv_by_w[w], f"{w}_{rule}"))

    for budget in budgets:
        for whitening, rule, stats, svs, label in rule_specs:
            alloc = _alloc_for_rule(rule, stats, svs, rank_grid, budget)
            cost = allocation_cost(stats, alloc)
            print(f"ablation budget={budget} label={label} cost={cost} alloc={alloc}", flush=True)
            set_seed(stable_seed(seed, "whitening_ablation", budget, label))
            model = _clone_base(base_state, vocab_size, seq_len, cfg["model"], device)
            set_lora_ranks(model, alloc, alpha=float(cfg["lora"].get("alpha", 8.0)))
            metrics, diverged = train_lora(model, adapt_task, cfg["lora"], device, eval_task=adapt_task)
            rows.append({
                "label": label,
                "rule": rule,
                "whitening": whitening,
                "budget": budget,
                "actual_cost": cost,
                "diverged": bool(diverged),
                "final_train_loss": metrics["train_loss_last"],
                "final_val_loss": metrics["val_loss"],
                "final_val_accuracy": metrics["val_accuracy"],
                "trainable_params": metrics["trainable_params"],
            })
            for site, rank in alloc.items():
                alloc_rows.append({
                    "label": label,
                    "rule": rule,
                    "whitening": whitening,
                    "budget": budget,
                    "site_name": site,
                    "rank": int(rank),
                    "actual_cost": cost,
                })
    out_dir = run_dir / "budget"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    ar = pd.DataFrame(alloc_rows)
    df.to_csv(out_dir / "whitening_ablation_results.csv", index=False)
    ar.to_csv(out_dir / "whitening_ablation_allocations.csv", index=False)
    # Also write budget_results.csv so existing summary tooling can inspect the run.
    budget_view = df.rename(columns={"rule": "allocation_rule", "label": "rule"})
    budget_view.to_csv(out_dir / "budget_results.csv", index=False)
    return df, ar


def _summarize_and_plot(df: pd.DataFrame, run_dir: Path) -> None:
    out_dir = run_dir / "budget"
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    base = df[df["label"] == "uniform_fill"][["budget", "final_val_loss", "final_val_accuracy"]]
    base = base.rename(columns={"final_val_loss": "uniform_fill_loss", "final_val_accuracy": "uniform_fill_acc"})
    merged = df.merge(base, on="budget", how="left")
    merged["loss_delta_vs_uniform_fill"] = merged["final_val_loss"] - merged["uniform_fill_loss"]
    merged["acc_delta_vs_uniform_fill"] = merged["final_val_accuracy"] - merged["uniform_fill_acc"]
    merged.to_csv(out_dir / "whitening_ablation_deltas.csv", index=False)
    summary = merged[merged["label"] != "uniform_fill"].groupby(["label", "whitening", "rule"], dropna=False).agg(
        mean_loss_delta=("loss_delta_vs_uniform_fill", "mean"),
        median_loss_delta=("loss_delta_vs_uniform_fill", "median"),
        mean_acc_delta=("acc_delta_vs_uniform_fill", "mean"),
        wins=("loss_delta_vs_uniform_fill", lambda x: int((x < 0).sum())),
        n=("loss_delta_vs_uniform_fill", "size"),
    ).reset_index().sort_values(["mean_loss_delta", "label"])
    summary.to_csv(out_dir / "whitening_ablation_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    for label, sub in df.groupby("label"):
        plt.plot(sub["budget"], sub["final_val_loss"], marker="o", label=label)
    plt.xlabel("parameter budget")
    plt.ylabel("validation loss")
    plt.title("Whitening ablation: budget curve")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(fig_dir / "whitening_ablation_budget_loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plot_df = summary.sort_values("mean_loss_delta")
    plt.barh(plot_df["label"], plot_df["mean_loss_delta"])
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("mean loss delta vs uniform_fill")
    plt.title("Whitening ablation summary")
    plt.tight_layout()
    plt.savefig(fig_dir / "whitening_ablation_mean_delta.png", dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run whitening ablation from an existing trained-base run.")
    ap.add_argument("--base-run", required=True, help="Existing run directory with config.yaml and base/base_model.pt")
    ap.add_argument("--name", default=None, help="Name suffix for ablation run")
    ap.add_argument("--whitenings", default="none,diag,full", help="Comma-separated whitening modes: none,diag,full")
    ap.add_argument("--spectral-rules", default="effective_rank,soft_dimension,marginal_gain_soft", help="Comma-separated spectral allocation rules")
    ap.add_argument("--device", default=None, help="Override device")
    ap.add_argument("--threads", type=int, default=None, help="Override torch thread count")
    args = ap.parse_args()

    base_run = Path(args.base_run)
    if not base_run.exists():
        raise FileNotFoundError(base_run)
    cfg0 = load_yaml(base_run / "config.yaml")
    threads = int(args.threads if args.threads is not None else cfg0.get("run", {}).get("torch_threads", 1))
    torch.set_num_threads(threads)
    seed = int(cfg0.get("run", {}).get("seed", 0))
    set_seed(seed)
    device = get_device(args.device or str(cfg0.get("run", {}).get("device", "cpu")))
    cfg, adapt_task, vocab_size, seq_len, base_state = _load_base_run(base_run, device)
    sites = list(cfg.get("sites", {}).get("include", []))
    if not sites:
        raise ValueError("config has no sites.include")

    whitenings = [x.strip() for x in args.whitenings.split(",") if x.strip()]
    spectral_rules = [x.strip() for x in args.spectral_rules.split(",") if x.strip()]
    name = args.name or f"whitening_ablation_from_{base_run.name}"
    run_dir = make_run_dir(cfg.get("run", {}).get("out_dir", "runs"), name)
    meta = copy.deepcopy(cfg)
    meta["ablation"] = {
        "base_run": str(base_run),
        "whitenings": whitenings,
        "spectral_rules": spectral_rules,
        "threads": threads,
        "device": str(device),
    }
    save_yaml(meta, run_dir / "config.yaml")
    print(f"run_dir={run_dir}")
    print(f"base_run={base_run}")
    print(f"whitenings={whitenings}")
    print(f"spectral_rules={spectral_rules}")

    stats_by_w, sv_by_w = _calibrate_all(cfg, adapt_task, vocab_size, seq_len, base_state, device, sites, whitenings, run_dir, seed=seed)
    df, _ar = _budget_ablation(cfg, adapt_task, vocab_size, seq_len, base_state, device, run_dir, stats_by_w, sv_by_w, whitenings, spectral_rules, seed=seed)
    _summarize_and_plot(df, run_dir)
    print("summary:")
    print(pd.read_csv(run_dir / "budget" / "whitening_ablation_summary.csv").to_string(index=False))
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
