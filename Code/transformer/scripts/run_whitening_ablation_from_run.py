#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
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
    uniform_exact_cost_allocations,
    uniform_fill_allocation,
)
from strank.calibrate import calibrate_module_spectra
from strank.model import build_model, clear_all_lora, make_lora_init_bank, set_lora_ranks
from strank.protocol import WHITENING_ABLATION_PROTOCOL_VERSION
from strank.scaling import resolve_lora_scaling, scaling_protocol
from strank.tasks import make_task_spec
from strank.train import materialize_batches, materialize_evaluation_batches, train_lora
from strank.utils import get_device, load_yaml, make_run_dir, save_yaml, stable_seed, write_json

PROTOCOL_VERSION = WHITENING_ABLATION_PROTOCOL_VERSION


def _task_key(task) -> str:
    return json.dumps(
        {"name": task.name, "params": task.params},
        sort_keys=True,
        separators=(",", ":"),
    )


def _exact_modular_evaluation(cfg: Dict, task) -> bool:
    return (
        bool(cfg.get("protocol", {}).get("exact_modular_evaluation", False))
        and task.name == "modular"
    )


def _clone_base(
    base_state: Mapping[str, torch.Tensor],
    vocab_size: int,
    seq_len: int,
    model_cfg: Dict,
    device: torch.device,
):
    model = build_model(vocab_size, seq_len, model_cfg).to(device)
    model.load_state_dict(base_state, strict=True)
    clear_all_lora(model)
    return model


def _save_arrays(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{key.replace(".", "__"): value for key, value in arrays.items()})


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
    return cfg, adapt_task, vocab_size, seq_len, _load_state_dict(base_state_path, device)


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
    stats_by_whitening: Dict[str, pd.DataFrame] = {}
    singular_values_by_whitening: Dict[str, Dict[str, np.ndarray]] = {}
    all_rows: list[dict] = []
    calibration_data_seed = stable_seed(
        seed, _task_key(adapt_task), "whitening_ablation", "calibration_data"
    )
    permutation_seed = stable_seed(
        seed, _task_key(adapt_task), "whitening_ablation", "calibration_null"
    )

    for whitening in whitenings:
        print(f"calibration whitening={whitening}", flush=True)
        model = _clone_base(base_state, vocab_size, seq_len, cfg["model"], device)
        calibration_cfg = copy.deepcopy(cfg["calibration"])
        calibration_cfg["whitening"] = whitening
        rows, singular_values, null_maxima = calibrate_module_spectra(
            model,
            adapt_task,
            calibration_cfg,
            sites,
            device,
            seed=permutation_seed,
            data_seed=calibration_data_seed,
            return_null_maxima=True,
        )
        for row in rows:
            row["whitening_ablation"] = whitening
            row["permutation_seed_base"] = int(permutation_seed)
        frame = pd.DataFrame(rows)
        stats_by_whitening[whitening] = frame
        singular_values_by_whitening[whitening] = singular_values
        all_rows.extend(rows)
        output_dir = run_dir / "calibration" / whitening
        output_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_dir / "module_stats.csv", index=False)
        _save_arrays(output_dir / "singular_values.npz", singular_values)
        _save_arrays(output_dir / "null_maxima.npz", null_maxima)

    pd.DataFrame(all_rows).to_csv(
        run_dir / "calibration" / "module_stats_all_whitenings.csv",
        index=False,
    )
    return stats_by_whitening, singular_values_by_whitening


def _allocation_for_rule(
    rule: str,
    stats: pd.DataFrame,
    singular_values: Dict[str, np.ndarray],
    rank_grid: list[int],
    budget: int,
):
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
        return marginal_gain_allocation(stats, singular_values, rank_grid, budget, mode="raw")
    if rule == "marginal_gain_edge":
        return marginal_gain_allocation(stats, singular_values, rank_grid, budget, mode="edge")
    if rule == "marginal_gain_soft":
        return marginal_gain_allocation(stats, singular_values, rank_grid, budget, mode="soft")
    raise ValueError(f"unknown rule: {rule}")


def _seed_bundle(comparison_seed: int) -> dict[str, int]:
    return {
        "comparison_seed": int(comparison_seed),
        "adapter_seed": int(stable_seed(comparison_seed, "adapter_init")),
        "train_data_seed": int(stable_seed(comparison_seed, "train_data")),
        "eval_data_seed": int(stable_seed(comparison_seed, "eval_data")),
        "dropout_seed": int(stable_seed(comparison_seed, "dropout")),
    }


def _allocation_signature(allocation: Mapping[str, int]) -> str:
    return json.dumps(
        sorted((str(site), int(rank)) for site, rank in allocation.items()),
        separators=(",", ":"),
    )


def _prepare_budget_conditions(
    budget: int,
    rank_grid: list[int],
    baseline_stats: pd.DataFrame,
    baseline_singular_values: Dict[str, np.ndarray],
    stats_by_whitening: Dict[str, pd.DataFrame],
    singular_values_by_whitening: Dict[str, Dict[str, np.ndarray]],
    whitenings: list[str],
    spectral_rules: list[str],
) -> list[dict]:
    specs = [
        ("baseline", "uniform_fill", baseline_stats, baseline_singular_values, "uniform_fill"),
        ("baseline", "gradient_norm", baseline_stats, baseline_singular_values, "gradient_norm"),
    ]
    for whitening in whitenings:
        for rule in spectral_rules:
            specs.append(
                (
                    whitening,
                    rule,
                    stats_by_whitening[whitening],
                    singular_values_by_whitening[whitening],
                    f"{whitening}_{rule}",
                )
            )

    conditions: list[dict] = []
    candidate_costs: dict[int, list[str]] = {}
    for whitening, rule, stats, singular_values, label in specs:
        allocation = _allocation_for_rule(rule, stats, singular_values, rank_grid, budget)
        cost = allocation_cost(baseline_stats, allocation)
        is_candidate = label != "uniform_fill"
        if is_candidate:
            candidate_costs.setdefault(cost, []).append(label)
        conditions.append(
            {
                "condition_id": label,
                "label": label,
                "rule": rule,
                "whitening": whitening,
                "comparison_role": "candidate" if is_candidate else "cap_baseline",
                "matched_cost": int(cost),
                "matched_to_rules": "[]",
                "matched_baseline_condition_id": (
                    f"uniform_exact_cost__{cost}" if is_candidate else ""
                ),
                "allocation": allocation,
            }
        )

    exact_allocations = uniform_exact_cost_allocations(
        baseline_stats,
        rank_grid,
        list(candidate_costs),
    )
    for cost, allocation in exact_allocations.items():
        conditions.append(
            {
                "condition_id": f"uniform_exact_cost__{cost}",
                "label": f"uniform_exact_cost__{cost}",
                "rule": "uniform_exact_cost",
                "whitening": "baseline",
                "comparison_role": "exact_cost_baseline",
                "matched_cost": int(cost),
                "matched_to_rules": json.dumps(
                    sorted(candidate_costs[cost]), separators=(",", ":")
                ),
                "matched_baseline_condition_id": "",
                "allocation": allocation,
            }
        )
    return conditions


def _budget_ablation(
    cfg: Dict,
    adapt_task,
    vocab_size: int,
    seq_len: int,
    base_state: Mapping[str, torch.Tensor],
    device: torch.device,
    run_dir: Path,
    stats_by_whitening: Dict[str, pd.DataFrame],
    singular_values_by_whitening: Dict[str, Dict[str, np.ndarray]],
    whitenings: list[str],
    spectral_rules: list[str],
    seed: int,
):
    rank_grid = [int(value) for value in cfg["lora"]["ranks"]]
    budgets = [int(value) for value in cfg.get("budgets", {}).get("param_budgets", [])]
    adaptation_replicates = int(cfg.get("protocol", {}).get("adaptation_replicates", 1))
    if adaptation_replicates < 1:
        raise ValueError("protocol.adaptation_replicates must be at least 1")

    baseline_stats = (
        stats_by_whitening["none"]
        if "none" in stats_by_whitening
        else next(iter(stats_by_whitening.values()))
    )
    baseline_singular_values = (
        singular_values_by_whitening["none"]
        if "none" in singular_values_by_whitening
        else next(iter(singular_values_by_whitening.values()))
    )
    lora_cfg = cfg["lora"]
    scaling = scaling_protocol(lora_cfg, cfg.get("protocol", {}))
    max_rank = max(rank_grid or [0])
    init_scale = float(lora_cfg.get("init_scale", 0.01))
    batch_size = int(adapt_task.params.get("train_batch_size", lora_cfg.get("batch_size", 64)))
    steps = int(lora_cfg.get("steps", 100))
    eval_count = int(adapt_task.params.get("eval_batches", 4))
    rows: list[dict] = []
    allocation_rows: list[dict] = []

    for budget in budgets:
        conditions = _prepare_budget_conditions(
            budget,
            rank_grid,
            baseline_stats,
            baseline_singular_values,
            stats_by_whitening,
            singular_values_by_whitening,
            whitenings,
            spectral_rules,
        )
        for replicate in range(adaptation_replicates):
            comparison_seed = stable_seed(
                seed,
                _task_key(adapt_task),
                "whitening_ablation",
                budget,
                replicate,
            )
            seeds = _seed_bundle(comparison_seed)
            template = _clone_base(base_state, vocab_size, seq_len, cfg["model"], device)
            init_bank = make_lora_init_bank(
                template,
                max_rank,
                seeds["adapter_seed"],
                init_scale=init_scale,
            )
            train_batches = materialize_batches(
                adapt_task,
                batch_size,
                steps,
                device,
                seeds["train_data_seed"],
            )
            validation_batches = materialize_evaluation_batches(
                adapt_task,
                batch_size,
                eval_count,
                device,
                seeds["eval_data_seed"],
                exact_modular=_exact_modular_evaluation(cfg, adapt_task),
            )

            for scaling_mode in scaling["scaling_modes"]:
                for condition in conditions:
                    allocation = condition["allocation"]
                    cost = allocation_cost(baseline_stats, allocation)
                    if cost != int(condition["matched_cost"]):
                        raise RuntimeError(
                            f"condition {condition['condition_id']} expected cost "
                            f"{condition['matched_cost']} but realized {cost}"
                        )
                    signature = _allocation_signature(allocation)
                    resolved_active = [
                        resolve_lora_scaling(
                            int(rank),
                            scaling["reference_alpha"],
                            scaling_mode,
                            scaling["reference_rank"],
                        )
                        for rank in allocation.values()
                        if int(rank) > 0
                    ]
                    scales = [resolved.lora_scale for resolved in resolved_active]
                    print(
                        f"ablation mode={scaling_mode} budget={budget} "
                        f"replicate={replicate} condition={condition['condition_id']} "
                        f"cost={cost} alloc={allocation}",
                        flush=True,
                    )
                    model = _clone_base(base_state, vocab_size, seq_len, cfg["model"], device)
                    set_lora_ranks(
                        model,
                        allocation,
                        alpha=scaling["reference_alpha"],
                        scaling_mode=scaling_mode,
                        scale_reference_rank=scaling["reference_rank"],
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
                    common = {
                        "protocol_version": PROTOCOL_VERSION,
                        "condition_seed": int(comparison_seed),
                        "adaptation_replicate": int(replicate),
                        **seeds,
                        "scaling_mode": scaling_mode,
                        "is_primary_scaling_mode": (
                            scaling_mode == scaling["primary_scaling_mode"]
                        ),
                        "reference_alpha": scaling["reference_alpha"],
                        "scale_reference_rank": scaling["reference_rank"],
                        "scale_at_reference_rank": scaling["scale_at_reference_rank"],
                        "min_lora_scale": min(scales) if scales else 0.0,
                        "max_lora_scale": max(scales) if scales else 0.0,
                        "mean_lora_scale": float(np.mean(scales)) if scales else 0.0,
                        "condition_id": condition["condition_id"],
                        "label": condition["label"],
                        "rule": condition["rule"],
                        "whitening": condition["whitening"],
                        "comparison_role": condition["comparison_role"],
                        "matched_cost": int(condition["matched_cost"]),
                        "matched_to_rules": condition["matched_to_rules"],
                        "matched_baseline_condition_id": condition[
                            "matched_baseline_condition_id"
                        ],
                        "budget": int(budget),
                        "requested_budget": int(budget),
                        "actual_cost": int(cost),
                        "allocation_signature": signature,
                        "exact_modular_evaluation": _exact_modular_evaluation(
                            cfg, adapt_task
                        ),
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
                            scaling["reference_alpha"],
                            scaling_mode,
                            scaling["reference_rank"],
                        )
                        allocation_rows.append(
                            {
                                **common,
                                "site_name": site,
                                "rank": int(rank),
                                "effective_alpha": resolved.effective_alpha,
                                "lora_scale": resolved.lora_scale,
                            }
                        )

    output_dir = run_dir / "budget"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    allocation_frame = pd.DataFrame(allocation_rows)
    atol = 1e-6 if device.type == "cuda" else 1e-10
    for _, group in frame.groupby(
        ["scaling_mode", "budget", "adaptation_replicate", "allocation_signature"]
    ):
        if len(group) < 2:
            continue
        for metric in ("final_train_loss", "final_val_loss", "final_val_accuracy"):
            values = group[metric].to_numpy(dtype=float)
            if float(values.max() - values.min()) > atol:
                raise RuntimeError(
                    f"identical-allocation CRN invariant failed for {metric}: "
                    f"{group['condition_id'].tolist()}"
                )
    frame.to_csv(output_dir / "whitening_ablation_results.csv", index=False)
    allocation_frame.to_csv(
        output_dir / "whitening_ablation_allocations.csv", index=False
    )
    frame.rename(columns={"rule": "allocation_rule", "label": "rule"}).to_csv(
        output_dir / "budget_results.csv", index=False
    )
    return frame, allocation_frame


def _summarize_and_plot(
    frame: pd.DataFrame,
    run_dir: Path,
    primary_scaling_mode: str,
) -> None:
    output_dir = run_dir / "budget"
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    pair_keys = ["scaling_mode", "budget", "adaptation_replicate"]

    cap_reference = frame[frame["condition_id"] == "uniform_fill"][
        pair_keys + ["final_val_loss", "final_val_accuracy", "actual_cost"]
    ].rename(
        columns={
            "final_val_loss": "uniform_fill_loss",
            "final_val_accuracy": "uniform_fill_accuracy",
            "actual_cost": "uniform_fill_cost",
        }
    )
    cap_candidates = frame[frame["comparison_role"] != "exact_cost_baseline"].copy()
    cap_deltas = cap_candidates.merge(
        cap_reference,
        on=pair_keys,
        how="left",
        validate="many_to_one",
    )
    cap_deltas["loss_delta_vs_uniform_fill"] = (
        cap_deltas["final_val_loss"] - cap_deltas["uniform_fill_loss"]
    )
    cap_deltas["acc_delta_vs_uniform_fill"] = (
        cap_deltas["final_val_accuracy"] - cap_deltas["uniform_fill_accuracy"]
    )
    cap_deltas["cost_gap_vs_uniform_fill"] = (
        cap_deltas["actual_cost"] - cap_deltas["uniform_fill_cost"]
    )
    cap_deltas.to_csv(
        output_dir / "whitening_ablation_deltas_vs_cap_uniform.csv", index=False
    )

    exact_reference = frame[frame["comparison_role"] == "exact_cost_baseline"][
        pair_keys
        + ["condition_id", "final_val_loss", "final_val_accuracy", "actual_cost"]
    ].rename(
        columns={
            "condition_id": "matched_baseline_condition_id",
            "final_val_loss": "exact_uniform_loss",
            "final_val_accuracy": "exact_uniform_accuracy",
            "actual_cost": "exact_uniform_cost",
        }
    )
    exact_candidates = frame[frame["comparison_role"] == "candidate"].copy()
    exact_deltas = exact_candidates.merge(
        exact_reference,
        on=pair_keys + ["matched_baseline_condition_id"],
        how="left",
        validate="many_to_one",
    )
    required = exact_deltas[["exact_uniform_loss", "exact_uniform_cost"]]
    if required.isna().any().any():
        raise ValueError("one or more whitening candidates lacks an exact-cost baseline")
    if not (exact_deltas["actual_cost"] == exact_deltas["exact_uniform_cost"]).all():
        raise ValueError("whitening exact-cost comparison contains unequal realized costs")
    exact_deltas["loss_delta_vs_exact_uniform"] = (
        exact_deltas["final_val_loss"] - exact_deltas["exact_uniform_loss"]
    )
    exact_deltas["acc_delta_vs_exact_uniform"] = (
        exact_deltas["final_val_accuracy"] - exact_deltas["exact_uniform_accuracy"]
    )
    exact_deltas.to_csv(
        output_dir / "whitening_ablation_deltas_exact_cost.csv", index=False
    )

    cap_summary = cap_deltas[cap_deltas["condition_id"] != "uniform_fill"].groupby(
        ["scaling_mode", "condition_id", "label", "whitening", "rule"],
        dropna=False,
    ).agg(
        mean_loss_delta=("loss_delta_vs_uniform_fill", "mean"),
        median_loss_delta=("loss_delta_vs_uniform_fill", "median"),
        mean_acc_delta=("acc_delta_vs_uniform_fill", "mean"),
        mean_cost_gap=("cost_gap_vs_uniform_fill", "mean"),
        wins=("loss_delta_vs_uniform_fill", lambda values: int((values < 0).sum())),
        n_pairs=("loss_delta_vs_uniform_fill", "size"),
    ).reset_index()
    cap_summary.to_csv(
        output_dir / "whitening_ablation_summary_vs_cap_uniform.csv", index=False
    )

    exact_summary = exact_deltas.groupby(
        ["scaling_mode", "condition_id", "label", "whitening", "rule"],
        dropna=False,
    ).agg(
        mean_loss_delta=("loss_delta_vs_exact_uniform", "mean"),
        median_loss_delta=("loss_delta_vs_exact_uniform", "median"),
        mean_acc_delta=("acc_delta_vs_exact_uniform", "mean"),
        mean_actual_cost=("actual_cost", "mean"),
        wins=("loss_delta_vs_exact_uniform", lambda values: int((values < 0).sum())),
        n_pairs=("loss_delta_vs_exact_uniform", "size"),
    ).reset_index().sort_values(["scaling_mode", "mean_loss_delta", "label"])
    exact_summary.to_csv(
        output_dir / "whitening_ablation_summary_exact_cost.csv", index=False
    )
    # Keep the historical name, but make it point to the publication-valid result.
    exact_summary.to_csv(output_dir / "whitening_ablation_summary.csv", index=False)

    plot_frame = frame[frame["comparison_role"] != "exact_cost_baseline"]
    for scaling_mode, mode_frame in plot_frame.groupby("scaling_mode"):
        curve = mode_frame.groupby(["label", "budget"], as_index=False).agg(
            final_val_loss=("final_val_loss", "mean")
        )
        plt.figure(figsize=(8, 5))
        for label, group in curve.groupby("label"):
            plt.plot(group["budget"], group["final_val_loss"], marker="o", label=label)
        plt.xlabel("parameter budget cap")
        plt.ylabel("mean validation loss")
        plt.title(f"Whitening ablation: {scaling_mode}")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(
            figure_dir / f"whitening_ablation_budget_loss_{scaling_mode}.png",
            dpi=160,
        )
        plt.close()

        mode_summary = exact_summary[exact_summary["scaling_mode"] == scaling_mode]
        plt.figure(figsize=(8, 5))
        plt.barh(mode_summary["label"], mode_summary["mean_loss_delta"])
        plt.axvline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("mean paired loss delta vs exact-cost uniform")
        plt.title(f"Whitening ablation exact-cost summary: {scaling_mode}")
        plt.tight_layout()
        plt.savefig(
            figure_dir / f"whitening_ablation_mean_delta_{scaling_mode}.png",
            dpi=160,
        )
        plt.close()

    primary_curve = plot_frame[
        plot_frame["scaling_mode"] == primary_scaling_mode
    ].groupby(["label", "budget"], as_index=False).agg(
        final_val_loss=("final_val_loss", "mean")
    )
    plt.figure(figsize=(8, 5))
    for label, group in primary_curve.groupby("label"):
        plt.plot(group["budget"], group["final_val_loss"], marker="o", label=label)
    plt.xlabel("parameter budget cap")
    plt.ylabel("mean validation loss")
    plt.title(f"Whitening ablation: {primary_scaling_mode}")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(figure_dir / "whitening_ablation_budget_loss.png", dpi=160)
    plt.close()

    primary_summary = exact_summary[
        exact_summary["scaling_mode"] == primary_scaling_mode
    ].sort_values("mean_loss_delta")
    plt.figure(figsize=(8, 5))
    plt.barh(primary_summary["label"], primary_summary["mean_loss_delta"])
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("mean paired loss delta vs exact-cost uniform")
    plt.title(f"Whitening ablation exact-cost summary: {primary_scaling_mode}")
    plt.tight_layout()
    plt.savefig(figure_dir / "whitening_ablation_mean_delta.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a scaling- and cost-controlled whitening ablation from a base run."
    )
    parser.add_argument(
        "--base-run",
        required=True,
        help="Existing run directory with config.yaml and base/base_model.pt",
    )
    parser.add_argument("--name", default=None, help="Name suffix for the ablation run")
    parser.add_argument(
        "--whitenings",
        default="none,diag,full",
        help="Comma-separated whitening modes: none,diag,full",
    )
    parser.add_argument(
        "--spectral-rules",
        default="effective_rank,soft_dimension,marginal_gain_soft",
        help="Comma-separated spectral allocation rules",
    )
    parser.add_argument("--device", default=None, help="Override device")
    parser.add_argument("--threads", type=int, default=None, help="Override thread count")
    parser.add_argument(
        "--scaling-modes",
        default=None,
        help="Override comma-separated LoRA scaling modes",
    )
    parser.add_argument(
        "--primary-scaling-mode",
        default=None,
        help="Override the pre-specified primary scaling mode",
    )
    args = parser.parse_args()

    base_run = Path(args.base_run)
    if not base_run.exists():
        raise FileNotFoundError(base_run)
    cfg0 = load_yaml(base_run / "config.yaml")
    threads = int(
        args.threads
        if args.threads is not None
        else cfg0.get("run", {}).get("torch_threads", 1)
    )
    torch.set_num_threads(threads)
    seed = int(cfg0.get("run", {}).get("seed", 0))
    device = get_device(args.device or str(cfg0.get("run", {}).get("device", "cpu")))
    cfg, adapt_task, vocab_size, seq_len, base_state = _load_base_run(base_run, device)
    if args.scaling_modes is not None:
        cfg.setdefault("lora", {})["scaling_modes"] = args.scaling_modes
    if args.primary_scaling_mode is not None:
        cfg.setdefault("protocol", {})["primary_scaling_mode"] = args.primary_scaling_mode
    scaling = scaling_protocol(cfg["lora"], cfg.get("protocol", {}))
    sites = list(cfg.get("sites", {}).get("include", []))
    if not sites:
        raise ValueError("config has no sites.include")

    whitenings = [value.strip() for value in args.whitenings.split(",") if value.strip()]
    spectral_rules = [
        value.strip() for value in args.spectral_rules.split(",") if value.strip()
    ]
    name = args.name or f"whitening_ablation_from_{base_run.name}"
    run_dir = make_run_dir(cfg.get("run", {}).get("out_dir", "runs"), name)
    metadata = copy.deepcopy(cfg)
    metadata["ablation"] = {
        "base_run": str(base_run),
        "whitenings": whitenings,
        "spectral_rules": spectral_rules,
        "threads": threads,
        "device": str(device),
        "scaling_modes": scaling["scaling_modes"],
        "primary_scaling_mode": scaling["primary_scaling_mode"],
        "reference_alpha": scaling["reference_alpha"],
        "scale_reference_rank": scaling["reference_rank"],
        "exact_cost_baselines": True,
    }
    save_yaml(metadata, run_dir / "config.yaml")
    print(f"run_dir={run_dir}")
    print(f"base_run={base_run}")
    print(f"whitenings={whitenings}")
    print(f"spectral_rules={spectral_rules}")
    print(f"scaling_modes={scaling['scaling_modes']}")
    print(f"primary_scaling_mode={scaling['primary_scaling_mode']}")

    stats_by_whitening, singular_values_by_whitening = _calibrate_all(
        cfg,
        adapt_task,
        vocab_size,
        seq_len,
        base_state,
        device,
        sites,
        whitenings,
        run_dir,
        seed=seed,
    )
    frame, _ = _budget_ablation(
        cfg,
        adapt_task,
        vocab_size,
        seq_len,
        base_state,
        device,
        run_dir,
        stats_by_whitening,
        singular_values_by_whitening,
        whitenings,
        spectral_rules,
        seed=seed,
    )
    _summarize_and_plot(frame, run_dir, scaling["primary_scaling_mode"])
    write_json(
        {
            "run_id": run_dir.name,
            "run_dir": ".",
            "protocol_version": PROTOCOL_VERSION,
            "base_seed": seed,
            "task_name": adapt_task.name,
            "source_base_run_id": base_run.name,
            "common_random_numbers": True,
            "rule_in_comparison_seed": False,
            "scaling_mode_in_comparison_seed": False,
            "scaling_modes": scaling["scaling_modes"],
            "primary_scaling_mode": scaling["primary_scaling_mode"],
            "reference_alpha": scaling["reference_alpha"],
            "scale_reference_rank": scaling["reference_rank"],
            "fixed_training_batches": True,
            "fixed_validation_batches": True,
            "exact_modular_evaluation": _exact_modular_evaluation(cfg, adapt_task),
            "calibration_dropout_disabled": True,
            "exact_cost_baselines": True,
            "whitenings": whitenings,
            "spectral_rules": spectral_rules,
        },
        run_dir / "run_info.json",
    )
    write_json(
        {
            "status": "PASS",
            "protocol_version": PROTOCOL_VERSION,
            "base_seed": seed,
            "task_name": adapt_task.name,
            "run_type": "whitening_ablation",
        },
        run_dir / "_SUCCESS.json",
    )
    print("summary:")
    print(
        pd.read_csv(
            run_dir / "budget" / "whitening_ablation_summary_exact_cost.csv"
        ).to_string(index=False)
    )
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
