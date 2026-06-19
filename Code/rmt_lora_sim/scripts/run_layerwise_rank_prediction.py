#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from rmt_lora.config import deep_get
from rmt_lora.io import make_run_dir, materialize_run, write_metrics
from rmt_lora.lora_sim import (
    make_linear_task,
    train_lora_fullbatch,
    estimate_gradient_detectable_rank,
    intruder_diagnostics,
    mse_loss,
)
from rmt_lora.nulls import bootstrap_edge
from rmt_lora.spectra import summarize_spectrum


def _power2_ceiling(x: int, ranks: list[int]) -> int:
    x = max(1, int(x))
    for r in sorted(ranks):
        if r >= x:
            return int(r)
    return int(max(ranks))


def _alpha_for_rank(rule: str, rank: int, fixed: float, multiplier: float) -> float:
    if rule == "fixed":
        return float(fixed)
    if rule == "proportional":
        return float(multiplier * rank)
    if rule == "sqrt":
        return float(multiplier * np.sqrt(rank))
    raise ValueError(f"unknown alpha_rule={rule!r}; use fixed, proportional, or sqrt")


def _random_spikes(rng: np.random.Generator, max_rank: int) -> tuple[np.ndarray, int, float, float, float]:
    """Generate a layer-dependent decaying spike spectrum with a soft knee."""
    k_strong = int(rng.integers(1, max_rank + 1))
    peak = float(rng.uniform(1.2, 3.4))
    decay = float(rng.uniform(0.08, 0.42))
    weak_factor = float(rng.uniform(0.12, 0.60))
    i = np.arange(max_rank, dtype=float)
    spikes = peak * np.exp(-decay * i)
    if k_strong < max_rank:
        spikes[k_strong:] *= weak_factor
    spikes *= rng.lognormal(mean=0.0, sigma=0.05, size=max_rank)
    spikes = np.sort(np.clip(spikes, 0.05, None))[::-1]
    return spikes, k_strong, peak, decay, weak_factor


def _summarize_layers(df: pd.DataFrame, ranks: list[int], recovery_target: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for layer, g in df.groupby("layer"):
        g = g.sort_values("rank")
        base = float(g["base_val_loss"].iloc[0])
        oracle = float(g["oracle_val_loss"].iloc[0])
        gap = max(base - oracle, 1e-12)
        min_idx = g["final_val_loss"].idxmin()
        best = g.loc[min_idx]
        min_loss = float(best["final_val_loss"])
        near_threshold = min_loss + 0.02 * gap
        near = g[g["final_val_loss"] <= near_threshold].sort_values("rank")
        near_best_rank = int(near["rank"].iloc[0]) if len(near) else int(best["rank"])
        recovery_threshold = oracle + (1.0 - recovery_target) * gap
        recovered = g[g["final_val_loss"] <= recovery_threshold].sort_values("rank")
        recovery_rank = float(recovered["rank"].iloc[0]) if len(recovered) else np.nan
        det = int(g["gradient_detectable_rank"].iloc[0])
        pred = _power2_ceiling(det, ranks)
        row: dict[str, Any] = {
            "layer": int(layer),
            "base_val_loss": base,
            "oracle_val_loss": oracle,
            "improvement_gap": gap,
            "gradient_matrix_mode": str(g["gradient_matrix_mode"].iloc[0]) if "gradient_matrix_mode" in g.columns else "legacy_raw",
            "gradient_detectable_rank": det,
            "gradient_stable_rank": float(g["gradient_stable_rank"].iloc[0]),
            "gradient_effective_rank": float(g["gradient_effective_rank"].iloc[0]),
            "predicted_rank_from_gradient": pred,
            "best_rank": int(best["rank"]),
            "best_val_loss": min_loss,
            "near_best_rank_2pct_gap": near_best_rank,
            "recovery_target": recovery_target,
            "recovery_rank": recovery_rank,
            "pred_matches_near_best": bool(pred == near_best_rank),
            "pred_within_factor2_near_best": bool((pred <= 2 * near_best_rank) and (near_best_rank <= 2 * pred)),
            "true_k_strong": int(g["true_k_strong"].iloc[0]),
            "spike_peak": float(g["spike_peak"].iloc[0]),
            "spike_decay": float(g["spike_decay"].iloc[0]),
            "weak_factor": float(g["weak_factor"].iloc[0]),
        }
        if "raw_gradient_effective_rank" in g.columns:
            raw_det = int(g["raw_gradient_detectable_rank"].iloc[0])
            raw_eff = float(g["raw_gradient_effective_rank"].iloc[0])
            raw_stb = float(g["raw_gradient_stable_rank"].iloc[0])
            row.update({
                "raw_gradient_detectable_rank": raw_det,
                "raw_gradient_stable_rank": raw_stb,
                "raw_gradient_effective_rank": raw_eff,
                "predicted_rank_raw_detectable": _power2_ceiling(raw_det, ranks),
                "predicted_rank_raw_effective": _power2_ceiling(raw_eff, ranks),
                "predicted_rank_raw_stable": _power2_ceiling(raw_stb, ranks),
            })
        if "whitened_gradient_effective_rank" in g.columns:
            wh_det = int(g["whitened_gradient_detectable_rank"].iloc[0])
            wh_eff = float(g["whitened_gradient_effective_rank"].iloc[0])
            wh_stb = float(g["whitened_gradient_stable_rank"].iloc[0])
            row.update({
                "whitened_gradient_detectable_rank": wh_det,
                "whitened_gradient_stable_rank": wh_stb,
                "whitened_gradient_effective_rank": wh_eff,
                "predicted_rank_whitened_detectable": _power2_ceiling(wh_det, ranks),
                "predicted_rank_whitened_effective": _power2_ceiling(wh_eff, ranks),
                "predicted_rank_whitened_stable": _power2_ceiling(wh_stb, ranks),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def _ols_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    targets = ["best_rank", "near_best_rank_2pct_gap", "recovery_rank"]
    predictors = [
        ("gradient_detectable_rank", "gradient_detectable_rank"),
        ("predicted_rank_from_gradient", "predicted_rank_from_gradient"),
        ("gradient_stable_rank", "gradient_stable_rank"),
        ("gradient_effective_rank", "gradient_effective_rank"),
        ("raw_gradient_detectable_rank", "raw_gradient_detectable_rank"),
        ("raw_gradient_stable_rank", "raw_gradient_stable_rank"),
        ("raw_gradient_effective_rank", "raw_gradient_effective_rank"),
        ("predicted_rank_raw_detectable", "predicted_rank_raw_detectable"),
        ("predicted_rank_raw_stable", "predicted_rank_raw_stable"),
        ("predicted_rank_raw_effective", "predicted_rank_raw_effective"),
        ("whitened_gradient_detectable_rank", "whitened_gradient_detectable_rank"),
        ("whitened_gradient_stable_rank", "whitened_gradient_stable_rank"),
        ("whitened_gradient_effective_rank", "whitened_gradient_effective_rank"),
        ("predicted_rank_whitened_detectable", "predicted_rank_whitened_detectable"),
        ("predicted_rank_whitened_stable", "predicted_rank_whitened_stable"),
        ("predicted_rank_whitened_effective", "predicted_rank_whitened_effective"),
        ("true_k_strong", "true_k_strong"),
    ]
    for target in targets:
        for name, pred_col in predictors:
            if pred_col not in summary.columns:
                continue
            sub = summary[[target, pred_col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) < 4:
                continue
            y = np.log2(np.asarray(sub[target], dtype=float))
            x = np.asarray(sub[pred_col], dtype=float)
            if pred_col.startswith("predicted_rank_") or pred_col in {"predicted_rank_from_gradient", "true_k_strong"}:
                x = np.log2(np.maximum(x, 1.0))
            xmat = np.column_stack([np.ones(len(sub)), x])
            beta, *_ = np.linalg.lstsq(xmat, y, rcond=None)
            resid = y - xmat @ beta
            sse = float(np.sum(resid ** 2))
            sst = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - sse / (sst + 1e-12)
            spearman = float(pd.Series(sub[pred_col]).corr(pd.Series(sub[target]), method="spearman"))
            rows.append({
                "target": target,
                "predictor": name,
                "n": int(len(sub)),
                "r2_log2_target": r2,
                "spearman": spearman,
                "rmse_log2": float(np.sqrt(sse / len(sub))),
            })
    out = pd.DataFrame(rows, columns=["target", "predictor", "n", "r2_log2_target", "spearman", "rmse_log2"])
    if len(out):
        out = out.sort_values(["target", "r2_log2_target"], ascending=[True, False])
    return out


def _plot(df: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots()
    ax.scatter(summary["gradient_detectable_rank"], summary["near_best_rank_2pct_gap"])
    ax.set_xlabel("early-gradient detectable rank")
    ax.set_ylabel("empirical near-best LoRA rank")
    ax.set_title("Layerwise rank prediction")
    p = out_dir / "layer_gradient_detectable_vs_near_best_rank.png"
    fig.tight_layout(); fig.savefig(p, dpi=180); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots()
    ax.scatter(summary["predicted_rank_from_gradient"], summary["near_best_rank_2pct_gap"])
    lim = [min(summary["predicted_rank_from_gradient"].min(), summary["near_best_rank_2pct_gap"].min()),
           max(summary["predicted_rank_from_gradient"].max(), summary["near_best_rank_2pct_gap"].max())]
    ax.plot(lim, lim, linestyle="--")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("predicted rank: next power of two above detectable rank")
    ax.set_ylabel("empirical near-best rank")
    ax.set_title("Predicted vs empirical rank")
    p = out_dir / "layer_predicted_vs_empirical_rank.png"
    fig.tight_layout(); fig.savefig(p, dpi=180); plt.close(fig); paths.append(p)

    g = df.copy()
    g["excess_val_loss"] = np.maximum(g["final_val_loss"] - g["oracle_val_loss"], 1e-12)
    h = g.groupby("rank", as_index=False).agg(mean_excess=("excess_val_loss", "mean"), sem=("excess_val_loss", lambda x: float(pd.Series(x).sem())))
    fig, ax = plt.subplots()
    ax.errorbar(h["rank"], h["mean_excess"], yerr=h["sem"], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("nominal rank")
    ax.set_ylabel("mean excess validation loss")
    ax.set_title("Layerwise rank sweep")
    p = out_dir / "layerwise_mean_excess_loss_vs_rank.png"
    fig.tight_layout(); fig.savefig(p, dpi=180); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots()
    ax.scatter(df["adapter_detectable_rank"], df["final_val_loss"])
    ax.set_xlabel("final adapter detectable rank")
    ax.set_ylabel("validation loss")
    ax.set_title("Final detectable rank remains the mechanistic diagnostic")
    p = out_dir / "layerwise_val_loss_vs_adapter_detectable_rank.png"
    fig.tight_layout(); fig.savefig(p, dpi=180); plt.close(fig); paths.append(p)
    return paths


def run(config: dict[str, Any], do_plot: bool = False) -> Path:
    run_dir = make_run_dir(deep_get(config, "output.base_dir", "runs"), deep_get(config, "output.name", "layerwise_rank_prediction"))
    materialize_run(run_dir, config)

    seed = int(deep_get(config, "seed", 0))
    rng = np.random.default_rng(seed)
    n_layers = int(deep_get(config, "layer_sweep.n_layers", 24))
    max_task_rank = int(deep_get(config, "layer_sweep.max_task_rank", 8))
    ranks = [int(v) for v in deep_get(config, "layer_sweep.ranks", [1, 2, 4, 8, 16, 32])]
    alpha_rule = str(deep_get(config, "layer_sweep.alpha_rule", "proportional"))
    alpha_fixed = float(deep_get(config, "layer_sweep.alpha_fixed", 16.0))
    alpha_multiplier = float(deep_get(config, "layer_sweep.alpha_multiplier", 1.0))
    recovery_target = float(deep_get(config, "layer_sweep.recovery_target", 0.95))

    task_cfg = config.get("task", {})
    train_cfg = config.get("train", {})
    detect_cfg = config.get("detect", {})

    lr = float(train_cfg.get("lr", 0.2))
    steps = int(train_cfg.get("steps", 900))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    init_scale = float(train_cfg.get("init_scale", 0.05))
    eval_every = int(train_cfg.get("eval_every", 100))
    batch_size = train_cfg.get("batch_size", None)
    batch_size = None if batch_size is None else int(batch_size)
    grad_clip = train_cfg.get("grad_clip", None)
    grad_clip = None if grad_clip is None else float(grad_clip)

    null_method = str(detect_cfg.get("null_method", "permute"))
    n_boot = int(detect_cfg.get("n_boot", 16))
    quantile = float(detect_cfg.get("quantile", 0.995))
    gradient_matrix_mode = str(detect_cfg.get("gradient_matrix_mode", "activation_whitened"))
    gradient_ridge_scale = float(detect_cfg.get("gradient_ridge_scale", 1e-6))
    include_raw_gradient = bool(detect_cfg.get("include_raw_gradient", True))
    intruder_overlap_threshold = float(detect_cfg.get("intruder_overlap_threshold", 0.10))

    rows: list[dict[str, Any]] = []
    for layer in range(n_layers):
        spikes, k_strong, peak, decay, weak_factor = _random_spikes(rng, max_task_rank)
        task_rng = np.random.default_rng(seed + 100_003 * layer)
        task = make_linear_task(
            d_in=int(task_cfg.get("d_in", 128)),
            d_out=int(task_cfg.get("d_out", 128)),
            n_train=int(task_cfg.get("n_train", 1024)),
            n_val=int(task_cfg.get("n_val", 1024)),
            task_rank=max_task_rank,
            task_spikes=spikes.tolist(),
            rng=task_rng,
            label_noise_std=float(task_cfg.get("label_noise_std", 0.05)),
            base_scale=float(task_cfg.get("base_scale", 0.2)),
            activation_spectrum=str(task_cfg.get("activation_spectrum", "powerlaw")),
            activation_decay=float(task_cfg.get("activation_decay", 0.7)),
        )
        diag_rng = np.random.default_rng(seed + 100_003 * layer + 777)
        grad_diag = estimate_gradient_detectable_rank(
            task,
            diag_rng,
            null_method=null_method,
            n_boot=n_boot,
            quantile=quantile,
            matrix_mode=gradient_matrix_mode,
            ridge_scale=gradient_ridge_scale,
            include_raw=include_raw_gradient,
        )
        base_val_loss = mse_loss(task.x_val, task.y_val, task.w_base, np.zeros_like(task.w_base))
        oracle_val_loss = mse_loss(task.x_val, task.y_val, task.w_base, task.delta_true)
        for rank in ranks:
            alpha = _alpha_for_rank(alpha_rule, rank, alpha_fixed, alpha_multiplier)
            run_rng = np.random.default_rng(seed + 100_003 * layer + 1009 * rank + int(17 * alpha))
            result = train_lora_fullbatch(
                task=task,
                rank=rank,
                alpha=alpha,
                lr=lr,
                steps=steps,
                rng=run_rng,
                weight_decay=weight_decay,
                init_scale=init_scale,
                eval_every=eval_every,
                batch_size=batch_size,
                grad_clip=grad_clip,
            )
            edge_rng = np.random.default_rng(seed + 100_003 * layer + 8123 * rank + int(31 * alpha))
            edge_result = bootstrap_edge(result.delta, edge_rng, method=null_method, n_boot=max(5, n_boot // 2), quantile=quantile)
            adapter_summary = summarize_spectrum(result.delta, edge=edge_result.edge)
            intr = intruder_diagnostics(task, result.delta, edge=edge_result.edge, overlap_threshold=intruder_overlap_threshold)
            rows.append({
                "experiment": "layerwise_rank_prediction",
                "seed": seed,
                "layer": layer,
                "rank": rank,
                "alpha": alpha,
                "alpha_rule": alpha_rule,
                "true_k_strong": k_strong,
                "spike_peak": peak,
                "spike_decay": decay,
                "weak_factor": weak_factor,
                "task_spikes": ";".join(f"{x:.6g}" for x in spikes),
                "lr": lr,
                "steps": steps,
                "steps_completed": result.steps_completed,
                "diverged": result.diverged,
                "base_val_loss": base_val_loss,
                "oracle_val_loss": oracle_val_loss,
                "final_train_loss": result.final_train_loss,
                "final_val_loss": result.final_val_loss,
                "forgetting_loss": result.forgetting_loss,
                "adapter_edge": edge_result.edge,
                "adapter_detectable_rank": adapter_summary.detectable_rank,
                **grad_diag,
                **result.diagnostics,
                **intr,
            })
    write_metrics(rows, run_dir / "metrics.csv")
    df = pd.DataFrame(rows)
    summary = _summarize_layers(df, ranks, recovery_target)
    fit = _ols_table(summary)
    summary.to_csv(run_dir / "layer_summary.csv", index=False)
    fit.to_csv(run_dir / "layer_rank_fit.csv", index=False)
    print("layer summary:")
    print(summary[["layer", "gradient_detectable_rank", "predicted_rank_from_gradient", "near_best_rank_2pct_gap", "best_rank", "recovery_rank"]].to_string(index=False))
    print("\nrank-prediction fit:")
    print(fit.to_string(index=False))
    if do_plot:
        paths = _plot(df, summary, run_dir / "figures")
        print("figures:")
        for p in paths:
            print(f"  {p}")
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    run_dir = run(config, do_plot=args.plot)
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
