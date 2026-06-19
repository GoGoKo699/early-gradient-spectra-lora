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
from rmt_lora.provenance import (
    git_state,
    runtime_versions,
    utc_now_iso,
    write_json,
    write_sha256s,
)
from rmt_lora.spectra import summarize_spectrum
from rmt_lora.stage4_analysis import (
    ANALYSIS_PLAN_VERSION,
    PREDICTOR_TRANSFORM,
    PRIMARY_PREDICTOR,
    PRIMARY_TARGET,
    TARGET_TRANSFORM,
    analysis_plan_document,
    spearman_or_nan,
    transform_spectral_predictor,
)
from rmt_lora.targets import (
    TARGET_DENOMINATOR,
    TARGET_ESTIMAND,
    TARGET_ESTIMAND_VERSION,
    TARGET_REFERENCE,
    annotate_observed_best_curves,
    summarize_observed_best_curve,
    target_definition_document,
    useful_rank_target_columns,
    validate_target_estimand_frame,
)


def _as_pair(x: Any, default: tuple[float, float]) -> tuple[float, float]:
    if x is None:
        return default
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return float(x[0]), float(x[1])
    raise ValueError(f"expected pair, got {x!r}")


def _as_int_pair(x: Any, default: tuple[int, int]) -> tuple[int, int]:
    if x is None:
        return default
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return int(x[0]), int(x[1])
    raise ValueError(f"expected int pair, got {x!r}")


def _power2_ceiling(x: float, ranks: list[int]) -> int:
    x = max(1, int(np.ceil(float(x))))
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


def _random_spikes(rng: np.random.Generator, cfg: dict[str, Any], max_rank: int) -> tuple[np.ndarray, int, float, float, float]:
    """Generate a layer-dependent task spectrum.

    v2 exposes the spike generator to config.  This matters because stage2 showed
    that a soft tail of weak directions makes the unconstrained best rank equal
    to the largest allowed rank.  The hard-knee config tests the BBP-style case:
    a small number of detectable spikes plus genuinely subthreshold weak modes.
    """
    k_min, k_max = _as_int_pair(cfg.get("k_strong_range"), (1, max_rank))
    k_min = max(1, min(max_rank, k_min))
    k_max = max(k_min, min(max_rank, k_max))
    k_strong = int(rng.integers(k_min, k_max + 1))

    peak_min, peak_max = _as_pair(cfg.get("spike_peak_range"), (1.2, 3.4))
    decay_min, decay_max = _as_pair(cfg.get("spike_decay_range"), (0.08, 0.42))
    weak_min, weak_max = _as_pair(cfg.get("weak_factor_range"), (0.12, 0.60))
    jitter = float(cfg.get("spike_jitter", 0.05))
    floor = float(cfg.get("spike_floor", 0.02))

    peak = float(rng.uniform(peak_min, peak_max))
    decay = float(rng.uniform(decay_min, decay_max))
    weak_factor = float(rng.uniform(weak_min, weak_max))

    i = np.arange(max_rank, dtype=float)
    spikes = peak * np.exp(-decay * i)
    if k_strong < max_rank:
        spikes[k_strong:] *= weak_factor
    if jitter > 0:
        spikes *= rng.lognormal(mean=0.0, sigma=jitter, size=max_rank)
    spikes = np.sort(np.clip(spikes, floor, None))[::-1]
    return spikes, k_strong, peak, decay, weak_factor


def _summarize_layers(df: pd.DataFrame, ranks: list[int], cfg: dict[str, Any]) -> pd.DataFrame:
    recovery_targets = [float(v) for v in cfg.get("recovery_targets", [0.50, 0.70, 0.80, 0.90, 0.95])]
    gap_tolerances = [float(v) for v in cfg.get("gap_tolerances", [0.02, 0.05, 0.10, 0.20])]
    penalty_lambdas = [float(v) for v in cfg.get("penalty_lambdas", [0.05, 0.10, 0.15, 0.20, 0.30])]
    gap_epsilon = float(cfg.get("gap_epsilon", 1e-12))
    rows: list[dict[str, Any]] = []

    for layer, g0 in df.groupby("layer"):
        g = g0.sort_values("rank").copy()
        target_row = summarize_observed_best_curve(
            g,
            recovery_targets=recovery_targets,
            gap_tolerances=gap_tolerances,
            penalty_lambdas=penalty_lambdas,
            gap_epsilon=gap_epsilon,
        )

        det = int(g["gradient_detectable_rank"].iloc[0])
        geff = float(g["gradient_effective_rank"].iloc[0])
        gstb = float(g["gradient_stable_rank"].iloc[0])

        row: dict[str, Any] = {
            "layer": int(layer),
            **target_row,
            "gradient_matrix_mode": str(g["gradient_matrix_mode"].iloc[0]) if "gradient_matrix_mode" in g.columns else "legacy_raw",
            "gradient_detectable_rank": det,
            "gradient_stable_rank": gstb,
            "gradient_effective_rank": geff,
            "predicted_rank_detectable": _power2_ceiling(det, ranks),
            "predicted_rank_effective": _power2_ceiling(geff, ranks),
            "predicted_rank_stable": _power2_ceiling(gstb, ranks),
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
    predictors = [
        ("gradient_detectable_rank", "gradient_detectable_rank", "log"),
        ("gradient_effective_rank", "gradient_effective_rank", "log"),
        ("gradient_stable_rank", "gradient_stable_rank", "log"),
        ("predicted_rank_detectable", "predicted_rank_detectable", "log"),
        ("predicted_rank_effective", "predicted_rank_effective", "log"),
        ("predicted_rank_stable", "predicted_rank_stable", "log"),
        ("raw_gradient_detectable_rank", "raw_gradient_detectable_rank", "log"),
        ("raw_gradient_effective_rank", "raw_gradient_effective_rank", "log"),
        ("raw_gradient_stable_rank", "raw_gradient_stable_rank", "log"),
        ("predicted_rank_raw_detectable", "predicted_rank_raw_detectable", "log"),
        ("predicted_rank_raw_effective", "predicted_rank_raw_effective", "log"),
        ("predicted_rank_raw_stable", "predicted_rank_raw_stable", "log"),
        ("whitened_gradient_detectable_rank", "whitened_gradient_detectable_rank", "log"),
        ("whitened_gradient_effective_rank", "whitened_gradient_effective_rank", "log"),
        ("whitened_gradient_stable_rank", "whitened_gradient_stable_rank", "log"),
        ("predicted_rank_whitened_detectable", "predicted_rank_whitened_detectable", "log"),
        ("predicted_rank_whitened_effective", "predicted_rank_whitened_effective", "log"),
        ("predicted_rank_whitened_stable", "predicted_rank_whitened_stable", "log"),
        ("true_k_strong", "true_k_strong", "log"),
    ]
    validate_target_estimand_frame(summary, context="layer summary")
    target_cols = ["best_rank"] + useful_rank_target_columns(summary.columns)
    rows: list[dict[str, Any]] = []
    for target in target_cols:
        for name, pred_col, transform in predictors:
            if pred_col not in summary.columns:
                continue
            sub = summary[[target, pred_col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) < 4:
                continue
            y = np.log2(np.asarray(sub[target], dtype=float))
            x = np.asarray(sub[pred_col], dtype=float)
            if transform == "log":
                x = transform_spectral_predictor(x)
            sst = float(np.sum((y - y.mean()) ** 2))
            if sst <= 1e-12:
                r2 = np.nan
                rmse = 0.0
            else:
                xmat = np.column_stack([np.ones(len(sub)), x])
                beta, *_ = np.linalg.lstsq(xmat, y, rcond=None)
                resid = y - xmat @ beta
                sse = float(np.sum(resid ** 2))
                r2 = 1.0 - sse / sst
                rmse = float(np.sqrt(sse / len(sub)))
            spearman = spearman_or_nan(sub[pred_col], sub[target])
            rows.append({
                "target_estimand": TARGET_ESTIMAND,
                "target_estimand_version": TARGET_ESTIMAND_VERSION,
                "target_reference": TARGET_REFERENCE,
                "target_denominator": TARGET_DENOMINATOR,
                "analysis_plan_version": ANALYSIS_PLAN_VERSION,
                "analysis_pair_role": (
                    "primary_pair"
                    if target == PRIMARY_TARGET and name == PRIMARY_PREDICTOR
                    else "secondary_or_exploratory"
                ),
                "target_transform": TARGET_TRANSFORM,
                "predictor_transform": PREDICTOR_TRANSFORM,
                "target": target,
                "target_role": "diagnostic" if target == "best_rank" else "useful_rank",
                "predictor": name,
                "n": int(len(sub)),
                "r2_log2_target": r2,
                "spearman": spearman,
                "rmse_log2": rmse,
            })
    out = pd.DataFrame(
        rows,
        columns=[
            "target_estimand",
            "target_estimand_version",
            "target_reference",
            "target_denominator",
            "analysis_plan_version",
            "analysis_pair_role",
            "target_transform",
            "predictor_transform",
            "target",
            "target_role",
            "predictor",
            "n",
            "r2_log2_target",
            "spearman",
            "rmse_log2",
        ],
    )
    if len(out):
        out = out.sort_values(["target", "r2_log2_target"], ascending=[True, False], na_position="last")
    return out


def _plot(df: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(fig, name: str) -> None:
        p = out_dir / name
        fig.tight_layout(); fig.savefig(p, dpi=180); plt.close(fig); paths.append(p)

    # Main predictive target: minimum rank within 10% of the best gap.
    target = "near_best_rank_gap_0.1" if "near_best_rank_gap_0.1" in summary.columns else "best_rank"
    fig, ax = plt.subplots()
    ax.scatter(summary["predicted_rank_detectable"], summary[target])
    lo = min(summary["predicted_rank_detectable"].min(), summary[target].min())
    hi = max(summary["predicted_rank_detectable"].max(), summary[target].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("predicted rank from early-gradient detectable count")
    ax.set_ylabel(target)
    ax.set_title("Predictive-rank test")
    save(fig, "predicted_detectable_vs_target_rank.png")

    fig, ax = plt.subplots()
    ax.scatter(summary["gradient_effective_rank"], summary[target])
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("early-gradient effective rank")
    ax.set_ylabel(target)
    ax.set_title("Effective-rank predictor")
    save(fig, "gradient_effective_vs_target_rank.png")

    g = df.copy()
    g["excess_val_loss"] = np.maximum(g["final_val_loss"] - g["oracle_val_loss"], 1e-12)
    h = g.groupby("rank", as_index=False).agg(mean_excess=("excess_val_loss", "mean"), sem=("excess_val_loss", lambda x: float(pd.Series(x).sem())))
    fig, ax = plt.subplots()
    ax.errorbar(h["rank"], h["mean_excess"], yerr=h["sem"], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("nominal rank")
    ax.set_ylabel("mean excess validation loss")
    ax.set_title("Rank sweep over synthetic layers")
    save(fig, "mean_excess_loss_vs_rank.png")

    if "penalized_rank_lambda_0.15" in summary.columns:
        fig, ax = plt.subplots()
        ax.scatter(summary["gradient_effective_rank"], summary["penalized_rank_lambda_0.15"])
        ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
        ax.set_xlabel("early-gradient effective rank")
        ax.set_ylabel("penalized optimal rank, lambda=0.15")
        ax.set_title("Rank/quality tradeoff prediction")
        save(fig, "gradient_effective_vs_penalized_rank.png")

    return paths


def run(config: dict[str, Any], do_plot: bool = False) -> Path:
    run_dir = make_run_dir(deep_get(config, "output.base_dir", "runs"), deep_get(config, "output.name", "layerwise_rank_prediction_v2"))
    materialize_run(run_dir, config)

    seed = int(deep_get(config, "seed", 0))
    rng = np.random.default_rng(seed)
    n_layers = int(deep_get(config, "layer_sweep.n_layers", 24))
    max_task_rank = int(deep_get(config, "layer_sweep.max_task_rank", 8))
    ranks = [int(v) for v in deep_get(config, "layer_sweep.ranks", [1, 2, 4, 8, 16, 32])]
    alpha_rule = str(deep_get(config, "layer_sweep.alpha_rule", "proportional"))
    alpha_fixed = float(deep_get(config, "layer_sweep.alpha_fixed", 16.0))
    alpha_multiplier = float(deep_get(config, "layer_sweep.alpha_multiplier", 1.0))

    task_cfg = config.get("task", {})
    train_cfg = config.get("train", {})
    detect_cfg = config.get("detect", {})
    spike_cfg = config.get("spikes", {})
    summary_cfg = config.get("summary", {})

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
        spikes, k_strong, peak, decay, weak_factor = _random_spikes(rng, spike_cfg, max_task_rank)
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
            row = {
                "experiment": deep_get(config, "output.name", "layerwise_rank_prediction_v2"),
                "seed": seed,
                "layer": layer,
                "rank": int(rank),
                "alpha": float(alpha),
                "alpha_rule": alpha_rule,
                "true_k_strong": int(k_strong),
                "spike_peak": peak,
                "spike_decay": decay,
                "weak_factor": weak_factor,
                "task_spikes": ";".join(f"{x:.6g}" for x in spikes),
                "lr": lr,
                "steps": steps,
                "steps_completed": result.steps_completed,
                "diverged": bool(result.diverged),
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
            }
            rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    annotated_metrics = annotate_observed_best_curves(
        metrics, gap_epsilon=float(summary_cfg.get("gap_epsilon", 1e-12))
    )
    annotated_metrics.to_csv(run_dir / "target_curve_metrics.csv", index=False)
    summary = _summarize_layers(metrics, ranks, summary_cfg)
    summary.to_csv(run_dir / "layer_summary.csv", index=False)
    fits = _ols_table(summary)
    fits.to_csv(run_dir / "layer_rank_fit.csv", index=False)

    paths: list[Path] = []
    if do_plot:
        paths = _plot(metrics, summary, run_dir / "figures")

    write_json(run_dir / "target_definition.json", target_definition_document())
    write_json(run_dir / "analysis_plan.json", analysis_plan_document())
    write_json(
        run_dir / "run_metadata.json",
        {
            "schema_version": 1,
            "created_utc": utc_now_iso(),
            "experiment": deep_get(config, "output.name", "layerwise_rank_prediction_v2"),
            "seed": seed,
            "n_layers": n_layers,
            "ranks": ranks,
            "target_estimand": TARGET_ESTIMAND,
            "target_estimand_version": TARGET_ESTIMAND_VERSION,
            "target_reference": TARGET_REFERENCE,
            "target_denominator": TARGET_DENOMINATOR,
            "analysis_plan_version": ANALYSIS_PLAN_VERSION,
            "target_transform": TARGET_TRANSFORM,
            "predictor_transform": PREDICTOR_TRANSFORM,
            "git": git_state(Path(__file__)),
            "runtime": runtime_versions(),
        },
    )
    write_sha256s(run_dir)

    print(f"wrote {run_dir}")
    if paths:
        print("figures:")
        for path in paths:
            print(f"  {path}")
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument(
        "--write-run-dir",
        type=Path,
        help="Write the exact created run directory to this file.",
    )
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_dir = run(cfg, do_plot=args.plot)
    if args.write_run_dir is not None:
        args.write_run_dir.parent.mkdir(parents=True, exist_ok=True)
        args.write_run_dir.write_text(str(run_dir.resolve()) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
