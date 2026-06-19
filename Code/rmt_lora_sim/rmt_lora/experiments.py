from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import deep_get
from .io import make_run_dir, materialize_run, write_metrics
from .spectra import mp_singular_edge, summarize_spectrum
from .spiked import (
    make_spiked_matrix,
    singular_vector_alignment,
    make_overlapping_spikes,
    matrix_from_svd,
    spectral_interference_score,
)
from .lora_sim import (
    make_linear_task,
    train_lora_fullbatch,
    estimate_gradient_detectable_rank,
    intruder_diagnostics,
    mse_loss,
    forgetting_loss,
)
from .nulls import bootstrap_edge


def _as_list(x: Any) -> list[Any]:
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def run_bbp(config: dict[str, Any]) -> Path:
    out_base = deep_get(config, "output.base_dir", "runs")
    run_dir = make_run_dir(out_base, deep_get(config, "output.name", "bbp"))
    materialize_run(run_dir, config)

    seed = int(deep_get(config, "seed", 0))
    rng = np.random.default_rng(seed)
    m = int(deep_get(config, "matrix.m", 256))
    n = int(deep_get(config, "matrix.n", 256))
    rank = int(deep_get(config, "matrix.rank", 1))
    noise_sigma = float(deep_get(config, "matrix.noise_sigma", 1.0))
    heavy_tail_df = deep_get(config, "matrix.heavy_tail_df", None)
    if heavy_tail_df is not None:
        heavy_tail_df = float(heavy_tail_df)
    spike_grid = [float(v) for v in deep_get(config, "sweep.spikes", [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0])]
    repetitions = int(deep_get(config, "sweep.repetitions", 20))
    margin = float(deep_get(config, "detect.margin", 1.0))

    edge = mp_singular_edge(m, n, noise_sigma=noise_sigma)
    rows: list[dict[str, Any]] = []
    for theta in spike_grid:
        for rep in range(repetitions):
            sm = make_spiked_matrix(
                m=m,
                n=n,
                spikes=[theta] * rank,
                noise_sigma=noise_sigma,
                rng=rng,
                heavy_tail_df=heavy_tail_df,
            )
            summary = summarize_spectrum(sm.matrix, edge=edge, margin=margin)
            align = singular_vector_alignment(sm.matrix, sm.u_true, sm.v_true, k=rank)
            rows.append(
                {
                    "experiment": "bbp",
                    "seed": seed,
                    "rep": rep,
                    "m": m,
                    "n": n,
                    "rank": rank,
                    "theta": theta,
                    "noise_sigma": noise_sigma,
                    "mp_edge": edge,
                    "top_sv": float(summary.singular_values[0]),
                    "second_sv": float(summary.singular_values[1]) if len(summary.singular_values) > 1 else np.nan,
                    "detectable_rank": summary.detectable_rank,
                    "stable_rank": summary.stable_rank,
                    "effective_rank": summary.effective_rank,
                    **align,
                }
            )
    write_metrics(rows, run_dir / "metrics.csv")
    return run_dir


def _make_task_from_config(config: dict[str, Any], rng: np.random.Generator):
    task_cfg = config.get("task", {})
    return make_linear_task(
        d_in=int(task_cfg.get("d_in", 128)),
        d_out=int(task_cfg.get("d_out", 128)),
        n_train=int(task_cfg.get("n_train", 2048)),
        n_val=int(task_cfg.get("n_val", 1024)),
        task_rank=int(task_cfg.get("task_rank", 8)),
        task_spikes=task_cfg.get("task_spikes", None),
        rng=rng,
        label_noise_std=float(task_cfg.get("label_noise_std", 0.05)),
        base_scale=float(task_cfg.get("base_scale", 0.2)),
        activation_spectrum=str(task_cfg.get("activation_spectrum", "isotropic")),
        activation_decay=float(task_cfg.get("activation_decay", 1.0)),
    )


def run_lora_rank(config: dict[str, Any]) -> Path:
    out_base = deep_get(config, "output.base_dir", "runs")
    run_dir = make_run_dir(out_base, deep_get(config, "output.name", "lora_rank"))
    materialize_run(run_dir, config)

    seed = int(deep_get(config, "seed", 0))
    task_repetitions = int(deep_get(config, "sweep.task_repetitions", 3))
    ranks = [int(v) for v in deep_get(config, "sweep.ranks", [1, 2, 4, 8, 16, 32, 64])]
    alphas = [float(v) for v in _as_list(deep_get(config, "sweep.alpha", 16.0))]

    train_cfg = config.get("train", {})
    lr = float(train_cfg.get("lr", 0.5))
    steps = int(train_cfg.get("steps", 400))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    init_scale = float(train_cfg.get("init_scale", 0.1))
    eval_every = int(train_cfg.get("eval_every", 25))
    batch_size = train_cfg.get("batch_size", None)
    if batch_size is not None:
        batch_size = int(batch_size)
    grad_clip = train_cfg.get("grad_clip", None)
    if grad_clip is not None:
        grad_clip = float(grad_clip)

    detect_cfg = config.get("detect", {})
    null_method = str(detect_cfg.get("null_method", "permute"))
    n_boot = int(detect_cfg.get("n_boot", 40))
    quantile = float(detect_cfg.get("quantile", 0.995))
    gradient_matrix_mode = str(detect_cfg.get("gradient_matrix_mode", "activation_whitened"))
    gradient_ridge_scale = float(detect_cfg.get("gradient_ridge_scale", 1e-6))
    include_raw_gradient = bool(detect_cfg.get("include_raw_gradient", True))
    intruder_overlap_threshold = float(detect_cfg.get("intruder_overlap_threshold", 0.05))

    rows: list[dict[str, Any]] = []
    for task_rep in range(task_repetitions):
        task_rng = np.random.default_rng(seed + 10_000 * task_rep)
        task = _make_task_from_config(config, task_rng)
        diag_rng = np.random.default_rng(seed + 10_000 * task_rep + 999)
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
        for alpha in alphas:
            for rank in ranks:
                run_rng = np.random.default_rng(seed + 10_000 * task_rep + int(rank * 101 + alpha * 17))
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
                edge_rng = np.random.default_rng(seed + 10_000 * task_rep + rank * 123 + int(alpha * 31) + 7)
                edge_result = bootstrap_edge(
                    result.delta,
                    edge_rng,
                    method=null_method,
                    n_boot=max(5, n_boot // 2),
                    quantile=quantile,
                )
                adapter_summary = summarize_spectrum(result.delta, edge=edge_result.edge)
                intr = intruder_diagnostics(
                    task,
                    result.delta,
                    edge=edge_result.edge,
                    overlap_threshold=intruder_overlap_threshold,
                )
                row = {
                    "experiment": "lora_rank",
                    "seed": seed,
                    "task_rep": task_rep,
                    "rank": rank,
                    "alpha": alpha,
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
                    "adapter_stable_rank_edge_summary": adapter_summary.stable_rank,
                    "adapter_effective_rank_edge_summary": adapter_summary.effective_rank,
                    **grad_diag,
                    **result.diagnostics,
                    **intr,
                }
                rows.append(row)
    write_metrics(rows, run_dir / "metrics.csv")
    return run_dir


def run_alpha_sweep(config: dict[str, Any]) -> Path:
    # Same engine as rank sweep, but with a fixed rank and alpha list.
    cfg = dict(config)
    sweep = dict(cfg.get("sweep", {}))
    fixed_rank = int(sweep.get("rank", 16))
    sweep["ranks"] = [fixed_rank]
    sweep["alpha"] = sweep.get("alphas", [1, 2, 4, 8, 16, 32, 64, 128])
    cfg["sweep"] = sweep
    cfg.setdefault("output", {})["name"] = deep_get(config, "output.name", "alpha_sweep")
    return run_lora_rank(cfg)


def run_merging(config: dict[str, Any]) -> Path:
    out_base = deep_get(config, "output.base_dir", "runs")
    run_dir = make_run_dir(out_base, deep_get(config, "output.name", "merge"))
    materialize_run(run_dir, config)

    seed = int(deep_get(config, "seed", 0))
    rng = np.random.default_rng(seed)
    d_in = int(deep_get(config, "task.d_in", 128))
    d_out = int(deep_get(config, "task.d_out", 128))
    n_val = int(deep_get(config, "task.n_val", 2048))
    rank = int(deep_get(config, "task.task_rank", 8))
    spikes = np.asarray(deep_get(config, "task.task_spikes", [2.5 / (1 + 0.25 * i) for i in range(rank)]), dtype=float)
    if len(spikes) != rank:
        raise ValueError("task_spikes length must equal task_rank")
    base_scale = float(deep_get(config, "task.base_scale", 0.2))
    update_noise_sigma = float(deep_get(config, "task.update_noise_sigma", 0.02))
    overlaps = [float(v) for v in deep_get(config, "sweep.overlaps", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])]
    repetitions = int(deep_get(config, "sweep.repetitions", 20))

    detect_cfg = config.get("detect", {})
    null_method = str(detect_cfg.get("null_method", "permute"))
    n_boot = int(detect_cfg.get("n_boot", 30))
    quantile = float(detect_cfg.get("quantile", 0.995))

    rows: list[dict[str, Any]] = []
    for overlap in overlaps:
        for rep in range(repetitions):
            w_base = rng.normal(0.0, base_scale / np.sqrt(d_in), size=(d_out, d_in))
            x_a = rng.normal(size=(n_val, d_in))
            x_b = rng.normal(size=(n_val, d_in))
            u1, v1, u2, v2 = make_overlapping_spikes(d_out, d_in, rank, overlap, rng)
            delta_a_true = matrix_from_svd(u1, spikes, v1)
            delta_b_true = matrix_from_svd(u2, spikes, v2)
            delta_a_hat = delta_a_true + rng.normal(0.0, update_noise_sigma / np.sqrt(d_in), size=(d_out, d_in))
            delta_b_hat = delta_b_true + rng.normal(0.0, update_noise_sigma / np.sqrt(d_in), size=(d_out, d_in))
            y_a = x_a @ (w_base + delta_a_true).T
            y_b = x_b @ (w_base + delta_b_true).T
            merge = 0.5 * (delta_a_hat + delta_b_hat)
            single_a_loss = mse_loss(x_a, y_a, w_base, delta_a_hat)
            single_b_loss = mse_loss(x_b, y_b, w_base, delta_b_hat)
            merge_a_loss = mse_loss(x_a, y_a, w_base, merge)
            merge_b_loss = mse_loss(x_b, y_b, w_base, merge)
            forgetting_merge = 0.5 * (forgetting_loss(x_a, merge) + forgetting_loss(x_b, merge))
            edge_a = bootstrap_edge(delta_a_hat, rng, method=null_method, n_boot=n_boot, quantile=quantile).edge
            edge_b = bootstrap_edge(delta_b_hat, rng, method=null_method, n_boot=n_boot, quantile=quantile).edge
            score = spectral_interference_score(delta_a_hat, delta_b_hat, edge_a=edge_a, edge_b=edge_b)
            sum_a = summarize_spectrum(delta_a_hat, edge=edge_a)
            sum_b = summarize_spectrum(delta_b_hat, edge=edge_b)
            rows.append(
                {
                    "experiment": "merge",
                    "seed": seed,
                    "rep": rep,
                    "overlap": overlap,
                    "rank": rank,
                    "single_a_loss": single_a_loss,
                    "single_b_loss": single_b_loss,
                    "merge_a_loss": merge_a_loss,
                    "merge_b_loss": merge_b_loss,
                    "mean_single_loss": 0.5 * (single_a_loss + single_b_loss),
                    "mean_merge_loss": 0.5 * (merge_a_loss + merge_b_loss),
                    "merge_degradation": 0.5 * (merge_a_loss + merge_b_loss) - 0.5 * (single_a_loss + single_b_loss),
                    "forgetting_merge": forgetting_merge,
                    "spectral_interference": score,
                    "edge_a": edge_a,
                    "edge_b": edge_b,
                    "detectable_rank_a": sum_a.detectable_rank,
                    "detectable_rank_b": sum_b.detectable_rank,
                    "stable_rank_a": sum_a.stable_rank,
                    "stable_rank_b": sum_b.stable_rank,
                }
            )
    write_metrics(rows, run_dir / "metrics.csv")
    return run_dir
