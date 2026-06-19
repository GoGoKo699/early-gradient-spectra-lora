#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from rmt_lora.io import make_run_dir, materialize_run, write_metrics
from rmt_lora.config import deep_get
from rmt_lora.spiked import random_orthonormal, spectral_interference_score
from rmt_lora.nulls import bootstrap_edge
from rmt_lora.spectra import summarize_spectrum
from rmt_lora.lora_sim import mse_loss, forgetting_loss


def orthogonal_completion(q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n, k = q.shape
    r = random_orthonormal(n, k, rng)
    r = r - q @ (q.T @ r)
    q2, rr = np.linalg.qr(r, mode="reduced")
    signs = np.sign(np.diag(rr))
    signs[signs == 0] = 1
    return q2[:, :k] * signs


def make_conflict_pair(d_out: int, d_in: int, rank: int, spikes: np.ndarray, overlap: float, conflict_fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    u = random_orthonormal(d_out, rank, rng)
    v = random_orthonormal(d_in, rank, rng)
    up = orthogonal_completion(u, rng)
    vp = orthogonal_completion(v, rng)

    # Pairwise-overlap construction: U2 and V2 remain column-orthonormal because
    # the components are orthogonal.
    c = np.sqrt(overlap)
    s = np.sqrt(max(0.0, 1.0 - overlap))
    u2 = c * u + s * up
    v2 = c * v + s * vp

    signs = np.ones(rank)
    n_flip = int(round(conflict_fraction * rank))
    if n_flip > 0:
        signs[:n_flip] = -1.0

    delta_a = u @ np.diag(spikes) @ v.T
    delta_b = u2 @ np.diag(signs * spikes) @ v2.T
    return delta_a, delta_b


def matrix_inner(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.sum(a * b) / denom)


def plot_conflict(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    g = df.groupby(["overlap", "conflict_fraction"], as_index=False).agg(
        degradation=("merge_degradation", "mean"),
        degradation_std=("merge_degradation", "std"),
        conflict_score=("conflict_score", "mean"),
        spectral_interference=("spectral_interference", "mean"),
    )

    fig, ax = plt.subplots()
    for cf, h in g.groupby("conflict_fraction"):
        ax.errorbar(h["overlap"], h["degradation"], yerr=h["degradation_std"], marker="o", label=f"conflict={cf:g}")
    ax.set_xlabel("shared singular-subspace overlap")
    ax.set_ylabel("merge degradation")
    ax.set_title("Merge degradation needs signed conflict, not overlap alone")
    ax.legend()
    p = out_dir / "merge_degradation_vs_overlap_and_conflict.png"
    fig.tight_layout(); fig.savefig(p, dpi=180); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots()
    ax.scatter(df["spectral_interference"], df["merge_degradation"])
    ax.set_xlabel("unsigned spectral interference")
    ax.set_ylabel("merge degradation")
    ax.set_title("Unsigned overlap is incomplete")
    p = out_dir / "merge_degradation_vs_unsigned_interference.png"
    fig.tight_layout(); fig.savefig(p, dpi=180); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots()
    ax.scatter(df["conflict_score"], df["merge_degradation"])
    ax.set_xlabel("signed conflict score")
    ax.set_ylabel("merge degradation")
    ax.set_title("Signed conflict predicts merge degradation")
    p = out_dir / "merge_degradation_vs_signed_conflict.png"
    fig.tight_layout(); fig.savefig(p, dpi=180); plt.close(fig); paths.append(p)
    return paths


def run(config: dict[str, Any], do_plot: bool = False) -> Path:
    run_dir = make_run_dir(deep_get(config, "output.base_dir", "runs"), deep_get(config, "output.name", "merge_conflict"))
    materialize_run(run_dir, config)
    seed = int(deep_get(config, "seed", 0))
    rng = np.random.default_rng(seed)

    d_in = int(deep_get(config, "task.d_in", 128))
    d_out = int(deep_get(config, "task.d_out", 128))
    n_val = int(deep_get(config, "task.n_val", 4096))
    rank = int(deep_get(config, "task.task_rank", 8))
    spikes = np.asarray(deep_get(config, "task.task_spikes", [2.5 / (1.0 + 0.25 * i) for i in range(rank)]), dtype=float)
    base_scale = float(deep_get(config, "task.base_scale", 0.2))
    update_noise_sigma = float(deep_get(config, "task.update_noise_sigma", 0.04))

    overlaps = [float(v) for v in deep_get(config, "sweep.overlaps", [0.0, 0.25, 0.5, 0.75, 1.0])]
    conflict_fractions = [float(v) for v in deep_get(config, "sweep.conflict_fractions", [0.0, 0.5, 1.0])]
    repetitions = int(deep_get(config, "sweep.repetitions", 8))

    null_method = str(deep_get(config, "detect.null_method", "permute"))
    n_boot = int(deep_get(config, "detect.n_boot", 10))
    quantile = float(deep_get(config, "detect.quantile", 0.995))

    rows: list[dict[str, Any]] = []
    for overlap in overlaps:
        for conflict_fraction in conflict_fractions:
            for rep in range(repetitions):
                w_base = rng.normal(0.0, base_scale / np.sqrt(d_in), size=(d_out, d_in))
                x_a = rng.normal(size=(n_val, d_in))
                x_b = rng.normal(size=(n_val, d_in))
                delta_a_true, delta_b_true = make_conflict_pair(d_out, d_in, rank, spikes, overlap, conflict_fraction, rng)
                delta_a_hat = delta_a_true + rng.normal(0.0, update_noise_sigma / np.sqrt(d_in), size=(d_out, d_in))
                delta_b_hat = delta_b_true + rng.normal(0.0, update_noise_sigma / np.sqrt(d_in), size=(d_out, d_in))

                y_a = x_a @ (w_base + delta_a_true).T
                y_b = x_b @ (w_base + delta_b_true).T
                merge = 0.5 * (delta_a_hat + delta_b_hat)

                edge_a = bootstrap_edge(delta_a_hat, rng, method=null_method, n_boot=n_boot, quantile=quantile).edge
                edge_b = bootstrap_edge(delta_b_hat, rng, method=null_method, n_boot=n_boot, quantile=quantile).edge
                sum_a = summarize_spectrum(delta_a_hat, edge=edge_a)
                sum_b = summarize_spectrum(delta_b_hat, edge=edge_b)

                signed_inner = matrix_inner(delta_a_hat, delta_b_hat)
                rows.append({
                    "experiment": "merge_conflict",
                    "seed": seed,
                    "rep": rep,
                    "overlap": overlap,
                    "conflict_fraction": conflict_fraction,
                    "rank": rank,
                    "single_a_loss": mse_loss(x_a, y_a, w_base, delta_a_hat),
                    "single_b_loss": mse_loss(x_b, y_b, w_base, delta_b_hat),
                    "merge_a_loss": mse_loss(x_a, y_a, w_base, merge),
                    "merge_b_loss": mse_loss(x_b, y_b, w_base, merge),
                    "mean_single_loss": 0.5 * (mse_loss(x_a, y_a, w_base, delta_a_hat) + mse_loss(x_b, y_b, w_base, delta_b_hat)),
                    "mean_merge_loss": 0.5 * (mse_loss(x_a, y_a, w_base, merge) + mse_loss(x_b, y_b, w_base, merge)),
                    "merge_degradation": 0.5 * (mse_loss(x_a, y_a, w_base, merge) + mse_loss(x_b, y_b, w_base, merge)) - 0.5 * (mse_loss(x_a, y_a, w_base, delta_a_hat) + mse_loss(x_b, y_b, w_base, delta_b_hat)),
                    "forgetting_merge": 0.5 * (forgetting_loss(x_a, merge) + forgetting_loss(x_b, merge)),
                    "edge_a": edge_a,
                    "edge_b": edge_b,
                    "detectable_rank_a": sum_a.detectable_rank,
                    "detectable_rank_b": sum_b.detectable_rank,
                    "spectral_interference": spectral_interference_score(delta_a_hat, delta_b_hat, edge_a=edge_a, edge_b=edge_b),
                    "signed_task_inner": signed_inner,
                    "conflict_score": max(0.0, -signed_inner),
                })
    write_metrics(rows, run_dir / "metrics.csv")
    if do_plot:
        paths = plot_conflict(pd.DataFrame(rows), run_dir / "figures")
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
