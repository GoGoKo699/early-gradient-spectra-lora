from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .experiments import run_bbp, run_lora_rank, run_alpha_sweep, run_merging
from .plotting import plot_from_metrics


def _config_argparser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config file.")
    parser.add_argument("--plot", action="store_true", help="Generate figures after the run.")
    return parser


def run_bbp_cli() -> None:
    parser = _config_argparser("Run the BBP/spiked-matrix simulation.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_dir = run_bbp(cfg)
    print(f"wrote {run_dir}")
    if args.plot:
        paths = plot_from_metrics(run_dir / "metrics.csv")
        print("figures:")
        for p in paths:
            print(f"  {p}")


def run_lora_rank_cli() -> None:
    parser = _config_argparser("Run the synthetic LoRA rank-sweep simulation.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_dir = run_lora_rank(cfg)
    print(f"wrote {run_dir}")
    if args.plot:
        paths = plot_from_metrics(run_dir / "metrics.csv")
        print("figures:")
        for p in paths:
            print(f"  {p}")


def run_alpha_cli() -> None:
    parser = _config_argparser("Run the synthetic LoRA alpha-sweep simulation.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_dir = run_alpha_sweep(cfg)
    print(f"wrote {run_dir}")
    if args.plot:
        paths = plot_from_metrics(run_dir / "metrics.csv")
        print("figures:")
        for p in paths:
            print(f"  {p}")


def run_merge_cli() -> None:
    parser = _config_argparser("Run the synthetic task-vector merging simulation.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_dir = run_merging(cfg)
    print(f"wrote {run_dir}")
    if args.plot:
        paths = plot_from_metrics(run_dir / "metrics.csv")
        print("figures:")
        for p in paths:
            print(f"  {p}")


def plot_cli() -> None:
    parser = argparse.ArgumentParser(description="Plot figures from a metrics.csv file.")
    parser.add_argument("metrics", help="Path to metrics.csv")
    parser.add_argument("--out", default=None, help="Output figure directory. Defaults to metrics parent / figures.")
    args = parser.parse_args()
    paths = plot_from_metrics(Path(args.metrics), args.out)
    print("figures:")
    for p in paths:
        print(f"  {p}")


def fit_cli() -> None:
    parser = argparse.ArgumentParser(description="Fit simple predictor models to decide which hypothesis best explains a simulation run.")
    parser.add_argument("metrics", help="Path to metrics.csv")
    parser.add_argument("--outcome", default=None, help="Outcome column. Defaults to final_val_loss or merge_degradation.")
    parser.add_argument("--out", default=None, help="Output CSV. Defaults to metrics parent / hypothesis_fit.csv.")
    args = parser.parse_args()
    from .analysis import fit_hypotheses

    result = fit_hypotheses(args.metrics, outcome=args.outcome, out_path=args.out)
    print(result.head(10).to_string(index=False))
