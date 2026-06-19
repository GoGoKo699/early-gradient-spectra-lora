#!/usr/bin/env python3
"""Import a Stage4 aggregate into the paper-facing Stage4 table.

Run from the clean package's `Paper/` directory.  By default this imports
the bundled released aggregate:

    python3 import_stage4_aggregate.py

To import a freshly rerun aggregate instead:

    python3 import_stage4_aggregate.py \
      --aggregate ../Code/rmt_lora_sim/runs/stage4_aggregate/stage4_key_table.csv

The output is `tables/stage4_gradient_effective_summary.csv`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TARGET_DISPLAY = {
    "near_best_rank_gap_0.1": "near-best 0.1 gap",
    "near_best_rank_gap_0.2": "near-best 0.2 gap",
    "penalized_rank_lambda_0.2": "penalty 0.2",
    "penalized_rank_lambda_0.3": "penalty 0.3",
    "recovery_rank_0.7": "70% recovery",
    "recovery_rank_0.8": "80% recovery",
}
TARGET_ORDER = {name: i for i, name in enumerate(TARGET_DISPLAY)}
OUT_COLUMNS = [
    "condition",
    "target_display",
    "n_runs",
    "mean_r2",
    "sem_r2",
    "mean_spearman",
    "sem_spearman",
    "mean_rmse_log2",
]


def default_aggregate_path() -> Path:
    candidates = [
        Path("../Code/rmt_lora_sim/results/released/stage4_aggregate/stage4_key_table.csv"),
        Path("../Code/rmt_lora_sim/runs/stage4_aggregate/stage4_key_table.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def import_stage4(aggregate: Path, out: Path, predictor: str) -> pd.DataFrame:
    df = pd.read_csv(aggregate)
    required = {
        "condition",
        "target",
        "predictor",
        "n_runs",
        "mean_r2",
        "sem_r2",
        "mean_spearman",
        "sem_spearman",
        "mean_rmse_log2",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{aggregate} is missing required columns: {missing}")

    sub = df[(df["predictor"] == predictor) & (df["target"].isin(TARGET_DISPLAY))].copy()
    if sub.empty:
        available = sorted(df["predictor"].dropna().unique())
        raise ValueError(f"predictor {predictor!r} not found for paper targets. Available predictors: {available}")
    expected_rows = 2 * len(TARGET_DISPLAY)
    if len(sub) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows for predictor {predictor!r}, found {len(sub)}")

    sub["target_display"] = sub["target"].map(TARGET_DISPLAY)
    sub["target_order"] = sub["target"].map(TARGET_ORDER)
    sub = sub.sort_values(["condition", "target_order"]).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub[OUT_COLUMNS].to_csv(out, index=False)
    return sub[OUT_COLUMNS]


def main() -> None:
    ap = argparse.ArgumentParser(description="Import Stage4 aggregate CSV into paper-facing summary table.")
    ap.add_argument("--aggregate", type=Path, default=default_aggregate_path(), help="Path to stage4_key_table.csv")
    ap.add_argument("--out", type=Path, default=Path("tables/stage4_gradient_effective_summary.csv"), help="Paper-facing output CSV.")
    ap.add_argument(
        "--predictor",
        default="gradient_effective_rank",
        help="Predictor to import. Publication configs use activation-whitened gradient_effective_rank.",
    )
    args = ap.parse_args()
    written = import_stage4(args.aggregate, args.out, args.predictor)
    print(f"read {args.aggregate}")
    print(f"wrote {args.out}")
    print(written.to_string(index=False))


if __name__ == "__main__":
    main()
