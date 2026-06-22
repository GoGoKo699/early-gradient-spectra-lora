#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.aggregate_step3_replicates import (
    cluster_bootstrap_interval,
    exact_sign_flip_p,
)


def _summary(run_clusters: pd.DataFrame, loss_column: str, accuracy_column: str) -> pd.DataFrame:
    group_keys = ["scaling_mode", "condition_id", "label", "whitening", "rule"]
    rows = []
    for keys, group in run_clusters.groupby(group_keys, dropna=False):
        loss = pd.to_numeric(group[loss_column], errors="coerce").dropna().to_numpy(dtype=float)
        accuracy = pd.to_numeric(group[accuracy_column], errors="coerce").dropna().to_numpy(dtype=float)
        ci_low, ci_high = cluster_bootstrap_interval(
            loss,
            seed_parts=tuple(keys) + ("whitening",),
        )
        rows.append(
            {
                **dict(zip(group_keys, keys)),
                "n_independent_runs": int(group["run"].nunique()),
                "mean_loss_delta": float(np.mean(loss)) if len(loss) else float("nan"),
                "sem_loss_delta": (
                    float(np.std(loss, ddof=1) / np.sqrt(len(loss))) if len(loss) > 1 else 0.0
                ),
                "median_loss_delta": float(np.median(loss)) if len(loss) else float("nan"),
                "cluster_bootstrap_ci_low": ci_low,
                "cluster_bootstrap_ci_high": ci_high,
                "exact_sign_flip_p_two_sided": exact_sign_flip_p(loss),
                "run_wins_loss": int(np.sum(loss < 0)),
                "run_ties_loss": int(np.sum(loss == 0)),
                "mean_accuracy_delta": float(np.mean(accuracy)) if len(accuracy) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["scaling_mode", "mean_loss_delta", "label"])


def _read_run(directory: Path) -> tuple[pd.DataFrame, str, str, str]:
    exact_path = directory / "budget" / "whitening_ablation_deltas_exact_cost.csv"
    if exact_path.is_file():
        frame = pd.read_csv(exact_path)
        return (
            frame,
            "exact_uniform",
            "loss_delta_vs_exact_uniform",
            "acc_delta_vs_exact_uniform",
        )
    legacy_path = directory / "budget" / "whitening_ablation_deltas.csv"
    if legacy_path.is_file():
        frame = pd.read_csv(legacy_path)
        return (
            frame,
            "uniform_fill_cap_legacy",
            "loss_delta_vs_uniform_fill",
            "acc_delta_vs_uniform_fill",
        )
    cap_path = directory / "budget" / "whitening_ablation_deltas_vs_cap_uniform.csv"
    if cap_path.is_file():
        frame = pd.read_csv(cap_path)
        return (
            frame,
            "uniform_fill_cap",
            "loss_delta_vs_uniform_fill",
            "acc_delta_vs_uniform_fill",
        )
    raise FileNotFoundError(f"no whitening delta file under {directory / 'budget'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate whitening ablations at the independent task/base-run level."
    )
    parser.add_argument(
        "runs",
        nargs="*",
        help="Ablation run directories. If omitted, uses runs/*whitening_ablation*",
    )
    parser.add_argument("--out", default="whitening_ablation_aggregate")
    args = parser.parse_args()
    run_dirs = (
        [Path(value) for value in args.runs]
        if args.runs
        else sorted(Path("runs").glob("*whitening_ablation*"))
    )

    frames: list[pd.DataFrame] = []
    reference_kinds: set[str] = set()
    loss_columns: set[str] = set()
    accuracy_columns: set[str] = set()
    accepted_runs: list[Path] = []
    for directory in run_dirs:
        try:
            frame, reference_kind, loss_column, accuracy_column = _read_run(directory)
        except FileNotFoundError as exc:
            print(f"skip {exc}")
            continue
        if "adaptation_replicate" not in frame:
            frame["adaptation_replicate"] = 0
        if "scaling_mode" not in frame:
            frame["scaling_mode"] = "standard"
        if "condition_id" not in frame:
            frame["condition_id"] = frame.get("label", frame.get("rule")).astype(str)
        if "label" not in frame:
            frame["label"] = frame["condition_id"].astype(str)
        if "whitening" not in frame:
            frame["whitening"] = "unknown"
        if "rule" not in frame:
            frame["rule"] = frame["label"].astype(str)
        frame.insert(0, "run", directory.name)
        frame["comparison_reference"] = reference_kind
        frame["loss_delta"] = frame[loss_column]
        frame["accuracy_delta"] = frame[accuracy_column]
        frames.append(frame)
        reference_kinds.add(reference_kind)
        loss_columns.add(loss_column)
        accuracy_columns.add(accuracy_column)
        accepted_runs.append(directory)
    if not frames:
        raise SystemExit("no whitening-ablation delta files found")
    if len(reference_kinds) != 1:
        raise ValueError(
            "do not pool exact-cost and cap-based whitening comparisons; "
            f"found {sorted(reference_kinds)}"
        )

    all_rows = pd.concat(frames, ignore_index=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(out / "all_whitening_ablation_deltas.csv", index=False)

    run_level = all_rows.groupby(
        [
            "run",
            "scaling_mode",
            "condition_id",
            "label",
            "whitening",
            "rule",
            "budget",
        ],
        dropna=False,
        as_index=False,
    ).agg(
        n_adaptation_replicates=("adaptation_replicate", "nunique"),
        loss_delta=("loss_delta", "mean"),
        accuracy_delta=("accuracy_delta", "mean"),
    )
    run_level.to_csv(out / "run_level_budget_whitening_ablation_deltas.csv", index=False)
    run_clusters = run_level.groupby(
        ["run", "scaling_mode", "condition_id", "label", "whitening", "rule"],
        dropna=False,
        as_index=False,
    ).agg(
        n_budgets=("budget", "nunique"),
        mean_adaptation_replicates=("n_adaptation_replicates", "mean"),
        loss_delta=("loss_delta", "mean"),
        accuracy_delta=("accuracy_delta", "mean"),
    )
    run_clusters.to_csv(out / "run_cluster_whitening_ablation_deltas.csv", index=False)
    summary = _summary(run_clusters, "loss_delta", "accuracy_delta")
    summary.to_csv(out / "whitening_ablation_summary.csv", index=False)

    for scaling_mode, mode_summary in summary.groupby("scaling_mode"):
        plt.figure(figsize=(8, 5))
        plot_frame = mode_summary.sort_values("mean_loss_delta")
        plt.barh(plot_frame["label"], plot_frame["mean_loss_delta"])
        plt.axvline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("mean independent-run loss delta")
        plt.title(f"Whitening ablation: {scaling_mode}")
        plt.tight_layout()
        plt.savefig(out / f"fig_whitening_ablation_delta_{scaling_mode}.png", dpi=160)
        plt.close()

    manifest = {
        "n_independent_runs": len(accepted_runs),
        "runs": [str(path) for path in accepted_runs],
        "unit_of_inference": (
            "source base-model run; adaptation replicates and budgets are averaged within run"
        ),
        "comparison_reference": next(iter(reference_kinds)),
        "publication_valid_exact_cost": next(iter(reference_kinds)) == "exact_uniform",
        "scaling_modes": sorted(all_rows["scaling_mode"].astype(str).unique().tolist()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("summary:")
    print(summary.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
