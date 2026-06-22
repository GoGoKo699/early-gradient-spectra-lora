#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def infer_seed_from_run(run_dir: Path) -> int | None:
    cfg = run_dir / "config.yaml"
    if cfg.exists():
        text = cfg.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"\n\s*seed:\s*(\d+)", "\n" + text)
        if match:
            return int(match.group(1))
    match = re.search(r"_s(\d+)", run_dir.name)
    return int(match.group(1)) if match else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_manifest_entry(run_dir: Path) -> dict:
    info_path = run_dir / "run_info.json"
    info = (
        json.loads(info_path.read_text(encoding="utf-8"))
        if info_path.is_file()
        else {}
    )
    required_hashes = {}
    for relative in (
        "config.yaml",
        "run_info.json",
        "_SUCCESS.json",
        "calibration/module_stats.csv",
        "calibration/singular_values.npz",
        "calibration/null_maxima.npz",
        "sweeps/site_rank_sweep_metrics.csv",
        "sweeps/site_target_summary.csv",
        "sweeps/site_prediction_fit.csv",
        "budget/budget_results.csv",
        "budget/allocation_comparison.csv",
    ):
        path = run_dir / relative
        if path.is_file():
            required_hashes[relative] = _file_sha256(path)
    return {
        "run_id": run_dir.name,
        "task_name": info.get("task_name"),
        "base_seed": info.get("base_seed", infer_seed_from_run(run_dir)),
        "protocol_version": info.get("protocol_version"),
        "primary_scaling_mode": info.get("primary_scaling_mode"),
        "primary_allocation_rule": info.get("primary_allocation_rule"),
        "primary_reference_rule": info.get("primary_reference_rule"),
        "primary_metric": info.get("primary_metric"),
        "independent_unit": info.get("independent_unit"),
        "sha256": required_hashes,
    }


def _common_non_null(entries: list[dict], key: str):
    values = [entry.get(key) for entry in entries if entry.get(key) is not None]
    if not values:
        return None
    canonical = {json.dumps(value, sort_keys=True) for value in values}
    if len(canonical) != 1:
        raise ValueError(f"source runs disagree on {key}: {values}")
    return values[0]


def collect_runs(root: Path, patterns: list[str]) -> list[Path]:
    runs = []
    for pattern in patterns:
        for path in root.glob(pattern):
            if (path / "budget" / "budget_results.csv").exists():
                runs.append(path)
    return sorted(set(runs), key=lambda path: path.name)


def sem(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) <= 1:
        return 0.0
    return float(numeric.std(ddof=1) / np.sqrt(len(numeric)))


def _stable_int_seed(*parts: object) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def exact_sign_flip_p(values: Iterable[float], *, max_exact_n: int = 20) -> float:
    data = np.asarray(list(values), dtype=float)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return float("nan")
    observed = abs(float(np.mean(data)))
    tolerance = 1e-15
    if len(data) <= max_exact_n:
        exceed = 0
        total = 0
        for signs in itertools.product((-1.0, 1.0), repeat=len(data)):
            statistic = abs(float(np.mean(data * np.asarray(signs))))
            exceed += statistic >= observed - tolerance
            total += 1
        return float(exceed / total)
    rng = np.random.default_rng(_stable_int_seed("sign_flip", data.tolist()))
    draws = 100_000
    signs = rng.choice(np.array([-1.0, 1.0]), size=(draws, len(data)))
    statistics = np.abs(np.mean(signs * data[None, :], axis=1))
    return float((1 + np.sum(statistics >= observed - tolerance)) / (draws + 1))


def cluster_bootstrap_interval(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed_parts: tuple[object, ...] = (),
) -> tuple[float, float]:
    data = np.asarray(list(values), dtype=float)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return float("nan"), float("nan")
    if len(data) == 1:
        value = float(data[0])
        return value, value
    rng = np.random.default_rng(_stable_int_seed("cluster_bootstrap", *seed_parts))
    indices = rng.integers(0, len(data), size=(int(n_resamples), len(data)))
    estimates = data[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(estimates, tail, method="linear")),
        float(np.quantile(estimates, 1.0 - tail, method="linear")),
    )


def _normalise_protocol_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize legacy and corrected result schemas without collapsing exact baselines."""
    df = df.copy()
    if "adaptation_replicate" not in df:
        df["adaptation_replicate"] = 0
    if "scaling_mode" not in df:
        df["scaling_mode"] = "standard"
    if "condition_id" not in df:
        df["condition_id"] = df["rule"].astype(str)
    if "allocation_rule" not in df:
        df["allocation_rule"] = df["rule"].astype(str)
    if "comparison_role" not in df:
        exact_mask = df["allocation_rule"].astype(str).eq("uniform_exact_cost")
        cap_mask = df["allocation_rule"].astype(str).isin(["uniform", "uniform_fill"])
        df["comparison_role"] = np.where(
            exact_mask,
            "exact_cost_baseline",
            np.where(cap_mask, "cap_baseline", "candidate"),
        )
    if "matched_baseline_condition_id" not in df:
        df["matched_baseline_condition_id"] = ""
    if "is_primary_allocation_rule" not in df:
        df["is_primary_allocation_rule"] = False
    if "matched_cost" not in df and "actual_cost" in df:
        df["matched_cost"] = df["actual_cost"]
    if "requested_budget" not in df and "budget" in df:
        df["requested_budget"] = df["budget"]

    # Keep historical ``rule``-based outputs unique. The unmodified allocator
    # identity remains available in ``allocation_rule``.
    exact = df["allocation_rule"].astype(str).eq("uniform_exact_cost")
    df.loc[exact, "rule"] = df.loc[exact, "condition_id"].astype(str)
    df["adaptation_replicate"] = pd.to_numeric(
        df["adaptation_replicate"], errors="raise"
    ).astype(int)
    keys = ["run", "scaling_mode", "budget", "adaptation_replicate", "condition_id"]
    missing = [column for column in keys if column not in df]
    if missing:
        raise ValueError(f"budget results are missing unique-key columns: {missing}")
    duplicated = df.duplicated(keys, keep=False)
    if duplicated.any():
        raise ValueError(
            f"duplicate budget result rows for keys {keys}:\n"
            f"{df.loc[duplicated, keys].to_string(index=False)}"
        )
    return df


def _run_level_budget(budget: pd.DataFrame, extra_keys: list[str] | None = None) -> pd.DataFrame:
    extra_keys = extra_keys or []
    group_keys = extra_keys + [
        "run",
        "seed",
        "scaling_mode",
        "condition_id",
        "rule",
        "allocation_rule",
        "comparison_role",
        "is_primary_allocation_rule",
        "budget",
    ]
    return budget.groupby(group_keys, dropna=False, as_index=False).agg(
        n_adaptation_replicates=("adaptation_replicate", "nunique"),
        actual_cost=("actual_cost", "mean"),
        requested_budget=("requested_budget", "mean"),
        final_val_loss=("final_val_loss", "mean"),
        final_val_accuracy=("final_val_accuracy", "mean"),
        diverged_count=("diverged", "sum"),
    )


def _cap_paired_deltas(budget: pd.DataFrame, extra_keys: list[str] | None = None) -> pd.DataFrame:
    extra_keys = extra_keys or []
    pair_keys = extra_keys + ["run", "scaling_mode", "budget", "adaptation_replicate"]
    reference = budget[budget["condition_id"].astype(str).eq("uniform_fill")][
        pair_keys + ["final_val_loss", "final_val_accuracy", "actual_cost"]
    ].rename(
        columns={
            "final_val_loss": "uniform_fill_loss",
            "final_val_accuracy": "uniform_fill_accuracy",
            "actual_cost": "uniform_fill_cost",
        }
    )
    if reference.duplicated(pair_keys).any():
        raise ValueError("uniform_fill must have one row per comparison block")
    candidates = budget[
        ~budget["condition_id"].astype(str).eq("uniform_fill")
        & ~budget["comparison_role"].astype(str).eq("exact_cost_baseline")
    ].copy()
    paired = candidates.merge(reference, on=pair_keys, how="inner", validate="many_to_one")
    paired["loss_delta_vs_uniform_fill"] = paired["final_val_loss"] - paired["uniform_fill_loss"]
    paired["acc_delta_vs_uniform_fill"] = (
        paired["final_val_accuracy"] - paired["uniform_fill_accuracy"]
    )
    paired["loss_ratio_vs_uniform_fill"] = paired["final_val_loss"] / paired[
        "uniform_fill_loss"
    ].replace(0, np.nan)
    paired["cost_gap_vs_uniform_fill"] = paired["actual_cost"] - paired["uniform_fill_cost"]
    keep = extra_keys + [
        "run",
        "seed",
        "scaling_mode",
        "budget",
        "adaptation_replicate",
        "condition_id",
        "rule",
        "allocation_rule",
        "comparison_role",
        "is_primary_allocation_rule",
        "loss_delta_vs_uniform_fill",
        "acc_delta_vs_uniform_fill",
        "loss_ratio_vs_uniform_fill",
        "actual_cost",
        "uniform_fill_cost",
        "cost_gap_vs_uniform_fill",
    ]
    return paired[keep]


def _exact_cost_paired_deltas(
    budget: pd.DataFrame,
    extra_keys: list[str] | None = None,
) -> pd.DataFrame:
    extra_keys = extra_keys or []
    pair_keys = extra_keys + ["run", "scaling_mode", "budget", "adaptation_replicate"]
    references = budget[budget["comparison_role"].astype(str).eq("exact_cost_baseline")][
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
    if references.duplicated(pair_keys + ["matched_baseline_condition_id"]).any():
        raise ValueError("exact-cost references are not unique within comparison blocks")
    candidates = budget[
        budget["comparison_role"].astype(str).eq("candidate")
        & budget["matched_baseline_condition_id"].fillna("").astype(str).ne("")
    ].copy()
    paired = candidates.merge(
        references,
        on=pair_keys + ["matched_baseline_condition_id"],
        how="left",
        validate="many_to_one",
    )
    if len(paired) and paired[["exact_uniform_loss", "exact_uniform_cost"]].isna().any().any():
        missing = paired.loc[
            paired["exact_uniform_cost"].isna(),
            pair_keys + ["condition_id", "matched_baseline_condition_id"],
        ]
        raise ValueError(f"candidate rows lack exact-cost baselines:\n{missing.to_string(index=False)}")
    if len(paired) and not (
        paired["actual_cost"].astype(int) == paired["exact_uniform_cost"].astype(int)
    ).all():
        raise ValueError("exact-cost pair contains unequal realized parameter costs")
    paired["loss_delta_vs_exact_uniform"] = (
        paired["final_val_loss"] - paired["exact_uniform_loss"]
    )
    paired["acc_delta_vs_exact_uniform"] = (
        paired["final_val_accuracy"] - paired["exact_uniform_accuracy"]
    )
    keep = extra_keys + [
        "run",
        "seed",
        "scaling_mode",
        "budget",
        "adaptation_replicate",
        "condition_id",
        "rule",
        "allocation_rule",
        "is_primary_allocation_rule",
        "matched_baseline_condition_id",
        "actual_cost",
        "exact_uniform_cost",
        "loss_delta_vs_exact_uniform",
        "acc_delta_vs_exact_uniform",
    ]
    return paired[keep]


def _run_level_deltas(
    deltas: pd.DataFrame,
    *,
    delta_prefix: str,
    extra_keys: list[str] | None = None,
) -> pd.DataFrame:
    extra_keys = extra_keys or []
    loss_column = f"loss_delta_vs_{delta_prefix}"
    accuracy_column = f"acc_delta_vs_{delta_prefix}"
    group_keys = extra_keys + [
        "run",
        "seed",
        "scaling_mode",
        "condition_id",
        "rule",
        "allocation_rule",
        "is_primary_allocation_rule",
        "budget",
    ]
    aggregations = {
        "n_adaptation_replicates": ("adaptation_replicate", "nunique"),
        loss_column: (loss_column, "mean"),
        accuracy_column: (accuracy_column, "mean"),
        "actual_cost": ("actual_cost", "mean"),
    }
    if delta_prefix == "uniform_fill":
        aggregations["reference_cost"] = ("uniform_fill_cost", "mean")
        aggregations["cost_gap"] = ("cost_gap_vs_uniform_fill", "mean")
    else:
        aggregations["reference_cost"] = ("exact_uniform_cost", "mean")
        aggregations["cost_gap"] = (
            "actual_cost",
            lambda values: 0.0,
        )
    return deltas.groupby(group_keys, dropna=False, as_index=False).agg(**aggregations)


def _independent_delta_summary(
    run_deltas: pd.DataFrame,
    *,
    delta_prefix: str,
    extra_keys: list[str] | None = None,
    include_budget: bool = True,
) -> pd.DataFrame:
    extra_keys = extra_keys or []
    loss_column = f"loss_delta_vs_{delta_prefix}"
    accuracy_column = f"acc_delta_vs_{delta_prefix}"
    group_keys = extra_keys + [
        "scaling_mode",
        "condition_id",
        "rule",
        "allocation_rule",
        "is_primary_allocation_rule",
    ]
    if include_budget:
        group_keys.append("budget")
    rows = []
    for keys, group in run_deltas.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip(group_keys, keys))
        loss = pd.to_numeric(group[loss_column], errors="coerce").dropna().to_numpy(dtype=float)
        accuracy = pd.to_numeric(group[accuracy_column], errors="coerce").dropna().to_numpy(dtype=float)
        ci_low, ci_high = cluster_bootstrap_interval(
            loss,
            seed_parts=tuple(identity.values()) + (delta_prefix,),
        )
        rows.append(
            {
                **identity,
                "n_independent_runs": int(group["run"].nunique()),
                "mean_loss_delta": float(np.mean(loss)) if len(loss) else float("nan"),
                "sem_loss_delta": float(np.std(loss, ddof=1) / np.sqrt(len(loss))) if len(loss) > 1 else 0.0,
                "median_loss_delta": float(np.median(loss)) if len(loss) else float("nan"),
                "cluster_bootstrap_ci_low": ci_low,
                "cluster_bootstrap_ci_high": ci_high,
                "exact_sign_flip_p_two_sided": exact_sign_flip_p(loss),
                "run_wins_loss": int(np.sum(loss < 0)),
                "run_ties_loss": int(np.sum(loss == 0)),
                "mean_accuracy_delta": float(np.mean(accuracy)) if len(accuracy) else float("nan"),
                "sem_accuracy_delta": (
                    float(np.std(accuracy, ddof=1) / np.sqrt(len(accuracy)))
                    if len(accuracy) > 1
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_delta_outputs(
    budget: pd.DataFrame,
    out: Path,
    *,
    extra_keys: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    extra_keys = extra_keys or []
    outputs: dict[str, pd.DataFrame] = {}
    cap = _cap_paired_deltas(budget, extra_keys)
    if len(cap):
        cap.to_csv(out / "deltas_vs_uniform_fill.csv", index=False)
        run_cap = _run_level_deltas(cap, delta_prefix="uniform_fill", extra_keys=extra_keys)
        run_cap.to_csv(out / "run_level_deltas_vs_uniform_fill.csv", index=False)
        summary_cap = _independent_delta_summary(
            run_cap,
            delta_prefix="uniform_fill",
            extra_keys=extra_keys,
        )
        summary_cap.to_csv(out / "delta_summary_vs_uniform_fill.csv", index=False)
        outputs.update(cap=cap, run_cap=run_cap, summary_cap=summary_cap)

    exact = _exact_cost_paired_deltas(budget, extra_keys)
    if len(exact):
        exact.to_csv(out / "deltas_exact_cost.csv", index=False)
        run_exact = _run_level_deltas(exact, delta_prefix="exact_uniform", extra_keys=extra_keys)
        run_exact.to_csv(out / "run_level_deltas_exact_cost.csv", index=False)
        summary_exact = _independent_delta_summary(
            run_exact,
            delta_prefix="exact_uniform",
            extra_keys=extra_keys,
        )
        summary_exact.to_csv(out / "delta_summary_exact_cost.csv", index=False)
        cluster_keys = extra_keys + [
            "run",
            "seed",
            "scaling_mode",
            "condition_id",
            "rule",
            "allocation_rule",
            "is_primary_allocation_rule",
        ]
        run_cluster = run_exact.groupby(cluster_keys, dropna=False, as_index=False).agg(
            n_budgets=("budget", "nunique"),
            mean_adaptation_replicates=("n_adaptation_replicates", "mean"),
            loss_delta_vs_exact_uniform=("loss_delta_vs_exact_uniform", "mean"),
            acc_delta_vs_exact_uniform=("acc_delta_vs_exact_uniform", "mean"),
            mean_actual_cost=("actual_cost", "mean"),
        )
        run_cluster.to_csv(out / "run_cluster_deltas_exact_cost.csv", index=False)
        cluster_summary = _independent_delta_summary(
            run_cluster,
            delta_prefix="exact_uniform",
            extra_keys=extra_keys,
            include_budget=False,
        )
        cluster_summary.to_csv(out / "run_cluster_summary_exact_cost.csv", index=False)
        outputs.update(
            exact=exact,
            run_exact=run_exact,
            summary_exact=summary_exact,
            run_cluster_exact=run_cluster,
            cluster_summary_exact=cluster_summary,
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate modular synthetic-transformer runs using independent run clusters."
    )
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--out", default="step3_aggregate")
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=["*_modular_small_step2", "*_modular_small_step3_s*"],
    )
    args = parser.parse_args()

    root = Path(args.runs_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dirs = collect_runs(root, args.patterns)
    if not run_dirs:
        raise SystemExit(f"No matching runs found under {root} for {args.patterns}")
    source_entries = [_source_manifest_entry(path) for path in run_dirs]

    budget_frames: list[pd.DataFrame] = []
    site_fit_frames: list[pd.DataFrame] = []
    module_frames: list[pd.DataFrame] = []
    base_frames: list[pd.DataFrame] = []
    target_frames: list[pd.DataFrame] = []
    allocation_frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        seed = infer_seed_from_run(run_dir)
        run_name = run_dir.name

        def add_meta(frame: pd.DataFrame) -> pd.DataFrame:
            frame = frame.copy()
            frame.insert(0, "run", run_name)
            frame.insert(1, "seed", seed)
            return frame

        budget_frames.append(add_meta(pd.read_csv(run_dir / "budget" / "budget_results.csv")))
        optional = [
            ("budget/allocation_comparison.csv", allocation_frames),
            ("sweeps/site_prediction_fit.csv", site_fit_frames),
            ("sweeps/site_target_summary.csv", target_frames),
            ("calibration/module_stats.csv", module_frames),
            ("base/base_metrics.csv", base_frames),
        ]
        for relative, destination in optional:
            path = run_dir / relative
            if path.exists():
                destination.append(add_meta(pd.read_csv(path)))

    budget = _normalise_protocol_columns(pd.concat(budget_frames, ignore_index=True))
    budget.to_csv(out / "all_budget_results.csv", index=False)
    for frames, filename in [
        (base_frames, "all_base_metrics.csv"),
        (module_frames, "all_module_stats.csv"),
        (site_fit_frames, "all_site_prediction_fit.csv"),
        (target_frames, "all_site_target_summary.csv"),
        (allocation_frames, "all_allocation_comparison.csv"),
    ]:
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out / filename, index=False)

    run_budget = _run_level_budget(budget)
    run_budget.to_csv(out / "run_level_budget_results.csv", index=False)
    summary = run_budget.groupby(
        [
            "scaling_mode",
            "condition_id",
            "rule",
            "allocation_rule",
            "comparison_role",
            "is_primary_allocation_rule",
            "budget",
        ],
        dropna=False,
        as_index=False,
    ).agg(
        n_runs=("run", "nunique"),
        mean_adaptation_replicates=("n_adaptation_replicates", "mean"),
        mean_actual_cost=("actual_cost", "mean"),
        mean_requested_budget=("requested_budget", "mean"),
        mean_val_loss=("final_val_loss", "mean"),
        sem_val_loss=("final_val_loss", sem),
        median_val_loss=("final_val_loss", "median"),
        mean_val_accuracy=("final_val_accuracy", "mean"),
        sem_val_accuracy=("final_val_accuracy", sem),
        median_val_accuracy=("final_val_accuracy", "median"),
        diverged_count=("diverged_count", "sum"),
    )
    summary.to_csv(out / "budget_summary.csv", index=False)
    delta_outputs = _write_delta_outputs(budget, out)

    winner_source = run_budget[
        ~run_budget["comparison_role"].astype(str).eq("exact_cost_baseline")
    ]
    winners = []
    for (run, scaling_mode, budget_id), group in winner_source.groupby(
        ["run", "scaling_mode", "budget"]
    ):
        best_loss = group.loc[group["final_val_loss"].idxmin()]
        best_accuracy = group.loc[group["final_val_accuracy"].idxmax()]
        winners.append(
            {
                "run": run,
                "seed": best_loss.get("seed"),
                "scaling_mode": scaling_mode,
                "budget": budget_id,
                "best_loss_condition_id": best_loss["condition_id"],
                "best_loss_rule": best_loss["allocation_rule"],
                "best_loss": best_loss["final_val_loss"],
                "best_accuracy_condition_id": best_accuracy["condition_id"],
                "best_accuracy_rule": best_accuracy["allocation_rule"],
                "best_accuracy": best_accuracy["final_val_accuracy"],
            }
        )
    winner_frame = pd.DataFrame(winners)
    winner_frame.to_csv(out / "winner_by_budget_run.csv", index=False)
    winner_frame.groupby(
        ["scaling_mode", "budget", "best_loss_rule"], as_index=False
    ).size().rename(columns={"size": "count"}).to_csv(
        out / "winner_counts_by_budget.csv", index=False
    )

    if site_fit_frames:
        fits = pd.concat(site_fit_frames, ignore_index=True)
        if "scaling_mode" not in fits:
            fits["scaling_mode"] = "standard"
        fits.groupby(
            ["scaling_mode", "target", "predictor"], as_index=False
        ).agg(
            n_runs=("run", "nunique"),
            mean_spearman=("spearman", "mean"),
            sem_spearman=("spearman", sem),
            median_spearman=("spearman", "median"),
        ).sort_values(
            ["scaling_mode", "target", "mean_spearman"],
            ascending=[True, True, False],
        ).to_csv(out / "site_prediction_summary.csv", index=False)

    figure_dir = out / "figures"
    figure_dir.mkdir(exist_ok=True)
    cap_summary = summary[
        ~summary["comparison_role"].astype(str).eq("exact_cost_baseline")
    ]
    for scaling_mode, mode_summary in cap_summary.groupby("scaling_mode"):
        plt.figure(figsize=(8, 5))
        for condition_id, group in mode_summary.groupby("condition_id"):
            group = group.sort_values("budget")
            plt.errorbar(
                group["budget"],
                group["mean_val_loss"],
                yerr=group["sem_val_loss"],
                marker="o",
                capsize=3,
                label=condition_id,
            )
        plt.xlabel("parameter budget cap")
        plt.ylabel("mean validation loss across independent runs")
        plt.title(f"Budgeted allocation: {scaling_mode}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(figure_dir / f"budget_loss_summary_{scaling_mode}.png", dpi=160)
        plt.close()

    manifest = {
        "n_independent_runs": len(run_dirs),
        "runs": [str(path) for path in run_dirs],
        "source_runs": source_entries,
        "protocol_version": _common_non_null(source_entries, "protocol_version"),
        "primary_scaling_mode": _common_non_null(source_entries, "primary_scaling_mode"),
        "primary_allocation_rule": _common_non_null(source_entries, "primary_allocation_rule"),
        "primary_reference_rule": _common_non_null(source_entries, "primary_reference_rule"),
        "primary_metric": _common_non_null(source_entries, "primary_metric"),
        "independent_unit": _common_non_null(source_entries, "independent_unit"),
        "unit_of_inference": (
            "base-model run; adaptation replicates are averaged within run and budgets "
            "are clustered within run for across-budget summaries"
        ),
        "scaling_modes": sorted(budget["scaling_mode"].astype(str).unique().tolist()),
        "scaling_conditions_analyzed_separately": True,
        "exact_cost_primary_outputs": bool(delta_outputs.get("exact") is not None),
        "smoke_or_publication_status": "not_inferred_from_directory_name",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"aggregated {len(run_dirs)} independent runs")
    print(f"wrote {out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
