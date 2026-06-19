#!/usr/bin/env python3
"""Regenerate paper-facing tables and vector figures from released CSV artifacts.

Run from the clean package ``Paper/`` directory:

    python3 make_paper_artifacts.py

The script is intentionally deterministic and uses only files shipped in this
repository.  It also forces Matplotlib PDF/PS output to Type 42 fonts so the
compiled paper avoids Type 3 fonts.
"""
from __future__ import annotations

from pathlib import Path
import math
import re

import matplotlib as mpl
mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
CODE = ROOT / "Code"
if not CODE.exists():
    CODE = ROOT
TABLES = PAPER / "tables"
GENERATED = TABLES / "generated"
FIGURES = PAPER / "figures"
RMT_RESULTS = CODE / "rmt_lora_sim" / "results" / "released"
RMT_RUNS_LEGACY = CODE / "rmt_lora_sim" / "runs"

RULE_DISPLAY = {
    "soft_dimension": "soft dimension",
    "gradient_norm": "gradient norm",
    "marginal_gain_soft": "marginal gain soft",
    "effective_rank": "effective rank",
    "marginal_gain_edge": "marginal gain edge",
    "marginal_gain_raw": "marginal gain raw",
}
CONDITION_DISPLAY = {"hard_knee": "Hard-knee", "sample_limited": "Sample-limited"}
TARGET_ORDER = [
    "near-best 0.1 gap",
    "near-best 0.2 gap",
    "70% recovery",
    "80% recovery",
    "penalty 0.2",
    "penalty 0.3",
]
TASK_ORDER = ["modular", "assoc"]
RULE_ORDER = ["soft_dimension", "gradient_norm", "marginal_gain_soft", "effective_rank", "marginal_gain_edge"]


def _ensure_dirs() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def _tex_escape(text: object) -> str:
    s = str(text)
    # Keep math symbols inserted by this script untouched; escape table text only.
    return (
        s.replace("%", r"\%")
        .replace("_", r"\_")
        .replace("&", r"\&")
    )


def _fmt(x: float, digits: int = 4, signed: bool = False) -> str:
    if pd.isna(x):
        return "--"
    spec = f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"
    return spec.format(x=x) if False else format(float(x), f"+.{digits}f" if signed else f".{digits}f")


def _sem(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) <= 1:
        return float("nan")
    return float(vals.std(ddof=1) / math.sqrt(len(vals)))


def _write(path: Path, text: str) -> None:
    r"""Write generated artifacts.

    LaTeX tabular bodies are input inside a live ``tabular`` environment.
    Some TeX installations do not accept a booktabs rule immediately after an
    ``\input`` file that ends with ``\\``.  For generated row files, leave
    the final row terminator to the caller (``\input{...}\\``) while keeping
    interior row terminators inside the file.
    """
    if path.suffix == ".tex" and path.parent == GENERATED:
        lines = text.rstrip("\n").splitlines()
        if lines:
            lines[-1] = re.sub(r"\s*\\\\\s*$", "", lines[-1])
            text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_stage4_rows() -> None:
    df = pd.read_csv(TABLES / "stage4_gradient_effective_summary.csv")
    lines: list[str] = []
    for ci, condition in enumerate(["hard_knee", "sample_limited"]):
        if ci:
            lines.append(r"\midrule")
        sub = df[df["condition"] == condition].copy()
        sub["target_order"] = sub["target_display"].map({t: i for i, t in enumerate(TARGET_ORDER)})
        sub = sub.sort_values("target_order")
        for _, row in sub.iterrows():
            lines.append(
                f"{CONDITION_DISPLAY[condition]} & {_tex_escape(row['target_display'])} & "
                f"{float(row['mean_r2']):.3f} ({float(row['sem_r2']):.3f}) & "
                f"{float(row['mean_spearman']):.3f} ({float(row['sem_spearman']):.3f}) \\\\" 
            )
    _write(GENERATED / "stage4_rows.tex", "\n".join(lines) + "\n")


def make_transformer_rows() -> None:
    per = pd.read_csv(TABLES / "per_run_rule_summary.csv")
    low = pd.read_csv(TABLES / "low_budget_by_rule_summary.csv")
    merged = per.merge(low[["rule", "wins", "mean_loss_delta"]], on="rule", suffixes=("", "_low"))
    merged = merged[merged["rule"].isin(RULE_ORDER)].copy()
    order = {r: i for i, r in enumerate(RULE_ORDER)}
    merged["order"] = merged["rule"].map(order).fillna(99)
    merged = merged.sort_values("order")
    lines = []
    for _, row in merged.iterrows():
        lines.append(
            f"{RULE_DISPLAY.get(row['rule'], row['rule'])} & "
            f"${_fmt(row['mean_loss_delta'], 4, signed=True)}$ & "
            f"${int(row['wins'])}$ & "
            f"${_fmt(row['mean_loss_delta_low'], 4, signed=True)}$ & "
            f"${int(row['wins_low'])}$ & "
            f"${_fmt(row['mean_acc_delta'], 4, signed=True)}$ \\\\" 
        )
    _write(GENERATED / "transformer_allocation_rows.tex", "\n".join(lines) + "\n")


def make_transformer_cluster_summary() -> None:
    df = pd.read_csv(TABLES / "combined_per_run_deltas.csv")
    rules = [r for r in RULE_ORDER if r in set(df["rule"])]
    run_rule = (
        df[df["rule"].isin(rules)]
        .groupby(["task", "run", "seed", "rule"], as_index=False)
        .agg(
            mean_loss_delta=("loss_delta_vs_uniform_fill", "mean"),
            wins=("loss_delta_vs_uniform_fill", lambda x: int((x < 0).sum())),
            n=("loss_delta_vs_uniform_fill", "size"),
            mean_acc_delta=("acc_delta_vs_uniform_fill", "mean"),
        )
    )
    cluster = (
        run_rule.groupby("rule", as_index=False)
        .agg(
            n_run_clusters=("mean_loss_delta", "size"),
            clusters_better=("mean_loss_delta", lambda x: int((x < 0).sum())),
            mean_cluster_loss_delta=("mean_loss_delta", "mean"),
            sem_cluster_loss_delta=("mean_loss_delta", _sem),
            median_cluster_loss_delta=("mean_loss_delta", "median"),
            mean_cluster_acc_delta=("mean_acc_delta", "mean"),
        )
    )
    cluster["order"] = cluster["rule"].map({r: i for i, r in enumerate(RULE_ORDER)}).fillna(99)
    cluster = cluster.sort_values("order").drop(columns="order")
    cluster.to_csv(TABLES / "transformer_cluster_summary.csv", index=False)
    lines = []
    for _, row in cluster.iterrows():
        lines.append(
            f"{RULE_DISPLAY.get(row['rule'], row['rule'])} & "
            f"${_fmt(row['mean_cluster_loss_delta'], 4, signed=True)}$ & "
            f"${_fmt(row['sem_cluster_loss_delta'], 4)}$ & "
            f"${int(row['clusters_better'])}/{int(row['n_run_clusters'])}$ \\\\" 
        )
    _write(GENERATED / "transformer_cluster_rows.tex", "\n".join(lines) + "\n")


def make_by_task_rows() -> None:
    df = pd.read_csv(TABLES / "by_task_rule_summary.csv")
    order = {r: i for i, r in enumerate(RULE_ORDER)}
    task_display = {"modular": "Modular", "assoc": "Associative recall"}
    lines = []
    for ti, task in enumerate(TASK_ORDER):
        if ti:
            lines.append(r"\midrule")
        sub = df[(df["task"] == task) & (df["rule"].isin(RULE_ORDER))].copy()
        sub["order"] = sub["rule"].map(order).fillna(99)
        sub = sub.sort_values("order")
        for _, row in sub.iterrows():
            lines.append(
                f"{task_display.get(task, task)} & {RULE_DISPLAY.get(row['rule'], row['rule'])} & "
                f"${_fmt(row['mean_loss_delta'], 4, signed=True)}$ & "
                f"${int(row['budgets_better'])}/{int(row['n_budgets'])}$ & "
                f"${_fmt(row['mean_acc_delta'], 4, signed=True)}$ & "
                f"${_fmt(row['mean_loss_ratio'], 3)}$ \\\\" 
            )
    _write(GENERATED / "by_task_transformer_rows.tex", "\n".join(lines) + "\n")


def make_whitening_rows() -> None:
    df = pd.read_csv(TABLES / "combined_whitening_summary.csv").sort_values("mean_loss_delta")
    low = pd.read_csv(TABLES / "combined_whitening_low_budget_summary.csv")[["label", "mean_loss_delta", "wins"]]
    merged = df.merge(low, on="label", suffixes=("", "_low"), how="left")
    lines = []
    for _, row in merged.iterrows():
        w = "--" if row["whitening"] == "baseline" else str(row["whitening"])
        lines.append(
            f"{RULE_DISPLAY.get(row['rule'], row['rule'])} & {_tex_escape(w)} & "
            f"${_fmt(row['mean_loss_delta'], 4, signed=True)}$ & "
            f"${int(row['wins'])}$ & "
            f"${_fmt(row['mean_loss_delta_low'], 4, signed=True)}$ & "
            f"${int(row['wins_low'])}$ & "
            f"${_fmt(row['mean_acc_delta'], 4, signed=True)}$ \\\\" 
        )
    _write(GENERATED / "whitening_rows.tex", "\n".join(lines) + "\n")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _find_run(name: str, legacy_pattern: str) -> Path:
    """Return a released RMT result directory.

    The cleaned release stores paper artifacts under semantic names in
    ``../Code/rmt_lora_sim/results/released``.  The fallback keeps the script usable
    with older working trees that still use timestamped ``runs`` folders.
    """
    direct = RMT_RESULTS / name
    if direct.exists():
        return direct
    matches = sorted(RMT_RUNS_LEGACY.glob(legacy_pattern))
    if matches:
        return matches[-1]
    raise FileNotFoundError(
        f"missing released RMT result {name!r} under {RMT_RESULTS}; "
        f"also tried legacy pattern {legacy_pattern!r} under {RMT_RUNS_LEGACY}"
    )


def make_bbp_figures() -> None:
    df = pd.read_csv(_find_run("bbp", "*_bbp") / "metrics.csv")
    g = df.groupby("theta", as_index=False).agg(
        top_sv_mean=("top_sv", "mean"),
        top_sv_sem=("top_sv", _sem),
        edge=("mp_edge", "mean"),
        overlap_mean=("mean_overlap", "mean"),
        overlap_sem=("mean_overlap", _sem),
    )

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.errorbar(g["theta"], g["overlap_mean"], yerr=g["overlap_sem"], marker="o", linewidth=1)
    ax.set_xlabel(r"spike strength $\theta$")
    ax.set_ylabel("mean singular-vector overlap")
    ax.set_title("BBP alignment")
    _save(fig, FIGURES / "bbp_alignment_summary.pdf")

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.errorbar(g["theta"], g["top_sv_mean"], yerr=g["top_sv_sem"], marker="o", linewidth=1)
    ax.plot(g["theta"], g["edge"], linestyle="--", linewidth=1)
    ax.set_xlabel(r"spike strength $\theta$")
    ax.set_ylabel("top singular value")
    ax.set_title("BBP top singular value")
    _save(fig, FIGURES / "bbp_top_sv_summary.pdf")


def make_lora_rank_figures() -> None:
    df = pd.read_csv(_find_run("lora_rank_clean", "*_lora_rank_clean") / "metrics.csv")
    g = df.groupby("rank", as_index=False).agg(
        val=("final_val_loss", "mean"),
        val_sem=("final_val_loss", _sem),
        det=("adapter_detectable_rank", "mean"),
        eff=("adapter_effective_rank", "mean"),
        stable=("adapter_stable_rank", "mean"),
    )
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.errorbar(g["rank"], g["val"], yerr=g["val_sem"], marker="o", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("nominal LoRA rank")
    ax.set_ylabel("validation MSE")
    ax.set_title("Clean rank sweep")
    _save(fig, FIGURES / "lora_rank_clean_val_loss.pdf")

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.plot(g["rank"], g["det"], marker="o", linewidth=1, label="detectable")
    ax.plot(g["rank"], g["eff"], marker="s", linewidth=1, label="effective")
    ax.plot(g["rank"], g["stable"], marker="^", linewidth=1, label="stable")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("nominal LoRA rank")
    ax.set_ylabel("final adapter statistic")
    ax.set_title("Final adapter spectra")
    ax.legend(fontsize=7)
    _save(fig, FIGURES / "lora_rank_clean_spectral_stats.pdf")


def make_alpha_figure() -> None:
    df = pd.read_csv(_find_run("alpha_noise_intruder", "*_alpha_noise_intruder") / "metrics.csv")
    g = df.groupby("alpha", as_index=False).agg(
        val=("final_val_loss", "mean"),
        val_sem=("final_val_loss", _sem),
        drift=("forgetting_loss", "mean"),
    )
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.errorbar(g["alpha"], g["val"], yerr=g["val_sem"], marker="o", linewidth=1, label="validation")
    ax2 = ax.twinx()
    ax2.plot(g["alpha"], g["drift"], marker="s", linewidth=1, label="drift")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"LoRA $\alpha$")
    ax.set_ylabel("validation MSE")
    ax2.set_ylabel("output drift proxy")
    ax.set_title("Alpha controls fit and drift")
    _save(fig, FIGURES / "alpha_noise_tradeoff.pdf")


def make_merge_figure() -> None:
    df = pd.read_csv(_find_run("merge_conflict", "*_merge_conflict") / "metrics.csv")
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.scatter(df["conflict_score"], df["merge_degradation"], s=12, alpha=0.7)
    x = df["conflict_score"].to_numpy(dtype=float)
    y = df["merge_degradation"].to_numpy(dtype=float)
    if len(x) > 1 and np.nanstd(x) > 0:
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
        ax.plot(xs, coef[0] * xs + coef[1], linewidth=1)
    ax.set_xlabel("signed conflict score")
    ax.set_ylabel("merge degradation")
    ax.set_title("Signed conflict predicts merge loss")
    _save(fig, FIGURES / "merge_conflict_score.pdf")


def make_stage4_figure() -> None:
    df = pd.read_csv(TABLES / "stage4_gradient_effective_summary.csv")
    df["target_order"] = df["target_display"].map({t: i for i, t in enumerate(TARGET_ORDER)})
    df = df.sort_values(["target_order", "condition"])
    hard = df[df["condition"] == "hard_knee"].sort_values("target_order")
    sample = df[df["condition"] == "sample_limited"].sort_values("target_order")
    labels = [re.sub(r" recovery", r" rec.", x).replace("near-best ", "near ").replace(" gap", "") for x in TARGET_ORDER]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.bar(x - width / 2, hard["mean_r2"], width, yerr=hard["sem_r2"], label="hard-knee")
    ax.bar(x + width / 2, sample["mean_r2"], width, yerr=sample["sem_r2"], label="sample-limited")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(r"mean $R^2$")
    ax.set_title("Stage4 effective-rank prediction")
    ax.legend(fontsize=7)
    _save(fig, FIGURES / "stage4_effective_rank_r2.pdf")


def main() -> None:
    _ensure_dirs()
    make_stage4_rows()
    make_transformer_rows()
    make_transformer_cluster_summary()
    make_by_task_rows()
    make_whitening_rows()
    make_bbp_figures()
    make_lora_rank_figures()
    make_alpha_figure()
    make_merge_figure()
    make_stage4_figure()
    print("wrote generated paper tables and Type-42 PDF figures")


if __name__ == "__main__":
    main()
