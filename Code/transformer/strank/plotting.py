from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt
import pandas as pd


def plot_rank_sweeps(metrics: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    collapsed = metrics.groupby(["site_name", "rank"], as_index=False).agg(final_val_loss=("final_val_loss", "mean"))
    for site, g in collapsed.groupby("site_name"):
        h = g.sort_values("rank")
        ax.plot(h["rank"], h["final_val_loss"], marker="o", label=site.split(".")[-1] + ":" + site.split(".")[1] if "." in site else site)
    ax.set_xscale("symlog", base=2, linthresh=1)
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel("validation loss")
    ax.set_title("Single-site LoRA rank sweeps")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_prediction_scatter(targets: pd.DataFrame, stats: pd.DataFrame, out: Path, target_col: str = "recovery_rank_0.7") -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df = targets.merge(stats, on="site_name", how="left")
    if target_col not in df:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["effective_rank"], df[target_col])
    for _, row in df.iterrows():
        ax.annotate(str(row["site_name"]).split(".")[-1], (row["effective_rank"], row[target_col]), fontsize=7)
    ax.set_xlabel("early-gradient effective rank")
    ax.set_ylabel(target_col)
    ax.set_title("Predicted spectral dimension vs useful rank")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_budget_curve(results: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    collapsed = results.groupby(["rule", "budget"], as_index=False).agg(final_val_loss=("final_val_loss", "mean"))
    for rule, g in collapsed.groupby("rule"):
        h = g.sort_values("budget")
        ax.plot(h["budget"], h["final_val_loss"], marker="o", label=rule)
    ax.set_xlabel("parameter budget")
    ax.set_ylabel("validation loss")
    ax.set_title("Budgeted rank allocation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_spectra(stats: pd.DataFrame, sv_dict: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for site, s in sv_dict.items():
        ax.plot(range(1, len(s) + 1), s, marker=".", linewidth=1, label=site.split(".")[-1])
    ax.set_xlabel("singular value index")
    ax.set_ylabel("singular value")
    ax.set_title("Module whitened-gradient spectra")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
