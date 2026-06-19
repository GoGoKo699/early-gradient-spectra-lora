from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_bbp(df: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    paths: list[Path] = []
    g = df.groupby("theta", as_index=False).agg(
        top_sv_mean=("top_sv", "mean"),
        top_sv_std=("top_sv", "std"),
        det_mean=("detectable_rank", "mean"),
        overlap_mean=("mean_overlap", "mean"),
        edge=("mp_edge", "mean"),
    )
    fig, ax = plt.subplots()
    ax.errorbar(g["theta"], g["top_sv_mean"], yerr=g["top_sv_std"], marker="o")
    ax.plot(g["theta"], g["edge"], linestyle="--")
    ax.set_xlabel("planted spike strength")
    ax.set_ylabel("top empirical singular value")
    ax.set_title("BBP-style separation: top singular value")
    p = out / "bbp_top_sv.png"
    _save(fig, p)
    paths.append(p)

    fig, ax = plt.subplots()
    ax.plot(g["theta"], g["overlap_mean"], marker="o")
    ax.set_xlabel("planted spike strength")
    ax.set_ylabel("mean singular-vector overlap")
    ax.set_title("Detectability transition: vector alignment")
    p = out / "bbp_alignment.png"
    _save(fig, p)
    paths.append(p)

    fig, ax = plt.subplots()
    ax.plot(g["theta"], g["det_mean"], marker="o")
    ax.set_xlabel("planted spike strength")
    ax.set_ylabel("mean detectable rank")
    ax.set_title("Detected outlier count")
    p = out / "bbp_detectable_rank.png"
    _save(fig, p)
    paths.append(p)
    return paths


def plot_lora_rank(df: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    paths: list[Path] = []
    group_cols = ["rank", "alpha"] if "alpha" in df.columns else ["rank"]
    g = df.groupby(group_cols, as_index=False).agg(
        val_loss=("final_val_loss", "mean"),
        val_loss_std=("final_val_loss", "std"),
        forgetting=("forgetting_loss", "mean"),
        adapter_det=("adapter_detectable_rank", "mean"),
        grad_det=("gradient_detectable_rank", "mean"),
        overlap=("adapter_mean_overlap_true", "mean"),
        intruders=("intruder_count", "mean"),
    )

    fig, ax = plt.subplots()
    for alpha, h in g.groupby("alpha") if "alpha" in g.columns else [(None, g)]:
        label = f"alpha={alpha:g}" if alpha is not None else None
        ax.errorbar(h["rank"], h["val_loss"], yerr=h["val_loss_std"], marker="o", label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("nominal LoRA rank")
    ax.set_ylabel("validation MSE")
    ax.set_title("Rank saturation")
    if "alpha" in g.columns:
        ax.legend()
    p = out / "lora_val_loss_vs_rank.png"
    _save(fig, p)
    paths.append(p)

    fig, ax = plt.subplots()
    ax.scatter(df["adapter_detectable_rank"], df["final_val_loss"])
    ax.set_xlabel("final adapter detectable rank")
    ax.set_ylabel("validation MSE")
    ax.set_title("Validation loss vs detectable rank")
    p = out / "lora_val_loss_vs_detectable_rank.png"
    _save(fig, p)
    paths.append(p)

    fig, ax = plt.subplots()
    for alpha, h in g.groupby("alpha") if "alpha" in g.columns else [(None, g)]:
        label = f"alpha={alpha:g}" if alpha is not None else None
        ax.plot(h["rank"], h["adapter_det"], marker="o", label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("nominal LoRA rank")
    ax.set_ylabel("adapter detectable rank")
    ax.set_title("Nominal rank vs detectable rank")
    if "alpha" in g.columns:
        ax.legend()
    p = out / "lora_detectable_rank_vs_nominal_rank.png"
    _save(fig, p)
    paths.append(p)

    fig, ax = plt.subplots()
    ax.scatter(df["intruder_count"], df["forgetting_loss"])
    ax.set_xlabel("intruder count")
    ax.set_ylabel("base-output drift / forgetting proxy")
    ax.set_title("Intruder directions and forgetting proxy")
    p = out / "lora_intruders_vs_forgetting.png"
    _save(fig, p)
    paths.append(p)
    return paths


def plot_alpha(df: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    paths: list[Path] = []
    g = df.groupby("alpha", as_index=False).agg(
        val_loss=("final_val_loss", "mean"),
        forgetting=("forgetting_loss", "mean"),
        adapter_det=("adapter_detectable_rank", "mean"),
        intruders=("intruder_count", "mean"),
    )
    fig, ax = plt.subplots()
    ax.plot(g["alpha"], g["val_loss"], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("LoRA alpha")
    ax.set_ylabel("validation MSE")
    ax.set_title("Alpha sweep: task loss")
    p = out / "alpha_val_loss.png"
    _save(fig, p)
    paths.append(p)

    fig, ax = plt.subplots()
    ax.plot(g["alpha"], g["forgetting"], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("LoRA alpha")
    ax.set_ylabel("base-output drift / forgetting proxy")
    ax.set_title("Alpha sweep: forgetting proxy")
    p = out / "alpha_forgetting.png"
    _save(fig, p)
    paths.append(p)

    fig, ax = plt.subplots()
    ax.plot(g["alpha"], g["intruders"], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("LoRA alpha")
    ax.set_ylabel("mean intruder count")
    ax.set_title("Alpha sweep: intruder directions")
    p = out / "alpha_intruders.png"
    _save(fig, p)
    paths.append(p)
    return paths


def plot_merge(df: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    paths: list[Path] = []
    g = df.groupby("overlap", as_index=False).agg(
        degradation=("merge_degradation", "mean"),
        degradation_std=("merge_degradation", "std"),
        interference=("spectral_interference", "mean"),
        merge_loss=("mean_merge_loss", "mean"),
    )
    fig, ax = plt.subplots()
    ax.errorbar(g["overlap"], g["degradation"], yerr=g["degradation_std"], marker="o")
    ax.set_xlabel("planted task-subspace overlap")
    ax.set_ylabel("merge degradation")
    ax.set_title("Task overlap and merge degradation")
    p = out / "merge_degradation_vs_overlap.png"
    _save(fig, p)
    paths.append(p)

    fig, ax = plt.subplots()
    ax.scatter(df["spectral_interference"], df["merge_degradation"])
    ax.set_xlabel("spectral interference score")
    ax.set_ylabel("merge degradation")
    ax.set_title("Spectral interference predicts merge degradation")
    p = out / "merge_degradation_vs_interference.png"
    _save(fig, p)
    paths.append(p)
    return paths


def plot_from_metrics(metrics_path: str | Path, out_dir: str | Path | None = None) -> list[Path]:
    metrics_path = Path(metrics_path)
    df = pd.read_csv(metrics_path)
    if out_dir is None:
        out_dir = metrics_path.parent / "figures"
    exp = str(df["experiment"].iloc[0]) if "experiment" in df.columns and len(df) else "unknown"
    if exp == "bbp":
        return plot_bbp(df, out_dir)
    if exp == "lora_rank":
        # If only one rank and alpha varies, alpha-specific plots are more useful.
        if "rank" in df.columns and df["rank"].nunique() == 1 and df["alpha"].nunique() > 1:
            return plot_alpha(df, out_dir) + plot_lora_rank(df, out_dir)
        return plot_lora_rank(df, out_dir)
    if exp == "merge":
        return plot_merge(df, out_dir)
    raise ValueError(f"Unsupported experiment type in {metrics_path}: {exp}")
