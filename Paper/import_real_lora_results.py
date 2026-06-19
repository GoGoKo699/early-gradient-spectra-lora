#!/usr/bin/env python3
"""Import released real GPT-2 LoRA validation CSVs into paper-facing rows.

Run from the Paper directory:

    python3 import_real_lora_results.py

The script reads compact released CSV summaries from ../Code/real_lora_validation/results/released
and writes static table rows under tables/generated/.
"""
from __future__ import annotations

from pathlib import Path
import math
import pandas as pd

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
CODE = ROOT / "Code"
REAL_RELEASED = CODE / "real_lora_validation" / "results" / "released"
TABLES = PAPER / "tables"
REAL_TABLES = TABLES / "real_lora"
GENERATED = TABLES / "generated"

SETTINGS = [
    ("gpt2_cattn_cfc_5seed", r"$c_{attn},c_{fc}$"),
    ("gpt2_attnproj_3seed", r"$+a_{proj}$"),
]
STRATEGY_DISPLAY = {
    "spectral_effective": "spectral effective",
    "uniform_r4": "uniform rank 4",
    "gradient_norm": "gradient norm",
}
STRATEGY_ORDER = ["spectral_effective", "uniform_r4", "gradient_norm"]


def fmt(x: float, digits: int = 4) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.{digits}f}"


def fmt_sem(mean: float, sem: float, digits: int = 4) -> str:
    if pd.isna(sem):
        return fmt(mean, digits)
    return f"{fmt(mean, digits)} ({fmt(sem, digits)})"


def main() -> None:
    if not REAL_RELEASED.exists():
        raise FileNotFoundError(f"missing released real LoRA results: {REAL_RELEASED}")
    REAL_TABLES.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)

    rows_tex: list[str] = []
    pair_rows_tex: list[str] = []
    for dirname, setting in SETTINGS:
        d = REAL_RELEASED / dirname
        summary_path = d / "summary_by_strategy.csv"
        pair_path = d / "spectral_pairwise_diffs.csv"
        winners_path = d / "per_run_winners.csv"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = pd.read_csv(summary_path)
        winners = pd.read_csv(winners_path) if winners_path.exists() else pd.DataFrame()
        summary.to_csv(REAL_TABLES / f"{dirname}_summary_by_strategy.csv", index=False)
        if pair_path.exists():
            pair = pd.read_csv(pair_path)
            pair.to_csv(REAL_TABLES / f"{dirname}_pairwise_diffs.csv", index=False)
            for comp, g in pair.groupby("comparison"):
                pair_rows_tex.append(
                    f"{setting} & {comp.replace('_', r'\_')} & "
                    f"{fmt(g['final_val_loss_diff'].mean(), 4)} & "
                    f"{fmt(g['final_val_loss_diff'].std(ddof=1) / math.sqrt(len(g)) if len(g) > 1 else float('nan'), 4)} & "
                    f"{fmt(g['perplexity_diff'].mean(), 3)} \\\\" 
                )
        if winners_path.exists():
            winners.to_csv(REAL_TABLES / f"{dirname}_per_run_winners.csv", index=False)

        # stable order
        summary["_order"] = summary["strategy"].map({s: i for i, s in enumerate(STRATEGY_ORDER)}).fillna(99)
        summary = summary.sort_values("_order")
        for _, r in summary.iterrows():
            strategy = str(r["strategy"])
            rows_tex.append(
                f"{setting} & {STRATEGY_DISPLAY.get(strategy, strategy.replace('_', ' '))} & "
                f"{fmt_sem(r['mean_final_val_loss'], r.get('sem_final_val_loss', float('nan')), 4)} & "
                f"{fmt(r['mean_perplexity'], 2)} & "
                f"{int(round(float(r['mean_trainable_params'])))} & "
                f"{int(r.get('wins', 0))}/{int(r.get('n', 0))} \\\\" 
            )

    (GENERATED / "real_lora_rows.tex").write_text("\n".join(rows_tex) + "\n", encoding="utf-8")
    (GENERATED / "real_lora_pairwise_rows.tex").write_text("\n".join(pair_rows_tex) + "\n", encoding="utf-8")
    print(f"wrote {GENERATED / 'real_lora_rows.tex'}")
    print(f"wrote {GENERATED / 'real_lora_pairwise_rows.tex'}")


if __name__ == "__main__":
    main()
