#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def maybe_print(path: Path, n: int = 10) -> None:
    print("\n" + "=" * 88)
    print(path)
    if not path.exists():
        print("missing")
        return
    df = pd.read_csv(path)
    print("shape", df.shape)
    print(df.head(n).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    r = Path(args.run_dir)
    maybe_print(r / "base" / "base_metrics.csv")
    maybe_print(r / "calibration" / "module_stats.csv")
    maybe_print(r / "sweeps" / "site_target_summary.csv")
    maybe_print(r / "sweeps" / "site_prediction_fit.csv", n=30)
    maybe_print(r / "budget" / "budget_results.csv")


if __name__ == "__main__":
    main()
