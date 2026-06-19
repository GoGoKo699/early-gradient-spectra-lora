from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping, Any

import pandas as pd

from .config import save_config


def make_run_dir(base: str | Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / f"{stamp}_{name}"
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    return run_dir


def write_metrics(rows: list[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False)


def read_metrics(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def materialize_run(run_dir: Path, config: Mapping[str, Any]) -> None:
    save_config(config, run_dir / "config.yaml")
