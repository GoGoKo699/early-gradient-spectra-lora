#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
import yaml


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _latest_run(name: str) -> Path:
    matches = sorted(Path("runs").glob(f"*{name}"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise RuntimeError(f"could not find run directory for {name!r}")
    return matches[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run multi-seed stage4 replication experiments.")
    ap.add_argument("--seeds", default="101,103,107", help="Comma-separated seeds. Use 5+ seeds for paper-grade results.")
    ap.add_argument("--n-layers", type=int, default=32, help="Layers per seed. Use 48+ for final run if compute allows.")
    ap.add_argument("--modes", default="hard,sample", help="Comma-separated modes: hard,sample")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--tag", default="stage4", help="Run-name tag.")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    base_map = {
        "hard": Path("configs/layerwise_hard_knee.yaml"),
        "sample": Path("configs/layerwise_sample_limited.yaml"),
    }

    env = os.environ.copy()
    env["PYTHONPATH"] = "." + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    made_runs: list[Path] = []

    tmp_cfg_dir = Path("configs/_stage4_generated")
    tmp_cfg_dir.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        if mode not in base_map:
            raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(base_map)}")
        base_cfg = _load_yaml(base_map[mode])
        for seed in seeds:
            cfg = dict(base_cfg)
            cfg["seed"] = int(seed)
            cfg.setdefault("output", {})
            cfg["output"] = dict(cfg["output"])
            name = f"{args.tag}_{mode}_s{seed}"
            cfg["output"]["name"] = name
            cfg.setdefault("layer_sweep", {})
            cfg["layer_sweep"] = dict(cfg["layer_sweep"])
            cfg["layer_sweep"]["n_layers"] = int(args.n_layers)
            cfg_path = tmp_cfg_dir / f"{name}.yaml"
            _write_yaml(cfg_path, cfg)

            cmd = ["python3", "scripts/run_layerwise_rank_prediction_v2.py", "--config", str(cfg_path)]
            if not args.no_plot:
                cmd.append("--plot")
            print("\n" + "=" * 100)
            print("running", " ".join(cmd))
            subprocess.run(cmd, check=True, env=env)
            run_dir = _latest_run(name)
            made_runs.append(run_dir)
            print("analyzing", run_dir)
            subprocess.run(["python3", "scripts/analyze_layerwise_targets.py", str(run_dir)], check=True, env=env)

    print("\nAggregating stage4 runs")
    agg_cmd = ["python3", "scripts/aggregate_stage4.py", *map(str, made_runs)]
    subprocess.run(agg_cmd, check=True, env=env)

    tar_name = f"rmt_lora_{args.tag}_runs.tar.gz"
    subprocess.run(["tar", "-czf", tar_name, *map(str, made_runs), "runs/stage4_aggregate"], check=True)
    print(f"wrote {tar_name}")


if __name__ == "__main__":
    main()
