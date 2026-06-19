#!/usr/bin/env python3
from __future__ import annotations

"""Run Stage4 replicates into a self-contained, publication-auditable release."""

import argparse
import copy
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import yaml

from rmt_lora.provenance import (
    git_state,
    runtime_versions,
    sha256_file,
    utc_now_iso,
    verify_sha256s,
    write_json,
    write_sha256s,
)
from rmt_lora.stage4_analysis import (
    ANALYSIS_PLAN_VERSION,
    PREDICTOR_TRANSFORM,
    TARGET_TRANSFORM,
    analysis_plan_document,
)
from rmt_lora.targets import (
    TARGET_DENOMINATOR,
    TARGET_ESTIMAND,
    TARGET_ESTIMAND_VERSION,
    TARGET_REFERENCE,
)

PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "paper": {
        "seeds": [101, 103, 107, 109, 113],
        "n_layers": 48,
        "configs": {
            "hard": Path("configs/layerwise_hard_knee.yaml"),
            "sample": Path("configs/layerwise_sample_limited.yaml"),
        },
    },
    "fast": {
        "seeds": [101, 103, 107],
        "n_layers": 32,
        "configs": {
            "hard": Path("configs/layerwise_hard_knee.yaml"),
            "sample": Path("configs/layerwise_sample_limited.yaml"),
        },
    },
    "smoke": {
        "seeds": [101],
        "n_layers": 6,
        "configs": {
            "hard": Path("configs/layerwise_hard_knee_smoke.yaml"),
            "sample": Path("configs/layerwise_sample_limited_smoke.yaml"),
        },
    },
}
CONDITION_BY_MODE = {"hard": "hard_knee", "sample": "sample_limited"}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"configuration root must be a mapping: {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _parse_ints(text: str) -> list[int]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("seeds must be a nonempty comma-separated list without duplicates")
    return values


def _parse_modes(text: str) -> list[str]:
    modes = [value.strip() for value in text.split(",") if value.strip()]
    invalid = sorted(set(modes) - set(CONDITION_BY_MODE))
    if not modes or invalid or len(modes) != len(set(modes)):
        raise ValueError(
            f"modes must be unique values from {sorted(CONDITION_BY_MODE)}; got {modes}"
        )
    return modes


def _safe_identifier(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip()).strip("-._")
    if not cleaned:
        raise ValueError("release identifier cannot be empty")
    return cleaned


def _run(command: list[str], *, env: dict[str, str]) -> None:
    print("\n" + "=" * 100, flush=True)
    print("running", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def _relative_to_project(path: Path, project_dir: Path) -> str:
    try:
        return path.resolve().relative_to(project_dir.resolve()).as_posix()
    except ValueError:
        return os.path.relpath(path.resolve(), project_dir.resolve())


def _archive_release(release_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        archive.add(release_dir, arcname=release_dir.name, recursive=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run multi-seed Stage4 experiments and emit a portable release containing "
            "every per-rank source row, aggregate table, manifest, and SHA-256 hash."
        )
    )
    parser.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS), default="paper")
    parser.add_argument(
        "--seeds",
        help="Comma-separated seeds. Defaults are profile-specific; paper uses five seeds.",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        help="Layers per seed. Defaults are profile-specific; paper uses 48.",
    )
    parser.add_argument("--modes", default="hard,sample", help="Comma-separated hard,sample")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--tag", default="stage4", help="Human-readable release tag")
    parser.add_argument("--release-id", help="Exact release directory name")
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("runs/stage4_releases"),
        help="Directory that receives release folders and the LATEST pointer.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Permit an uncommitted Git worktree. Never use this for publication results.",
    )
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args()

    project_dir = Path.cwd().resolve()
    if not (project_dir / "rmt_lora").is_dir() or not (project_dir / "scripts").is_dir():
        raise SystemExit("run this command from Code/rmt_lora_sim")

    defaults = PROFILE_DEFAULTS[args.profile]
    seeds = _parse_ints(args.seeds) if args.seeds else list(defaults["seeds"])
    n_layers = int(args.n_layers if args.n_layers is not None else defaults["n_layers"])
    modes = _parse_modes(args.modes)
    if n_layers < 4:
        raise ValueError("Stage4 regression fits require at least four layers per run")

    code_state = git_state(Path(__file__))
    if code_state.get("git_dirty") and not args.allow_dirty:
        status = "\n".join(code_state.get("git_status_porcelain") or [])
        raise SystemExit(
            "Git worktree is dirty; commit/stash changes before a traceable Stage4 run.\n"
            + status
        )

    timestamp = utc_now_iso().replace("-", "").replace(":", "").replace("+00:00", "Z")
    timestamp = timestamp.split(".", 1)[0]
    release_id = _safe_identifier(
        args.release_id or f"{args.tag}_{args.profile}_{timestamp}"
    )
    release_root = args.release_root.resolve()
    release_dir = release_root / release_id
    if release_dir.exists():
        raise FileExistsError(f"release already exists: {release_dir}")

    configs_dir = release_dir / "configs"
    source_root = release_dir / "source_runs"
    aggregate_dir = release_dir / "aggregate"
    pointer_dir = release_dir / ".run_pointers"
    for directory in (configs_dir, source_root, aggregate_dir, pointer_dir):
        directory.mkdir(parents=True, exist_ok=False)

    env = os.environ.copy()
    env["PYTHONPATH"] = "." + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    source_runs: list[Path] = []
    run_records: list[dict[str, Any]] = []

    try:
        for mode in modes:
            condition = CONDITION_BY_MODE[mode]
            base_path = Path(defaults["configs"][mode])
            base_config = _load_yaml(base_path)
            for seed in seeds:
                config = copy.deepcopy(base_config)
                name = f"{args.tag}_{mode}_s{seed}"
                config["seed"] = int(seed)
                config.setdefault("output", {})
                config["output"] = dict(config["output"])
                config["output"]["base_dir"] = _relative_to_project(
                    source_root, project_dir
                )
                config["output"]["name"] = name
                config.setdefault("layer_sweep", {})
                config["layer_sweep"] = dict(config["layer_sweep"])
                config["layer_sweep"]["n_layers"] = n_layers
                config["release"] = {
                    "release_id": release_id,
                    "profile": args.profile,
                    "condition": condition,
                    "seed": int(seed),
                    "target_estimand_version": TARGET_ESTIMAND_VERSION,
                }

                config_path = configs_dir / f"{name}.yaml"
                _write_yaml(config_path, config)
                run_pointer = pointer_dir / f"{name}.txt"
                command = [
                    sys.executable,
                    "scripts/run_layerwise_rank_prediction_v2.py",
                    "--config",
                    _relative_to_project(config_path, project_dir),
                    "--write-run-dir",
                    str(run_pointer),
                ]
                if not args.no_plot:
                    command.append("--plot")
                _run(command, env=env)

                run_dir = Path(run_pointer.read_text(encoding="utf-8").strip()).resolve()
                try:
                    run_dir.relative_to(source_root)
                except ValueError as error:
                    raise RuntimeError(
                        f"producer wrote outside the release source directory: {run_dir}"
                    ) from error
                _run(
                    [
                        sys.executable,
                        "scripts/analyze_layerwise_targets.py",
                        str(run_dir),
                        "--check-existing",
                    ],
                    env=env,
                )
                checksum_errors = verify_sha256s(run_dir, reject_unlisted=True)
                if checksum_errors:
                    raise RuntimeError(
                        f"source run failed checksum verification: {run_dir}\n  "
                        + "\n  ".join(checksum_errors)
                    )
                source_runs.append(run_dir)
                run_records.append(
                    {
                        "condition": condition,
                        "mode": mode,
                        "seed": int(seed),
                        "config": config_path.relative_to(release_dir).as_posix(),
                        "source_run": run_dir.relative_to(release_dir).as_posix(),
                    }
                )

        shutil.rmtree(pointer_dir)
        aggregate_command = [
            sys.executable,
            "scripts/aggregate_stage4.py",
            *[str(path) for path in source_runs],
            "--out",
            str(aggregate_dir),
            "--expected-seeds",
            ",".join(str(seed) for seed in seeds),
            "--expected-conditions",
            ",".join(CONDITION_BY_MODE[mode] for mode in modes),
        ]
        _run(aggregate_command, env=env)

        write_json(release_dir / "analysis_plan.json", analysis_plan_document())
        write_json(
            release_dir / "release_manifest.json",
            {
                "schema_version": 1,
                "release_id": release_id,
                "created_utc": utc_now_iso(),
                "profile": args.profile,
                "tag": args.tag,
                "publication_grade_profile": args.profile == "paper",
                "conditions": [CONDITION_BY_MODE[mode] for mode in modes],
                "seeds": seeds,
                "n_layers_per_run": n_layers,
                "n_source_runs": len(run_records),
                "source_runs": run_records,
                "aggregate": "aggregate",
                "target_estimand": TARGET_ESTIMAND,
                "target_estimand_version": TARGET_ESTIMAND_VERSION,
                "target_reference": TARGET_REFERENCE,
                "target_denominator": TARGET_DENOMINATOR,
                "analysis_plan_version": ANALYSIS_PLAN_VERSION,
                "target_transform": TARGET_TRANSFORM,
                "predictor_transform": PREDICTOR_TRANSFORM,
                "git": code_state,
                "runtime": runtime_versions(),
                "command": sys.argv,
            },
        )
        (release_dir / "README.txt").write_text(
            "Stage4 publication evidence bundle\n"
            "==================================\n\n"
            "The source_runs directory contains every raw per-rank row used by the "
            "aggregate. target_definition.json fixes the observed-best estimand. "
            "Validate this directory with:\n\n"
            "  PYTHONPATH=. python3 scripts/validate_stage4_release.py "
            f"{_relative_to_project(release_dir, project_dir)}\n",
            encoding="utf-8",
        )
        write_sha256s(release_dir)

        _run(
            [
                sys.executable,
                "scripts/validate_stage4_release.py",
                str(release_dir),
                "--expected-seeds",
                ",".join(str(seed) for seed in seeds),
                "--expected-conditions",
                ",".join(CONDITION_BY_MODE[mode] for mode in modes),
            ],
            env=env,
        )

        release_root.mkdir(parents=True, exist_ok=True)
        (release_root / "LATEST").write_text(release_id + "\n", encoding="utf-8")

        archive_path: Path | None = None
        if not args.no_archive:
            archive_path = release_root / f"{release_id}.tar.gz"
            _archive_release(release_dir, archive_path)
            (release_root / f"{release_id}.tar.gz.sha256").write_text(
                f"{sha256_file(archive_path)}  {archive_path.name}\n", encoding="utf-8"
            )

        print("\nStage4 release complete")
        print(f"release: {release_dir}")
        print(f"aggregate: {aggregate_dir / 'stage4_key_table.csv'}")
        if archive_path is not None:
            print(f"archive: {archive_path}")
    except Exception:
        print(f"\nFAILED release retained for diagnosis: {release_dir}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
