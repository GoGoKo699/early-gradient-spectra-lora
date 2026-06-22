#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_rank_scaling_run import validate as validate_source_run
from scripts.validate_transformer_release import validate_release
from strank.protocol import (
    SYNTHETIC_TRANSFORMER_PROTOCOL_VERSION,
    TRANSFORMER_AGGREGATE_SCHEMA_VERSION,
    TRANSFORMER_RELEASE_SCHEMA_VERSION,
)

PROTOCOL_VERSION = SYNTHETIC_TRANSFORMER_PROTOCOL_VERSION
_RELEASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _run(command: list[str], *, cwd: Path | None = None) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.returncode, completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_code_snapshot(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        ROOT / "strank",
        destination / "strank",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )
    scripts = destination / "scripts"
    scripts.mkdir()
    for path in sorted((ROOT / "scripts").iterdir()):
        if path.is_file() and path.suffix in {".py", ".sh"}:
            shutil.copy2(path, scripts / path.name)
    configs = destination / "configs"
    configs.mkdir()
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        shutil.copy2(path, configs / path.name)
    for name in ("requirements.txt", "pyproject.toml", "README.md"):
        shutil.copy2(ROOT / name, destination / name)
    if (ROOT / "docs").is_dir():
        shutil.copytree(ROOT / "docs", destination / "docs")
    if (ROOT / "tests").is_dir():
        shutil.copytree(
            ROOT / "tests",
            destination / "tests",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )


def _write_provenance(destination: Path, command_log: Path | None) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    repo_code, repo_root_text = _run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT)
    repo_root = Path(repo_root_text) if repo_code == 0 and repo_root_text else ROOT
    head_code, head = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    status_code, status = _run(["git", "status", "--porcelain=v1"], cwd=repo_root)
    diff_code, diff = _run(["git", "diff", "--binary", "HEAD"], cwd=repo_root)
    _write_json(
        {
            "repository_root_name": repo_root.name,
            "head": head if head_code == 0 else None,
            "status_porcelain": status.splitlines() if status_code == 0 and status else [],
            "snapshot_note": (
                "code_snapshot contains the exact experiment files copied after source-run "
                "validation; git metadata records whether they came from a clean revision"
            ),
        },
        destination / "git.json",
    )
    if diff_code == 0 and diff:
        (destination / "working_tree.patch").write_text(diff + "\n", encoding="utf-8")

    package_names = ["torch", "numpy", "pandas", "matplotlib", "PyYAML"]
    versions: dict[str, str | None] = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    torch_details: dict[str, object] = {}
    try:
        import torch

        torch_details = {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "num_threads": torch.get_num_threads(),
        }
    except Exception as exc:  # pragma: no cover - provenance must not break packaging
        torch_details = {"torch_probe_error": repr(exc)}
    _write_json(
        {
            "python": sys.version,
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "packages": versions,
            **torch_details,
        },
        destination / "system.json",
    )
    freeze_code, freeze = _run([sys.executable, "-m", "pip", "freeze"])
    if freeze_code != 0:
        freeze = f"pip freeze failed with exit code {freeze_code}\n{freeze}\n"
    (destination / "python_environment.txt").write_text(freeze + "\n", encoding="utf-8")
    if command_log is not None:
        if not command_log.is_file():
            raise ValueError(f"command log does not exist: {command_log}")
        shutil.copy2(command_log, destination / "execution.log")


def _write_checksums(release_dir: Path) -> None:
    checksum_path = release_dir / "SHA256SUMS.txt"
    files = sorted(
        path for path in release_dir.rglob("*") if path.is_file() and path != checksum_path
    )
    lines = [f"{_sha256(path)}  {path.relative_to(release_dir).as_posix()}" for path in files]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _normalise_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    if info.isfile():
        info.mode = 0o755 if info.mode & 0o111 else 0o644
    elif info.isdir():
        info.mode = 0o755
    return info


def _write_archive(release_dir: Path, archive_path: Path) -> None:
    with archive_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                archive.add(
                    release_dir,
                    arcname=release_dir.name,
                    recursive=True,
                    filter=_normalise_tar_info,
                )


def _copy_aggregate(aggregate_dir: Path, destination: Path) -> dict:
    aggregate_dir = aggregate_dir.resolve()
    if not aggregate_dir.is_dir():
        raise ValueError(f"aggregate directory does not exist: {aggregate_dir}")
    if any(path.is_symlink() for path in aggregate_dir.rglob("*")):
        raise ValueError("aggregate directory must not contain symbolic links")
    if not (aggregate_dir / "manifest.json").is_file():
        raise ValueError("aggregate directory is missing manifest.json")
    shutil.copytree(
        aggregate_dir,
        destination,
        symlinks=False,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )
    manifest = json.loads((aggregate_dir / "manifest.json").read_text(encoding="utf-8"))
    return {
        "relative_path": destination.relative_to(destination.parents[1]).as_posix(),
        "aggregate_schema_version": manifest.get("aggregate_schema_version"),
        "n_independent_runs": manifest.get("n_independent_runs"),
        "unit_of_inference": manifest.get("unit_of_inference"),
        "analysis_plan_sha256": manifest.get("analysis_plan_sha256"),
        "analysis_plan_version": manifest.get("analysis_plan_version"),
        "publication_release_id": manifest.get("publication_release_id"),
    }


def _common_summary_value(summaries: list[dict], key: str):
    if not summaries:
        raise ValueError("at least one source run is required")
    values = [summary.get(key) for summary in summaries]
    canonical = [json.dumps(value, sort_keys=True) for value in values]
    if len(set(canonical)) != 1:
        raise ValueError(f"source runs disagree on {key}: {values!r}")
    return values[0]


def build_release(
    source_runs: list[Path],
    release_root: Path,
    release_id: str,
    release_kind: str,
    command_log: Path | None,
    aggregate_dir: Path | None = None,
) -> tuple[Path, Path]:
    if not _RELEASE_ID.fullmatch(release_id):
        raise ValueError("release_id must contain only letters, digits, '.', '_', and '-'")
    source_runs = [path.resolve() for path in source_runs]
    if len(set(source_runs)) != len(source_runs):
        raise ValueError("duplicate source-run path")
    release_root = release_root.resolve()
    release_root.mkdir(parents=True, exist_ok=True)
    final_dir = release_root / release_id
    temporary_dir = release_root / f".{release_id}.building"
    if final_dir.exists() or temporary_dir.exists():
        raise ValueError(f"release id already exists: {release_id}")

    summaries = [validate_source_run(path, fail_on_divergence=True) for path in source_runs]
    common_protocol = {
        key: _common_summary_value(summaries, key)
        for key in (
            "protocol_version",
            "scaling_modes",
            "primary_scaling_mode",
            "scale_reference_rank",
            "primary_allocation_rule",
            "primary_reference_rule",
            "primary_metric",
            "independent_unit",
            "exact_cost_matched_rules",
        )
    }
    if common_protocol["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("source-run protocol version does not match the release builder")
    if common_protocol["primary_scaling_mode"] not in common_protocol["scaling_modes"]:
        raise ValueError("primary scaling mode is absent from the scaling design")
    if common_protocol["primary_allocation_rule"] not in common_protocol[
        "exact_cost_matched_rules"
    ]:
        raise ValueError("primary allocation rule lacks an exact-cost comparator")
    if release_kind == "publication" and aggregate_dir is None:
        raise ValueError("publication releases require --aggregate-dir")

    temporary_dir.mkdir()
    try:
        source_root = temporary_dir / "source_runs"
        source_root.mkdir()
        manifest_runs = []
        seen_names: set[str] = set()
        for source, summary in zip(source_runs, summaries):
            name = source.name
            if name in seen_names:
                raise ValueError(f"duplicate source-run directory name: {name}")
            seen_names.add(name)
            relative = Path("source_runs") / name
            shutil.copytree(
                source,
                temporary_dir / relative,
                symlinks=False,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
            )
            manifest_runs.append(
                {
                    "relative_path": relative.as_posix(),
                    "protocol_version": summary["protocol_version"],
                    "task_name": summary["task_name"],
                    "base_seed": summary["base_seed"],
                    "scaling_modes": summary["scaling_modes"],
                    "primary_scaling_mode": summary["primary_scaling_mode"],
                    "scale_reference_rank": summary["scale_reference_rank"],
                    "primary_allocation_rule": summary["primary_allocation_rule"],
                    "primary_reference_rule": summary["primary_reference_rule"],
                    "primary_metric": summary["primary_metric"],
                    "independent_unit": summary["independent_unit"],
                    "n_sites": summary["n_sites"],
                    "n_sweep_rows": summary["n_sweep_rows"],
                    "n_budget_rows": summary["n_budget_rows"],
                    "n_allocation_rows": summary["n_allocation_rows"],
                    "adaptation_replicates": summary["adaptation_replicates"],
                    "null_bootstrap": summary["null_bootstrap"],
                    "null_quantile": summary["null_quantile"],
                    "null_uncertainty_resamples": summary["null_uncertainty_resamples"],
                    "exact_modular_evaluation": summary["exact_modular_evaluation"],
                    "exact_cost_matched_rules": summary["exact_cost_matched_rules"],
                }
            )

        aggregate_entry = None
        if aggregate_dir is not None:
            aggregate_entry = _copy_aggregate(
                aggregate_dir,
                temporary_dir / "aggregate" / aggregate_dir.resolve().name,
            )
            if (
                aggregate_entry.get("aggregate_schema_version")
                != TRANSFORMER_AGGREGATE_SCHEMA_VERSION
            ):
                raise ValueError("aggregate schema version is missing or stale")
            if aggregate_entry.get("publication_release_id") != release_id:
                raise ValueError(
                    "aggregate publication_release_id does not match --release-id"
                )
        _copy_code_snapshot(temporary_dir / "code_snapshot")
        _write_provenance(temporary_dir / "provenance", command_log)
        _write_json(
            {
                "release_id": release_id,
                "release_kind": release_kind,
                "release_schema_version": TRANSFORMER_RELEASE_SCHEMA_VERSION,
                "built_utc": datetime.now(timezone.utc).isoformat(),
                "protocol_version": PROTOCOL_VERSION,
                "checksum_algorithm": "sha256",
                "unit_of_inference": common_protocol["independent_unit"],
                "cluster_structure_note": (
                    "adaptation replicates and budgets are nested within each independent "
                    "task/base-model seed run"
                ),
                "primary_scaling_mode": common_protocol["primary_scaling_mode"],
                "sensitivity_scaling_modes": sorted(
                    set(common_protocol["scaling_modes"])
                    - {common_protocol["primary_scaling_mode"]}
                ),
                "scale_reference_rank": common_protocol["scale_reference_rank"],
                "primary_allocation_rule": common_protocol["primary_allocation_rule"],
                "primary_reference_rule": common_protocol["primary_reference_rule"],
                "primary_metric": common_protocol["primary_metric"],
                "exact_cost_matched_rules": common_protocol["exact_cost_matched_rules"],
                "source_runs": manifest_runs,
                "aggregate": aggregate_entry,
            },
            temporary_dir / "release_manifest.json",
        )
        _write_checksums(temporary_dir)
        temporary_dir.rename(final_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    validate_release(final_dir, expected_kind=release_kind)
    archive_path = release_root / f"{release_id}.tar.gz"
    if archive_path.exists():
        raise ValueError(f"archive already exists: {archive_path}")
    _write_archive(final_dir, archive_path)
    digest = _sha256(archive_path)
    sidecar = Path(str(archive_path) + ".sha256")
    sidecar.write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")
    latest_tmp = release_root / ".LATEST.tmp"
    latest_tmp.write_text(release_id + "\n", encoding="utf-8")
    latest_tmp.replace(release_root / "LATEST")
    print("Transformer release build: PASS")
    print(f"release_dir: {final_dir}")
    print(f"archive: {archive_path}")
    print(f"archive_sha256: {digest}")
    return final_dir, archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a self-contained transformer evidence release.")
    parser.add_argument("--run", action="append", required=True, dest="source_runs")
    parser.add_argument("--release-root", default="runs/transformer_releases")
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--kind", choices=("smoke", "publication"), required=True)
    parser.add_argument("--command-log", default=None)
    parser.add_argument("--aggregate-dir", default=None)
    args = parser.parse_args()
    build_release(
        [Path(value).resolve() for value in args.source_runs],
        Path(args.release_root),
        args.release_id,
        args.kind,
        Path(args.command_log).resolve() if args.command_log else None,
        Path(args.aggregate_dir).resolve() if args.aggregate_dir else None,
    )


if __name__ == "__main__":
    main()
