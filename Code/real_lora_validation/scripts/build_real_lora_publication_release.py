#!/usr/bin/env python3
"""Build a portable multi-source real-model LoRA publication release."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import shutil
import subprocess
import sys
sys.dont_write_bytecode = True
import tarfile
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in [ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from publication_protocol import (  # noqa: E402
    PUBLICATION_CODE_SNAPSHOT_FILES,
    PUBLICATION_RELEASE_VERSION,
    sha256_file,
    validate_publication_plan,
)
from real_protocol import canonical_json_sha256  # noqa: E402
from validate_real_lora_publication_release import validate_release  # noqa: E402
from validate_real_lora_run import validate_run  # noqa: E402

CODE_SNAPSHOT_FILES = list(PUBLICATION_CODE_SNAPSHOT_FILES)


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def command_output(command: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_checksums(directory: Path) -> None:
    target = directory / "SHA256SUMS.txt"
    rows = [
        f"{sha256_file(path)}  {path.relative_to(directory).as_posix()}"
        for path in sorted(
            item for item in directory.rglob("*") if item.is_file() and item != target
        )
    ]
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")


def normalized_tarinfo(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo:
    tarinfo.uid = 0
    tarinfo.gid = 0
    tarinfo.uname = ""
    tarinfo.gname = ""
    tarinfo.mtime = 0
    tarinfo.mode = 0o755 if tarinfo.isdir() else 0o644
    return tarinfo


def write_deterministic_tar_gz(source_dir: Path, archive_path: Path) -> None:
    with archive_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in [source_dir, *sorted(source_dir.rglob("*"))]:
                    arcname = Path(source_dir.name) / path.relative_to(source_dir)
                    archive.add(
                        path,
                        arcname=arcname.as_posix(),
                        recursive=False,
                        filter=normalized_tarinfo,
                    )


def verify_aggregate_checksums(aggregate_dir: Path) -> int:
    manifest = aggregate_dir / "AGGREGATE_SHA256SUMS.txt"
    if not manifest.is_file():
        raise ValueError("aggregate lacks AGGREGATE_SHA256SUMS.txt")
    recorded: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        if relative in recorded:
            raise ValueError(f"duplicate aggregate checksum path: {relative}")
        recorded.add(relative)
        path = aggregate_dir / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"aggregate checksum mismatch: {relative}")
    actual = {
        path.relative_to(aggregate_dir).as_posix()
        for path in aggregate_dir.rglob("*")
        if path.is_file() and path != manifest
    }
    if actual != recorded:
        raise ValueError(
            f"aggregate checksum coverage mismatch: missing={sorted(actual-recorded)}, stale={sorted(recorded-actual)}"
        )
    return len(recorded)


def build_release(
    *,
    run_dirs: list[Path],
    aggregate_dir: Path,
    analysis_plan: Path,
    release_root: Path,
    release_id: str,
    release_kind: str,
    command_log: Path | None,
) -> tuple[Path, Path]:
    validated_plan = validate_publication_plan(analysis_plan, ROOT)
    if validated_plan["release_kind"] != release_kind:
        raise ValueError("release kind differs from analysis plan")
    if not release_id or any(not (ch.isalnum() or ch in "._-") for ch in release_id):
        raise ValueError("release_id contains unsupported characters")
    aggregate_dir = aggregate_dir.resolve()
    verify_aggregate_checksums(aggregate_dir)
    aggregate_manifest = json.loads((aggregate_dir / "manifest.json").read_text(encoding="utf-8"))
    if aggregate_manifest.get("release_id") != release_id:
        raise ValueError("aggregate release_id mismatch")
    if aggregate_manifest.get("analysis_plan_sha256") != validated_plan["sha256"]:
        raise ValueError("aggregate analysis-plan binding mismatch")

    run_dirs = [Path(path).resolve() for path in run_dirs]
    if len(run_dirs) != validated_plan["n_source_runs"] or len(run_dirs) != len(set(run_dirs)):
        raise ValueError("source run count/uniqueness differs from plan")
    source_summaries = [validate_run(path, expected_kind=release_kind) for path in run_dirs]
    source_ids = [str(summary["run_id"]) for summary in source_summaries]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("duplicate source run id")
    source_configs = [
        json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        for run_dir in run_dirs
    ]
    source_commits = {str(config.get("source_git_commit", "")) for config in source_configs}
    if len(source_commits) != 1:
        raise ValueError("source runs do not share one Git commit")
    source_commit = next(iter(source_commits))
    git_commit = command_output(["git", "rev-parse", "HEAD"], PROJECT_ROOT)
    git_status = command_output(["git", "status", "--porcelain"], PROJECT_ROOT)
    if not git_commit or git_commit != source_commit:
        raise ValueError(
            "release code Git commit does not match the source-run Git commit"
        )
    if git_status is None or git_status:
        raise ValueError("controlled release requires a clean Git worktree")
    if aggregate_manifest.get("source_git_commit") != source_commit:
        raise ValueError("aggregate/source Git commit mismatch")

    release_root = release_root.resolve()
    release_root.mkdir(parents=True, exist_ok=True)
    final_dir = release_root / release_id
    archive_path = release_root / f"{release_id}.tar.gz"
    sidecar_path = Path(f"{archive_path}.sha256")
    for path in [final_dir, archive_path, sidecar_path]:
        if path.exists():
            raise FileExistsError(f"release output already exists: {path}")

    staging = Path(tempfile.mkdtemp(prefix=f".{release_id}.", dir=release_root))
    try:
        source_root = staging / "source_runs"
        for run_dir, run_id in sorted(zip(run_dirs, source_ids), key=lambda pair: pair[1]):
            shutil.copytree(run_dir, source_root / run_id)
        shutil.copytree(aggregate_dir, staging / "aggregate")
        shutil.copy2(validated_plan["path"], staging / "analysis_plan.json")
        if validated_plan["input_manifest_path"] is not None:
            shutil.copy2(validated_plan["input_manifest_path"], staging / "input_manifest.json")
        if command_log is not None:
            command_log = command_log.resolve()
            if not command_log.is_file():
                raise FileNotFoundError(f"command log does not exist: {command_log}")
            shutil.copy2(command_log, staging / "publication_driver.log")

        snapshot = staging / "code_snapshot"
        code_hashes: dict[str, str] = {}
        for relative in CODE_SNAPSHOT_FILES:
            source = ROOT / relative
            if not source.is_file():
                raise FileNotFoundError(f"code snapshot input missing: {source}")
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            code_hashes[relative] = sha256_file(destination)

        environment_dir = staging / "environment"
        environment_dir.mkdir(parents=True)
        freeze = command_output([sys.executable, "-m", "pip", "freeze"], ROOT)
        (environment_dir / "pip_freeze.txt").write_text(
            (freeze or "pip freeze unavailable") + "\n", encoding="utf-8"
        )
        environment: dict[str, Any] = {
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "packages": {
                name: package_version(name)
                for name in [
                    "torch",
                    "numpy",
                    "scipy",
                    "pandas",
                    "transformers",
                    "datasets",
                    "accelerate",
                    "tokenizers",
                    "safetensors",
                ]
            },
            "git_commit": git_commit,
            "git_status_porcelain": git_status,
        }
        (environment_dir / "environment.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        source_entries: list[dict[str, Any]] = []
        for run_id in sorted(source_ids):
            copied = source_root / run_id
            config = json.loads((copied / "config.json").read_text(encoding="utf-8"))
            source_entries.append(
                {
                    "run_id": run_id,
                    "suite_id": config["suite_id"],
                    "suite_role": config["suite_role"],
                    "seed": int(config["seed"]),
                    "path": f"source_runs/{run_id}",
                    "config_sha256": sha256_file(copied / "config.json"),
                    "run_checksums_sha256": sha256_file(copied / "RUN_SHA256SUMS.txt"),
                    "source_git_commit": str(config["source_git_commit"]),
                }
            )
        manifest = {
            "schema_version": PUBLICATION_RELEASE_VERSION,
            "release_id": release_id,
            "release_kind": release_kind,
            "protocol_version": validated_plan["plan"]["protocol_version"],
            "analysis_plan_version": validated_plan["plan"]["plan_version"],
            "analysis_plan_sha256": sha256_file(staging / "analysis_plan.json"),
            "aggregate_manifest_sha256": sha256_file(staging / "aggregate" / "manifest.json"),
            "aggregate_checksums_sha256": sha256_file(staging / "aggregate" / "AGGREGATE_SHA256SUMS.txt"),
            "n_source_runs": len(source_entries),
            "source_runs": source_entries,
            "source_run_set_sha256": canonical_json_sha256(source_entries),
            "code_snapshot_sha256": code_hashes,
            "code_snapshot_manifest_sha256": canonical_json_sha256(code_hashes),
            "git_commit": git_commit,
            "source_git_commit": source_commit,
        }
        (staging / "release_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_checksums(staging)
        os.replace(staging, final_dir)
        validate_release(final_dir, expected_kind=release_kind)
        write_deterministic_tar_gz(final_dir, archive_path)
        archive_hash = sha256_file(archive_path)
        sidecar_path.write_text(f"{archive_hash}  {archive_path.name}\n", encoding="utf-8")
        (release_root / "LATEST").write_text(release_id + "\n", encoding="utf-8")
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)
        archive_path.unlink(missing_ok=True)
        sidecar_path.unlink(missing_ok=True)
        raise

    print("Real LoRA publication release build: PASS")
    print("release_dir:", final_dir)
    print("archive:", archive_path)
    print("archive_sha256:", sha256_file(archive_path))
    return final_dir, archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, dest="runs")
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--analysis-plan", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, default=ROOT / "runs" / "real_lora_releases")
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--kind", choices=["smoke", "publication"], required=True)
    parser.add_argument("--command-log", type=Path, default=None)
    args = parser.parse_args()
    plan = args.analysis_plan if args.analysis_plan.is_absolute() else ROOT / args.analysis_plan
    build_release(
        run_dirs=[Path(value) for value in args.runs],
        aggregate_dir=args.aggregate_dir,
        analysis_plan=plan,
        release_root=args.release_root,
        release_id=args.release_id,
        release_kind=args.kind,
        command_log=args.command_log,
    )


if __name__ == "__main__":
    main()
