#!/usr/bin/env python3
"""Build and validate a portable real-model LoRA source-run release."""
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

from real_protocol import (  # noqa: E402
    RELEASE_MANIFEST_VERSION,
    canonical_json_sha256,
    sha256_file,
)
from validate_real_lora_release import validate_release  # noqa: E402
from validate_real_lora_run import validate_run  # noqa: E402

CODE_SNAPSHOT_FILES = [
    "run_real_lora_validation.py",
    "hf_gpu_smoke.py",
    "real_protocol.py",
    "spectral_metrics.py",
    "requirements.txt",
    "requirements-tested-rocm721.txt",
    "README.md",
    "scripts/run_protocol_smoke.sh",
    "scripts/validate_real_lora_run.py",
    "scripts/validate_real_lora_release.py",
    "scripts/build_real_lora_release.py",
    "tests/conftest.py",
    "tests/test_real_protocol.py",
    "tests/test_runner_protocol.py",
    "tests/test_spectral_metrics.py",
]


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
            value
            for value in directory.rglob("*")
            if value.is_file() and value != target
        )
    ]
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")


def normalized_tarinfo(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo:
    tarinfo.uid = 0
    tarinfo.gid = 0
    tarinfo.uname = ""
    tarinfo.gname = ""
    tarinfo.mtime = 0
    if tarinfo.isdir():
        tarinfo.mode = 0o755
    elif tarinfo.isfile():
        tarinfo.mode = 0o644
    return tarinfo


def write_deterministic_tar_gz(source_dir: Path, archive_path: Path) -> None:
    with archive_path.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9
        ) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                paths = [source_dir, *sorted(source_dir.rglob("*"))]
                for path in paths:
                    arcname = Path(source_dir.name) / path.relative_to(source_dir)
                    archive.add(
                        path,
                        arcname=arcname.as_posix(),
                        recursive=False,
                        filter=normalized_tarinfo,
                    )


def build_release(
    source_run: Path,
    release_root: Path,
    release_id: str | None,
    release_kind: str,
) -> tuple[Path, Path]:
    source_run = source_run.resolve()
    source_summary = validate_run(source_run, expected_kind=release_kind)
    release_id = release_id or str(source_summary["run_id"])
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
        shutil.copytree(source_run, staging / "source_run")
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

        freeze = command_output([sys.executable, "-m", "pip", "freeze"], ROOT)
        (staging / "environment").mkdir(parents=True, exist_ok=True)
        (staging / "environment" / "pip_freeze.txt").write_text(
            (freeze or "pip freeze unavailable") + "\n", encoding="utf-8"
        )
        git_commit = command_output(["git", "rev-parse", "HEAD"], PROJECT_ROOT)
        git_status = command_output(["git", "status", "--porcelain"], PROJECT_ROOT)
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
        (staging / "environment" / "environment.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        source_config_hash = sha256_file(staging / "source_run" / "config.json")
        manifest = {
            "schema_version": RELEASE_MANIFEST_VERSION,
            "release_id": release_id,
            "release_kind": release_kind,
            "source_run_id": source_summary["run_id"],
            "source_run_path": "source_run",
            "source_run_config_sha256": source_config_hash,
            "protocol_version": source_summary["protocol_version"],
            "code_snapshot_sha256": code_hashes,
            "code_snapshot_manifest_sha256": canonical_json_sha256(code_hashes),
            "git_commit": git_commit,
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
        sidecar_path.write_text(
            f"{archive_hash}  {archive_path.name}\n", encoding="utf-8"
        )
        (release_root / "LATEST").write_text(release_id + "\n", encoding="utf-8")
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)
        archive_path.unlink(missing_ok=True)
        sidecar_path.unlink(missing_ok=True)
        raise

    print("Real LoRA release build: PASS")
    print("release_dir:", final_dir)
    print("archive:", archive_path)
    print("archive_sha256:", sha256_file(archive_path))
    return final_dir, archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("--release-root", type=Path, default=ROOT / "runs" / "real_lora_releases")
    parser.add_argument("--release-id", default=None)
    parser.add_argument(
        "--kind", choices=["exploratory", "smoke", "publication"], required=True
    )
    args = parser.parse_args()
    build_release(args.source_run, args.release_root, args.release_id, args.kind)


if __name__ == "__main__":
    main()
