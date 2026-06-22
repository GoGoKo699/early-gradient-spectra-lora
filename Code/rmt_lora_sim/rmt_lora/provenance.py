from __future__ import annotations

"""Small provenance helpers for publication release bundles."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import platform
import subprocess
import sys

import matplotlib
import numpy as np
import pandas as pd
import scipy
import threadpoolctl
import yaml


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state(start: str | Path) -> dict[str, Any]:
    """Return git revision and cleanliness without raising outside a repository."""

    start_path = Path(start).resolve()
    if start_path.is_file():
        start_path = start_path.parent

    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(start_path), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return completed.stdout.strip()

    try:
        root = Path(run("rev-parse", "--show-toplevel"))
        revision = run("rev-parse", "HEAD")
        status = run("status", "--porcelain")
        return {
            "git_root": str(root),
            "git_revision": revision,
            "git_dirty": bool(status),
            "git_status_porcelain": status.splitlines(),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {
            "git_root": None,
            "git_revision": None,
            "git_dirty": None,
            "git_status_porcelain": [],
        }


def runtime_versions() -> dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "threadpoolctl": threadpoolctl.__version__,
        "pyyaml": yaml.__version__,
    }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_sha256s(
    root: str | Path,
    *,
    output_name: str = "SHA256SUMS.txt",
    exclude_names: tuple[str, ...] = ("SHA256SUMS.txt",),
) -> Path:
    """Hash every regular file below ``root`` using portable relative paths."""

    root_path = Path(root).resolve()
    output = root_path / output_name
    excluded = set(exclude_names) | {output_name}
    files = sorted(
        p for p in root_path.rglob("*") if p.is_file() and p.name not in excluded
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(root_path).as_posix()}" for path in files]
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output


def verify_sha256s(
    root: str | Path,
    *,
    checksum_name: str = "SHA256SUMS.txt",
    reject_unlisted: bool = False,
) -> list[str]:
    """Return checksum errors; optionally reject files absent from the manifest."""

    root_path = Path(root).resolve()
    checksum_path = root_path / checksum_name
    if not checksum_path.exists():
        return [f"missing checksum file: {checksum_path}"]
    errors: list[str] = []
    listed: set[str] = set()
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"line {line_number}: malformed checksum entry")
            continue
        if relative in listed:
            errors.append(f"line {line_number}: duplicate checksum entry for {relative}")
            continue
        listed.add(relative)
        path = root_path / relative
        if not path.is_file():
            errors.append(f"line {line_number}: missing file {relative}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            errors.append(
                f"line {line_number}: checksum mismatch for {relative}: "
                f"expected {expected}, got {actual}"
            )
    if reject_unlisted:
        actual = {
            path.relative_to(root_path).as_posix()
            for path in root_path.rglob("*")
            if path.is_file() and path.name != checksum_name
        }
        for relative in sorted(actual - listed):
            errors.append(f"unlisted file: {relative}")
        for relative in sorted(listed - actual):
            # Missing files are already reported above; avoid duplicate messages.
            if not (root_path / relative).is_file():
                continue
    return errors
