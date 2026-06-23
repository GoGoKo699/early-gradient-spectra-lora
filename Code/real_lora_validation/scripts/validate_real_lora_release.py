#!/usr/bin/env python3
"""Validate a packaged controlled real-model LoRA release directory."""
from __future__ import annotations

import argparse
import json
import sys

sys.dont_write_bytecode = True
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from real_protocol import RELEASE_MANIFEST_VERSION, sha256_file  # noqa: E402
from validate_real_lora_run import validate_run  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_release_checksums(release_dir: Path) -> int:
    manifest_path = release_dir / "SHA256SUMS.txt"
    require(manifest_path.is_file(), "missing release SHA256SUMS.txt")
    recorded: set[str] = set()
    count = 0
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        require(len(parts) == 2, f"invalid release checksum line {line_number}")
        expected, relative = parts
        require(relative not in recorded, f"duplicate release checksum path {relative}")
        recorded.add(relative)
        path = release_dir / relative
        require(path.is_file(), f"release file missing: {relative}")
        require(
            sha256_file(path) == expected,
            f"release checksum mismatch for {relative}",
        )
        count += 1
    actual = {
        path.relative_to(release_dir).as_posix()
        for path in release_dir.rglob("*")
        if path.is_file() and path != manifest_path
    }
    require(
        recorded == actual,
        "release checksum coverage mismatch: "
        f"missing={sorted(actual - recorded)}, stale={sorted(recorded - actual)}",
    )
    return count


def validate_release(
    release_dir: Path, expected_kind: str | None = None
) -> dict[str, Any]:
    release_dir = release_dir.resolve()
    require(release_dir.is_dir(), f"release directory does not exist: {release_dir}")
    checksum_count = verify_release_checksums(release_dir)
    manifest = json.loads((release_dir / "release_manifest.json").read_text(encoding="utf-8"))
    require(
        manifest.get("schema_version") == RELEASE_MANIFEST_VERSION,
        "release schema mismatch",
    )
    kind = str(manifest.get("release_kind"))
    if expected_kind is not None:
        require(kind == expected_kind, f"release kind {kind!r} != {expected_kind!r}")
    source_rel = manifest.get("source_run_path")
    require(isinstance(source_rel, str), "release manifest has no source_run_path")
    source_summary = validate_run(release_dir / source_rel, expected_kind=kind)

    code_hashes = manifest.get("code_snapshot_sha256")
    require(isinstance(code_hashes, dict) and code_hashes, "release code snapshot is empty")
    for relative, expected in code_hashes.items():
        path = release_dir / "code_snapshot" / relative
        require(path.is_file(), f"missing code snapshot file {relative}")
        require(sha256_file(path) == expected, f"code snapshot hash mismatch for {relative}")

    summary = {
        "release_id": manifest["release_id"],
        "release_kind": kind,
        "source_run_id": source_summary["run_id"],
        "protocol_version": source_summary["protocol_version"],
        "n_source_runs": 1,
        "n_checksummed_files": checksum_count,
    }
    print("Real LoRA release validation: PASS")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_dir", type=Path)
    parser.add_argument(
        "--expected-kind", choices=["exploratory", "smoke", "publication"], default=None
    )
    args = parser.parse_args()
    validate_release(args.release_dir, expected_kind=args.expected_kind)


if __name__ == "__main__":
    main()
