#!/usr/bin/env python3
"""Verify byte-for-byte reproducibility of publication artifacts.

The verifier builds in two independent temporary copies, compares the two
builds to each other, and compares both with hashes in
``publication-toolchain.lock.json``. The canonical full check is:

    python Paper/verify_publication_reproducibility.py --scope full

Use ``--scope python`` to check generated tables and Matplotlib PDFs without a
TeX installation. ``--allow-python-version-mismatch`` is diagnostic only; the
canonical release gate remains strict CPython 3.12.3 on linux/amd64.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
LOCK_PATH = PAPER / "publication-toolchain.lock.json"
BUILD_DEBRIS = (
    "*.aux",
    "*.fdb_latexmk",
    "*.fls",
    "*.log",
    "*.out",
    "*.synctex.gz",
    "*.toc",
)


class VerificationError(RuntimeError):
    """Raised when a publication reproducibility invariant fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_lock() -> dict[str, Any]:
    try:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read toolchain lock {LOCK_PATH}: {exc}") from exc
    if lock.get("schema_version") != 1:
        raise VerificationError("unsupported publication toolchain lock schema")
    return lock


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        output = completed.stdout[-12000:]
        raise VerificationError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{output}"
        )
    return completed.stdout


def _check_python_environment(
    lock: dict[str, Any], *, allow_python_version_mismatch: bool
) -> list[str]:
    environment = lock["environment"]
    expected_python = str(environment["python"])
    observed_python = platform.python_version()
    notices: list[str] = []
    if observed_python != expected_python:
        message = f"Python {observed_python} is active; lock requires {expected_python}"
        if allow_python_version_mismatch:
            notices.append("diagnostic override: " + message)
        else:
            raise VerificationError(message)

    observed_machine = platform.machine().lower()
    if observed_machine not in {"x86_64", "amd64"}:
        raise VerificationError(
            f"architecture {observed_machine!r} is not the locked linux/amd64 target"
        )

    mismatches: list[str] = []
    for distribution, expected in environment["packages"].items():
        try:
            observed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(f"{distribution}: missing (expected {expected})")
            continue
        if observed != str(expected):
            mismatches.append(f"{distribution}: {observed} (expected {expected})")
    if mismatches:
        raise VerificationError(
            "Python package lock mismatch:\n  " + "\n  ".join(mismatches)
        )
    return notices


def _check_tex_environment(lock: dict[str, Any]) -> str:
    executable = shutil.which("pdflatex")
    if executable is None:
        raise VerificationError("pdflatex is missing; use Dockerfile.publication")
    completed = subprocess.run(
        [executable, "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    banner = completed.stdout.splitlines()[0] if completed.stdout else ""
    expected = str(lock["environment"]["tex"]["version_substring"])
    if completed.returncode != 0 or expected not in completed.stdout:
        raise VerificationError(
            f"TeX engine mismatch: {banner!r}; required substring {expected!r}"
        )
    return banner


def _copy_source(destination: Path) -> Path:
    ignored_names = {
        ".git",
        ".matplotlib-cache",
        ".pytest_cache",
        ".venv",
        "__pycache__",
    }

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored = {name for name in names if name in ignored_names}
        if Path(directory).resolve() == (ROOT / "Paper").resolve():
            ignored.add("paper.pdf")
        return ignored

    target = destination / "source"
    shutil.copytree(ROOT, target, ignore=ignore)
    for pattern in BUILD_DEBRIS:
        for path in target.rglob(pattern):
            path.unlink()
    return target


def _deterministic_environment(
    lock: dict[str, Any], *, workspace: Path
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {str(key): str(value) for key, value in lock["deterministic_environment"].items()}
    )
    env["HOME"] = str(workspace / "home")
    env["MPLCONFIGDIR"] = str(workspace / "matplotlib")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    (workspace / "home").mkdir(parents=True, exist_ok=True)
    (workspace / "matplotlib").mkdir(parents=True, exist_ok=True)
    return env


def _remove_expected_outputs(source: Path, lock: dict[str, Any], *, full: bool) -> None:
    for relative in lock["artifacts"]["python_generated"]:
        path = source / relative
        if path.exists():
            path.unlink()
    paper = source / lock["artifacts"]["paper"]["path"]
    if full and paper.exists():
        paper.unlink()


def _build_once(
    source: Path,
    lock: dict[str, Any],
    *,
    full: bool,
    workspace: Path,
) -> dict[str, str]:
    env = _deterministic_environment(lock, workspace=workspace)
    paper_dir = source / "Paper"
    _remove_expected_outputs(source, lock, full=full)

    _run([sys.executable, "make_paper_artifacts.py"], cwd=paper_dir, env=env)

    observed: dict[str, str] = {}
    for relative in lock["artifacts"]["python_generated"]:
        path = source / relative
        if not path.is_file():
            raise VerificationError(f"build did not create {relative}")
        observed[relative] = _sha256(path)

    if full:
        for _ in range(2):
            _run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "paper.tex",
                ],
                cwd=paper_dir,
                env=env,
            )
        built_paper = paper_dir / "paper.pdf"
        if not built_paper.is_file():
            raise VerificationError("TeX build did not create Paper/paper.pdf")
        relative = str(lock["artifacts"]["paper"]["path"])
        observed[relative] = _sha256(built_paper)
    return observed


def _expected_hashes(lock: dict[str, Any], *, full: bool) -> dict[str, str]:
    expected = {
        str(path): str(digest)
        for path, digest in lock["artifacts"]["python_generated"].items()
    }
    if full:
        paper = lock["artifacts"]["paper"]
        expected[str(paper["path"])] = str(paper["sha256"])
    return expected


def _compare_hashes(
    label: str, observed: dict[str, str], expected: dict[str, str]
) -> None:
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    mismatched = sorted(
        path for path in set(observed) & set(expected) if observed[path] != expected[path]
    )
    if not (missing or extra or mismatched):
        return
    lines = [f"{label} hash verification failed"]
    if missing:
        lines.append("missing: " + ", ".join(missing))
    if extra:
        lines.append("unexpected: " + ", ".join(extra))
    for path in mismatched:
        lines.append(
            f"{path}: observed {observed[path]}, expected {expected[path]}"
        )
    raise VerificationError("\n".join(lines))


def _verify_pdf_quality(source: Path, lock: dict[str, Any], *, full: bool) -> None:
    pdffonts = shutil.which("pdffonts")
    if pdffonts is None:
        return
    pdfs = [source / path for path in lock["artifacts"]["python_generated"] if path.endswith(".pdf")]
    if full:
        pdfs.append(source / "Paper" / "paper.pdf")
    for pdf in pdfs:
        completed = subprocess.run(
            [pdffonts, str(pdf)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode != 0:
            raise VerificationError(f"pdffonts failed for {pdf}:\n{completed.stdout}")
        rows = completed.stdout.splitlines()[2:]
        if any("Type 3" in row for row in rows):
            raise VerificationError(f"Type 3 font detected in {pdf}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=("python", "full"),
        default="full",
        help="check Python-generated artifacts only, or include the manuscript PDF",
    )
    parser.add_argument(
        "--allow-python-version-mismatch",
        action="store_true",
        help="diagnostic override; package versions and artifact hashes remain strict",
    )
    parser.add_argument(
        "--keep-workspaces",
        action="store_true",
        help="retain temporary clean source copies for inspection",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    temp_root: Path | None = None
    try:
        lock = _load_lock()
        notices = _check_python_environment(
            lock, allow_python_version_mismatch=args.allow_python_version_mismatch
        )
        tex_banner = None
        full = args.scope == "full"
        if full:
            tex_banner = _check_tex_environment(lock)

        temp_root = Path(tempfile.mkdtemp(prefix="publication-repro-"))
        results: list[dict[str, str]] = []
        sources: list[Path] = []
        for index in (1, 2):
            workspace = temp_root / f"build-{index}"
            workspace.mkdir(parents=True)
            source = _copy_source(workspace)
            sources.append(source)
            results.append(
                _build_once(
                    source,
                    lock,
                    full=full,
                    workspace=workspace,
                )
            )

        expected = _expected_hashes(lock, full=full)
        _compare_hashes("clean build 1", results[0], expected)
        _compare_hashes("clean build 2", results[1], expected)
        _compare_hashes("independent clean builds", results[0], results[1])
        _verify_pdf_quality(sources[0], lock, full=full)

        for notice in notices:
            print(f"WARNING: {notice}")
        if tex_banner:
            print(f"TeX: {tex_banner}")
        print(
            "publication reproducibility: PASS "
            f"({len(expected)} artifacts; two independent clean builds; scope={args.scope})"
        )
        if args.keep_workspaces:
            print(f"workspaces retained at {temp_root}")
        return 0
    except VerificationError as exc:
        print(f"publication reproducibility: FAIL\n{exc}", file=sys.stderr)
        if args.keep_workspaces and temp_root is not None:
            print(f"workspaces retained at {temp_root}", file=sys.stderr)
        return 1
    finally:
        if not args.keep_workspaces and temp_root is not None:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
