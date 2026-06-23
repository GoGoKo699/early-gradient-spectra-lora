#!/usr/bin/env python3
"""Run the frozen multi-seed real-model LoRA plan end to end."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any, Iterable

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in [ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from publication_protocol import sha256_file, validate_publication_plan  # noqa: E402
from validate_real_lora_run import validate_run  # noqa: E402

PROXY_KEYS = (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "NO_PROXY",
    "no_proxy",
)
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_clean_or_raise() -> str:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ValueError("the publication driver requires a Git worktree") from exc
    if status:
        raise ValueError(
            "the publication driver requires a clean Git worktree; git status --short:\n"
            + status
        )
    return head


def child_environment(parent: dict[str, str] | None = None) -> dict[str, str]:
    """Return an offline child environment without mutating the parent shell."""
    environment = dict(os.environ if parent is None else parent)
    for key in PROXY_KEYS:
        environment.pop(key, None)
    environment.update(OFFLINE_ENVIRONMENT)
    return environment


def _write_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip("\n") + "\n")


def _run_logged(command: list[str], log_path: Path, *, environment: dict[str, str]) -> None:
    rendered = shlex.join(command)
    print(f"$ {rendered}", flush=True)
    _write_line(log_path, f"$ {rendered}")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    with log_path.open("a", encoding="utf-8") as handle:
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"command failed with exit code {return_code}: {rendered}"
        )


def _append_flag(command: list[str], flag: str, enabled: object) -> None:
    if bool(enabled):
        command.append(flag)


def _source_run_command(
    *,
    validated_plan: dict[str, Any],
    suite: dict[str, Any],
    seed: int,
    release_id: str,
    run_root: Path,
) -> tuple[str, list[str]]:
    plan = validated_plan["plan"]
    model = validated_plan["model"]
    dataset = validated_plan["dataset"]
    run = validated_plan["run"]
    run_id = f"{release_id}_{suite['suite_id']}_seed{seed}"
    command = [
        sys.executable,
        "run_real_lora_validation.py",
        "--run-id",
        run_id,
        "--run-kind",
        validated_plan["release_kind"],
        "--out-dir",
        str(run_root),
        "--model",
        str(model["argument"]),
        "--dataset_mode",
        str(dataset["mode"]),
        "--seed",
        str(seed),
        "--target_suffixes",
        str(suite["target_suffixes"]),
        "--max_targets",
        str(suite["max_targets"]),
        "--publication-plan-version",
        str(plan["plan_version"]),
        "--publication-plan-sha256",
        str(validated_plan["sha256"]),
        "--publication-release-id",
        release_id,
        "--suite-id",
        str(suite["suite_id"]),
        "--suite-role",
        str(suite["role"]),
    ]
    if bool(model["local_files_only"]):
        command.append("--local_files_only")
    if dataset["mode"] == "local":
        command.extend(
            [
                "--train_text_file",
                str(dataset["train_text_file"]),
                "--val_text_file",
                str(dataset["val_text_file"]),
            ]
        )

    scalar_flags = {
        "block_size": "--block_size",
        "batch_size": "--batch_size",
        "calib_batches": "--calib_batches",
        "eval_batches": "--eval_batches",
        "steps": "--steps",
        "lr": "--lr",
        "weight_decay": "--weight_decay",
        "grad_clip": "--grad_clip",
        "uniform_rank": "--uniform_rank",
        "min_rank": "--min_rank",
        "max_rank": "--max_rank",
        "fisher_reference_rank": "--fisher_reference_rank",
        "ridge_scale": "--ridge_scale",
        "lora_alpha_scale": "--lora_alpha_scale",
        "lora_init_std": "--lora_init_std",
        "max_train_texts": "--max_train_texts",
        "max_val_texts": "--max_val_texts",
        "max_train_blocks": "--max_train_blocks",
        "max_val_blocks": "--max_val_blocks",
        "dtype": "--dtype",
        "device": "--device",
        "log_every": "--log_every",
        "identity_tolerance": "--identity_tolerance",
        "minimum_first_step_gradient_l2": "--minimum_first_step_gradient_l2",
        "minimum_adapter_delta_l2": "--minimum_adapter_delta_l2",
        "minimum_effective_update_l2": "--minimum_effective_update_l2",
    }
    for key, flag in scalar_flags.items():
        command.extend([flag, str(run[key])])
    command.extend(["--strategies", ",".join(run["strategies"])])
    _append_flag(command, "--include_identity_control", run["include_identity_control"])
    _append_flag(command, "--deterministic_algorithms", run["deterministic_algorithms"])
    return run_id, command


def _validate_reusable_source(
    run_dir: Path,
    *,
    validated_plan: dict[str, Any],
    suite: dict[str, Any],
    seed: int,
    release_id: str,
    source_git_commit: str,
) -> None:
    validate_run(run_dir, expected_kind=validated_plan["release_kind"])
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    expected = {
        "run_id": run_dir.name,
        "publication_plan_version": validated_plan["plan"]["plan_version"],
        "publication_plan_sha256": validated_plan["sha256"],
        "publication_release_id": release_id,
        "suite_id": suite["suite_id"],
        "suite_role": suite["role"],
        "seed": seed,
        "target_suffixes": suite["target_suffixes"],
        "max_targets": suite["max_targets"],
        "source_git_commit": source_git_commit,
        "source_git_status_porcelain": "",
    }
    for key, value in expected.items():
        require(config.get(key) == value, f"resume source {run_dir.name}: {key} mismatch")


def _quarantine_invalid_source(run_dir: Path, error: Exception) -> Path:
    """Preserve an invalid partial run before recreating its canonical path."""
    quarantine_root = run_dir.parent.parent / "real_lora_quarantine"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = quarantine_root / f"{run_dir.name}.invalid.{stamp}"
    counter = 1
    while destination.exists():
        destination = quarantine_root / f"{run_dir.name}.invalid.{stamp}.{counter}"
        counter += 1
    shutil.move(str(run_dir), str(destination))
    reason = destination / "QUARANTINE_REASON.txt"
    reason.write_text(
        f"source_run={run_dir.name}\nerror_type={type(error).__name__}\n"
        f"error={error}\n",
        encoding="utf-8",
    )
    print(
        f"resume: quarantined invalid source run {run_dir.name} at {destination}",
        flush=True,
    )
    return destination


def _verify_archive_sidecar(archive: Path) -> None:
    sidecar = Path(str(archive) + ".sha256")
    require(archive.is_file(), f"archive does not exist: {archive}")
    require(sidecar.is_file(), f"archive sidecar does not exist: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    require(
        len(fields) == 2 and fields[1] == archive.name,
        "archive SHA-256 sidecar has an invalid format",
    )
    import hashlib

    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    require(fields[0] == digest.hexdigest(), "archive SHA-256 mismatch")
    print(f"{archive.name}: OK", flush=True)


def _prepare_release_outputs_for_resume(
    final_release_dir: Path, archive: Path, *, resume: bool
) -> bool:
    sidecar = Path(str(archive) + ".sha256")
    paths = (final_release_dir, archive, sidecar)
    if not any(path.exists() for path in paths):
        return False
    if not resume:
        raise ValueError(f"release output already exists: {final_release_dir.name}")
    if final_release_dir.is_dir() and archive.is_file() and sidecar.is_file():
        return True
    if final_release_dir.exists():
        require(final_release_dir.is_dir(), f"release path is not a directory: {final_release_dir}")
        shutil.rmtree(final_release_dir)
    for path in [archive, sidecar]:
        if path.exists():
            require(path.is_file(), f"release artifact is not a file: {path}")
            path.unlink()
    print(f"resume: removed incomplete release outputs for {final_release_dir.name}", flush=True)
    return False


def _reuse_complete_release_if_valid(
    *,
    final_release_dir: Path,
    archive: Path,
    validated_plan: dict[str, Any],
    source_git_commit: str,
    resume: bool,
    master_log: Path,
    environment: dict[str, str],
) -> bool:
    sidecar = Path(str(archive) + ".sha256")
    if not (final_release_dir.is_dir() and archive.is_file() and sidecar.is_file()):
        return False
    if not resume:
        raise ValueError(f"release output already exists: {final_release_dir.name}")
    release_plan = final_release_dir / "analysis_plan.json"
    require(release_plan.is_file(), "existing release is missing analysis_plan.json")
    require(
        sha256_file(release_plan) == validated_plan["sha256"],
        "existing release is bound to a different analysis plan",
    )
    existing_manifest = json.loads(
        (final_release_dir / "release_manifest.json").read_text(encoding="utf-8")
    )
    require(
        existing_manifest.get("source_git_commit") == source_git_commit,
        "existing release was generated from a different Git commit",
    )
    _run_logged(
        [
            sys.executable,
            "-B",
            "scripts/validate_real_lora_publication_release.py",
            str(final_release_dir),
            "--expected-kind",
            validated_plan["release_kind"],
        ],
        master_log,
        environment=environment,
    )
    _verify_archive_sidecar(archive)
    print(f"resume: reusing complete validated release {final_release_dir.name}", flush=True)
    return True


def _preflight_command(validated_plan: dict[str, Any]) -> list[str]:
    if validated_plan["release_kind"] == "publication":
        return [sys.executable, "scripts/preflight_publication_rerun.py"]
    command = [
        sys.executable,
        "hf_gpu_smoke.py",
        "--model",
        str(validated_plan["model"]["argument"]),
    ]
    if validated_plan["model"]["local_files_only"]:
        command.append("--local-files-only")
    return command


def run_plan(
    plan_path: Path,
    *,
    release_id: str | None = None,
    resume: bool = False,
) -> Path:
    validated_plan = validate_publication_plan(plan_path, ROOT)
    git_head = _git_clean_or_raise()
    if release_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        release_id = f"{validated_plan['release_prefix']}_{timestamp}"
    require(
        bool(release_id)
        and all(ch.isalnum() or ch in "._-" for ch in str(release_id)),
        "release_id contains unsupported characters",
    )
    release_id = str(release_id)

    child_env = child_environment()
    run_root = ROOT / "runs" / "real_lora_source_runs"
    aggregate_dir = ROOT / "runs" / "real_lora_aggregates" / f"{release_id}_aggregate"
    release_root = ROOT / "runs" / "real_lora_releases"
    log_root = ROOT / "runs" / "real_lora_logs" / release_id
    master_log = log_root / "publication_driver.log"
    master_log.parent.mkdir(parents=True, exist_ok=True)
    if master_log.exists() and not resume:
        raise ValueError(f"driver log already exists: {master_log}")
    _write_line(master_log, f"release_id={release_id}")
    _write_line(master_log, f"release_kind={validated_plan['release_kind']}")
    _write_line(master_log, f"git_head={git_head}")
    _write_line(master_log, f"plan={validated_plan['path']}")
    _write_line(master_log, f"plan_sha256={validated_plan['sha256']}")
    _write_line(master_log, "child_network_mode=offline")
    _write_line(master_log, "child_proxy_environment=stripped")
    print(f"release_id: {release_id}", flush=True)
    print(f"release_kind: {validated_plan['release_kind']}", flush=True)
    print(f"plan_sha256: {validated_plan['sha256']}", flush=True)
    print(f"git_head: {git_head}", flush=True)
    print("child network mode: offline; parent VPN/proxy environment is unchanged", flush=True)

    final_release_dir = release_root / release_id
    archive = release_root / f"{release_id}.tar.gz"
    if _reuse_complete_release_if_valid(
        final_release_dir=final_release_dir,
        archive=archive,
        validated_plan=validated_plan,
        source_git_commit=git_head,
        resume=resume,
        master_log=master_log,
        environment=child_env,
    ):
        print("Real LoRA publication driver: PASS", flush=True)
        print(f"ARCHIVE={archive}", flush=True)
        return archive

    _run_logged(_preflight_command(validated_plan), master_log, environment=child_env)

    source_runs: list[Path] = []
    for suite in validated_plan["suites"]:
        for seed in suite["seeds"]:
            run_id, command = _source_run_command(
                validated_plan=validated_plan,
                suite=suite,
                seed=int(seed),
                release_id=release_id,
                run_root=run_root,
            )
            run_dir = run_root / run_id
            reuse_source = False
            if run_dir.exists():
                if not resume:
                    raise ValueError(f"source run already exists: {run_dir}")
                try:
                    _validate_reusable_source(
                        run_dir,
                        validated_plan=validated_plan,
                        suite=suite,
                        seed=int(seed),
                        release_id=release_id,
                        source_git_commit=git_head,
                    )
                    reuse_source = True
                except Exception as exc:
                    _quarantine_invalid_source(run_dir, exc)
            if reuse_source:
                print(f"resume: reusing validated source run {run_id}", flush=True)
            else:
                _run_logged(command, master_log, environment=child_env)
                _run_logged(
                    [
                        sys.executable,
                        "-B",
                        "scripts/validate_real_lora_run.py",
                        str(run_dir),
                        "--expected-kind",
                        validated_plan["release_kind"],
                    ],
                    master_log,
                    environment=child_env,
                )
                _validate_reusable_source(
                    run_dir,
                    validated_plan=validated_plan,
                    suite=suite,
                    seed=int(seed),
                    release_id=release_id,
                    source_git_commit=git_head,
                )
            source_runs.append(run_dir)

    if aggregate_dir.exists():
        if not resume:
            raise ValueError(f"aggregate directory already exists: {aggregate_dir}")
        shutil.rmtree(aggregate_dir)
    aggregate_command = [
        sys.executable,
        "-B",
        "scripts/aggregate_real_lora_publication.py",
        "--analysis-plan",
        str(validated_plan["path"]),
        "--out",
        str(aggregate_dir),
        "--release-id",
        release_id,
    ]
    for run_dir in source_runs:
        aggregate_command.extend(["--run", str(run_dir)])
    _run_logged(aggregate_command, master_log, environment=child_env)

    if _prepare_release_outputs_for_resume(final_release_dir, archive, resume=resume):
        _run_logged(
            [
                sys.executable,
                "-B",
                "scripts/validate_real_lora_publication_release.py",
                str(final_release_dir),
                "--expected-kind",
                validated_plan["release_kind"],
            ],
            master_log,
            environment=child_env,
        )
        _verify_archive_sidecar(archive)
        print("Real LoRA publication driver: PASS", flush=True)
        print(f"ARCHIVE={archive}", flush=True)
        return archive

    build_command = [
        sys.executable,
        "-B",
        "scripts/build_real_lora_publication_release.py",
        "--aggregate-dir",
        str(aggregate_dir),
        "--analysis-plan",
        str(validated_plan["path"]),
        "--release-root",
        str(release_root),
        "--release-id",
        release_id,
        "--kind",
        validated_plan["release_kind"],
        "--command-log",
        str(master_log),
    ]
    for run_dir in source_runs:
        build_command.extend(["--run", str(run_dir)])
    _run_logged(build_command, master_log, environment=child_env)
    _run_logged(
        [
            sys.executable,
            "-B",
            "scripts/validate_real_lora_publication_release.py",
            str(final_release_dir),
            "--expected-kind",
            validated_plan["release_kind"],
        ],
        master_log,
        environment=child_env,
    )
    _verify_archive_sidecar(archive)
    print("Real LoRA publication driver: PASS", flush=True)
    print(f"ARCHIVE={archive}", flush=True)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--release-id", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only source runs that pass the complete source-run validator.",
    )
    parser.add_argument(
        "--validate-plan-only",
        action="store_true",
        help="Validate the frozen plan without running a model.",
    )
    args = parser.parse_args()
    plan_path = Path(args.plan)
    if not plan_path.is_absolute():
        plan_path = ROOT / plan_path
    if args.validate_plan_only:
        validated = validate_publication_plan(plan_path, ROOT)
        print("Real LoRA publication plan validation: PASS")
        print(f"release_kind: {validated['release_kind']}")
        print(
            "suites:",
            [
                {
                    "suite_id": suite["suite_id"],
                    "role": suite["role"],
                    "seeds": suite["seeds"],
                }
                for suite in validated["suites"]
            ],
        )
        print(f"n_source_runs: {validated['n_source_runs']}")
        print(f"plan_sha256: {validated['sha256']}")
        return
    run_plan(plan_path, release_id=args.release_id, resume=args.resume)


if __name__ == "__main__":
    main()
