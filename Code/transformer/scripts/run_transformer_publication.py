#!/usr/bin/env python3
"""Run a pre-specified multi-task transformer evidence plan end to end."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strank.protocol import (  # noqa: E402
    SYNTHETIC_TRANSFORMER_PROTOCOL_VERSION,
    validate_synthetic_transformer_config,
)

PLAN_VERSION = "transformer_publication_plan_v1"
EXPECTED_SCALING_MODES = {"fixed_update_scale", "standard", "rslora"}
PUBLICATION_MIN_TASKS = 2
PUBLICATION_MIN_RUNS_PER_TASK = 5
PUBLICATION_MIN_NULL_MAXIMA = 4096
PUBLICATION_MIN_NULL_UNCERTAINTY_RESAMPLES = 2000
PUBLICATION_PRIMARY_NULL_QUANTILE = 0.995


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def _safe_package_path(value: str, label: str) -> Path:
    pure = PurePosixPath(str(value))
    if pure.is_absolute() or not pure.parts or "." in pure.parts or ".." in pure.parts:
        raise ValueError(f"unsafe {label}: {value!r}")
    path = ROOT.joinpath(*pure.parts).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the transformer package: {value!r}") from exc
    return path


def validate_publication_plan(plan_path: Path) -> dict[str, Any]:
    """Validate the plan and every task template before any experiment starts."""

    plan_path = plan_path.resolve()
    plan = _load_yaml(plan_path)
    if plan.get("plan_version") != PLAN_VERSION:
        raise ValueError(f"plan_version must be {PLAN_VERSION}")
    if plan.get("protocol_version") != SYNTHETIC_TRANSFORMER_PROTOCOL_VERSION:
        raise ValueError("publication plan protocol version is stale")
    release_kind = str(plan.get("release_kind", ""))
    if release_kind not in {"smoke", "publication"}:
        raise ValueError("release_kind must be smoke or publication")
    prefix = str(plan.get("release_prefix", ""))
    if not prefix or not all(ch.isalnum() or ch in "._-" for ch in prefix):
        raise ValueError("release_prefix contains unsupported characters")

    seeds = [int(value) for value in plan.get("seeds", [])]
    if not seeds or len(set(seeds)) != len(seeds) or seeds != sorted(seeds):
        raise ValueError("plan seeds must be a non-empty sorted unique integer list")
    if any(seed < 0 for seed in seeds):
        raise ValueError("plan seeds must be non-negative")

    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("plan tasks must be a non-empty list")
    task_names: list[str] = []
    templates: list[dict[str, Any]] = []
    analysis = plan.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("plan analysis must be a mapping")
    required_analysis = {
        "primary_scope": "task_stratified_omnibus",
        "task_weighting": "equal_weight_across_task_families",
        "generalization_scope": "conditional_on_prespecified_task_families",
        "secondary_task_specific": True,
        "primary_scaling_mode": "fixed_update_scale",
        "primary_allocation_rule": "soft_dimension",
        "primary_reference_rule": "uniform_exact_cost",
        "primary_metric": "final_val_loss",
        "independent_unit": "task_seed_run",
        "budget_weighting": "equal_weight_within_run",
        "scaling_conditions_analyzed_separately": True,
        "exact_sign_flip_test": "two_sided_task_stratified",
    }
    for key, expected in required_analysis.items():
        if analysis.get(key) != expected:
            raise ValueError(
                f"analysis.{key}={analysis.get(key)!r}; expected {expected!r}"
            )
    if not str(analysis.get("primary_estimand", "")).strip():
        raise ValueError("analysis.primary_estimand must be declared")
    if abs(float(analysis.get("two_sided_alpha", 0.0)) - 0.05) > 1e-15:
        raise ValueError("analysis.two_sided_alpha must be 0.05")
    if abs(float(analysis.get("confidence", 0.0)) - 0.95) > 1e-15:
        raise ValueError("analysis.confidence must be 0.95")
    if int(analysis.get("cluster_bootstrap_resamples", 0)) != 10000:
        raise ValueError("analysis.cluster_bootstrap_resamples must be 10000")

    for entry in tasks:
        if not isinstance(entry, dict):
            raise ValueError("each task entry must be a mapping")
        task_name = str(entry.get("task_name", ""))
        if not task_name or task_name in task_names:
            raise ValueError(f"invalid or duplicate task_name: {task_name!r}")
        config_path = _safe_package_path(str(entry.get("config", "")), "task config")
        if not config_path.is_file():
            raise ValueError(f"task config does not exist: {config_path}")
        cfg = _load_yaml(config_path)
        validated = validate_synthetic_transformer_config(cfg)
        config_task = str(cfg.get("task", {}).get("name", ""))
        if config_task != task_name:
            raise ValueError(
                f"plan task {task_name!r} uses config for {config_task!r}: {config_path}"
            )
        expected_roles = {
            "primary_scaling_mode": analysis["primary_scaling_mode"],
            "primary_allocation_rule": analysis["primary_allocation_rule"],
            "primary_reference_rule": analysis["primary_reference_rule"],
            "primary_metric": analysis["primary_metric"],
            "independent_unit": analysis["independent_unit"],
        }
        for key, expected in expected_roles.items():
            if validated.get(key) != expected:
                raise ValueError(
                    f"{config_path.name} {key}={validated.get(key)!r}; expected {expected!r}"
                )
        if set(validated["scaling_modes"]) != EXPECTED_SCALING_MODES:
            raise ValueError(f"{config_path.name} lacks the full scaling design")
        candidate_rules = set(validated["rules"]) - {"uniform", "uniform_fill"}
        if candidate_rules != set(validated["cost_matched_rules"]):
            raise ValueError(
                f"{config_path.name} must exact-cost-match every candidate rule; "
                f"candidates={sorted(candidate_rules)}, matched={validated['cost_matched_rules']}"
            )
        task_names.append(task_name)
        templates.append(
            {
                "task_name": task_name,
                "config_path": config_path,
                "config_relative": config_path.relative_to(ROOT).as_posix(),
                "config": cfg,
                "validated": validated,
            }
        )

    if release_kind == "publication":
        if len(tasks) < PUBLICATION_MIN_TASKS:
            raise ValueError(
                f"publication plan requires at least {PUBLICATION_MIN_TASKS} task families"
            )
        if len(seeds) < PUBLICATION_MIN_RUNS_PER_TASK:
            raise ValueError(
                "publication plan requires at least "
                f"{PUBLICATION_MIN_RUNS_PER_TASK} independent seeds per task"
            )
        exact_test_units = len(tasks) * len(seeds)
        if exact_test_units > 20:
            raise ValueError(
                "publication plan has more than 20 independent units; the declared "
                "sign-flip test would no longer be exactly enumerated"
            )
        nominal_min_p = 2.0 / (2**exact_test_units)
        if nominal_min_p > float(analysis["two_sided_alpha"]):
            raise ValueError(
                "publication plan has insufficient exact-test resolution: "
                f"minimum nominal two-sided p={nominal_min_p}"
            )
        for template in templates:
            cfg = template["config"]
            validated = template["validated"]
            if int(validated["adaptation_replicates"]) < 3:
                raise ValueError("publication configs require at least 3 adaptation replicates")
            if int(validated["null_bootstrap"]) < PUBLICATION_MIN_NULL_MAXIMA:
                raise ValueError("publication configs require at least 4096 null maxima")
            if (
                abs(
                    float(validated["null_quantile"])
                    - PUBLICATION_PRIMARY_NULL_QUANTILE
                )
                > 1e-15
            ):
                raise ValueError("publication null quantile must be 0.995")
            if (
                int(validated["null_uncertainty_resamples"])
                < PUBLICATION_MIN_NULL_UNCERTAINTY_RESAMPLES
            ):
                raise ValueError(
                    "publication configs require at least 2000 null-uncertainty resamples"
                )
            if (
                template["task_name"] == "modular"
                and cfg.get("protocol", {}).get("exact_modular_evaluation") is not True
            ):
                raise ValueError("publication modular config must use exhaustive evaluation")

    return {
        "path": plan_path,
        "sha256": _sha256(plan_path),
        "plan": plan,
        "release_kind": release_kind,
        "release_prefix": prefix,
        "seeds": seeds,
        "tasks": templates,
        "analysis": analysis,
    }


def _git_clean_or_raise() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if head.returncode != 0:
        raise ValueError("publication driver requires a Git worktree")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if status.returncode != 0:
        raise ValueError(f"git status failed:\n{status.stdout}")
    if status.stdout.strip():
        raise ValueError(
            "publication driver requires a clean committed worktree; remaining paths:\n"
            + status.stdout.strip()
        )
    return head.stdout.strip()


def _write_line(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")
        handle.flush()


def _run_logged(command: list[str], log_path: Path) -> None:
    rendered = " ".join(json.dumps(part) for part in command)
    stamp = datetime.now(timezone.utc).isoformat()
    header = f"[{stamp}] $ {rendered}"
    print(header, flush=True)
    _write_line(log_path, header)
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MPLBACKEND", "Agg")
    matplotlib_cache = ROOT / "runs" / "matplotlib_cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    with subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        close_fds=True,
    ) as process:
        assert process.stdout is not None
        with log_path.open("a", encoding="utf-8") as handle:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"command failed with exit code {return_code}: {rendered}")


def _generated_config(
    template: dict[str, Any],
    *,
    seed: int,
    run_id: str,
    release_id: str,
    plan_version: str,
    plan_sha256: str,
) -> dict[str, Any]:
    cfg = deepcopy(template)
    cfg.setdefault("run", {})["seed"] = int(seed)
    cfg["run"]["name"] = run_id
    cfg["run"]["out_dir"] = "runs/transformer_source_runs"
    protocol = cfg.setdefault("protocol", {})
    protocol["publication_plan_version"] = plan_version
    protocol["publication_plan_sha256"] = plan_sha256
    protocol["publication_release_id"] = release_id
    return cfg


def _write_or_check_config(path: Path, cfg: dict[str, Any], *, resume: bool) -> None:
    text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not resume:
            raise ValueError(f"generated config already exists: {path}")
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"resume config differs from the pre-specified plan: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _verify_archive_sidecar(archive: Path) -> None:
    sidecar = Path(str(archive) + ".sha256")
    if not archive.is_file() or not sidecar.is_file():
        raise ValueError("release archive or SHA-256 sidecar is missing")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != archive.name:
        raise ValueError("archive SHA-256 sidecar has an invalid format")
    actual = _sha256(archive)
    if fields[0] != actual:
        raise ValueError(f"archive checksum mismatch: expected {fields[0]}, got {actual}")
    print(f"{archive.name}: OK", flush=True)


def _prepare_release_outputs_for_resume(
    final_release_dir: Path, archive: Path, *, resume: bool
) -> bool:
    """Return True for a complete release; remove only incomplete resume outputs."""

    sidecar = Path(str(archive) + ".sha256")
    paths = (final_release_dir, archive, sidecar)
    if not any(path.exists() for path in paths):
        return False
    if not resume:
        raise ValueError(f"release already exists: {final_release_dir.name}")
    if final_release_dir.is_dir() and archive.is_file() and sidecar.is_file():
        return True

    if final_release_dir.exists():
        if not final_release_dir.is_dir():
            raise ValueError(f"release path is not a directory: {final_release_dir}")
        shutil.rmtree(final_release_dir)
    for path in (archive, sidecar):
        if path.exists():
            if not path.is_file():
                raise ValueError(f"release artifact path is not a file: {path}")
            path.unlink()
    print(
        f"resume: removed incomplete release outputs for {final_release_dir.name}",
        flush=True,
    )
    return False


def run_plan(
    plan_path: Path,
    *,
    release_id: str | None = None,
    resume: bool = False,
) -> Path:
    validated_plan = validate_publication_plan(plan_path)
    git_head = _git_clean_or_raise()
    plan = validated_plan["plan"]
    if release_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        release_id = f"{validated_plan['release_prefix']}_{timestamp}"
    if not release_id or not all(ch.isalnum() or ch in "._-" for ch in release_id):
        raise ValueError("release_id contains unsupported characters")

    run_root = ROOT / "runs" / "transformer_source_runs"
    config_root = ROOT / "runs" / "transformer_generated_configs" / release_id
    log_root = ROOT / "runs" / "transformer_logs" / release_id
    aggregate_dir = ROOT / "runs" / "transformer_aggregates" / f"{release_id}_aggregate"
    release_root = ROOT / "runs" / "transformer_releases"
    master_log = log_root / "publication_driver.log"
    master_log.parent.mkdir(parents=True, exist_ok=True)
    if master_log.exists() and not resume:
        raise ValueError(f"driver log already exists: {master_log}")
    _write_line(master_log, f"release_id={release_id}")
    _write_line(master_log, f"release_kind={validated_plan['release_kind']}")
    _write_line(master_log, f"git_head={git_head}")
    _write_line(master_log, f"plan={validated_plan['path']}")
    _write_line(master_log, f"plan_sha256={validated_plan['sha256']}")
    print(f"release_id: {release_id}", flush=True)
    print(f"plan_sha256: {validated_plan['sha256']}", flush=True)
    print(f"git_head: {git_head}", flush=True)

    source_runs: list[Path] = []
    for task in validated_plan["tasks"]:
        for seed in validated_plan["seeds"]:
            run_id = f"{release_id}_{task['task_name']}_seed{seed}"
            run_dir = run_root / run_id
            config_path = config_root / f"{task['task_name']}_seed{seed}.yaml"
            cfg = _generated_config(
                task["config"],
                seed=seed,
                run_id=run_id,
                release_id=release_id,
                plan_version=str(plan["plan_version"]),
                plan_sha256=validated_plan["sha256"],
            )
            _write_or_check_config(config_path, cfg, resume=resume)
            if run_dir.exists():
                if not resume:
                    raise ValueError(f"source run already exists: {run_dir}")
                try:
                    _run_logged(
                        [
                            sys.executable,
                            "scripts/validate_rank_scaling_run.py",
                            str(run_dir),
                        ],
                        master_log,
                    )
                except Exception as exc:
                    raise ValueError(
                        "resume found an incomplete or invalid source run; inspect or remove "
                        f"{run_dir} before retrying"
                    ) from exc
                print(f"resume: reusing validated source run {run_id}", flush=True)
            else:
                _run_logged(
                    [
                        sys.executable,
                        "scripts/run_synthetic_transformer.py",
                        "--config",
                        str(config_path),
                        "--run-dir",
                        str(run_dir),
                    ],
                    master_log,
                )
                _run_logged(
                    [
                        sys.executable,
                        "scripts/validate_rank_scaling_run.py",
                        str(run_dir),
                    ],
                    master_log,
                )
            source_runs.append(run_dir)

    if aggregate_dir.exists():
        if not resume:
            raise ValueError(f"aggregate directory already exists: {aggregate_dir}")
        import shutil

        shutil.rmtree(aggregate_dir)
    aggregate_command = [
        sys.executable,
        "scripts/aggregate_step4_tasks.py",
        "--out",
        str(aggregate_dir),
        "--analysis-plan",
        str(validated_plan["path"]),
    ]
    for run_dir in source_runs:
        aggregate_command.extend(["--run", str(run_dir)])
    _run_logged(aggregate_command, master_log)

    final_release_dir = release_root / release_id
    archive = release_root / f"{release_id}.tar.gz"
    if _prepare_release_outputs_for_resume(
        final_release_dir, archive, resume=resume
    ):
        _run_logged(
            [
                sys.executable,
                "scripts/validate_transformer_release.py",
                str(final_release_dir),
                "--expected-kind",
                validated_plan["release_kind"],
            ],
            master_log,
        )
        _verify_archive_sidecar(archive)
        return archive

    build_command = [
        sys.executable,
        "scripts/build_transformer_release.py",
        "--release-root",
        str(release_root),
        "--release-id",
        release_id,
        "--kind",
        validated_plan["release_kind"],
        "--command-log",
        str(master_log),
        "--aggregate-dir",
        str(aggregate_dir),
    ]
    for run_dir in source_runs:
        build_command.extend(["--run", str(run_dir)])
    _run_logged(build_command, master_log)
    _run_logged(
        [
            sys.executable,
            "scripts/validate_transformer_release.py",
            str(final_release_dir),
            "--expected-kind",
            validated_plan["release_kind"],
        ],
        master_log,
    )
    _verify_archive_sidecar(archive)
    print("Transformer publication driver: PASS", flush=True)
    print(f"ARCHIVE={archive}", flush=True)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a pre-specified multi-task transformer evidence plan."
    )
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
        help="Validate the plan and templates without running experiments.",
    )
    args = parser.parse_args()
    plan_path = Path(args.plan)
    if not plan_path.is_absolute():
        plan_path = ROOT / plan_path
    if args.validate_plan_only:
        result = validate_publication_plan(plan_path)
        print("Transformer publication plan validation: PASS")
        print(f"release_kind: {result['release_kind']}")
        print(f"tasks: {[item['task_name'] for item in result['tasks']]}")
        print(f"seeds: {result['seeds']}")
        print(f"plan_sha256: {result['sha256']}")
        return
    run_plan(plan_path, release_id=args.release_id, resume=args.resume)


if __name__ == "__main__":
    main()
