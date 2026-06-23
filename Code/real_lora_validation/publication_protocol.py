"""Pre-specified multi-seed real-model LoRA publication protocol helpers."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import t as student_t

from real_protocol import (
    PUBLICATION_AGGREGATE_VERSION,
    PUBLICATION_PLAN_VERSION,
    PUBLICATION_RELEASE_VERSION,
    REAL_PROTOCOL_VERSION,
)

FULL_PRIMARY_SEEDS = [101, 103, 107, 109, 113, 127, 131, 137]
FULL_BOUNDARY_SEEDS = [101, 103, 107]
FULL_STRATEGIES = [
    "uniform",
    "gradient_norm",
    "spectral_effective",
    "eva_activation",
    "fim_gradient_variance",
    "gora_sensitivity",
]
BOOTSTRAP_SEED_SCHEME = "sha256(plan_sha256|label)_first8_mod_2^63_v1"
PUBLICATION_CODE_SNAPSHOT_FILES = (
    "run_real_lora_validation.py",
    "hf_gpu_smoke.py",
    "real_protocol.py",
    "publication_protocol.py",
    "spectral_metrics.py",
    "requirements.txt",
    "requirements-tested-rocm721.txt",
    "README.md",
    "README_LOCAL_DATA.md",
    "INPUT_MANIFEST.json",
    "configs/real_lora_publication_plan.json",
    "configs/real_lora_publication_smoke_plan.json",
    "scripts/preflight_publication_rerun.py",
    "scripts/prepare_publication_inputs.py",
    "scripts/run_protocol_smoke.sh",
    "scripts/run_real_lora_publication.py",
    "scripts/run_real_lora_publication.sh",
    "scripts/run_real_lora_publication_smoke.sh",
    "scripts/aggregate_real_lora_publication.py",
    "scripts/validate_real_lora_run.py",
    "scripts/validate_real_lora_release.py",
    "scripts/validate_real_lora_publication_release.py",
    "scripts/build_real_lora_release.py",
    "scripts/build_real_lora_publication_release.py",
    "tests/conftest.py",
    "tests/test_real_protocol.py",
    "tests/test_runner_protocol.py",
    "tests/test_spectral_metrics.py",
    "tests/test_publication_protocol.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_mapping(path: Path, label: str = "JSON file") -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def safe_package_path(root: Path, value: str, label: str) -> Path:
    pure = PurePosixPath(str(value))
    if pure.is_absolute() or not pure.parts or "." in pure.parts or ".." in pure.parts:
        raise ValueError(f"unsafe {label}: {value!r}")
    path = root.joinpath(*pure.parts).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes package root: {value!r}") from exc
    return path


def _require_keys(mapping: dict[str, Any], keys: Iterable[str], label: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{label} is missing keys: {missing}")


def _validate_identifier(value: object, label: str) -> str:
    text = str(value or "")
    if not text or any(not (ch.isalnum() or ch in "._-") for ch in text):
        raise ValueError(f"invalid {label}: {text!r}")
    return text


def _float_equal(value: object, expected: float) -> bool:
    try:
        return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-15)
    except (TypeError, ValueError):
        return False


def validate_publication_plan(plan_path: Path, package_root: Path) -> dict[str, Any]:
    plan_path = Path(plan_path).resolve()
    package_root = Path(package_root).resolve()
    plan = load_json_mapping(plan_path, "publication plan")
    if plan.get("plan_version") != PUBLICATION_PLAN_VERSION:
        raise ValueError(f"plan_version must be {PUBLICATION_PLAN_VERSION}")
    if plan.get("protocol_version") != REAL_PROTOCOL_VERSION:
        raise ValueError("publication plan protocol_version is stale")
    release_kind = str(plan.get("release_kind", ""))
    if release_kind not in {"smoke", "publication"}:
        raise ValueError("release_kind must be smoke or publication")
    release_prefix = _validate_identifier(plan.get("release_prefix"), "release_prefix")

    model = plan.get("model")
    dataset = plan.get("dataset")
    run = plan.get("run")
    analysis = plan.get("analysis")
    suites = plan.get("suites")
    for value, label in [
        (model, "model"),
        (dataset, "dataset"),
        (run, "run"),
        (analysis, "analysis"),
    ]:
        if not isinstance(value, dict):
            raise ValueError(f"plan.{label} must be an object")
    if not isinstance(suites, list) or not suites:
        raise ValueError("plan.suites must be a nonempty list")

    _require_keys(model, ["argument", "local_files_only"], "plan.model")
    _require_keys(dataset, ["mode"], "plan.dataset")
    if not str(model["argument"]).strip():
        raise ValueError("plan.model.argument must be nonempty")
    if not bool(model["local_files_only"]):
        raise ValueError("all controlled plans require model.local_files_only=true")
    if dataset["mode"] not in {"builtin", "local"}:
        raise ValueError("controlled plans require dataset.mode builtin or local")
    required_run = [
        "block_size",
        "batch_size",
        "calib_batches",
        "eval_batches",
        "steps",
        "lr",
        "weight_decay",
        "grad_clip",
        "uniform_rank",
        "min_rank",
        "max_rank",
        "fisher_reference_rank",
        "ridge_scale",
        "lora_alpha_scale",
        "lora_init_std",
        "max_train_texts",
        "max_val_texts",
        "max_train_blocks",
        "max_val_blocks",
        "dtype",
        "device",
        "log_every",
        "strategies",
        "include_identity_control",
        "identity_tolerance",
        "minimum_first_step_gradient_l2",
        "minimum_adapter_delta_l2",
        "minimum_effective_update_l2",
        "deterministic_algorithms",
    ]
    _require_keys(run, required_run, "plan.run")
    positive_integer_fields = [
        "block_size",
        "batch_size",
        "calib_batches",
        "eval_batches",
        "steps",
        "max_train_texts",
        "max_val_texts",
        "max_train_blocks",
        "max_val_blocks",
        "log_every",
    ]
    for key in positive_integer_fields:
        if int(run[key]) < 1:
            raise ValueError(f"plan.run.{key} must be >= 1")
    positive_float_fields = ["lr", "lora_alpha_scale", "lora_init_std"]
    for key in positive_float_fields:
        value = float(run[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"plan.run.{key} must be finite and positive")
    nonnegative_float_fields = ["weight_decay", "grad_clip", "ridge_scale"]
    for key in nonnegative_float_fields:
        value = float(run[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"plan.run.{key} must be finite and nonnegative")
    strategies = list(run["strategies"])
    if strategies != FULL_STRATEGIES:
        raise ValueError(
            f"plan.run.strategies must be exactly {FULL_STRATEGIES}; got {strategies}"
        )
    if not bool(run["include_identity_control"]):
        raise ValueError("publication plan requires include_identity_control=true")
    if not bool(run["deterministic_algorithms"]):
        raise ValueError("publication plan requires deterministic_algorithms=true")
    if int(run["fisher_reference_rank"]) != int(run["uniform_rank"]):
        raise ValueError("fisher_reference_rank must equal uniform_rank")
    if not 1 <= int(run["min_rank"]) <= int(run["uniform_rank"]) <= int(run["max_rank"]):
        raise ValueError("rank bounds/reference are inconsistent")
    if str(run["dtype"]) != "float32":
        raise ValueError("the frozen real-model design requires float32")
    if str(run["device"]) != "cuda":
        raise ValueError("the frozen real-model design requires device=cuda")
    if not _float_equal(run["identity_tolerance"], 1e-7):
        raise ValueError("identity_tolerance must be 1e-7")
    for key in [
        "minimum_first_step_gradient_l2",
        "minimum_adapter_delta_l2",
        "minimum_effective_update_l2",
    ]:
        value = float(run[key])
        if value < 0 or not math.isfinite(value):
            raise ValueError(f"plan.run.{key} must be finite and nonnegative")

    suite_rows: list[dict[str, Any]] = []
    suite_ids: set[str] = set()
    independent_units: set[tuple[str, int]] = set()
    for entry in suites:
        if not isinstance(entry, dict):
            raise ValueError("each suite entry must be an object")
        _require_keys(
            entry,
            ["suite_id", "role", "seeds", "target_suffixes", "max_targets"],
            "suite",
        )
        suite_id = _validate_identifier(entry["suite_id"], "suite_id")
        role = _validate_identifier(entry["role"], "suite role")
        if suite_id in suite_ids:
            raise ValueError(f"duplicate suite_id: {suite_id}")
        suite_ids.add(suite_id)
        seeds = [int(value) for value in entry["seeds"]]
        if not seeds or seeds != sorted(seeds) or len(seeds) != len(set(seeds)):
            raise ValueError(f"suite {suite_id} seeds must be sorted and unique")
        if any(seed < 0 for seed in seeds):
            raise ValueError(f"suite {suite_id} has a negative seed")
        suffixes = [part.strip() for part in str(entry["target_suffixes"]).split(",") if part.strip()]
        if not suffixes or len(suffixes) != len(set(suffixes)):
            raise ValueError(f"suite {suite_id} target_suffixes are invalid")
        if int(entry["max_targets"]) < 0:
            raise ValueError(f"suite {suite_id} max_targets must be nonnegative")
        for seed in seeds:
            unit = (suite_id, seed)
            if unit in independent_units:
                raise ValueError(f"duplicate independent unit: {unit}")
            independent_units.add(unit)
        suite_rows.append(
            {
                **entry,
                "suite_id": suite_id,
                "role": role,
                "seeds": seeds,
                "target_suffixes": ",".join(suffixes),
                "max_targets": int(entry["max_targets"]),
            }
        )

    required_analysis = {
        "primary_metric": "final_val_loss",
        "primary_candidate_strategy": "spectral_effective",
        "primary_reference_requested_strategy": "uniform",
        "independent_unit": "suite_seed_run",
        "delta_definition": "candidate_minus_reference",
        "lower_is_better": True,
        "exact_sign_flip_test": "two_sided_within_suite_nonzero_deltas",
        "secondary_multiplicity": "holm_within_suite",
    }
    for key, expected in required_analysis.items():
        if analysis.get(key) != expected:
            raise ValueError(
                f"analysis.{key}={analysis.get(key)!r}; expected {expected!r}"
            )
    primary_suite = str(analysis.get("primary_suite", ""))
    if primary_suite not in suite_ids:
        raise ValueError("analysis.primary_suite is not a declared suite")
    if not _float_equal(analysis.get("confidence"), 0.95):
        raise ValueError("analysis.confidence must be 0.95")
    if not _float_equal(analysis.get("two_sided_alpha"), 0.05):
        raise ValueError("analysis.two_sided_alpha must be 0.05")
    if int(analysis.get("bootstrap_resamples", 0)) != 10000:
        raise ValueError("analysis.bootstrap_resamples must be 10000")
    if not str(analysis.get("primary_estimand", "")).strip():
        raise ValueError("analysis.primary_estimand must be declared")
    minimum_nonzero = int(
        analysis.get("minimum_nonzero_primary_deltas_for_resolution", 0)
    )
    planned_tie_tolerance = int(analysis.get("planned_tie_tolerance", -1))
    if minimum_nonzero < 1:
        raise ValueError(
            "analysis.minimum_nonzero_primary_deltas_for_resolution must be >= 1"
        )
    if planned_tie_tolerance < 0:
        raise ValueError("analysis.planned_tie_tolerance must be nonnegative")

    input_manifest_path: Path | None = None
    if release_kind == "publication":
        expected_suites = {
            "cattn_cfc": {
                "role": "primary",
                "seeds": FULL_PRIMARY_SEEDS,
                "target_suffixes": "c_attn,c_fc",
                "max_targets": 0,
            },
            "attnproj": {
                "role": "boundary",
                "seeds": FULL_BOUNDARY_SEEDS,
                "target_suffixes": "c_attn,attn.c_proj,c_fc",
                "max_targets": 0,
            },
        }
        actual_suites = {
            row["suite_id"]: {
                "role": row["role"],
                "seeds": row["seeds"],
                "target_suffixes": row["target_suffixes"],
                "max_targets": row["max_targets"],
            }
            for row in suite_rows
        }
        if actual_suites != expected_suites:
            raise ValueError(
                f"publication suite design differs from the frozen 8+3 design: {actual_suites}"
            )
        expected_run = {
            "block_size": 128,
            "batch_size": 4,
            "calib_batches": 8,
            "eval_batches": 32,
            "steps": 200,
            "lr": 2e-4,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "uniform_rank": 4,
            "min_rank": 1,
            "max_rank": 16,
            "fisher_reference_rank": 4,
            "ridge_scale": 1e-3,
            "lora_alpha_scale": 2.0,
            "lora_init_std": 0.01,
            "max_train_texts": 5000,
            "max_val_texts": 1000,
            "max_train_blocks": 1024,
            "max_val_blocks": 256,
            "dtype": "float32",
            "device": "cuda",
            "log_every": 20,
            "include_identity_control": True,
            "identity_tolerance": 1e-7,
            "minimum_first_step_gradient_l2": 0.0,
            "minimum_adapter_delta_l2": 0.0,
            "minimum_effective_update_l2": 0.0,
            "deterministic_algorithms": True,
        }
        for key, expected in expected_run.items():
            actual = run.get(key)
            if isinstance(expected, float):
                if not _float_equal(actual, expected):
                    raise ValueError(f"publication run.{key}={actual!r}; expected {expected}")
            elif actual != expected:
                raise ValueError(f"publication run.{key}={actual!r}; expected {expected!r}")
        if model.get("argument") != "models/gpt2_local" or not bool(
            model.get("local_files_only")
        ):
            raise ValueError("publication model must be pinned local GPT-2")
        if dataset.get("mode") != "local":
            raise ValueError("publication dataset mode must be local")
        if dataset.get("train_text_file") != "data/wikitext2_local/train.txt":
            raise ValueError("publication train_text_file is stale")
        if dataset.get("val_text_file") != "data/wikitext2_local/validation.txt":
            raise ValueError("publication val_text_file is stale")
        manifest_rel = str(plan.get("input_manifest", ""))
        manifest_hash = str(plan.get("input_manifest_sha256", ""))
        input_manifest_path = safe_package_path(package_root, manifest_rel, "input manifest")
        if not input_manifest_path.is_file():
            raise ValueError(f"input manifest does not exist: {input_manifest_path}")
        actual_hash = sha256_file(input_manifest_path)
        if actual_hash != manifest_hash:
            raise ValueError(
                f"input manifest hash mismatch: {actual_hash}; expected {manifest_hash}"
            )
        if primary_suite != "cattn_cfc":
            raise ValueError("the frozen publication primary suite is cattn_cfc")
        if analysis.get("boundary_inference") != "descriptive_only_not_pooled":
            raise ValueError(
                "the attnproj boundary suite must be descriptive_only_not_pooled"
            )
        primary_n = len(expected_suites[primary_suite]["seeds"])
        if minimum_nonzero != primary_n - planned_tie_tolerance:
            raise ValueError(
                "primary resolution fields are inconsistent with the planned tie tolerance"
            )
        attainable_min_p = 2.0 / (2**minimum_nonzero)
        if attainable_min_p > float(analysis["two_sided_alpha"]):
            raise ValueError(
                "the primary exact sign-flip test has insufficient prespecified "
                f"resolution after ties: nonzero={minimum_nonzero}, "
                f"minimum two-sided p={attainable_min_p}"
            )
    else:
        expected_smoke_suites = {
            "tiny_core": {
                "role": "pipeline_primary",
                "seeds": [321, 323],
                "target_suffixes": "c_attn,c_fc",
                "max_targets": 4,
            },
            "tiny_attnproj": {
                "role": "pipeline_boundary",
                "seeds": [321, 323],
                "target_suffixes": "c_attn,attn.c_proj,c_fc",
                "max_targets": 4,
            },
        }
        actual_smoke_suites = {
            row["suite_id"]: {
                "role": row["role"],
                "seeds": row["seeds"],
                "target_suffixes": row["target_suffixes"],
                "max_targets": row["max_targets"],
            }
            for row in suite_rows
        }
        if actual_smoke_suites != expected_smoke_suites:
            raise ValueError(
                f"smoke suite design differs from the frozen 2+2 design: "
                f"{actual_smoke_suites}"
            )
        expected_smoke_run = {
            "block_size": 32,
            "batch_size": 2,
            "calib_batches": 2,
            "eval_batches": 2,
            "steps": 4,
            "lr": 2e-4,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "uniform_rank": 2,
            "min_rank": 1,
            "max_rank": 3,
            "fisher_reference_rank": 2,
            "ridge_scale": 1e-3,
            "lora_alpha_scale": 2.0,
            "lora_init_std": 0.01,
            "max_train_texts": 32,
            "max_val_texts": 16,
            "max_train_blocks": 16,
            "max_val_blocks": 8,
            "dtype": "float32",
            "device": "cuda",
            "log_every": 1,
            "include_identity_control": True,
            "identity_tolerance": 1e-7,
            "minimum_first_step_gradient_l2": 0.0,
            "minimum_adapter_delta_l2": 0.0,
            "minimum_effective_update_l2": 0.0,
            "deterministic_algorithms": True,
        }
        for key, expected in expected_smoke_run.items():
            actual = run.get(key)
            if isinstance(expected, float):
                if not _float_equal(actual, expected):
                    raise ValueError(
                        f"smoke run.{key}={actual!r}; expected {expected}"
                    )
            elif actual != expected:
                raise ValueError(
                    f"smoke run.{key}={actual!r}; expected {expected!r}"
                )
        if model.get("argument") != "sshleifer/tiny-gpt2":
            raise ValueError("smoke model must be sshleifer/tiny-gpt2")
        if dataset.get("mode") != "builtin":
            raise ValueError("smoke dataset mode must be builtin")
        if primary_suite != "tiny_core":
            raise ValueError("the frozen smoke primary suite is tiny_core")
        primary_n = len(expected_smoke_suites[primary_suite]["seeds"])
        if minimum_nonzero != 1 or planned_tie_tolerance != 1:
            raise ValueError("smoke resolution fields differ from the frozen plan")
        if minimum_nonzero != primary_n - planned_tie_tolerance:
            raise ValueError(
                "smoke resolution fields are inconsistent with the planned tie tolerance"
            )

    plan_hash = sha256_file(plan_path)
    return {
        "path": plan_path,
        "sha256": plan_hash,
        "plan": plan,
        "release_kind": release_kind,
        "release_prefix": release_prefix,
        "model": model,
        "dataset": dataset,
        "run": run,
        "analysis": analysis,
        "suites": suite_rows,
        "input_manifest_path": input_manifest_path,
        "n_source_runs": sum(len(row["seeds"]) for row in suite_rows),
    }


def validate_input_provenance_against_manifest(
    provenance: dict[str, Any], manifest: dict[str, Any]
) -> None:
    """Verify one source-run provenance record against the frozen input manifest."""
    recorded_hash = provenance.get("provenance_sha256")
    unhashed = dict(provenance)
    unhashed.pop("provenance_sha256", None)
    if recorded_hash != hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest():
        raise ValueError("input provenance self-hash mismatch")

    expected_model = {
        Path(str(row["path"])).name: (int(row["bytes"]), str(row["sha256"]))
        for row in manifest["model"]["files"]
    }
    actual_model_rows = provenance.get("local_model_files")
    if not isinstance(actual_model_rows, list):
        raise ValueError("publication input provenance lacks local_model_files")
    actual_model: dict[str, tuple[int, str]] = {}
    for row in actual_model_rows:
        if not isinstance(row, dict):
            raise ValueError("local_model_files contains a non-object row")
        relative = str(row.get("path", ""))
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"unsafe local model provenance path: {relative!r}")
        if relative in actual_model:
            raise ValueError(f"duplicate local model provenance path: {relative}")
        actual_model[relative] = (int(row["size"]), str(row["sha256"]))
    if actual_model != expected_model:
        raise ValueError("local model provenance differs from INPUT_MANIFEST.json")

    text_rows = provenance.get("local_text_files")
    if not isinstance(text_rows, dict):
        raise ValueError("publication input provenance lacks local_text_files")
    expected_text = manifest["dataset"]["text_files"]
    for split in ["train", "validation"]:
        row = text_rows.get(split)
        if not isinstance(row, dict):
            raise ValueError(f"local text provenance lacks {split}")
        expected = expected_text[split]
        if str(row.get("path")) != str(expected["path"]):
            raise ValueError(f"local {split} text path differs from the manifest")
        if int(row.get("size", -1)) != int(expected["bytes"]):
            raise ValueError(f"local {split} text size differs from the manifest")
        if str(row.get("sha256")) != str(expected["sha256"]):
            raise ValueError(f"local {split} text hash differs from the manifest")

    if provenance.get("model_argument") != "models/gpt2_local":
        raise ValueError("publication model_argument is not the pinned local path")
    if provenance.get("local_model_path") != "models/gpt2_local":
        raise ValueError("publication local_model_path is not portable/pinned")
    if provenance.get("dataset_mode") != "local":
        raise ValueError("publication input provenance dataset_mode is not local")


def stable_seed(plan_sha256: str, label: str) -> int:
    payload = f"{plan_sha256}|{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def exact_sign_flip_p(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 1 or not np.isfinite(array).all():
        raise ValueError("exact sign-flip values must be a finite nonempty vector")
    nonzero = array[array != 0.0]
    if nonzero.size == 0:
        return 1.0
    observed = abs(float(nonzero.mean()))
    count = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=nonzero.size):
        statistic = abs(float(np.mean(nonzero * np.asarray(signs))))
        count += statistic >= observed - 1e-15
        total += 1
    return float(count / total)


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    confidence: float,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 1 or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be a finite nonempty vector")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be inside (0, 1)")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(n_resamples), array.size))
    means = array[indices].mean(axis=1)
    alpha = 1.0 - float(confidence)
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def t_mean_interval(values: Sequence[float], confidence: float) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 1 or not np.isfinite(array).all():
        raise ValueError("t-interval values must be a finite nonempty vector")
    mean = float(array.mean())
    if array.size == 1:
        return float("nan"), float("nan")
    sem = float(array.std(ddof=1) / math.sqrt(array.size))
    critical = float(student_t.ppf((1.0 + confidence) / 2.0, df=array.size - 1))
    return mean - critical * sem, mean + critical * sem


def summarize_paired_values(
    values: Sequence[float],
    *,
    confidence: float,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 1 or not np.isfinite(array).all():
        raise ValueError("paired values must be a finite nonempty vector")
    t_low, t_high = t_mean_interval(array, confidence)
    boot_low, boot_high = bootstrap_mean_interval(
        array,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )
    n_nonzero = int(np.count_nonzero(array))
    attainable_min_p = float(2.0 / (2**n_nonzero)) if n_nonzero else 1.0
    return {
        "n_independent_runs": int(array.size),
        "n_nonzero_deltas": n_nonzero,
        "mean_loss_delta": float(array.mean()),
        "median_loss_delta": float(np.median(array)),
        "sd_loss_delta": float(array.std(ddof=1)) if array.size > 1 else float("nan"),
        "sem_loss_delta": float(array.std(ddof=1) / math.sqrt(array.size))
        if array.size > 1
        else float("nan"),
        "t_ci_low": t_low,
        "t_ci_high": t_high,
        "bootstrap_ci_low": boot_low,
        "bootstrap_ci_high": boot_high,
        "exact_sign_flip_p_two_sided": exact_sign_flip_p(array),
        "wins": int(np.sum(array < 0)),
        "ties": int(np.sum(array == 0)),
        "losses": int(np.sum(array > 0)),
        "planned_no_tie_min_two_sided_p": float(2.0 / (2**array.size)),
        "attainable_min_two_sided_p": attainable_min_p,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = [float(value) for value in p_values]
    if any(not 0 <= value <= 1 or not math.isfinite(value) for value in values):
        raise ValueError("Holm p-values must be finite values in [0, 1]")
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [0.0] * len(values)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


__all__ = [
    "BOOTSTRAP_SEED_SCHEME",
    "FULL_BOUNDARY_SEEDS",
    "FULL_PRIMARY_SEEDS",
    "FULL_STRATEGIES",
    "PUBLICATION_AGGREGATE_VERSION",
    "PUBLICATION_CODE_SNAPSHOT_FILES",
    "PUBLICATION_PLAN_VERSION",
    "PUBLICATION_RELEASE_VERSION",
    "bootstrap_mean_interval",
    "exact_sign_flip_p",
    "holm_adjust",
    "load_json_mapping",
    "safe_package_path",
    "sha256_file",
    "stable_seed",
    "summarize_paired_values",
    "t_mean_interval",
    "validate_input_provenance_against_manifest",
    "validate_publication_plan",
]
