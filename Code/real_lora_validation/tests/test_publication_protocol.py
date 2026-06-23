from __future__ import annotations

from configparser import ConfigParser
import importlib.util
import json
from pathlib import Path

import pytest

from publication_protocol import (
    FULL_BOUNDARY_SEEDS,
    FULL_PRIMARY_SEEDS,
    PUBLICATION_CODE_SNAPSHOT_FILES,
    bootstrap_mean_interval,
    exact_sign_flip_p,
    holm_adjust,
    stable_seed,
    summarize_paired_values,
    validate_input_provenance_against_manifest,
    validate_publication_plan,
)
from real_protocol import canonical_json_sha256

ROOT = Path(__file__).resolve().parents[1]


def load_driver_module():
    path = ROOT / "scripts" / "run_real_lora_publication.py"
    spec = importlib.util.spec_from_file_location("real_lora_publication_driver_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_publication_plan_has_exact_test_resolution() -> None:
    result = validate_publication_plan(
        ROOT / "configs" / "real_lora_publication_plan.json", ROOT
    )
    assert result["release_kind"] == "publication"
    assert result["n_source_runs"] == 11
    assert FULL_PRIMARY_SEEDS == [101, 103, 107, 109, 113, 127, 131, 137]
    assert FULL_BOUNDARY_SEEDS == [101, 103, 107]
    analysis = result["analysis"]
    assert analysis["minimum_nonzero_primary_deltas_for_resolution"] == 6
    assert analysis["planned_tie_tolerance"] == 2
    assert 2.0 / (2**6) < 0.05


def test_smoke_plan_is_four_source_runs_and_non_evidential() -> None:
    result = validate_publication_plan(
        ROOT / "configs" / "real_lora_publication_smoke_plan.json", ROOT
    )
    assert result["release_kind"] == "smoke"
    assert result["n_source_runs"] == 4
    assert {suite["suite_id"]: suite["seeds"] for suite in result["suites"]} == {
        "tiny_core": [321, 323],
        "tiny_attnproj": [321, 323],
    }
    assert "not scientific evidence" in result["analysis"]["primary_estimand"]
    assert result["analysis"]["minimum_nonzero_primary_deltas_for_resolution"] == 1
    assert result["analysis"]["planned_tie_tolerance"] == 1


def test_exact_sign_flip_and_bootstrap_are_deterministic() -> None:
    values = [-1.0] * 8
    assert exact_sign_flip_p(values) == pytest.approx(2.0 / 256.0)
    seed = stable_seed("a" * 64, "primary")
    assert seed == stable_seed("a" * 64, "primary")
    first = bootstrap_mean_interval(values, confidence=0.95, n_resamples=1000, seed=seed)
    second = bootstrap_mean_interval(values, confidence=0.95, n_resamples=1000, seed=seed)
    assert first == second == pytest.approx((-1.0, -1.0))


def test_exact_sign_flip_drops_exact_ties_and_records_attainable_resolution() -> None:
    values = [-1.0] * 6 + [0.0, 0.0]
    assert exact_sign_flip_p(values) == pytest.approx(2.0 / (2**6))
    assert exact_sign_flip_p([0.0, 0.0]) == 1.0
    summary = summarize_paired_values(
        values,
        confidence=0.95,
        n_resamples=128,
        seed=7,
    )
    assert summary["n_independent_runs"] == 8
    assert summary["n_nonzero_deltas"] == 6
    assert summary["ties"] == 2
    assert summary["planned_no_tie_min_two_sided_p"] == pytest.approx(2.0 / 256.0)
    assert summary["attainable_min_two_sided_p"] == pytest.approx(2.0 / 64.0)


def test_holm_adjustment_preserves_original_order() -> None:
    assert holm_adjust([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])


def test_child_environment_does_not_mutate_or_inherit_proxy_settings() -> None:
    driver = load_driver_module()
    parent = {
        "PATH": "/bin",
        "ALL_PROXY": "socks://127.0.0.1:7890",
        "https_proxy": "http://127.0.0.1:7890",
    }
    before = dict(parent)
    child = driver.child_environment(parent)
    assert parent == before
    assert "ALL_PROXY" not in child
    assert "https_proxy" not in child
    assert child["HF_HUB_OFFLINE"] == "1"
    assert child["TRANSFORMERS_OFFLINE"] == "1"
    assert child["PYTHONDONTWRITEBYTECODE"] == "1"


def test_invalid_resume_source_is_preserved_in_quarantine(tmp_path: Path) -> None:
    driver = load_driver_module()
    run_root = tmp_path / "runs" / "real_lora_source_runs"
    run_dir = run_root / "release_suite_seed1"
    run_dir.mkdir(parents=True)
    (run_dir / "partial.txt").write_text("partial", encoding="utf-8")
    destination = driver._quarantine_invalid_source(run_dir, ValueError("broken"))
    assert not run_dir.exists()
    assert (destination / "partial.txt").read_text(encoding="utf-8") == "partial"
    reason = (destination / "QUARANTINE_REASON.txt").read_text(encoding="utf-8")
    assert "error_type=ValueError" in reason
    assert "error=broken" in reason


def test_publication_boundary_suite_is_explicitly_descriptive(tmp_path: Path) -> None:
    source = ROOT / "configs" / "real_lora_publication_plan.json"
    plan = json.loads(source.read_text(encoding="utf-8"))
    plan["analysis"]["boundary_inference"] = "pooled_confirmatory"
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="descriptive_only_not_pooled"):
        validate_publication_plan(path, ROOT)


def test_controlled_plans_require_offline_local_model_loading(tmp_path: Path) -> None:
    source = ROOT / "configs" / "real_lora_publication_smoke_plan.json"
    plan = json.loads(source.read_text(encoding="utf-8"))
    plan["model"]["local_files_only"] = False
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="local_files_only=true"):
        validate_publication_plan(path, ROOT)


def test_publication_input_provenance_matches_frozen_manifest() -> None:
    manifest = json.loads((ROOT / "INPUT_MANIFEST.json").read_text(encoding="utf-8"))
    provenance = {
        "model_argument": "models/gpt2_local",
        "model_revision_requested": None,
        "resolved_model_commit": None,
        "dataset_mode": "local",
        "dataset_name": "wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "dataset_revision_requested": None,
        "local_model_path": "models/gpt2_local",
        "local_model_files": [
            {
                "path": Path(row["path"]).name,
                "size": row["bytes"],
                "sha256": row["sha256"],
            }
            for row in manifest["model"]["files"]
        ],
        "local_text_files": {
            split: {
                "path": row["path"],
                "size": row["bytes"],
                "sha256": row["sha256"],
            }
            for split, row in manifest["dataset"]["text_files"].items()
        },
    }
    provenance["provenance_sha256"] = canonical_json_sha256(provenance)
    validate_input_provenance_against_manifest(provenance, manifest)

    tampered = json.loads(json.dumps(provenance))
    tampered["local_model_files"][0]["size"] += 1
    tampered.pop("provenance_sha256")
    tampered["provenance_sha256"] = canonical_json_sha256(tampered)
    with pytest.raises(ValueError, match="local model provenance differs"):
        validate_input_provenance_against_manifest(tampered, manifest)


def test_publication_code_snapshot_contract_is_complete() -> None:
    assert len(PUBLICATION_CODE_SNAPSHOT_FILES) == len(
        set(PUBLICATION_CODE_SNAPSHOT_FILES)
    )
    assert all((ROOT / relative).is_file() for relative in PUBLICATION_CODE_SNAPSHOT_FILES)


def test_pytest_collection_excludes_generated_release_trees() -> None:
    parser = ConfigParser()
    assert parser.read(ROOT / "pytest.ini", encoding="utf-8")
    assert parser.get("pytest", "testpaths").split() == ["tests"]
    ignored = set(parser.get("pytest", "norecursedirs").split())
    assert {"runs", "real_lora_runs", "results", "models", "data"} <= ignored


def test_five_run_exact_two_sided_test_cannot_cross_point_zero_five() -> None:
    assert 2.0 / (2**5) == pytest.approx(0.0625)
    assert 2.0 / (2**5) > 0.05
