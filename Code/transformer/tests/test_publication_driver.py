from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.build_transformer_release import _copy_code_snapshot
from scripts.run_transformer_publication import (
    _generated_config,
    validate_publication_plan,
)
from scripts.validate_transformer_release import _validate_analysis_plan_roles
from strank.protocol import SYNTHETIC_TRANSFORMER_PROTOCOL_VERSION


ROOT = Path(__file__).resolve().parents[1]


def test_publication_and_design_smoke_plans_pass_preflight() -> None:
    publication = validate_publication_plan(
        ROOT / "configs" / "transformer_publication_plan.yaml"
    )
    assert publication["release_kind"] == "publication"
    assert publication["seeds"] == [101, 103, 107, 109, 113]
    assert {item["task_name"] for item in publication["tasks"]} == {
        "modular",
        "associative_recall",
    }

    smoke = validate_publication_plan(
        ROOT / "configs" / "transformer_publication_smoke_plan.yaml"
    )
    assert smoke["release_kind"] == "smoke"
    assert smoke["seeds"] == [321, 323]


def test_publication_plan_rejects_underpowered_seed_set(tmp_path: Path) -> None:
    source = yaml.safe_load(
        (ROOT / "configs" / "transformer_publication_plan.yaml").read_text(
            encoding="utf-8"
        )
    )
    source["seeds"] = [101]
    plan = tmp_path / "underpowered.yaml"
    plan.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="at least 5 independent seeds"):
        validate_publication_plan(plan)


def test_generated_source_config_binds_plan_seed_and_release() -> None:
    plan = validate_publication_plan(
        ROOT / "configs" / "transformer_publication_smoke_plan.yaml"
    )
    template = plan["tasks"][0]["config"]
    generated = _generated_config(
        template,
        seed=323,
        run_id="release_modular_seed323",
        release_id="release",
        plan_version="transformer_publication_plan_v1",
        plan_sha256="a" * 64,
    )
    assert generated["run"]["seed"] == 323
    assert generated["run"]["name"] == "release_modular_seed323"
    assert generated["protocol"]["publication_release_id"] == "release"
    assert generated["protocol"]["publication_plan_sha256"] == "a" * 64


def test_analysis_plan_protocol_is_top_level() -> None:
    plan = yaml.safe_load(
        (ROOT / "configs" / "transformer_publication_smoke_plan.yaml").read_text(
            encoding="utf-8"
        )
    )
    expected = {
        "protocol_version": SYNTHETIC_TRANSFORMER_PROTOCOL_VERSION,
        "primary_scaling_mode": "fixed_update_scale",
        "primary_allocation_rule": "soft_dimension",
        "primary_reference_rule": "uniform_exact_cost",
        "primary_metric": "final_val_loss",
        "independent_unit": "task_seed_run",
    }
    _validate_analysis_plan_roles(plan, expected)

    stale = dict(plan)
    stale["protocol_version"] = "stale"
    with pytest.raises(ValueError, match="analysis plan protocol_version"):
        _validate_analysis_plan_roles(stale, expected)


def test_release_snapshot_contains_publication_driver_and_plan(tmp_path: Path) -> None:
    destination = tmp_path / "snapshot"
    _copy_code_snapshot(destination)
    assert (destination / "scripts" / "run_transformer_publication.py").is_file()
    assert (destination / "scripts" / "run_transformer_publication.sh").is_file()
    assert (destination / "configs" / "transformer_publication_plan.yaml").is_file()


def _write_primary_budget_fixture(path: Path, *, unequal_cost: bool = False) -> None:
    import pandas as pd

    rows = []
    values = {
        (64, 0): (3.0, 2.0, 0.60, 0.50, 12),
        (64, 1): (5.0, 3.0, 0.70, 0.50, 12),
        (128, 0): (8.0, 4.0, 0.80, 0.60, 16),
        (128, 1): (10.0, 6.0, 0.90, 0.70, 16),
    }
    for key, metrics in values.items():
        budget, replicate = key
        candidate_loss, reference_loss, candidate_acc, reference_acc, cost = metrics
        baseline_id = f"uniform_exact_cost__{cost}"
        rows.append(
            {
                "scaling_mode": "fixed_update_scale",
                "condition_id": "soft_dimension",
                "rule": "soft_dimension",
                "comparison_role": "candidate",
                "matched_baseline_condition_id": baseline_id,
                "is_primary_allocation_rule": True,
                "budget": budget,
                "adaptation_replicate": replicate,
                "actual_cost": cost,
                "final_val_loss": candidate_loss,
                "final_val_accuracy": candidate_acc,
            }
        )
        rows.append(
            {
                "scaling_mode": "fixed_update_scale",
                "condition_id": baseline_id,
                "rule": "uniform_exact_cost",
                "comparison_role": "exact_cost_baseline",
                "matched_baseline_condition_id": "",
                "is_primary_allocation_rule": False,
                "budget": budget,
                "adaptation_replicate": replicate,
                "actual_cost": cost
                + (1 if unequal_cost and budget == 64 and replicate == 0 else 0),
                "final_val_loss": reference_loss,
                "final_val_accuracy": reference_acc,
            }
        )
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_primary_run_delta_is_recomputed_from_raw_exact_cost_rows(
    tmp_path: Path,
) -> None:
    from scripts.validate_transformer_release import _recompute_primary_run_deltas

    release_dir = tmp_path / "release"
    run_id = "release_modular_seed101"
    relative = Path("source_runs") / run_id
    _write_primary_budget_fixture(
        release_dir / relative / "budget" / "budget_results.csv"
    )
    manifest = {
        "primary_scaling_mode": "fixed_update_scale",
        "primary_allocation_rule": "soft_dimension",
    }
    release_by_id = {run_id: {"relative_path": relative.as_posix()}}
    summary_by_id = {
        run_id: {
            "task_name": "modular",
            "base_seed": 101,
            "adaptation_replicates": 2,
        }
    }
    result = _recompute_primary_run_deltas(
        manifest, release_dir, release_by_id, summary_by_id
    )
    assert len(result) == 1
    row = result.iloc[0]
    assert row["n_budgets"] == 2
    assert row["mean_adaptation_replicates"] == pytest.approx(2.0)
    assert row["loss_delta_vs_exact_uniform"] == pytest.approx(2.75)
    assert row["acc_delta_vs_exact_uniform"] == pytest.approx(0.175)
    assert row["mean_actual_cost"] == pytest.approx(14.0)


def test_primary_run_delta_rejects_unequal_exact_cost_pair(
    tmp_path: Path,
) -> None:
    from scripts.validate_transformer_release import _recompute_primary_run_deltas

    release_dir = tmp_path / "release"
    run_id = "release_modular_seed101"
    relative = Path("source_runs") / run_id
    _write_primary_budget_fixture(
        release_dir / relative / "budget" / "budget_results.csv",
        unequal_cost=True,
    )
    with pytest.raises(ValueError, match="unequal costs"):
        _recompute_primary_run_deltas(
            {
                "primary_scaling_mode": "fixed_update_scale",
                "primary_allocation_rule": "soft_dimension",
            },
            release_dir,
            {run_id: {"relative_path": relative.as_posix()}},
            {
                run_id: {
                    "task_name": "modular",
                    "base_seed": 101,
                    "adaptation_replicates": 2,
                }
            },
        )
