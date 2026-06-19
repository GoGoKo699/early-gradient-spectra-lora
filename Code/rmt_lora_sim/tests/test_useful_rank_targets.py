from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rmt_lora.provenance import verify_sha256s, write_sha256s
from rmt_lora.targets import (
    TARGET_DENOMINATOR,
    TARGET_ESTIMAND,
    TARGET_ESTIMAND_VERSION,
    TARGET_REFERENCE,
    annotate_observed_best_curve,
    summarize_observed_best_curve,
    target_definition_document,
    validate_target_estimand_frame,
)


def _curve(oracle: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "layer": [0, 0, 0],
            "rank": [1, 2, 4],
            "base_val_loss": [10.0, 10.0, 10.0],
            "oracle_val_loss": [oracle, oracle, oracle],
            "final_val_loss": [9.0, 9.0, 8.0],
        }
    )


def _summary(frame: pd.DataFrame) -> dict:
    return summarize_observed_best_curve(
        frame,
        recovery_targets=[0.7],
        gap_tolerances=[0.1],
        penalty_lambdas=[0.2],
    )


def test_targets_use_observed_best_gap_not_oracle_gap() -> None:
    first = _summary(_curve(oracle=0.0))
    second = _summary(_curve(oracle=-1000.0))

    assert first["best_observed_val_loss"] == pytest.approx(8.0)
    assert first["observed_best_gap"] == pytest.approx(2.0)
    assert first["near_best_rank_gap_0.1"] == 4
    assert first["recovery_rank_0.7"] == 4
    assert first["penalized_rank_lambda_0.2"] == 4

    for key in (
        "best_observed_val_loss",
        "observed_best_gap",
        "near_best_rank_gap_0.1",
        "recovery_rank_0.7",
        "penalized_rank_lambda_0.2",
    ):
        assert second[key] == first[key]
    assert second["oracle_improvement_gap"] != first["oracle_improvement_gap"]


def test_curve_annotations_are_observed_best_normalized() -> None:
    annotated = annotate_observed_best_curve(_curve())
    assert annotated["normalized_residual_observed_best"].tolist() == pytest.approx(
        [0.5, 0.5, 0.0]
    )
    assert annotated["recovery_fraction_observed_best"].tolist() == pytest.approx(
        [0.5, 0.5, 1.0]
    )
    assert set(annotated["target_estimand"]) == {TARGET_ESTIMAND}


def test_nonpositive_observed_improvement_is_undefined() -> None:
    frame = pd.DataFrame(
        {
            "rank": [1, 2, 4],
            "base_val_loss": [1.0, 1.0, 1.0],
            "oracle_val_loss": [0.0, 0.0, 0.0],
            "final_val_loss": [1.0, 1.1, 1.2],
        }
    )
    result = _summary(frame)
    assert result["target_valid"] is False
    assert result["target_status"] == "nonpositive_observed_improvement"
    assert np.isnan(result["near_best_rank_gap_0.1"])
    assert np.isnan(result["recovery_rank_0.7"])
    assert np.isnan(result["penalized_rank_lambda_0.2"])


def test_smallest_rank_breaks_target_and_best_ties() -> None:
    frame = pd.DataFrame(
        {
            "rank": [1, 2, 4],
            "base_val_loss": [10.0] * 3,
            "oracle_val_loss": [0.0] * 3,
            "final_val_loss": [8.0, 8.0, 8.0],
        }
    )
    result = summarize_observed_best_curve(
        frame,
        recovery_targets=[1.0],
        gap_tolerances=[0.0],
        penalty_lambdas=[0.0],
    )
    assert result["best_rank"] == 1
    assert result["near_best_rank_gap_0"] == 1
    assert result["recovery_rank_1"] == 1
    assert result["penalized_rank_lambda_0"] == 1


def test_metadata_validator_rejects_legacy_or_mixed_frames() -> None:
    legacy = pd.DataFrame({"target": ["recovery_rank_0.7"]})
    with pytest.raises(ValueError, match="legacy oracle-gap artifacts"):
        validate_target_estimand_frame(legacy, context="legacy.csv")

    mixed = pd.DataFrame(
        {
            "target_estimand": [TARGET_ESTIMAND, "oracle_loss_gap"],
            "target_estimand_version": [TARGET_ESTIMAND_VERSION] * 2,
            "target_reference": [TARGET_REFERENCE] * 2,
            "target_denominator": [TARGET_DENOMINATOR] * 2,
        }
    )
    with pytest.raises(ValueError, match="invalid or mixed"):
        validate_target_estimand_frame(mixed, context="mixed.csv")


def test_target_definition_is_machine_readable_and_explicit() -> None:
    payload = target_definition_document()
    encoded = json.dumps(payload, allow_nan=False)
    assert TARGET_ESTIMAND_VERSION in encoded
    assert "oracle_val_loss is diagnostic only" in encoded
    assert payload["target_denominator"] == TARGET_DENOMINATOR


def test_strict_checksums_reject_unlisted_files(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("one\n", encoding="utf-8")
    write_sha256s(tmp_path)
    assert verify_sha256s(tmp_path, reject_unlisted=True) == []
    (tmp_path / "two.txt").write_text("two\n", encoding="utf-8")
    errors = verify_sha256s(tmp_path, reject_unlisted=True)
    assert errors == ["unlisted file: two.txt"]


def test_stage4_predictor_transform_matches_manuscript() -> None:
    from rmt_lora.stage4_analysis import (
        PREDICTOR_TRANSFORM,
        spearman_or_nan,
        transform_spectral_predictor,
    )

    assert PREDICTOR_TRANSFORM == "log2(1+s)"
    assert transform_spectral_predictor([0.0, 1.0, 3.0]).tolist() == pytest.approx(
        [0.0, 1.0, 2.0]
    )
    with pytest.raises(ValueError, match="nonnegative"):
        transform_spectral_predictor([-1.0])
    assert spearman_or_nan([1, 2, 3], [2, 4, 8]) == pytest.approx(1.0)
    assert np.isnan(spearman_or_nan([1, 1, 1], [2, 3, 4]))
    with pytest.raises(ValueError, match="matching shapes"):
        spearman_or_nan([1, 2], [1])


def test_stage4_analysis_plan_declares_one_primary_pair() -> None:
    from rmt_lora.stage4_analysis import (
        PRIMARY_CONDITION,
        PRIMARY_PREDICTOR,
        PRIMARY_TARGET,
        analysis_plan_document,
        analysis_role,
    )

    plan = analysis_plan_document()
    primary = plan["primary_hypothesis"]
    assert primary["condition"] == PRIMARY_CONDITION
    assert primary["target"] == PRIMARY_TARGET
    assert primary["predictor"] == PRIMARY_PREDICTOR
    assert analysis_role(PRIMARY_CONDITION, PRIMARY_TARGET, PRIMARY_PREDICTOR) == (
        "confirmatory_primary"
    )
    assert analysis_role("sample_limited", PRIMARY_TARGET, PRIMARY_PREDICTOR) == (
        "confirmatory_robustness"
    )
