#!/usr/bin/env python3
"""Regenerate paper-facing tables and vector figures from released CSV artifacts.

Run from the clean package ``Paper/`` directory:

    python3 make_paper_artifacts.py

The script is intentionally deterministic and uses only files shipped in this
repository.  It also forces Matplotlib PDF/PS output to Type 42 fonts so the
compiled paper avoids Type 3 fonts.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import sys

import matplotlib as mpl

# Ignore user/site Matplotlib configuration and force a headless backend. The
# exact Matplotlib wheel and its bundled fonts are checksum-locked separately.
mpl.use("Agg")
mpl.rcdefaults()
mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
CODE = ROOT / "Code"
if not CODE.exists():
    CODE = ROOT
TABLES = PAPER / "tables"
GENERATED = TABLES / "generated"
FIGURES = PAPER / "figures"
RMT_RESULTS = CODE / "rmt_lora_sim" / "results" / "released"
RMT_RUNS_LEGACY = CODE / "rmt_lora_sim" / "runs"
sys.path.insert(0, str(CODE / "rmt_lora_sim"))

from rmt_lora.targets import (  # noqa: E402
    TARGET_ESTIMAND_VERSION,
    validate_target_estimand_frame,
)

CONDITION_DISPLAY = {"hard_knee": "Hard-knee", "sample_limited": "Sample-limited"}
TARGET_ORDER = [
    "near-best 0.1 gap",
    "near-best 0.2 gap",
    "70% recovery",
    "80% recovery",
    "penalty 0.2",
    "penalty 0.3",
]
TRANSFORMER_TABLE_ROOT = TABLES / "transformer_publication"
TRANSFORMER_PLAN_SHA256 = "89c74a8d3e773cd20950bf2a9854bcdf39cd6f0a38a8eed78ca0d3345c85be6d"
TRANSFORMER_PLAN_VERSION = "transformer_publication_plan_v1"
TRANSFORMER_PROTOCOL = "synthetic_transformer_publication_protocol_v4"
TRANSFORMER_RELEASE_ID = "transformer_publication_20260622T101458Z"
TRANSFORMER_ARCHIVE_SHA256 = "b1deb63807b04a4cd1032a7152d9664ff4b47929dd68ce99afc4317d500482e3"
TRANSFORMER_PROVENANCE_COLUMNS = (
    "source_release_id",
    "source_archive_sha256",
    "analysis_plan_sha256",
    "analysis_plan_version",
    "protocol_version",
)
TRANSFORMER_OUTPUT_FILES = {
    "primary": "primary_analysis.csv",
    "by_task": "primary_analysis_by_task.csv",
    "run_deltas": "primary_run_deltas.csv",
    "scaling": "scaling_sensitivity.csv",
    "site": "site_prediction_selected.csv",
}
TRANSFORMER_TASK_DISPLAY = {
    "associative_recall": "Associative recall",
    "modular": "Modular arithmetic",
}
TRANSFORMER_SCALING_DISPLAY = {
    "fixed_update_scale": r"Fixed update scale",
    "standard": r"Standard $\alpha/r$",
    "rslora": r"rsLoRA",
}
TRANSFORMER_PREDICTOR_DISPLAY = {
    "effective_rank": "Effective rank",
    "soft_dimension": "Soft dimension",
}
REAL_LORA_TABLE_ROOT = TABLES / "real_lora_publication"
REAL_LORA_PLAN_SHA256 = "e407ba9f34946cfaa6a1246ebd55247b4fdc208d22de5bc1bd279d0c63a7da7b"
REAL_LORA_PLAN_VERSION = "real_lora_publication_plan_v1"
REAL_LORA_PROTOCOL = "real_lora_publication_protocol_v3"
REAL_LORA_RELEASE_ID = "real_lora_publication_20260623T074520Z"
REAL_LORA_ARCHIVE_SHA256 = "4368ca32afd32fde04c68e1b889bcba2add57a801d8360db4e99557953c991de"
REAL_LORA_SOURCE_GIT_COMMIT = "9d8547ca172ffcc8e059351b8d91a4fc92d36b7e"
REAL_LORA_PROVENANCE_COLUMNS = (
    "source_release_id",
    "source_archive_sha256",
    "analysis_plan_sha256",
    "analysis_plan_version",
    "protocol_version",
    "source_git_commit",
)
REAL_LORA_OUTPUT_FILES = {
    "primary": "primary_analysis.csv",
    "primary_seeds": "primary_seed_deltas.csv",
    "analysis": "analysis_by_suite_strategy.csv",
    "summary": "strategy_summary.csv",
    "spectral_seeds": "spectral_seed_deltas.csv",
}
REAL_LORA_SUITE_DISPLAY = {
    "cattn_cfc": r"$c_{attn},c_{fc}$",
    "attnproj": r"$+a_{proj}$",
}
REAL_LORA_STRATEGY_DISPLAY = {
    "spectral_effective": "spectral effective",
    "eva_activation": "EVA-style allocation",
    "uniform_r4": "uniform rank 4",
    "gora_sensitivity": "GoRA-style allocation",
    "gradient_norm": "gradient norm",
    "fim_gradient_variance": "FIM-LoRA-style allocation",
}
REAL_LORA_STRATEGY_ORDER = (
    "spectral_effective",
    "eva_activation",
    "uniform_r4",
    "gora_sensitivity",
    "gradient_norm",
    "fim_gradient_variance",
)
ROW_END = r" \\"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_dirs() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def _tex_escape(text: object) -> str:
    s = str(text)
    # Keep math symbols inserted by this script untouched; escape table text only.
    return (
        s.replace("%", r"\%")
        .replace("_", r"\_")
        .replace("&", r"\&")
    )


def _fmt(x: float, digits: int = 4, signed: bool = False) -> str:
    if pd.isna(x):
        return "--"
    return format(float(x), f"+.{digits}f" if signed else f".{digits}f")


def _sem(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) <= 1:
        return float("nan")
    return float(vals.std(ddof=1) / math.sqrt(len(vals)))


def _write(path: Path, text: str) -> None:
    r"""Write generated artifacts.

    LaTeX tabular bodies are input inside a live ``tabular`` environment.
    Some TeX installations do not accept a booktabs rule immediately after an
    ``\input`` file that ends with ``\\``.  For generated row files, leave
    the final row terminator to the caller (``\input{...}\\``) while keeping
    interior row terminators inside the file.
    """
    if path.suffix == ".tex" and path.parent == GENERATED:
        lines = text.rstrip("\n").splitlines()
        if lines:
            lines[-1] = re.sub(r"\s*\\\\\s*$", "", lines[-1])
            text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_stage4_table() -> pd.DataFrame:
    path = TABLES / "stage4_gradient_effective_summary.csv"
    frame = pd.read_csv(path)
    validate_target_estimand_frame(frame, context=str(path))
    if set(frame["target_estimand_version"].astype(str)) != {TARGET_ESTIMAND_VERSION}:
        raise ValueError(f"{path} has a non-current Stage4 target version")
    if set(pd.to_numeric(frame["n_runs"], errors="raise")) != {5}:
        raise ValueError(f"{path} is not a five-seed publication table")
    if frame["source_aggregate_sha256"].nunique() != 1:
        raise ValueError(f"{path} mixes multiple Stage4 source aggregates")
    return frame


def make_stage4_rows() -> None:
    df = _load_stage4_table()
    lines: list[str] = []
    for ci, condition in enumerate(["hard_knee", "sample_limited"]):
        if ci:
            lines.append(r"\midrule")
        sub = df[df["condition"] == condition].copy()
        sub["target_order"] = sub["target_display"].map({t: i for i, t in enumerate(TARGET_ORDER)})
        sub = sub.sort_values("target_order")
        for _, row in sub.iterrows():
            lines.append(
                f"{CONDITION_DISPLAY[condition]} & {_tex_escape(row['target_display'])} & "
                f"{float(row['mean_r2']):.3f} ({float(row['sem_r2']):.3f}) & "
                f"{float(row['mean_spearman']):.3f} ({float(row['sem_spearman']):.3f}) \\\\" 
            )
    _write(GENERATED / "stage4_rows.tex", "\n".join(lines) + "\n")


def _load_transformer_publication_tables() -> dict[str, pd.DataFrame]:
    """Load only checksum-bound tables produced by the post-CRN importer."""

    source_path = TRANSFORMER_TABLE_ROOT / "source.json"
    if not source_path.is_file():
        raise FileNotFoundError(
            f"missing {source_path}; run Paper/import_transformer_publication.py first"
        )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError(f"{source_path} must contain a JSON object")

    expected_source = {
        "source_release_id": TRANSFORMER_RELEASE_ID,
        "source_archive_sha256": TRANSFORMER_ARCHIVE_SHA256,
        "analysis_plan_sha256": TRANSFORMER_PLAN_SHA256,
        "analysis_plan_version": TRANSFORMER_PLAN_VERSION,
        "protocol_version": TRANSFORMER_PROTOCOL,
        "n_recursive_checksums": 350,
        "n_source_runs": 10,
        "n_exact_candidate_pairs": 2925,
        "n_identical_allocation_groups": 819,
        "max_identical_metric_spread": 0.0,
        "max_identical_seed_disagreement": 0,
        "n_diverged_rows": 0,
        "primary_run_wins": 3,
        "primary_task_wins": 0,
        "importer": "import_transformer_publication.py",
    }
    for key, expected in expected_source.items():
        observed = source.get(key)
        if isinstance(expected, float):
            if not math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=0.0):
                raise ValueError(
                    f"{source_path}: {key}={observed!r}; expected {expected!r}"
                )
        elif observed != expected:
            raise ValueError(
                f"{source_path}: {key}={observed!r}; expected {expected!r}"
            )

    expected_primary = {
        "primary_mean_loss_delta": 0.014638354179882738,
        "primary_exact_sign_flip_p_two_sided": 0.240234375,
    }
    for key, expected in expected_primary.items():
        if not math.isclose(
            float(source.get(key)), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{source_path}: {key} disagrees with the verified release")
    interval = source.get("primary_cluster_bootstrap_ci")
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError(f"{source_path} lacks the primary bootstrap interval")
    for observed, expected in zip(
        interval, (-0.007750680089594088, 0.03521469451846864)
    ):
        if not math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{source_path}: primary interval disagrees with release")

    output_manifest = source.get("outputs")
    if not isinstance(output_manifest, dict):
        raise ValueError(f"{source_path} lacks paper-output checksums")
    expected_names = set(TRANSFORMER_OUTPUT_FILES.values())
    observed_names = {path.name for path in TRANSFORMER_TABLE_ROOT.glob("*.csv")}
    if observed_names != expected_names:
        raise ValueError(
            "transformer publication table set is not exact; "
            f"unexpected={sorted(observed_names - expected_names)}, "
            f"missing={sorted(expected_names - observed_names)}"
        )

    frames: dict[str, pd.DataFrame] = {}
    expected_provenance = {
        "source_release_id": TRANSFORMER_RELEASE_ID,
        "source_archive_sha256": TRANSFORMER_ARCHIVE_SHA256,
        "analysis_plan_sha256": TRANSFORMER_PLAN_SHA256,
        "analysis_plan_version": TRANSFORMER_PLAN_VERSION,
        "protocol_version": TRANSFORMER_PROTOCOL,
    }
    for key, filename in TRANSFORMER_OUTPUT_FILES.items():
        path = TRANSFORMER_TABLE_ROOT / filename
        record = output_manifest.get(filename)
        if not isinstance(record, dict):
            raise ValueError(f"{source_path} lacks output record {filename!r}")
        if record.get("relative_path") != path.relative_to(PAPER).as_posix():
            raise ValueError(f"{source_path} records the wrong path for {filename}")
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {path}; run Paper/import_transformer_publication.py first"
            )
        if _sha256(path) != record.get("sha256"):
            raise ValueError(f"paper transformer table checksum mismatch: {path}")
        frame = pd.read_csv(path)
        if frame.empty or int(record.get("rows", -1)) != len(frame):
            raise ValueError(f"{path} is empty or its recorded row count is stale")
        missing = sorted(set(TRANSFORMER_PROVENANCE_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"{path} lacks provenance columns {missing}")
        for column, expected in expected_provenance.items():
            values = set(frame[column].dropna().astype(str))
            if values != {str(expected)}:
                raise ValueError(f"{path} has mixed or stale {column}: {values}")
        frames[key] = frame

    primary = frames["primary"]
    if len(primary) != 1:
        raise ValueError("transformer primary table must contain exactly one row")
    row = primary.iloc[0]
    required_primary = {
        "analysis_scope": "task_stratified_omnibus",
        "task_weighting": "equal_weight_across_task_families",
        "generalization_scope": "conditional_on_prespecified_task_families",
        "scaling_mode": "fixed_update_scale",
        "allocation_rule": "soft_dimension",
        "reference_rule": "uniform_exact_cost",
        "metric": "final_val_loss",
    }
    for column, expected in required_primary.items():
        if str(row[column]) != expected:
            raise ValueError(f"primary {column}={row[column]!r}; expected {expected!r}")
    for column, expected in {
        "n_task_families": 2,
        "n_independent_runs": 10,
        "min_runs_per_task": 5,
        "max_runs_per_task": 5,
        "run_wins_loss": 3,
        "task_wins_loss": 0,
    }.items():
        if int(row[column]) != expected:
            raise ValueError(f"primary {column}={row[column]!r}; expected {expected}")
    for column, expected in {
        "mean_loss_delta": expected_primary["primary_mean_loss_delta"],
        "cluster_bootstrap_ci_low": float(interval[0]),
        "cluster_bootstrap_ci_high": float(interval[1]),
        "exact_sign_flip_p_two_sided": expected_primary[
            "primary_exact_sign_flip_p_two_sided"
        ],
        "mean_accuracy_delta": -0.0039236059494904465,
    }.items():
        if not math.isclose(float(row[column]), expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"primary {column} disagrees with {source_path}")

    by_task = frames["by_task"]
    if len(by_task) != 2 or set(by_task["task_family"].astype(str)) != set(
        TRANSFORMER_TASK_DISPLAY
    ):
        raise ValueError("transformer task table is incomplete")
    if set(pd.to_numeric(by_task["n_independent_runs"], errors="raise")) != {5}:
        raise ValueError("each transformer task must contain five independent runs")

    runs = frames["run_deltas"]
    observed_units = set(
        zip(
            runs["task_family"].astype(str),
            pd.to_numeric(runs["seed"], errors="raise").astype(int),
        )
    )
    expected_units = {
        (task, seed)
        for task in TRANSFORMER_TASK_DISPLAY
        for seed in (101, 103, 107, 109, 113)
    }
    if len(runs) != 10 or observed_units != expected_units:
        raise ValueError("transformer run table differs from the frozen design")
    if not np.allclose(
        pd.to_numeric(runs["mean_adaptation_replicates"], errors="raise"), 3.0
    ):
        raise ValueError("run summaries must average three adaptation replicates")

    scaling = frames["scaling"]
    if len(scaling) != 3 or set(scaling["scaling_mode"].astype(str)) != set(
        TRANSFORMER_SCALING_DISPLAY
    ):
        raise ValueError("transformer scaling table is incomplete")
    fixed = scaling[scaling["scaling_mode"].astype(str).eq("fixed_update_scale")]
    if len(fixed) != 1 or str(fixed.iloc[0]["analysis_role"]) != "confirmatory_primary":
        raise ValueError("fixed-update-scale row is not marked as confirmatory")

    site = frames["site"]
    expected_site = {
        (task, predictor)
        for task in TRANSFORMER_TASK_DISPLAY
        for predictor in TRANSFORMER_PREDICTOR_DISPLAY
    }
    observed_site = set(
        zip(site["task_family"].astype(str), site["predictor"].astype(str))
    )
    if len(site) != 4 or observed_site != expected_site:
        raise ValueError("transformer site-prediction table is incomplete")
    if set(site["scaling_mode"].astype(str)) != {"fixed_update_scale"}:
        raise ValueError("transformer site-prediction table is not fixed-scale")
    if set(site["target"].astype(str)) != {"near_best_rank_gap_0.1"}:
        raise ValueError("transformer site table uses the wrong useful-rank target")

    return frames


def _load_real_lora_publication_tables() -> dict[str, pd.DataFrame]:
    """Load only checksum-bound tables from the controlled 8+3 GPT-2 release."""

    source_path = REAL_LORA_TABLE_ROOT / "source.json"
    if not source_path.is_file():
        raise FileNotFoundError(
            f"missing {source_path}; run Paper/import_real_lora_results.py first"
        )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError(f"{source_path} must contain a JSON object")

    expected_source = {
        "source_release_id": REAL_LORA_RELEASE_ID,
        "source_archive_sha256": REAL_LORA_ARCHIVE_SHA256,
        "analysis_plan_sha256": REAL_LORA_PLAN_SHA256,
        "analysis_plan_version": REAL_LORA_PLAN_VERSION,
        "protocol_version": REAL_LORA_PROTOCOL,
        "source_git_commit": REAL_LORA_SOURCE_GIT_COMMIT,
        "release_schema_version": "real_lora_publication_release_v1",
        "n_recursive_checksums": 354,
        "n_aggregate_checksums": 9,
        "n_source_runs": 11,
        "n_run_checksum_entries": 297,
        "n_primary_runs": 8,
        "n_boundary_runs": 3,
        "n_paired_candidate_rows": 55,
        "n_result_rows": 77,
        "n_allocation_rows": 2100,
        "identity_control_runs": 11,
        "exact_cost_runs": 11,
        "activity_gate_runs": 11,
        "primary_wins": 8,
        "primary_ties": 0,
        "primary_losses": 0,
        "importer": "import_real_lora_results.py",
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise ValueError(
                f"{source_path}: {key}={source.get(key)!r}; expected {expected!r}"
            )

    expected_floats = {
        "primary_mean_loss_delta": -0.0070257661864162,
        "primary_exact_sign_flip_p_two_sided": 0.0078125,
        "primary_holm_p_within_suite": 0.0390625,
        "boundary_spectral_mean_loss_delta": 0.0010871663689611,
    }
    for key, expected in expected_floats.items():
        if not math.isclose(float(source.get(key)), expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{source_path}: {key} disagrees with the verified release")
    interval = source.get("primary_bootstrap_ci")
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError(f"{source_path} lacks the primary bootstrap interval")
    for observed, expected in zip(interval, (-0.0083426544442773, -0.0060258745914326)):
        if not math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{source_path}: primary interval disagrees with release")

    output_manifest = source.get("outputs")
    if not isinstance(output_manifest, dict):
        raise ValueError(f"{source_path} lacks paper-output checksums")
    expected_names = set(REAL_LORA_OUTPUT_FILES.values())
    observed_names = {path.name for path in REAL_LORA_TABLE_ROOT.glob("*.csv")}
    if observed_names != expected_names:
        raise ValueError(
            "real-model publication table set is not exact; "
            f"unexpected={sorted(observed_names - expected_names)}, "
            f"missing={sorted(expected_names - observed_names)}"
        )

    expected_provenance = {
        "source_release_id": REAL_LORA_RELEASE_ID,
        "source_archive_sha256": REAL_LORA_ARCHIVE_SHA256,
        "analysis_plan_sha256": REAL_LORA_PLAN_SHA256,
        "analysis_plan_version": REAL_LORA_PLAN_VERSION,
        "protocol_version": REAL_LORA_PROTOCOL,
        "source_git_commit": REAL_LORA_SOURCE_GIT_COMMIT,
    }
    frames: dict[str, pd.DataFrame] = {}
    for key, filename in REAL_LORA_OUTPUT_FILES.items():
        path = REAL_LORA_TABLE_ROOT / filename
        record = output_manifest.get(filename)
        if not isinstance(record, dict):
            raise ValueError(f"{source_path} lacks output record {filename!r}")
        if record.get("relative_path") != path.relative_to(PAPER).as_posix():
            raise ValueError(f"{source_path} records the wrong path for {filename}")
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {path}; run Paper/import_real_lora_results.py first"
            )
        if _sha256(path) != record.get("sha256"):
            raise ValueError(f"paper real-model table checksum mismatch: {path}")
        frame = pd.read_csv(path)
        if frame.empty or int(record.get("rows", -1)) != len(frame):
            raise ValueError(f"{path} is empty or its recorded row count is stale")
        missing = sorted(set(REAL_LORA_PROVENANCE_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"{path} lacks provenance columns {missing}")
        for column, expected in expected_provenance.items():
            values = set(frame[column].dropna().astype(str))
            if values != {str(expected)}:
                raise ValueError(f"{path} has mixed or stale {column}: {values}")
        frames[key] = frame

    primary = frames["primary"]
    if len(primary) != 1:
        raise ValueError("real-model primary table must contain exactly one row")
    row = primary.iloc[0]
    required_primary = {
        "analysis_scope": "prespecified_primary_suite",
        "suite_id": "cattn_cfc",
        "suite_role": "primary",
        "candidate_strategy": "spectral_effective",
        "reference_strategy": "uniform_r4",
        "delta_definition": "candidate_minus_reference",
    }
    for column, expected in required_primary.items():
        if str(row[column]) != expected:
            raise ValueError(f"real-model primary {column}={row[column]!r}; expected {expected!r}")
    for column, expected in {
        "n_independent_runs": 8,
        "n_nonzero_deltas": 8,
        "wins": 8,
        "ties": 0,
        "losses": 0,
        "distinct_assignment_runs": 8,
        "identical_assignment_runs": 0,
        "bootstrap_resamples": 10000,
    }.items():
        if int(row[column]) != expected:
            raise ValueError(f"real-model primary {column}={row[column]!r}; expected {expected}")
    for column, expected in {
        "mean_loss_delta": expected_floats["primary_mean_loss_delta"],
        "bootstrap_ci_low": float(interval[0]),
        "bootstrap_ci_high": float(interval[1]),
        "exact_sign_flip_p_two_sided": expected_floats[
            "primary_exact_sign_flip_p_two_sided"
        ],
        "holm_p_within_suite": expected_floats["primary_holm_p_within_suite"],
    }.items():
        if not math.isclose(float(row[column]), expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"real-model primary {column} disagrees with {source_path}")

    analysis = frames["analysis"]
    candidates = {
        "gradient_norm",
        "spectral_effective",
        "eva_activation",
        "fim_gradient_variance",
        "gora_sensitivity",
    }
    expected_pairs = {(suite, strategy) for suite in REAL_LORA_SUITE_DISPLAY for strategy in candidates}
    observed_pairs = set(zip(analysis["suite_id"].astype(str), analysis["candidate_strategy"].astype(str)))
    if len(analysis) != 10 or observed_pairs != expected_pairs:
        raise ValueError("real-model suite-by-strategy analysis is incomplete")
    if set(analysis["reference_strategy"].astype(str)) != {"uniform_r4"}:
        raise ValueError("real-model comparisons do not use the frozen uniform reference")
    run_counts = analysis.groupby("suite_id")["n_independent_runs"].first().to_dict()
    if {str(k): int(v) for k, v in run_counts.items()} != {"attnproj": 3, "cattn_cfc": 8}:
        raise ValueError("real-model suite run counts differ from the frozen design")

    summary = frames["summary"]
    expected_summary = {
        (suite, strategy)
        for suite in REAL_LORA_SUITE_DISPLAY
        for strategy in REAL_LORA_STRATEGY_ORDER
    }
    observed_summary = set(zip(summary["suite_id"].astype(str), summary["strategy"].astype(str)))
    if len(summary) != 12 or observed_summary != expected_summary:
        raise ValueError("real-model strategy summary is incomplete")
    expected_params = {"cattn_cfc": 331776.0, "attnproj": 405504.0}
    for suite, expected in expected_params.items():
        subset = summary[summary["suite_id"].astype(str).eq(suite)]
        if set(pd.to_numeric(subset["mean_trainable_params"], errors="raise")) != {expected}:
            raise ValueError(f"{suite} strategies are not exact-cost matched")

    primary_seeds = frames["primary_seeds"]
    if len(primary_seeds) != 8 or set(pd.to_numeric(primary_seeds["seed"], errors="raise").astype(int)) != {101, 103, 107, 109, 113, 127, 131, 137}:
        raise ValueError("real-model primary seed table differs from the frozen design")
    spectral_seeds = frames["spectral_seeds"]
    spectral_units = set(
        zip(
            spectral_seeds["suite_id"].astype(str),
            pd.to_numeric(spectral_seeds["seed"], errors="raise").astype(int),
        )
    )
    expected_units = {
        *(("cattn_cfc", seed) for seed in (101, 103, 107, 109, 113, 127, 131, 137)),
        *(("attnproj", seed) for seed in (101, 103, 107)),
    }
    if len(spectral_seeds) != 11 or spectral_units != expected_units:
        raise ValueError("real-model spectral seed table differs from the frozen design")

    return frames


def _latex_row(*cells: object) -> str:
    return " & ".join(str(cell) for cell in cells) + ROW_END


def make_transformer_publication_rows() -> None:
    frames = _load_transformer_publication_tables()
    primary = frames["primary"].iloc[0]
    by_task = frames["by_task"]

    rows: list[str] = [
        _latex_row(
            "Omnibus (equal task weight)",
            int(primary["n_independent_runs"]),
            f"${_fmt(primary['mean_loss_delta'], 4, signed=True)}$",
            f"$[{_fmt(primary['cluster_bootstrap_ci_low'], 4, signed=True)},\\,{_fmt(primary['cluster_bootstrap_ci_high'], 4, signed=True)}]$",
            f"${_fmt(primary['exact_sign_flip_p_two_sided'], 4)}$",
            f"${int(primary['run_wins_loss'])}/{int(primary['n_independent_runs'])}$",
            f"${_fmt(primary['mean_accuracy_delta'], 4, signed=True)}$",
        )
    ]
    for task in ("associative_recall", "modular"):
        row = by_task[by_task["task_family"].astype(str).eq(task)].iloc[0]
        rows.append(
            _latex_row(
                TRANSFORMER_TASK_DISPLAY[task],
                int(row["n_independent_runs"]),
                f"${_fmt(row['mean_loss_delta'], 4, signed=True)}$",
                f"$[{_fmt(row['cluster_bootstrap_ci_low'], 4, signed=True)},\\,{_fmt(row['cluster_bootstrap_ci_high'], 4, signed=True)}]$",
                f"${_fmt(row['exact_sign_flip_p_two_sided'], 4)}$",
                f"${int(row['run_wins_loss'])}/{int(row['n_independent_runs'])}$",
                f"${_fmt(row['mean_accuracy_delta'], 4, signed=True)}$",
            )
        )
    _write(GENERATED / "transformer_primary_rows.tex", "\n".join(rows) + "\n")

    scaling = frames["scaling"].copy()
    scaling["_order"] = scaling["scaling_mode"].map(
        {"fixed_update_scale": 0, "standard": 1, "rslora": 2}
    )
    scaling = scaling.sort_values("_order")
    rows = []
    for _, row in scaling.iterrows():
        role = "Primary" if row["analysis_role"] == "confirmatory_primary" else "Exploratory"
        rows.append(
            _latex_row(
                TRANSFORMER_SCALING_DISPLAY[str(row["scaling_mode"])],
                role,
                f"${_fmt(row['mean_loss_delta'], 4, signed=True)}$",
                f"${_fmt(row['exact_sign_flip_p_two_sided'], 4)}$",
                f"${int(row['run_wins_loss'])}/{int(row['n_independent_runs'])}$",
                f"${_fmt(row['associative_recall_mean_loss_delta'], 4, signed=True)}$",
                f"${_fmt(row['modular_mean_loss_delta'], 4, signed=True)}$",
            )
        )
    _write(GENERATED / "transformer_scaling_rows.tex", "\n".join(rows) + "\n")

    site = frames["site"].copy()
    site["_task_order"] = site["task_family"].map(
        {"associative_recall": 0, "modular": 1}
    )
    site["_predictor_order"] = site["predictor"].map(
        {"effective_rank": 0, "soft_dimension": 1}
    )
    site = site.sort_values(["_task_order", "_predictor_order"])
    rows = []
    previous_task: str | None = None
    for _, row in site.iterrows():
        task = str(row["task_family"])
        if previous_task is not None and task != previous_task:
            rows.append(r"\midrule")
        rows.append(
            _latex_row(
                TRANSFORMER_TASK_DISPLAY[task],
                TRANSFORMER_PREDICTOR_DISPLAY[str(row["predictor"])],
                f"${_fmt(row['mean_spearman'], 3, signed=True)}$",
                f"${_fmt(row['sem_spearman'], 3)}$",
            )
        )
        previous_task = task
    _write(GENERATED / "transformer_sitewise_rows.tex", "\n".join(rows) + "\n")


def make_real_lora_publication_rows() -> None:
    frames = _load_real_lora_publication_tables()
    summary = frames["summary"].copy()
    analysis = frames["analysis"].copy()
    rows: list[str] = []
    for suite_index, suite in enumerate(("cattn_cfc", "attnproj")):
        if suite_index:
            rows.append(r"\midrule")
        for strategy in REAL_LORA_STRATEGY_ORDER:
            srow = summary[
                summary["suite_id"].astype(str).eq(suite)
                & summary["strategy"].astype(str).eq(strategy)
            ]
            if len(srow) != 1:
                raise ValueError(f"missing real-model summary row: {suite}/{strategy}")
            srow = srow.iloc[0]
            if strategy == "uniform_r4":
                delta = "$0$ (reference)"
                interval = "--"
                wins = "--"
            else:
                arow = analysis[
                    analysis["suite_id"].astype(str).eq(suite)
                    & analysis["candidate_strategy"].astype(str).eq(strategy)
                ]
                if len(arow) != 1:
                    raise ValueError(f"missing real-model analysis row: {suite}/{strategy}")
                arow = arow.iloc[0]
                delta = f"${_fmt(arow['mean_loss_delta'], 4, signed=True)}$"
                interval = (
                    f"$[{_fmt(arow['bootstrap_ci_low'], 4, signed=True)},\\,"
                    f"{_fmt(arow['bootstrap_ci_high'], 4, signed=True)}]$"
                )
                wins = f"${int(arow['wins'])}/{int(arow['n_independent_runs'])}$"
            rows.append(
                _latex_row(
                    REAL_LORA_SUITE_DISPLAY[suite],
                    REAL_LORA_STRATEGY_DISPLAY[strategy],
                    f"{_fmt(srow['mean_final_val_loss'], 4)} ({_fmt(srow['sem_final_val_loss'], 4)})",
                    delta,
                    interval,
                    wins,
                )
            )
    _write(GENERATED / "real_lora_rows.tex", "\n".join(rows) + "\n")


def make_transformer_publication_figure() -> None:
    frames = _load_transformer_publication_tables()
    runs = frames["run_deltas"].copy()
    task_order = ("associative_recall", "modular")
    runs["_task_order"] = runs["task_family"].map(
        {task: index for index, task in enumerate(task_order)}
    )
    runs = runs.sort_values(["_task_order", "seed"]).reset_index(drop=True)
    x = np.arange(len(runs), dtype=float)
    values = runs["loss_delta_vs_exact_uniform"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(5.3, 2.55))
    for task in task_order:
        mask = runs["task_family"].astype(str).eq(task).to_numpy()
        positions = x[mask]
        task_values = values[mask]
        ax.scatter(positions, task_values, label=TRANSFORMER_TASK_DISPLAY[task], zorder=3)
        ax.hlines(
            float(np.mean(task_values)),
            float(positions.min()) - 0.35,
            float(positions.max()) + 0.35,
            linestyles=":",
            linewidth=1.2,
        )
    ax.axhline(0.0, linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            ("AR" if task == "associative_recall" else "Mod") + f" {int(seed)}"
            for task, seed in zip(runs["task_family"], runs["seed"])
        ],
        rotation=35,
        ha="right",
    )
    ax.set_ylabel(r"loss $\Delta$: soft dimension $-$ exact uniform")
    ax.set_title("Confirmatory transformer run deltas")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, FIGURES / "transformer_primary_run_deltas.pdf")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        path,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def _find_run(name: str, legacy_pattern: str) -> Path:
    """Return a released RMT result directory.

    The cleaned release stores paper artifacts under semantic names in
    ``../Code/rmt_lora_sim/results/released``.  The fallback keeps the script usable
    with older working trees that still use timestamped ``runs`` folders.
    """
    direct = RMT_RESULTS / name
    if direct.exists():
        return direct
    matches = sorted(RMT_RUNS_LEGACY.glob(legacy_pattern))
    if matches:
        return matches[-1]
    raise FileNotFoundError(
        f"missing released RMT result {name!r} under {RMT_RESULTS}; "
        f"also tried legacy pattern {legacy_pattern!r} under {RMT_RUNS_LEGACY}"
    )


def make_bbp_figures() -> None:
    df = pd.read_csv(_find_run("bbp", "*_bbp") / "metrics.csv")
    g = df.groupby("theta", as_index=False).agg(
        top_sv_mean=("top_sv", "mean"),
        top_sv_sem=("top_sv", _sem),
        edge=("mp_edge", "mean"),
        overlap_mean=("mean_overlap", "mean"),
        overlap_sem=("mean_overlap", _sem),
    )

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.errorbar(g["theta"], g["overlap_mean"], yerr=g["overlap_sem"], marker="o", linewidth=1)
    ax.set_xlabel(r"spike strength $\theta$")
    ax.set_ylabel("mean singular-vector overlap")
    ax.set_title("BBP alignment")
    _save(fig, FIGURES / "bbp_alignment_summary.pdf")

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.errorbar(g["theta"], g["top_sv_mean"], yerr=g["top_sv_sem"], marker="o", linewidth=1)
    ax.plot(g["theta"], g["edge"], linestyle="--", linewidth=1)
    ax.set_xlabel(r"spike strength $\theta$")
    ax.set_ylabel("top singular value")
    ax.set_title("BBP top singular value")
    _save(fig, FIGURES / "bbp_top_sv_summary.pdf")


def make_lora_rank_figures() -> None:
    df = pd.read_csv(_find_run("lora_rank_clean", "*_lora_rank_clean") / "metrics.csv")
    g = df.groupby("rank", as_index=False).agg(
        val=("final_val_loss", "mean"),
        val_sem=("final_val_loss", _sem),
        det=("adapter_detectable_rank", "mean"),
        eff=("adapter_effective_rank", "mean"),
        stable=("adapter_stable_rank", "mean"),
    )
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.errorbar(g["rank"], g["val"], yerr=g["val_sem"], marker="o", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("nominal LoRA rank")
    ax.set_ylabel("validation MSE")
    ax.set_title("Clean rank sweep")
    _save(fig, FIGURES / "lora_rank_clean_val_loss.pdf")

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.plot(g["rank"], g["det"], marker="o", linewidth=1, label="detectable")
    ax.plot(g["rank"], g["eff"], marker="s", linewidth=1, label="effective")
    ax.plot(g["rank"], g["stable"], marker="^", linewidth=1, label="stable")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("nominal LoRA rank")
    ax.set_ylabel("final adapter statistic")
    ax.set_title("Final adapter spectra")
    ax.legend(fontsize=7)
    _save(fig, FIGURES / "lora_rank_clean_spectral_stats.pdf")


def make_alpha_figure() -> None:
    df = pd.read_csv(_find_run("alpha_noise_intruder", "*_alpha_noise_intruder") / "metrics.csv")
    g = df.groupby("alpha", as_index=False).agg(
        val=("final_val_loss", "mean"),
        val_sem=("final_val_loss", _sem),
        drift=("forgetting_loss", "mean"),
    )
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.errorbar(g["alpha"], g["val"], yerr=g["val_sem"], marker="o", linewidth=1, label="validation")
    ax2 = ax.twinx()
    ax2.plot(g["alpha"], g["drift"], marker="s", linewidth=1, label="drift")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"LoRA $\alpha$")
    ax.set_ylabel("validation MSE")
    ax2.set_ylabel("output drift proxy")
    ax.set_title("Alpha controls fit and drift")
    _save(fig, FIGURES / "alpha_noise_tradeoff.pdf")


def make_merge_figure() -> None:
    df = pd.read_csv(_find_run("merge_conflict", "*_merge_conflict") / "metrics.csv")
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.scatter(df["conflict_score"], df["merge_degradation"], s=12, alpha=0.7)
    x = df["conflict_score"].to_numpy(dtype=float)
    y = df["merge_degradation"].to_numpy(dtype=float)
    if len(x) > 1 and np.nanstd(x) > 0:
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
        ax.plot(xs, coef[0] * xs + coef[1], linewidth=1)
    ax.set_xlabel("signed conflict score")
    ax.set_ylabel("merge degradation")
    ax.set_title("Signed conflict predicts merge loss")
    _save(fig, FIGURES / "merge_conflict_score.pdf")


def make_stage4_figure() -> None:
    df = _load_stage4_table()
    df["target_order"] = df["target_display"].map({t: i for i, t in enumerate(TARGET_ORDER)})
    df = df.sort_values(["target_order", "condition"])
    hard = df[df["condition"] == "hard_knee"].sort_values("target_order")
    sample = df[df["condition"] == "sample_limited"].sort_values("target_order")
    labels = [re.sub(r" recovery", r" rec.", x).replace("near-best ", "near ").replace(" gap", "") for x in TARGET_ORDER]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.bar(x - width / 2, hard["mean_r2"], width, yerr=hard["sem_r2"], label="hard-knee")
    ax.bar(x + width / 2, sample["mean_r2"], width, yerr=sample["sem_r2"], label="sample-limited")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(r"mean $R^2$")
    ax.set_title("Stage4 effective-rank prediction")
    ax.legend(fontsize=7)
    _save(fig, FIGURES / "stage4_effective_rank_r2.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate or regenerate paper-facing tables and vector figures."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate Stage4, transformer, and real-model paper inputs without writing artifacts.",
    )
    args = parser.parse_args()
    if args.validate_only:
        _load_stage4_table()
        _load_transformer_publication_tables()
        _load_real_lora_publication_tables()
        print("paper artifact input validation: PASS")
        return

    _ensure_dirs()
    make_stage4_rows()
    make_transformer_publication_rows()
    make_real_lora_publication_rows()
    make_bbp_figures()
    make_lora_rank_figures()
    make_alpha_figure()
    make_merge_figure()
    make_stage4_figure()
    make_transformer_publication_figure()
    print("wrote validated paper tables and deterministic Type-42 PDF figures")


if __name__ == "__main__":
    main()
