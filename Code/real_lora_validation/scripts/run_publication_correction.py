#!/usr/bin/env python3
"""Run the complete corrected GPT-2 study, publish tables, and build an upload bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
RESULTS = ROOT / "results"
EXPECTED_STRATEGIES = ["uniform_r4", "gradient_norm", "spectral_effective"]
DEFINITION = "p_i = sigma_i^2 / sum_j sigma_j^2"
SUITES = {
    "cattn_cfc": {
        "release": "gpt2_cattn_cfc_5seed",
        "seeds": [101, 103, 107, 109, 123],
        "suffixes": "c_attn,c_fc",
    },
    "attnproj": {
        "release": "gpt2_attnproj_3seed",
        "seeds": [101, 103, 107],
        "suffixes": "c_attn,attn.c_proj,c_fc",
    },
}

PROTOCOL_CONFIG = {
    "model": "models/gpt2_local",
    "dataset_mode": "local",
    "train_text_file": "data/wikitext2_local/train.txt",
    "val_text_file": "data/wikitext2_local/validation.txt",
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
    "max_train_texts": 5000,
    "max_val_texts": 1000,
    "max_train_blocks": 1024,
    "max_val_blocks": 256,
    "ridge_scale": 1e-3,
    "lora_alpha_scale": 2.0,
    "lora_init_std": 0.01,
    "dtype": "float32",
    "log_every": 20,
    "max_targets": 0,
    "device": "cuda",
    "trust_remote_code": False,
    "calibrate_only": False,
    "train_blocks": 1024,
    "val_blocks": 256,
    "effective_rank_definition": DEFINITION,
    "matched_strategy_batch_order": True,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sem(series: pd.Series) -> float:
    return float("nan") if len(series) <= 1 else float(series.std(ddof=1) / math.sqrt(len(series)))


def write_checksums(directory: Path) -> None:
    checksum_path = directory / "SHA256SUMS.txt"
    rows = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file() and p != checksum_path):
        rows.append(f"{sha256(path)}  {path.relative_to(directory).as_posix()}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def context_record() -> dict:
    packages = [
        "torch", "torchvision", "torchaudio", "numpy", "transformers", "datasets",
        "accelerate", "peft", "safetensors", "tokenizers", "huggingface_hub",
        "pandas", "pyarrow", "scikit-learn", "evaluate", "tqdm", "pytest",
    ]
    code_paths = [
        ROOT / "run_real_lora_validation.py",
        ROOT / "spectral_metrics.py",
        ROOT / "scripts/prepare_publication_inputs.py",
        ROOT / "scripts/preflight_publication_rerun.py",
        ROOT / "scripts/run_publication_correction.py",
    ]
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in packages},
        "pip_freeze": freeze,
        "code_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in code_paths},
        "torch": {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": getattr(torch.version, "cuda", None),
            "hip_runtime": getattr(torch.version, "hip", None),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else [],
        },
        "effective_rank_definition": DEFINITION,
        "matched_strategy_batch_order": True,
    }


def context_signature(context: dict) -> dict:
    """Fields that must remain identical when resuming a publication run."""
    return {
        "python": context.get("python"),
        "platform": context.get("platform"),
        "packages": context.get("packages"),
        "pip_freeze": context.get("pip_freeze"),
        "code_sha256": context.get("code_sha256"),
        "torch": context.get("torch"),
        "effective_rank_definition": context.get("effective_rank_definition"),
        "matched_strategy_batch_order": context.get("matched_strategy_batch_order"),
    }


def input_signature(manifest: dict) -> dict:
    return {
        "model_repo_id": manifest["model"].get("repo_id"),
        "model_revision": manifest["model"].get("revision"),
        "model_files": [(x["path"], x["sha256"]) for x in manifest["model"]["files"]],
        "dataset_repo_id": manifest["dataset"].get("repo_id"),
        "dataset_revision": manifest["dataset"].get("revision"),
        "dataset_config": manifest["dataset"].get("config"),
        "parquet_files": {
            key: (value["path"], value["sha256"])
            for key, value in sorted(manifest["dataset"]["parquet_files"].items())
        },
        "text_files": {
            key: (value["path"], value["sha256"], value.get("source_rows"))
            for key, value in sorted(manifest["dataset"]["text_files"].items())
        },
    }


def run_and_tee(command: list[str], cwd: Path, log_path: Path, env: dict[str, str]) -> None:
    print("COMMAND:", " ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND: " + " ".join(command) + "\n")
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code != 0:
        raise SystemExit(f"Command failed with exit code {code}; see {log_path}")


def _matches_protocol(cfg: dict, seed: int, suffixes: str) -> bool:
    if int(cfg.get("seed", -1)) != seed or cfg.get("target_suffixes") != suffixes:
        return False
    for key, expected in PROTOCOL_CONFIG.items():
        actual = cfg.get(key)
        if isinstance(expected, float):
            try:
                if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-15):
                    return False
            except (TypeError, ValueError):
                return False
        elif actual != expected:
            return False
    return bool(cfg.get("cuda_available"))


def complete_run(seed_dir: Path, seed: int, suffixes: str) -> Path | None:
    complete = []
    for result_path in sorted(seed_dir.glob("*/results.csv")):
        config_path = result_path.parent / "config.json"
        if not config_path.is_file():
            continue
        try:
            cfg = json.loads(config_path.read_text())
            df = pd.read_csv(result_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        numeric_columns = [
            "initial_val_loss", "final_val_loss", "val_loss_delta", "perplexity",
            "trainable_params", "rank_min", "rank_mean", "rank_max",
        ]
        if not all(column in df.columns for column in ["strategy", *numeric_columns]):
            continue
        numeric_ok = all(pd.to_numeric(df[column], errors="coerce").notna().all() for column in numeric_columns)
        finite_ok = all(
            pd.Series(pd.to_numeric(df[column], errors="coerce")).map(math.isfinite).all()
            for column in numeric_columns
        )
        if (
            _matches_protocol(cfg, seed, suffixes)
            and list(df["strategy"]) == EXPECTED_STRATEGIES
            and len(df) == 3
            and numeric_ok
            and finite_ok
        ):
            complete.append(result_path.parent)
    if len(complete) > 1:
        raise SystemExit(f"Multiple complete runs found for seed {seed}: {complete}")
    return complete[0] if complete else None


def aggregate_suite(suite_root: Path, suite_name: str, spec: dict) -> tuple[Path, dict[int, Path]]:
    rows = []
    run_dirs = {}
    for seed in spec["seeds"]:
        run_dir = complete_run(suite_root / f"seed{seed}", seed, spec["suffixes"])
        if run_dir is None:
            raise SystemExit(f"Missing complete run for {suite_name} seed {seed}")
        run_dirs[seed] = run_dir
        cfg = json.loads((run_dir / "config.json").read_text())
        df = pd.read_csv(run_dir / "results.csv")
        for _, row in df.iterrows():
            item = row.to_dict()
            item.update({
                "run_dir": f"{suite_name}/seed{seed}", "seed": seed,
                "model": cfg["model"], "steps": cfg["steps"],
                "target_suffixes": cfg["target_suffixes"],
            })
            rows.append(item)

    out = suite_root / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    all_df = pd.DataFrame(rows)
    all_df.to_csv(out / "all_results.csv", index=False)

    winners = []
    for seed, group in all_df.groupby("seed"):
        best = group.loc[group["final_val_loss"].idxmin()]
        winners.append({
            "run_dir": f"{suite_name}/seed{int(seed)}", "seed": int(seed),
            "winner": best["strategy"], "winner_final_val_loss": best["final_val_loss"],
        })
    winners_df = pd.DataFrame(winners)
    winners_df.to_csv(out / "per_run_winners.csv", index=False)

    summary = all_df.groupby("strategy").agg(
        n=("final_val_loss", "size"),
        mean_initial_val_loss=("initial_val_loss", "mean"),
        mean_final_val_loss=("final_val_loss", "mean"),
        sem_final_val_loss=("final_val_loss", sem),
        mean_val_loss_delta=("val_loss_delta", "mean"),
        sem_val_loss_delta=("val_loss_delta", sem),
        mean_perplexity=("perplexity", "mean"),
        sem_perplexity=("perplexity", sem),
        mean_trainable_params=("trainable_params", "mean"),
        min_rank=("rank_min", "mean"), mean_rank=("rank_mean", "mean"), max_rank=("rank_max", "mean"),
    ).reset_index()
    wins = winners_df["winner"].value_counts().rename_axis("strategy").reset_index(name="wins")
    summary = summary.merge(wins, on="strategy", how="left")
    summary["wins"] = summary["wins"].fillna(0).astype(int)
    summary = summary.sort_values("mean_final_val_loss").reset_index(drop=True)
    summary.to_csv(out / "summary_by_strategy.csv", index=False)

    pairs = []
    for seed, group in all_df.groupby("seed"):
        by = group.set_index("strategy")
        spectral = by.loc["spectral_effective"]
        for other in ["uniform_r4", "gradient_norm"]:
            control = by.loc[other]
            pairs.append({
                "run_dir": f"{suite_name}/seed{int(seed)}", "seed": int(seed),
                "comparison": f"spectral_minus_{other}",
                "final_val_loss_diff": spectral["final_val_loss"] - control["final_val_loss"],
                "perplexity_diff": spectral["perplexity"] - control["perplexity"],
                "params_diff": spectral["trainable_params"] - control["trainable_params"],
            })
    pd.DataFrame(pairs).to_csv(out / "spectral_pairwise_diffs.csv", index=False)
    return out, run_dirs


def copy_evidence(run_dirs: dict[int, Path], destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    for seed, source in sorted(run_dirs.items()):
        target = destination / f"seed{seed}"
        target.mkdir(parents=True, exist_ok=True)
        for name in [
            "config.json", "calibration_metrics.csv", "allocations.csv", "results.csv", "summary.txt",
            "train_history_uniform_r4.csv", "train_history_gradient_norm.csv",
            "train_history_spectral_effective.csv",
        ]:
            path = source / name
            if path.is_file():
                shutil.copy2(path, target / name)
        write_checksums(target)


def publish(run_root: Path, aggregates: dict, all_run_dirs: dict, context: dict, input_manifest: dict) -> Path:
    archive_root = RESULTS / "archive_raw_singular_value_effective_rank"
    evidence_root = RESULTS / "corrected_full_runs" / run_root.name
    evidence_root.mkdir(parents=True, exist_ok=True)
    (evidence_root / "RERUN_CONTEXT.json").write_text(json.dumps(context, indent=2, sort_keys=True) + "\n")
    (evidence_root / "INPUT_MANIFEST.json").write_text(json.dumps(input_manifest, indent=2, sort_keys=True) + "\n")

    report_lines = [
        "# Corrected real GPT-2 effective-rank results", "",
        f"Effective-rank definition: `{DEFINITION}`.", "",
        "All three strategies were rerun together for every seed; no old control measurements were reused.", "",
        "Each strategy used a fresh, identically seeded training loader, so mini-batch order was matched within each seed.", "",
    ]

    for suite_name, spec in SUITES.items():
        release_name = spec["release"]
        current = RESULTS / "released" / release_name
        archived = archive_root / release_name
        if current.exists() and not archived.exists():
            archived.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(current, archived)
            write_checksums(archived)
        if current.exists():
            shutil.rmtree(current)
        current.mkdir(parents=True)
        aggregate = aggregates[suite_name]
        for name in ["all_results.csv", "per_run_winners.csv", "summary_by_strategy.csv", "spectral_pairwise_diffs.csv"]:
            shutil.copy2(aggregate / name, current / name)
        provenance = {
            "correction": "effective rank now uses normalized squared singular values",
            "effective_rank_definition": DEFINITION,
            "rerun_scope": "all three strategies rerun together for every seed",
            "matched_strategy_batch_order": True,
            "suite": suite_name, "seeds": spec["seeds"], "target_suffixes": spec["suffixes"],
            "model": input_manifest["model"], "dataset": input_manifest["dataset"],
            "environment_file": f"results/corrected_full_runs/{run_root.name}/RERUN_CONTEXT.json",
            "evidence_directory": f"results/corrected_full_runs/{run_root.name}/{suite_name}",
            "superseded_release_archive": f"results/archive_raw_singular_value_effective_rank/{release_name}",
        }
        (current / "CORRECTION_PROVENANCE.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        (current / "README.md").write_text(
            "# Corrected GPT-2 validation\n\n"
            "These tables were regenerated after changing entropy effective rank to use normalized squared singular values. "
            "Uniform, gradient-norm, and spectral-effective strategies were all rerun together with matched mini-batch order.\n",
            encoding="utf-8",
        )
        copy_evidence(all_run_dirs[suite_name], evidence_root / suite_name)
        write_checksums(current)

        summary = pd.read_csv(current / "summary_by_strategy.csv")
        pairwise = pd.read_csv(current / "spectral_pairwise_diffs.csv")
        report_lines += [f"## {release_name}", "", summary.to_markdown(index=False), "", "Pairwise spectral differences:", "", pairwise.groupby("comparison")[["final_val_loss_diff", "perplexity_diff"]].mean().reset_index().to_markdown(index=False), ""]

    write_checksums(evidence_root)
    report_lines += [
        "## Paper follow-up", "",
        "The generated LaTeX table rows have been refreshed. The prose in `Paper/paper.tex` contains hard-coded numerical claims and must be checked against these corrected values before submission.", "",
    ]
    report_path = PROJECT / "EFFECTIVE_RANK_CORRECTION_RESULTS.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


def add_tree(zf: zipfile.ZipFile, path: Path, base: Path) -> None:
    if path.is_file():
        zf.write(path, path.relative_to(base).as_posix())
    elif path.is_dir():
        for file in sorted(p for p in path.rglob("*") if p.is_file()):
            zf.write(file, file.relative_to(base).as_posix())


def make_bundle(run_root: Path, report_path: Path) -> Path:
    bundle = PROJECT / f"CORRECTED_REAL_GPT2_RESULTS_{run_root.name}.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in [
            report_path,
            ROOT / "INPUT_MANIFEST.json",
            RESULTS / "released",
            RESULTS / "archive_raw_singular_value_effective_rank",
            RESULTS / "corrected_full_runs" / run_root.name,
            PROJECT / "Paper/tables/generated/real_lora_rows.tex",
            PROJECT / "Paper/tables/generated/real_lora_pairwise_rows.tex",
            PROJECT / "Paper/tables/real_lora",
            ROOT / "run_real_lora_validation.py",
            ROOT / "spectral_metrics.py",
            ROOT / "tests/test_spectral_metrics.py",
        ]:
            if path.exists():
                add_tree(zf, path, PROJECT)
    return bundle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resume", type=Path, help="existing corrected run root to resume on the same machine/environment")
    args = ap.parse_args()

    subprocess.run(
        [sys.executable, str(ROOT / "scripts/preflight_publication_rerun.py")],
        cwd=ROOT,
        check=True,
    )

    input_manifest = json.loads((ROOT / "INPUT_MANIFEST.json").read_text())
    context = context_record()
    if args.resume:
        run_root = args.resume.expanduser().resolve()
        if not run_root.is_dir():
            raise SystemExit(f"Resume directory does not exist: {run_root}")
        old_context_path = run_root / "RERUN_CONTEXT.json"
        old_manifest_path = run_root / "INPUT_MANIFEST.json"
        if not old_context_path.is_file() or not old_manifest_path.is_file():
            raise SystemExit("Resume directory lacks RERUN_CONTEXT.json or INPUT_MANIFEST.json")
        old_context = json.loads(old_context_path.read_text())
        old_manifest = json.loads(old_manifest_path.read_text())
        if context_signature(old_context) != context_signature(context):
            raise SystemExit("Environment/code changed since this run started; resume on the original environment")
        if input_signature(old_manifest) != input_signature(input_manifest):
            raise SystemExit("Model or dataset inputs changed since this run started")
        with (run_root / "RESUME_EVENTS.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"resumed_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    else:
        run_root = ROOT / "real_lora_runs" / f"corrected_full_{time.strftime('%Y%m%d_%H%M%S')}"
        run_root.mkdir(parents=True, exist_ok=False)
        (run_root / "RERUN_CONTEXT.json").write_text(json.dumps(context, indent=2, sort_keys=True) + "\n")
        (run_root / "INPUT_MANIFEST.json").write_text(json.dumps(input_manifest, indent=2, sort_keys=True) + "\n")

    (ROOT / "LATEST_CORRECTION_RUN.txt").write_text(str(run_root) + "\n")

    env = os.environ.copy()
    env.update({
        "HF_HUB_DISABLE_XET": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1", "TOKENIZERS_PARALLELISM": "false",
    })
    command_keys = [
        "model", "dataset_mode", "train_text_file", "val_text_file", "block_size",
        "batch_size", "calib_batches", "eval_batches", "steps", "lr", "weight_decay",
        "grad_clip", "uniform_rank", "min_rank", "max_rank", "max_train_texts",
        "max_val_texts", "max_train_blocks", "max_val_blocks", "ridge_scale",
        "lora_alpha_scale", "lora_init_std", "dtype",
    ]
    common = []
    for key in command_keys:
        common.extend([f"--{key}", str(PROTOCOL_CONFIG[key])])
    common.extend(["--log_every", str(PROTOCOL_CONFIG["log_every"])])

    for suite_name, spec in SUITES.items():
        for seed in spec["seeds"]:
            seed_dir = run_root / suite_name / f"seed{seed}"
            existing = complete_run(seed_dir, seed, spec["suffixes"])
            if existing is not None:
                print(f"SKIP complete {suite_name} seed {seed}: {existing}")
                continue
            command = [
                sys.executable, "-u", "run_real_lora_validation.py", *common,
                "--target_suffixes", spec["suffixes"], "--seed", str(seed),
                "--out_dir", str(seed_dir),
            ]
            run_and_tee(command, ROOT, seed_dir / "console.log", env)
            if complete_run(seed_dir, seed, spec["suffixes"]) is None:
                raise SystemExit(f"Run finished without a complete result: {suite_name} seed {seed}")

    aggregates, all_run_dirs = {}, {}
    for suite_name, spec in SUITES.items():
        aggregates[suite_name], all_run_dirs[suite_name] = aggregate_suite(run_root / suite_name, suite_name, spec)

    report_path = publish(run_root, aggregates, all_run_dirs, context, input_manifest)
    subprocess.run([sys.executable, str(PROJECT / "Paper/import_real_lora_results.py")], cwd=PROJECT / "Paper", check=True)
    bundle = make_bundle(run_root, report_path)
    print("\nPUBLICATION CORRECTION RUN COMPLETE")
    print("Run root:", run_root)
    print("Correction report:", report_path)
    print("Upload this bundle for paper update:", bundle)


if __name__ == "__main__":
    main()
