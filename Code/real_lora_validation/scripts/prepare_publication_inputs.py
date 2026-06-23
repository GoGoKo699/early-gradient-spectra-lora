#!/usr/bin/env python3
"""Materialize and verify the immutable GPT-2/WikiText-2 publication inputs.

The tracked INPUT_MANIFEST.json is the frozen specification. This helper never
rewrites it, so preparing local inputs cannot dirty the Git worktree or silently
change the prespecified analysis plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer

MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
]
DATASET_ID = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_FILES = {
    "train": f"{DATASET_CONFIG}/train-00000-of-00001.parquet",
    "validation": f"{DATASET_CONFIG}/validation-00000-of-00001.parquet",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit("INPUT_MANIFEST.json must contain a JSON object")
    return value


def manifest_record_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records: list[dict[str, Any]] = list(manifest["model"]["files"])
    records.extend(manifest["dataset"]["parquet_files"].values())
    records.extend(manifest["dataset"]["text_files"].values())
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        relative = str(record["path"])
        if relative in result:
            raise SystemExit(f"duplicate manifest path: {relative}")
        result[relative] = record
    return result


def verify_file(path: Path, record: dict[str, Any], root: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"missing publication input: {path}")
    actual_size = path.stat().st_size
    actual_hash = sha256(path)
    expected_size = int(record["bytes"])
    expected_hash = str(record["sha256"])
    if actual_size != expected_size or actual_hash != expected_hash:
        raise SystemExit(
            "publication input mismatch: "
            f"{path.relative_to(root)} size={actual_size} sha256={actual_hash}; "
            f"expected size={expected_size} sha256={expected_hash}"
        )


def write_split(source: Path, target: Path) -> int:
    values = pq.read_table(source, columns=["text"]).column("text").to_pylist()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            text = "" if value is None else str(value)
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
    return len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove and rematerialize local model/data files; never changes the manifest.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "INPUT_MANIFEST.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing frozen input manifest: {manifest_path}")
    manifest = load_manifest(manifest_path)
    if (manifest["model"].get("repo_id"), manifest["model"].get("revision")) != (
        MODEL_ID,
        MODEL_REVISION,
    ):
        raise SystemExit("tracked input manifest has an unexpected model identity")
    if (
        manifest["dataset"].get("repo_id"),
        manifest["dataset"].get("revision"),
        manifest["dataset"].get("config"),
    ) != (DATASET_ID, DATASET_REVISION, DATASET_CONFIG):
        raise SystemExit("tracked input manifest has an unexpected dataset identity")
    records = manifest_record_map(manifest)

    model_dir = root / "models/gpt2_local"
    source_dir = root / "data/wikitext2_source"
    text_dir = root / "data/wikitext2_local"
    if args.force:
        for path in [model_dir, source_dir, text_dir]:
            shutil.rmtree(path, ignore_errors=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    print(f"Materializing {MODEL_ID}@{MODEL_REVISION}")
    for name in MODEL_FILES:
        path = Path(
            hf_hub_download(
                repo_id=MODEL_ID,
                filename=name,
                revision=MODEL_REVISION,
                local_dir=model_dir,
            )
        ).resolve()
        relative = path.relative_to(root).as_posix()
        if relative not in records:
            raise SystemExit(f"downloaded model file is not in the frozen manifest: {relative}")
        verify_file(path, records[relative], root)
        print("  verified:", relative)

    print(f"Materializing {DATASET_ID}@{DATASET_REVISION}")
    parquet_paths: dict[str, Path] = {}
    for split, name in DATASET_FILES.items():
        path = Path(
            hf_hub_download(
                repo_id=DATASET_ID,
                repo_type="dataset",
                filename=name,
                revision=DATASET_REVISION,
                local_dir=source_dir,
            )
        ).resolve()
        relative = path.relative_to(root).as_posix()
        if relative not in records:
            raise SystemExit(f"downloaded dataset file is not in the frozen manifest: {relative}")
        verify_file(path, records[relative], root)
        parquet_paths[split] = path
        print("  verified:", relative)

    expected_rows = {"train": 36718, "validation": 3760}
    for split, source in parquet_paths.items():
        target = text_dir / f"{split}.txt"
        rows = write_split(source, target)
        relative = target.relative_to(root).as_posix()
        if relative not in records:
            raise SystemExit(f"generated text file is not in the frozen manifest: {relative}")
        verify_file(target, records[relative], root)
        if rows != expected_rows[split] or int(records[relative]["source_rows"]) != rows:
            raise SystemExit(f"unexpected {split} row count: {rows}")
        print(f"  verified: {relative} ({rows} rows)")

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, use_fast=False, local_files_only=True
    )
    if (config.model_type, config.n_layer, config.n_embd) != ("gpt2", 12, 768):
        raise SystemExit("materialized model is not GPT-2 124M")
    if tokenizer.vocab_size != 50257:
        raise SystemExit(f"unexpected tokenizer size: {tokenizer.vocab_size}")

    print("PUBLICATION INPUT PREPARATION: PASS")
    print("manifest_sha256:", sha256(manifest_path))
    print("The tracked INPUT_MANIFEST.json was not modified.")


if __name__ == "__main__":
    main()
