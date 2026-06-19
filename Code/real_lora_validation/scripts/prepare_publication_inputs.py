#!/usr/bin/env python3
"""Download immutable GPT-2 and WikiText-2 inputs for the publication rerun."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer

MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
MODEL_FILES = [
    "config.json", "generation_config.json", "model.safetensors",
    "vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json",
]
DATASET_ID = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_SHA256 = {
    "train": "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7",
    "validation": "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c",
}
DATASET_FILES = {
    "train": f"{DATASET_CONFIG}/train-00000-of-00001.parquet",
    "validation": f"{DATASET_CONFIG}/validation-00000-of-00001.parquet",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def record(path: Path, root: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_split(source: Path, target: Path) -> int:
    values = pq.read_table(source, columns=["text"]).column("text").to_pylist()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as f:
        for value in values:
            text = "" if value is None else str(value)
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
    return len(values)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "models/gpt2_local"
    source_dir = root / "data/wikitext2_source"
    text_dir = root / "data/wikitext2_local"
    manifest_path = root / "INPUT_MANIFEST.json"

    if args.force:
        for path in [model_dir, source_dir, text_dir]:
            shutil.rmtree(path, ignore_errors=True)
        manifest_path.unlink(missing_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {MODEL_ID}@{MODEL_REVISION}")
    model_paths = []
    for name in MODEL_FILES:
        model_paths.append(Path(hf_hub_download(
            repo_id=MODEL_ID, filename=name, revision=MODEL_REVISION, local_dir=model_dir
        )).resolve())
        print("  model:", name)

    print(f"Downloading {DATASET_ID}@{DATASET_REVISION}")
    parquet_paths = {}
    for split, name in DATASET_FILES.items():
        parquet_paths[split] = Path(hf_hub_download(
            repo_id=DATASET_ID, repo_type="dataset", filename=name,
            revision=DATASET_REVISION, local_dir=source_dir
        )).resolve()
        actual = sha256(parquet_paths[split])
        expected = DATASET_SHA256[split]
        if actual != expected:
            raise SystemExit(
                f"Pinned WikiText-2 {split} checksum mismatch: {actual}; expected {expected}"
            )
        print("  dataset:", name, actual)

    text_paths, row_counts = {}, {}
    for split, source in parquet_paths.items():
        target = text_dir / f"{split}.txt"
        row_counts[split] = write_split(source, target)
        text_paths[split] = target
        print(f"  wrote {target.relative_to(root)}: {row_counts[split]} rows")

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, local_files_only=True)
    if (config.model_type, config.n_layer, config.n_embd) != ("gpt2", 12, 768):
        raise SystemExit("Downloaded model is not GPT-2 124M")
    if tokenizer.vocab_size != 50257:
        raise SystemExit(f"Unexpected tokenizer size: {tokenizer.vocab_size}")
    if row_counts != {"train": 36718, "validation": 3760}:
        raise SystemExit(f"Unexpected WikiText row counts: {row_counts}")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "repo_id": MODEL_ID, "revision": MODEL_REVISION,
            "model_type": config.model_type, "n_layer": config.n_layer,
            "n_embd": config.n_embd, "vocab_size": tokenizer.vocab_size,
            "files": [record(p, root) for p in sorted(model_paths)],
        },
        "dataset": {
            "repo_id": DATASET_ID, "revision": DATASET_REVISION, "config": DATASET_CONFIG,
            "parquet_files": {k: record(v, root) for k, v in parquet_paths.items()},
            "text_files": {k: {**record(v, root), "source_rows": row_counts[k]} for k, v in text_paths.items()},
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Wrote", manifest_path)
    print("INPUT PREPARATION COMPLETE")


if __name__ == "__main__":
    main()
