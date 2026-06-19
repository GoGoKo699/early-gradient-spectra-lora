#!/usr/bin/env python3
"""Fail fast unless code, packages, inputs, and hardware match the rerun protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from spectral_metrics import EFFECTIVE_RANK_DEFINITION, singular_effective_rank

EXPECTED_MODEL = ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e")
EXPECTED_DATASET = ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3", "wikitext-2-raw-v1")
EXPECTED_DATASET_SHA256 = {
    "train": "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7",
    "validation": "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c",
}

EXPECTED = {
    "numpy": "1.26.4", "transformers": "5.12.0", "datasets": "5.0.0",
    "accelerate": "1.14.0", "peft": "0.19.1", "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_blocks(tokenizer, path: Path, text_limit: int, block_size: int, block_limit: int) -> int:
    texts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                texts.append(text)
            if len(texts) >= text_limit:
                break
    ids = []
    target = block_limit * block_size
    for text in texts:
        ids.extend(tokenizer.encode(text, add_special_tokens=False))
        ids.append(int(tokenizer.eos_token_id))
        if len(ids) >= target + block_size:
            break
    return min(len(ids) // block_size, block_limit)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    root = ROOT

    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"Python 3.12 required; found {platform.python_version()}")
    if not torch.__version__.startswith("2.9.1"):
        raise SystemExit(f"PyTorch 2.9.1 required; found {torch.__version__}")
    errors = []
    for name, expected in EXPECTED.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = "MISSING"
        if actual != expected:
            errors.append(f"{name}: expected {expected}, found {actual}")
    if errors:
        raise SystemExit("Package mismatch:\n  " + "\n  ".join(errors))
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("No CUDA/ROCm GPU visible. Use --allow-cpu only for a diagnostic run.")

    manifest_path = root / "INPUT_MANIFEST.json"
    if not manifest_path.is_file():
        raise SystemExit("Run scripts/prepare_publication_inputs.py first")
    manifest = json.loads(manifest_path.read_text())
    model_identity = (manifest["model"].get("repo_id"), manifest["model"].get("revision"))
    dataset_identity = (
        manifest["dataset"].get("repo_id"), manifest["dataset"].get("revision"),
        manifest["dataset"].get("config"),
    )
    if model_identity != EXPECTED_MODEL:
        raise SystemExit(f"Unexpected model identity: {model_identity}")
    if dataset_identity != EXPECTED_DATASET:
        raise SystemExit(f"Unexpected dataset identity: {dataset_identity}")
    for split, expected_hash in EXPECTED_DATASET_SHA256.items():
        actual_hash = manifest["dataset"]["parquet_files"][split].get("sha256")
        if actual_hash != expected_hash:
            raise SystemExit(
                f"Unexpected pinned WikiText-2 {split} checksum: {actual_hash}; expected {expected_hash}"
            )

    records = list(manifest["model"]["files"])
    records.extend(manifest["dataset"]["parquet_files"].values())
    records.extend(manifest["dataset"]["text_files"].values())
    for item in records:
        path = root / item["path"]
        if not path.is_file():
            raise SystemExit(f"Missing input: {path}")
        if sha256(path) != item["sha256"]:
            raise SystemExit(f"Checksum mismatch: {path}")

    model_dir = root / "models/gpt2_local"
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, local_files_only=True)
    if (config.model_type, config.n_layer, config.n_embd) != ("gpt2", 12, 768):
        raise SystemExit("Expected GPT-2 124M (12 layers, width 768)")
    train_blocks = count_blocks(tokenizer, root / "data/wikitext2_local/train.txt", 5000, 128, 1024)
    val_blocks = count_blocks(tokenizer, root / "data/wikitext2_local/validation.txt", 1000, 128, 256)
    if (train_blocks, val_blocks) != (1024, 256):
        raise SystemExit(f"Incomplete local data: train_blocks={train_blocks}, val_blocks={val_blocks}")
    expected_rank = 1.6493848884661177
    actual_rank = singular_effective_rank(torch.tensor([2.0, 1.0]))
    if abs(actual_rank - expected_rank) > 1e-12:
        raise SystemExit("Squared-singular-value effective-rank test failed")

    print("PUBLICATION PREFLIGHT PASSED")
    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("GPU visible:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("HIP:", getattr(torch.version, "hip", None), "CUDA:", getattr(torch.version, "cuda", None))
    print("Model revision:", manifest["model"]["revision"])
    print("Dataset revision:", manifest["dataset"]["revision"])
    print("Blocks:", train_blocks, val_blocks)
    print("Effective rank:", EFFECTIVE_RANK_DEFINITION)


if __name__ == "__main__":
    main()
