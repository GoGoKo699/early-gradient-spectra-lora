#!/usr/bin/env python3
"""Minimal Hugging Face model/tokenizer/GPU execution preflight."""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    print("torch:", torch.__version__)
    print("hip:", getattr(torch.version, "hip", None))
    print("cuda available:", torch.cuda.is_available())
    print(
        "device:",
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    )
    if not torch.cuda.is_available():
        raise SystemExit("GPU is not visible to PyTorch")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        local_files_only=args.local_files_only,
    ).to("cuda")
    model.eval()

    inputs = tokenizer(
        "Early gradient spectra help LoRA because", return_tensors="pt"
    ).to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print("HF GPU smoke OK")


if __name__ == "__main__":
    main()
