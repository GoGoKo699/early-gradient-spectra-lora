#!/usr/bin/env python3
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "sshleifer/tiny-gpt2"

print("torch:", torch.__version__)
print("hip:", getattr(torch.version, "hip", None))
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    raise SystemExit("GPU is not visible to PyTorch")

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to("cuda")
model.eval()

inp = tok("Early gradient spectra help LoRA because", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**inp, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
torch.cuda.synchronize()
print(tok.decode(out[0], skip_special_tokens=True))
print("HF GPU smoke OK")
