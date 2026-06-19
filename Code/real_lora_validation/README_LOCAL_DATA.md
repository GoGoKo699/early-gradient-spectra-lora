# Local model/data preparation

The clean release does not include GPT-2 weights or Wikitext data.

Expected local paths for the provided scripts:

```text
models/gpt2_local/
data/wikitext2_local/train.txt
data/wikitext2_local/validation.txt
```

Minimum expected GPT-2 local files:

```text
models/gpt2_local/config.json
models/gpt2_local/generation_config.json
models/gpt2_local/model.safetensors
models/gpt2_local/vocab.json
models/gpt2_local/merges.txt
models/gpt2_local/tokenizer_config.json
models/gpt2_local/special_tokens_map.json
```

The released result CSVs in `results/released/` are sufficient to regenerate the paper tables. The local model/data files are needed only to rerun the GPT-2 validation.

For a full local rerun, record the exact model revision, Wikitext source revision, and SHA256 checksums of the local files you used. Those external files are intentionally excluded from this compact artifact package.
