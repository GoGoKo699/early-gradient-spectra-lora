# Zenodo data record

This record should archive the full raw releases and pinned real-model inputs.

Recommended contents:

```text
releases/stage4_paper_20260622T061351.tar.gz
releases/transformer_publication_20260622T101458Z.tar.gz
releases/real_lora_publication_20260623T074520Z.tar.gz
inputs/real_lora_validation_inputs_20260623.tar.gz
manifests/SHA256SUMS.txt
README_ZENODO_DATA.md
LICENSE-DATA.md
THIRD_PARTY_NOTICES.md
```

The three raw release archives should not be renamed or recompressed unless
they are intentionally rebuilt and their new hashes are recorded.

The real-model input archive should be built from:

```text
Local/inputs/real_lora_validation/
```

and verified against its `INPUT_MANIFEST.json`.
