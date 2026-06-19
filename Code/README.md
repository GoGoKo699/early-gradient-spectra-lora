# Early Gradient Spectra for Low-Rank Adaptation — Code Package

This directory contains the code and compact released numerical artifacts for the paper.

## Structure

```text
rmt_lora_sim/          Matrix/RMT and Stage4 synthetic LoRA simulations
transformer/           Synthetic transformer allocation experiments
real_lora_validation/  Real GPT-2/Wikitext LoRA allocation validation
```

## Quick checks

```bash
make test
make smoke-rmt
make smoke-transformer
```

## Released artifacts

```text
rmt_lora_sim/results/released/
real_lora_validation/results/released/
```

Raw local run folders, model weights, Wikitext downloads, ROCm setup logs, and failed historical attempts are intentionally excluded.

## Real-model validation

The released real-model summaries are under:

```text
real_lora_validation/results/released/gpt2_cattn_cfc_5seed/
real_lora_validation/results/released/gpt2_attnproj_3seed/
```

The GPT-2 checkpoint and Wikitext files are not bundled. Reproduction scripts in `real_lora_validation/scripts/` expect local model/data preparation as described in that subpackage.
