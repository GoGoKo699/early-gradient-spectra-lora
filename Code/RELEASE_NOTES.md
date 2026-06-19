# Clean release notes

Top-level layout:

```text
LoRA Project/
  Code/
  Paper/
  paper.pdf
```

`Code/` is the GitHub-facing package. `Paper/` is the Overleaf-facing source package. The top-level `paper.pdf` is the compiled manuscript.

The Stage4 paper table is generated from `Code/rmt_lora_sim/results/released/stage4_aggregate/stage4_key_table.csv` with predictor `gradient_effective_rank`, which is activation-whitened in the publication configs. Stale per-seed Stage4 release folders are intentionally excluded from this clean package; rerun `Code/rmt_lora_sim/scripts/run_stage4.sh` to regenerate full per-seed runs.

Large model weights and prepared Wikitext text files remain excluded. The
compact corrected per-seed GPT-2 evidence used by the manuscript is included
under `Code/real_lora_validation/results/corrected_full_runs/`, together with
input and environment manifests and checksums.

RMT released-artifact checksums are provided in `Code/rmt_lora_sim/results/released/SHA256SUMS.txt`; each compact RMT released subdirectory also has a local `SHA256SUMS.txt`.

The real-LoRA package includes `requirements-tested-rocm721.txt`, a core tested environment record rather than a complete transitive lockfile.

Real-LoRA released-summary `run_path`/`run_dir` strings are provenance labels from local runs; seed, strategy, metric, rank, and parameter-count columns are the paper-relevant fields.


The real-LoRA release was corrected on June 19, 2026 to compute entropy
effective rank from normalized squared singular values. The prior real-model
tables are preserved only in the explicitly named superseded archive.
