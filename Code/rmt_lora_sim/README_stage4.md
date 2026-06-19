# Stage4 replication

This repeats the layerwise useful-rank experiments across multiple random seeds and aggregates the results used in the paper.

Run from the `rmt_lora_sim` project directory:

```bash
bash scripts/run_stage4.sh
```

The default `run_stage4.sh` is the paper-grade run:

```text
5 seeds x 48 synthetic layers x 2 conditions
```

For a shorter check, use:

```bash
bash scripts/run_stage4_fast.sh
```

The fast script runs:

```text
3 seeds x 32 synthetic layers x 2 conditions
```

Outputs:

```text
runs/stage4_aggregate/stage4_all_fits.csv
runs/stage4_aggregate/stage4_key_table.csv
runs/stage4_aggregate/stage4_best_predictor_counts.csv
runs/stage4_aggregate/stage4_spearman_hard_knee.png
runs/stage4_aggregate/stage4_spearman_sample_limited.png
rmt_lora_stage4_runs.tar.gz
```

The key success criterion is whether `gradient_effective_rank` or `gradient_detectable_rank` predicts useful-rank targets such as:

```text
near_best_rank_gap_0.1
near_best_rank_gap_0.2
recovery_rank_0.7
recovery_rank_0.8
penalized_rank_lambda_0.2
penalized_rank_lambda_0.3
```

In the publication configs, `gradient_*` means the activation-whitened early-gradient matrix. Raw unwhitened ablations are kept as `raw_gradient_*` and included in aggregate predictor tables when present.

## Importing Stage4 into the paper

To use a fresh Stage4 aggregate in the manuscript tables, run from `Code/`:

```bash
make stage4-to-paper
make paper
```

`make stage4-to-paper` reads a locally rerun `Code/rmt_lora_sim/runs/stage4_aggregate/stage4_key_table.csv` and rewrites `Paper/tables/stage4_gradient_effective_summary.csv`.

To import the bundled released aggregate instead, run from `Code/`:

```bash
make stage4-release-to-paper
make paper
```

## Clean release provenance

The clean release intentionally exposes `results/released/stage4_aggregate/` as the authoritative compact Stage4 artifact. Historical per-seed Stage4 release folders are excluded because stale raw-gradient runs can contradict the activation-whitened aggregate used by the paper.

Use `make stage4-release-to-paper` from the top-level `Code/` directory to import the bundled released aggregate into the paper table. Use `make stage4-to-paper` only after rerunning Stage4 locally.
