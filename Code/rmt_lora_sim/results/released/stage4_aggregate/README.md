# Stage4 released aggregate

This directory is the authoritative compact Stage4 release used by the paper.

The paper-facing table in `Paper/tables/stage4_gradient_effective_summary.csv` is generated from:

```text
stage4_key_table.csv
```

using the `gradient_effective_rank` predictor.  In the publication configs this predictor is the activation-whitened early-gradient effective rank; raw unwhitened ablations remain in the aggregate as `raw_gradient_*` rows.

Full per-seed Stage4 run directories are intentionally not bundled in the clean release because stale pre-whitening per-seed directories can contradict this aggregate.  To regenerate full runs, execute from `Code/rmt_lora_sim`:

```bash
bash scripts/run_stage4.sh
```

then import the fresh aggregate with:

```bash
cd ../..
cd Code
make stage4-to-paper
make paper
```
