# SUPERSEDED Stage4 aggregate — do not use for publication

This compact aggregate was generated before the useful-rank estimand was made
consistent with the manuscript. Its target thresholds and penalties can depend
on the simulation oracle-loss gap, whereas the paper defines them using the best
validation loss observed on the tested rank grid.

It is retained only as historical evidence. It lacks the complete per-seed,
per-layer, per-rank source rows needed for a publication-grade correction and is
rejected by `Paper/import_stage4_aggregate.py`.

Generate the replacement from `Code/rmt_lora_sim`:

```bash
bash scripts/run_stage4.sh
```

The replacement release will appear under `runs/stage4_releases/` with raw
source runs, manifests, and recursive SHA-256 checksums.
