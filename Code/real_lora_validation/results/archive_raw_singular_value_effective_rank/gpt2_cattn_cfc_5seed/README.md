# Released GPT-2 c_attn/c_fc 5-seed summary

This directory contains the compact CSV summaries used by the paper for the GPT-2/Wikitext-2 `c_attn,c_fc` setting. The released seed set is `101, 103, 107, 109, 123`.

The `run_dir` column is a historical local provenance label retained for auditability; only the seed/result columns are used to regenerate paper tables. The clean rerun script `scripts/run_gpt2_validation_local_replicates.sh` reruns seed 123 directly.
