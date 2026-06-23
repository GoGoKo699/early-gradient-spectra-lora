# Real-model paper evidence status

The legacy five-seed `c_attn,c_fc` and three-seed `+attn.c_proj` CSVs are
superseded and are not manuscript inputs. They predate the controlled
`real_lora_publication_protocol_v3` release.

Current paper-facing evidence is generated only by
`Paper/import_real_lora_results.py` from:

- release: `real_lora_publication_20260623T074520Z`;
- archive SHA-256: `4368ca32afd32fde04c68e1b889bcba2add57a801d8360db4e99557953c991de`;
- frozen plan SHA-256: `e407ba9f34946cfaa6a1246ebd55247b4fdc208d22de5bc1bd279d0c63a7da7b`;
- source Git commit: `9d8547ca172ffcc8e059351b8d91a4fc92d36b7e`.

The importer verifies recursive checksums, source-run provenance, exact
parameter cost, paired stochastic controls, duplicate-uniform identity,
nonzero adapter activity, and all released aggregate statistics before writing
`Paper/tables/real_lora_publication/`. `Paper/make_paper_artifacts.py` rejects
any stale or mixed table set.
