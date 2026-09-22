# Artifact ledger

The maintained code, compact evidence, and Markdown research report live in
this repository. The cleanup preserves the historical raw archive identities;
it does not create or republish a data artifact.

| Artifact | Location | What is checked locally |
|---|---|---|
| Research report | [Paper/paper.md](Paper/paper.md) | Documentation and generated table consistency |
| Compact evidence | [evidence/](evidence/README.md) | Checksums and primary comparison reconstruction |
| Experiment code | [Code/](Code/README.md) | Unit tests in a recorded CPU environment |
| Full raw releases and pinned inputs | [Recorded Zenodo data record](https://doi.org/10.5281/zenodo.21061917) | Archive identities recorded; external archives not downloaded in this cleanup |
| Historical software release | [v1.0.4-publication](https://github.com/GoGoKo699/early-gradient-spectra-lora/releases/tag/v1.0.4-publication) | Historical reference, not the current cleanup version |

## Recorded companion archive identities

| Path in data artifact | Bytes | SHA-256 |
|---|---:|---|
| `releases/stage4_paper_20260622T061351.tar.gz` | 2729153 | `e1bec75d05b0716b7f0e314c0f18d64fa99875bd07d1e634e31bb1b8c268f2c2` |
| `releases/transformer_publication_20260622T101458Z.tar.gz` | 16611798 | `b1deb63807b04a4cd1032a7152d9664ff4b47929dd68ce99afc4317d500482e3` |
| `releases/real_lora_publication_20260623T074520Z.tar.gz` | 115868168 | `4368ca32afd32fde04c68e1b889bcba2add57a801d8360db4e99557953c991de` |
| `inputs/real_lora_validation_inputs_20260623.tar.gz` | 474014914 | `e152ecb90e09dd79739f02f77dbf688c5a3ac06374b60d319da73e5fe537b9b1` |
| `stage4_source/stage4_code_snapshot_e78df5898c949b003dfde4a8ac568465a5188b6d.tar.gz` | 3313564 | `c4006d6dcebf9a95ff7a7136c7eac3a2203b92a389709c072010819ddb032708` |

The Stage4 source sidecar is associated with source commit
`e78df5898c949b003dfde4a8ac568465a5188b6d`. It supplies the code omitted from
the original Stage4 raw archive while preserving that archive's identity.

[RELEASE_ARTIFACTS.json](RELEASE_ARTIFACTS.json) retains machine-readable hashes,
input-manifest identity, external URLs, historical build metadata, and the
current report path. Old build paths under `historical_publication` are
historical records; those files are no longer required in the working tree.
