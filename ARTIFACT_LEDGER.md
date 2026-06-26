# Artifact ledger

This project is distributed as linked software, data, and manuscript artifacts.
The three local folders used to build the final release have different roles:

| Local source folder | Final role | Public destination |
|---|---|---|
| `Publication/GitHub` | Code, compact evidence, validators, reproducibility docs, paper mirror | GitHub tag and Zenodo software record |
| `Publication/Overleaf` | Manuscript source for editing/compilation | Overleaf and optional GitHub `Paper/` mirror |
| `Publication/paper.pdf` | Final manuscript PDF | GitHub release and Zenodo software record |
| `Local` | Large raw releases and pinned GPT-2/WikiText-2 inputs | Zenodo data record |
| `Lora_Project` | Mother workspace used for provenance recovery | Not published wholesale; selected records copied to `docs/provenance/` |

## Public artifacts

| Artifact | Planned public location | Built from | Verification |
|---|---|---|---|
| Software and compact evidence | GitHub tag `v1.0.0-publication`; Zenodo software DOI `TBD` | `Publication/GitHub` | `sha256sum -c SHA256SUMS.txt`, unit tests, compact evidence checks |
| Manuscript source | Overleaf; optional GitHub `Paper/` mirror | `Publication/Overleaf` | locked publication build / Overleaf compile |
| Paper PDF | GitHub release and Zenodo software record | `Publication/paper.pdf` | SHA-256 in `RELEASE_ARTIFACTS.json` and `SHA256SUMS.txt` |
| Full raw releases | Zenodo data DOI `TBD` | `Local/releases` | archive SHA-256 plus release validators |
| Real-model pinned inputs | Zenodo data DOI `TBD` | `Local/inputs/real_lora_validation` | `INPUT_MANIFEST.json` |
| Provenance notes | GitHub `docs/provenance/` | selected files from `Lora_Project` | source-history/provenance audit |

## Raw release identities

| Study | Filename | SHA-256 |
|---|---|---|
| Stage4 | `stage4_paper_20260622T061351.tar.gz` | `e1bec75d05b0716b7f0e314c0f18d64fa99875bd07d1e634e31bb1b8c268f2c2` |
| Synthetic transformer | `transformer_publication_20260622T101458Z.tar.gz` | `b1deb63807b04a4cd1032a7152d9664ff4b47929dd68ce99afc4317d500482e3` |
| Real LoRA | `real_lora_publication_20260623T074520Z.tar.gz` | `4368ca32afd32fde04c68e1b889bcba2add57a801d8360db4e99557953c991de` |

## Stage4 source coupling

The current Stage4 raw release is valid as raw evidence but is not fully
self-contained as a software artifact. It relies on the repository source at
the recorded Stage4 source commit:

```text
e78df5898c949b003dfde4a8ac568465a5188b6d
```

Publication-quality options are:

1. rebuild the Stage4 archive with an internal `code_snapshot/`; or
2. keep the audited Stage4 archive unchanged and publish a source sidecar, such
   as `stage4_code_snapshot_e78df5898c949b003dfde4a8ac568465a5188b6d.tar.gz`.

The selected option must be recorded in `RELEASE_ARTIFACTS.json` before final
tagging.

## Staged Zenodo data package

The current staged Zenodo data folder is:

```text
/home/ubuntu/下载/Release_Staging/zenodo_data_20260626T090248Z
```

Primary artifact identities:

| Path in Zenodo data record | Bytes | SHA-256 |
|---|---:|---|
| `releases/stage4_paper_20260622T061351.tar.gz` | 2729153 | `e1bec75d05b0716b7f0e314c0f18d64fa99875bd07d1e634e31bb1b8c268f2c2` |
| `releases/transformer_publication_20260622T101458Z.tar.gz` | 16611798 | `b1deb63807b04a4cd1032a7152d9664ff4b47929dd68ce99afc4317d500482e3` |
| `releases/real_lora_publication_20260623T074520Z.tar.gz` | 115868168 | `4368ca32afd32fde04c68e1b889bcba2add57a801d8360db4e99557953c991de` |
| `inputs/real_lora_validation_inputs_20260623.tar.gz` | 474014914 | `e152ecb90e09dd79739f02f77dbf688c5a3ac06374b60d319da73e5fe537b9b1` |
| `stage4_source/stage4_code_snapshot_e78df5898c949b003dfde4a8ac568465a5188b6d.tar.gz` | 3313564 | `c4006d6dcebf9a95ff7a7136c7eac3a2203b92a389709c072010819ddb032708` |

The final Zenodo DOI is still `TBD`; after the draft DOI is reserved, update
this ledger, `RELEASE_ARTIFACTS.json`, `CITATION.cff`, and the manuscript.
