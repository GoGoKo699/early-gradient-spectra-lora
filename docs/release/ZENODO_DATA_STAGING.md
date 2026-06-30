# Zenodo data staging registration

Generated UTC: `20260626T100837Z`

Staged directory:

```text
zenodo_data_20260626T090248Z
```

The staged directory passed:

```bash
sha256sum -c SHA256SUMS.txt --ignore-missing
```

## Primary data artifacts

| Path in Zenodo data record | Bytes | SHA-256 |
|---|---:|---|
| `releases/stage4_paper_20260622T061351.tar.gz` | 2729153 | `e1bec75d05b0716b7f0e314c0f18d64fa99875bd07d1e634e31bb1b8c268f2c2` |
| `releases/transformer_publication_20260622T101458Z.tar.gz` | 16611798 | `b1deb63807b04a4cd1032a7152d9664ff4b47929dd68ce99afc4317d500482e3` |
| `releases/real_lora_publication_20260623T074520Z.tar.gz` | 115868168 | `4368ca32afd32fde04c68e1b889bcba2add57a801d8360db4e99557953c991de` |
| `inputs/real_lora_validation_inputs_20260623.tar.gz` | 474014914 | `e152ecb90e09dd79739f02f77dbf688c5a3ac06374b60d319da73e5fe537b9b1` |
| `stage4_source/stage4_code_snapshot_e78df5898c949b003dfde4a8ac568465a5188b6d.tar.gz` | 3313564 | `c4006d6dcebf9a95ff7a7136c7eac3a2203b92a389709c072010819ddb032708` |

## How this connects the release

- The three raw release archives are copied unchanged from `Local/releases`.
- The real-model pinned input archive is built from `Local/inputs/real_lora_validation`.
- The Stage4 source sidecar is built from the `Lora_Project` Git history at commit `e78df5898c949b003dfde4a8ac568465a5188b6d`.
- Zenodo data DOI: `10.5281/zenodo.21061917`.

- Zenodo data record: `https://zenodo.org/records/21061917`
