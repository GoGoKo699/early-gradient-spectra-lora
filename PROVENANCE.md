# Provenance summary

This release reconciles three local sources:

- `Publication/GitHub`: the reviewer-facing repository and compact evidence package.
- `Local`: the large raw releases and pinned offline GPT-2/WikiText-2 inputs.
- `Lora_Project`: the mother working environment used to recover source-history
  and correction-provenance information.

The mother workspace was clean at:

```text
bbfa459a4ceca34abac2dba3d522eaab970b9f2e
```

The publication-facing repository continues from that scientific source state
with packaging-only commits. At preflight, `Publication/GitHub` was at:

```text
fdfe5a1b2569e7b4d827530a892c6b34ad68fe81
```

## What each source resolved

| Question | Resolved by | Final handling |
|---|---|---|
| Where are the full raw releases? | `Local/releases` | Publish on Zenodo data record; reference from `RELEASE_ARTIFACTS.json` |
| Where are pinned GPT-2/WikiText-2 inputs? | `Local/inputs/real_lora_validation` | Package as Zenodo data companion input archive |
| Is Stage4 source lost? | `Lora_Project` history | Reference source commit or publish source sidecar/rebuilt archive |
| Were important fixes uncommitted? | `Lora_Project` Git state | Mother workspace was clean at the recorded HEAD |
| Are paper PDF copies aligned? | `Publication/paper.pdf` and `Publication/GitHub/paper.pdf` | Same SHA-256 at preflight |

## Historical records

Selected records from the mother workspace are copied under:

```text
docs/provenance/
docs/provenance/historical/
```

The mother workspace root `SHA256SUMS.txt` and `PATCH_MANIFEST.json` are
historical records only. They are not the active release checksum manifest.
