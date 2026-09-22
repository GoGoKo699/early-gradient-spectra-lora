# Provenance

## Current cleanup

The Markdown cleanup starts from repository commit
`5bf03dc1c5472b9297208a2df5da8bb44854f5da`.
All 523 entries in its root checksum manifest passed before edits.

The cleanup updates documentation, narrows claims, fixes identified code
defects, and changes report generation. It is not described as a packaging-only
change. Numerical evidence and existing licenses are preserved.
[docs/CLEANUP.md](docs/CLEANUP.md) records the completed checks and their limits.

The earlier manuscript and document-build system are recoverable from Git
history and [v1.0.4-publication](https://github.com/GoGoKo699/early-gradient-spectra-lora/releases/tag/v1.0.4-publication).
That tag is a historical artifact, not the version of the current working tree.

## Recorded origin of the historical release

The previous release reconciled three author-side folders:

| Recorded source | Role |
|---|---|
| `Publication/GitHub` | Repository and compact evidence |
| `Local` | Large raw releases and pinned offline model/data inputs |
| `Lora_Project` | Working environment used for provenance recovery |

The recorded scientific source commit was
`bbfa459a4ceca34abac2dba3d522eaab970b9f2e`.
The recorded publication preflight commit was
`fdfe5a1b2569e7b4d827530a892c6b34ad68fe81`.
These are historical source labels; they do not imply that the present
filesystem contains those external folders.

Selected source-history and correction records remain under
[docs/provenance/](docs/provenance/README.md). Their old paths, build commands,
and status statements describe earlier checkpoints. Historical checksum and
patch manifests are not current repository manifests. In particular, a
historical validation statement is not evidence that this cleanup reran it.
