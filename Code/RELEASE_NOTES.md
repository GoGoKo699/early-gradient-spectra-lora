# Code and evidence history

The current cleanup is documented in [docs/CLEANUP.md](../docs/CLEANUP.md).
The maintained report is [Paper/paper.md](../Paper/paper.md).

The historical real-LoRA release was corrected in June 2026 to compute entropy
effective rank from normalized **squared** singular values. Earlier summaries
are preserved only in the explicitly named superseded archive. The publication
study subsequently expanded to eight primary and three boundary seeds; use
`evidence/real_lora/` for its compact aggregates, not the earlier five/three-seed
tables.

The transformer study was rerun with common random numbers and exact realized
parameter-cost controls. Earlier budget-win summaries do not support current
claims. The corrected primary comparison is null.

The matrix subpackage retains compact historical simulations and per-directory
checksums. The current Stage4 publication aggregate is under
`evidence/stage4/`. Run-path strings in historical CSVs are provenance labels,
not portable filesystem locations.
