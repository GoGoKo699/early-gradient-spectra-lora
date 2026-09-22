# Experiment code

| Directory | Study |
|---|---|
| `rmt_lora_sim/` | Matrix/RMT and Stage4 simulations |
| `transformer/` | Synthetic-transformer allocation experiments |
| `real_lora_validation/` | GPT-2/WikiText-2 allocation experiments |

After installing the dependencies described in
[the reproducibility guide](../REPRODUCIBILITY.md), run `make -C Code test`
from the repository root. Lightweight checks are `python verify.py`.

Current compact publication evidence is under [evidence/](../evidence/README.md).
Several older `results/released/` folders are retained for provenance and are
superseded; their names alone do not identify the current evidence. The
[research report](../Paper/paper.md) and its table status notes identify the
supported results. Full raw runs, model weights, and prepared WikiText inputs
are separate artifacts.
