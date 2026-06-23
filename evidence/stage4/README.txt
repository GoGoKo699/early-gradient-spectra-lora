Stage4 publication evidence bundle
==================================

The source_runs directory contains every raw per-rank row used by the aggregate. target_definition.json fixes the observed-best estimand. Validate this directory with:

  PYTHONPATH=. python3 scripts/validate_stage4_release.py runs/stage4_releases/stage4_paper_20260622T061351
