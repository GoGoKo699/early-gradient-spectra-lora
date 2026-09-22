.PHONY: stage4-to-paper stage4-release-to-paper transformer-to-paper test test-transformer test-rmt test-real smoke-rmt smoke-transformer smoke-transformer-scaling smoke-transformer-publication transformer-publication real-lora-to-paper paper-artifacts paper report report-artifacts

RMT_SMOKE_CFG = configs/smoke

test: test-transformer test-rmt test-real

test-transformer:
	cd transformer && PYTHONPATH=. python3 -m pytest -q

test-rmt:
	cd rmt_lora_sim && PYTHONPATH=. python3 -m pytest -q
	cd rmt_lora_sim && python3 -m compileall -q rmt_lora scripts

test-real:
	cd real_lora_validation && PYTHONPATH=. python3 -m pytest -q tests
	cd real_lora_validation && python3 -m py_compile run_real_lora_validation.py spectral_metrics.py real_protocol.py publication_protocol.py hf_gpu_smoke.py scripts/aggregate_real_replicates.py scripts/aggregate_real_lora_publication.py scripts/prepare_publication_inputs.py scripts/preflight_publication_rerun.py scripts/run_publication_correction.py scripts/run_real_lora_publication.py scripts/build_real_lora_publication_release.py scripts/validate_real_lora_publication_release.py

smoke-rmt:
	cd rmt_lora_sim && PYTHONPATH=. python3 scripts/run_lora_rank.py --config $(RMT_SMOKE_CFG).yaml

smoke-transformer:
	cd transformer && PYTHONPATH=. python3 scripts/run_synthetic_transformer.py --config configs/smoke_modular.yaml

smoke-transformer-scaling:
	cd transformer && bash scripts/run_rank_scaling_smoke.sh

smoke-transformer-publication:
	cd transformer && bash scripts/run_transformer_publication_smoke.sh

transformer-publication:
	cd transformer && bash scripts/run_transformer_publication.sh

stage4-release-to-paper:
	@echo "ERROR: bundled Stage4 aggregate is superseded; run rmt_lora_sim/scripts/run_stage4.sh" >&2
	@false

stage4-to-paper:
	cd ../Paper && python3 import_stage4_aggregate.py

transformer-to-paper:
	cd ../Paper && python3 import_transformer_publication.py

real-lora-to-paper:
	cd ../Paper && python3 import_real_lora_results.py

# Regenerate readable artifacts from committed, validated CSV inputs.
report-artifacts:
	cd ../Paper && python3 make_paper_artifacts.py

# Compatibility aliases keep existing automation useful without a compiler.
paper-artifacts: report-artifacts

report: report-artifacts
	@echo "Research report: ../Paper/paper.md"

paper: report
