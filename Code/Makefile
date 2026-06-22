.PHONY: stage4-to-paper stage4-release-to-paper test test-transformer test-rmt test-real smoke-rmt smoke-transformer smoke-transformer-scaling smoke-transformer-publication transformer-publication real-lora-to-paper paper-artifacts paper clean-paper

RMT_SMOKE_CFG = configs/smoke

test: test-transformer test-rmt test-real

test-transformer:
	cd transformer && PYTHONPATH=. python3 -m pytest -q

test-rmt:
	cd rmt_lora_sim && PYTHONPATH=. python3 -m pytest -q
	cd rmt_lora_sim && python3 -m compileall -q rmt_lora scripts

test-real:
	cd real_lora_validation && PYTHONPATH=. python3 -m pytest -q tests
	cd real_lora_validation && python3 -m py_compile run_real_lora_validation.py spectral_metrics.py hf_gpu_smoke.py scripts/aggregate_real_replicates.py scripts/prepare_publication_inputs.py scripts/preflight_publication_rerun.py scripts/run_publication_correction.py

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

real-lora-to-paper:
	cd ../Paper && python3 import_real_lora_results.py

paper-artifacts:
	cd ../Paper && python3 make_paper_artifacts.py

paper: stage4-to-paper real-lora-to-paper paper-artifacts
	cd ../Paper && pdflatex -interaction=nonstopmode paper.tex
	cd ../Paper && pdflatex -interaction=nonstopmode paper.tex
	cp ../Paper/paper.pdf ../paper.pdf

clean-paper:
	rm -f ../Paper/*.aux ../Paper/*.log ../Paper/*.out ../Paper/*.toc ../Paper/paper.pdf
