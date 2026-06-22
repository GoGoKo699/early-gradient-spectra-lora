.PHONY: stage4-to-paper stage4-release-to-paper transformer-to-paper real-lora-to-paper paper-artifacts paper clean

SOURCE_DATE_EPOCH ?= 1782123298
PDF_ENV = SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) FORCE_SOURCE_DATE=1

stage4-to-paper:
	python3 import_stage4_aggregate.py

transformer-to-paper:
	python3 import_transformer_publication.py

stage4-release-to-paper:
	@echo "ERROR: bundled Stage4 aggregate is superseded; run ../Code/rmt_lora_sim/scripts/run_stage4.sh" >&2
	@false

real-lora-to-paper:
	python3 import_real_lora_results.py

paper-artifacts:
	python3 make_paper_artifacts.py

paper: stage4-to-paper transformer-to-paper real-lora-to-paper paper-artifacts
	$(PDF_ENV) pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	$(PDF_ENV) pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	cp paper.pdf ../paper.pdf

clean:
	rm -f *.aux *.log *.out *.toc paper.pdf
