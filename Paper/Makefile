.PHONY: stage4-to-paper stage4-release-to-paper transformer-to-paper real-lora-to-paper refresh-paper-inputs paper-artifacts paper paper-from-releases publication-check publication-check-python publication-image clean

PYTHON ?= python3
SOURCE_DATE_EPOCH ?= 1782123298
PDF_ENV = SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) FORCE_SOURCE_DATE=1 TZ=UTC LANG=C.UTF-8 LC_ALL=C.UTF-8
ARTIFACT_ENV = $(PDF_ENV) PYTHONHASHSEED=0 MPLBACKEND=Agg MPLCONFIGDIR=$(CURDIR)/.matplotlib-cache

stage4-to-paper:
	$(PYTHON) import_stage4_aggregate.py

transformer-to-paper:
	$(PYTHON) import_transformer_publication.py

stage4-release-to-paper:
	@echo "ERROR: bundled Stage4 aggregate is superseded; run ../Code/rmt_lora_sim/scripts/run_stage4.sh" >&2
	@false

real-lora-to-paper:
	$(PYTHON) import_real_lora_results.py

refresh-paper-inputs: stage4-to-paper transformer-to-paper real-lora-to-paper

paper-artifacts:
	@mkdir -p .matplotlib-cache
	$(ARTIFACT_ENV) $(PYTHON) make_paper_artifacts.py

# Compile the committed, reviewer-facing tables and figures. This target does
# not require the large raw releases and does not regenerate data imports.
paper:
	$(PDF_ENV) pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	$(PDF_ENV) pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	cp paper.pdf ../paper.pdf

# Refresh all paper-facing inputs from external full releases, then rebuild.
paper-from-releases: refresh-paper-inputs paper-artifacts paper

publication-check:
	$(PYTHON) verify_publication_reproducibility.py --scope full

publication-check-python:
	$(PYTHON) verify_publication_reproducibility.py --scope python

publication-image:
	docker build --platform linux/amd64 -f ../Dockerfile.publication -t early-gradient-spectra-publication ..

clean:
	rm -rf .matplotlib-cache
	rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz paper.pdf
