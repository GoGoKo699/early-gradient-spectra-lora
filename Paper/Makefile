.PHONY: stage4-to-paper stage4-release-to-paper real-lora-to-paper paper-artifacts paper clean

stage4-to-paper:
	python3 import_stage4_aggregate.py

stage4-release-to-paper:
	@echo "ERROR: bundled Stage4 aggregate is superseded; run ../Code/rmt_lora_sim/scripts/run_stage4.sh" >&2
	@false

real-lora-to-paper:
	python3 import_real_lora_results.py

paper-artifacts:
	python3 make_paper_artifacts.py

paper: stage4-to-paper real-lora-to-paper paper-artifacts
	pdflatex -interaction=nonstopmode paper.tex
	pdflatex -interaction=nonstopmode paper.tex
	cp paper.pdf ../paper.pdf

clean:
	rm -f *.aux *.log *.out *.toc paper.pdf
