.PHONY: stage4-release-to-paper  real-lora-to-paper paper-artifacts paper clean

stage4-release-to-paper:
	python3 import_stage4_aggregate.py --aggregate ../Code/rmt_lora_sim/results/released/stage4_aggregate/stage4_key_table.csv

real-lora-to-paper:
	python3 import_real_lora_results.py

paper-artifacts:
	python3 make_paper_artifacts.py

paper: stage4-release-to-paper real-lora-to-paper paper-artifacts
	pdflatex -interaction=nonstopmode paper.tex
	pdflatex -interaction=nonstopmode paper.tex
	cp paper.pdf ../paper.pdf

clean:
	rm -f *.aux *.log *.out *.toc paper.pdf
