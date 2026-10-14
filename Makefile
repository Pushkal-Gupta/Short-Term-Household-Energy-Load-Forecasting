PY ?= .venv/bin/python
WORKERS ?= 8

.PHONY: help venv install flower clean test demo data preprocess baselines fedavg fedprox matrix report paper docx

help:
	@echo "make venv        - create .venv and install every dependency "
	@echo "make flower      - also install Flower (optional --backend flower)"
	@echo "make test        - unit tests"
	@echo "make demo        - end-to-end check on SYNTHETIC data (results/demo/)"
	@echo "make data        - download Low Carbon London (~800 MB) into data/raw/"
	@echo "make preprocess  - clean + select households (data/processed/)"
	@echo "make baselines   - naive, local-only and centralised arms (one seed)"
	@echo "make fedavg      - one FedAvg run (one seed)"
	@echo "make fedprox     - one FedProx run (one seed)"
	@echo "make matrix      - full experiment matrix, 3 seeds, WORKERS=$(WORKERS) in parallel"
	@echo "make report      - tables, statistics, plots, paper/generated/*.tex"
	@echo "make paper       - compile paper/manuscript.pdf (needs tectonic or pdflatex)"
	@echo "make docx        - Word manuscript (12 pages), supplement, full-length version and review response"
	@echo "make clean       - remove caches"

venv:
	python3 -m venv .venv
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -r requirements.txt

install: venv

flower:
	$(PY) -m pip install --quiet -r requirements-flower.txt

test:
	$(PY) -m pytest

demo:
	$(PY) -m src.experiments.run_demo

data:
	$(PY) -m src.data.download

preprocess:
	$(PY) -m src.data.preprocess

baselines:
	$(PY) -m src.experiments.run_baselines

fedavg:
	$(PY) -m src.experiments.run_fedavg

fedprox:
	$(PY) -m src.experiments.run_fedprox

matrix:
	$(PY) -m src.experiments.run_matrix --workers $(WORKERS)

report:
	$(PY) -m src.experiments.make_report

paper:
	cd paper && (command -v tectonic >/dev/null && tectonic manuscript.tex || (pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript))

docx:
	$(PY) -m src.evaluation.revision
	$(PY) paper/docx/workflow_figure.py
	$(PY) paper/docx/build_docx.py
	$(PY) paper/docx/build_docx.py --long
	$(PY) paper/docx/build_docx.py --supp
	$(PY) paper/docx/response2.py

clean:
	rm -rf __pycache__ src/__pycache__ .pytest_cache
