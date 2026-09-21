PY ?= .venv/bin/python
DECK = deck/23BCT0253.pptx

.PHONY: help venv deck preview check clean

help:
	@echo "make venv     - create .venv and install dependencies"
	@echo "make deck     - regenerate $(DECK) from src/"
	@echo "make preview  - render every slide to preview/ as PNG (macOS only)"
	@echo "make check    - report out-of-bounds shapes and likely text overflow"

venv:
	python3 -m venv .venv
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -r requirements.txt

deck:
	$(PY) src/build.py

preview: deck
	rm -rf preview
	$(PY) src/preview.py $(DECK) preview

check: deck
	$(PY) src/validate.py $(DECK)

clean:
	rm -rf preview __pycache__ src/__pycache__
