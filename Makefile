# Works with GNU make on Linux/macOS, and on Windows from Git Bash.
ifeq ($(OS),Windows_NT)
    BOOTSTRAP_PYTHON ?= py -3.11
    VENV_BIN := .venv/Scripts
else
    BOOTSTRAP_PYTHON ?= python3.11
    VENV_BIN := .venv/bin
endif

PYTHON  := $(VENV_BIN)/python
REVIEWS ?= tests/fixtures/sample_reviews.json
GAME    ?= Fake Game

EDA       := experiments/task2_aspect/eda
EDA_STEPS := 00_ingest 01_overview 02_sentences 03_lexicon 04_topics 05_select_games              06_candidate_sample 07_report

.PHONY: setup lint format test test-cov demo clean eda-setup eda eda-smoke

setup:
	$(BOOTSTRAP_PYTHON) -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"
	$(PYTHON) -m pre_commit install

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m mypy

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

test:
	$(PYTHON) -m pytest

test-cov:
	$(PYTHON) -m pytest --cov=steam_review_translator --cov-report=term-missing

demo:
	$(PYTHON) -m steam_review_translator demo --reviews $(REVIEWS) --game "$(GAME)"

# Task 2 EDA (needs the task2 extra + spaCy model; see experiments/task2_aspect/eda/README.md)
eda-setup:
	$(PYTHON) -m pip install -e ".[dev,task2]"
	$(PYTHON) -m spacy download en_core_web_sm

eda:
	@for s in $(EDA_STEPS); do echo "== $$s"; $(PYTHON) $(EDA)/$$s.py $(EDA_ARGS) || exit 1; done

eda-smoke:
	@$(MAKE) --no-print-directory eda EDA_ARGS=--smoke

clean:
	$(BOOTSTRAP_PYTHON) -c "import pathlib, shutil; \
	[shutil.rmtree(p, ignore_errors=True) for pat in ('**/__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', 'htmlcov', 'build', 'dist', 'src/*.egg-info') \
	 for p in pathlib.Path('.').glob(pat) if '.venv' not in p.parts]; \
	pathlib.Path('.coverage').unlink(missing_ok=True)"
