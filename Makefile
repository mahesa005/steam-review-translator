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

.PHONY: setup lint format test test-cov demo clean

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

clean:
	$(BOOTSTRAP_PYTHON) -c "import pathlib, shutil; \
	[shutil.rmtree(p, ignore_errors=True) for pat in ('**/__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', 'htmlcov', 'build', 'dist', 'src/*.egg-info') \
	 for p in pathlib.Path('.').glob(pat) if '.venv' not in p.parts]; \
	pathlib.Path('.coverage').unlink(missing_ok=True)"
