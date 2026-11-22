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

.PHONY: setup lint format test test-cov demo clean eda-setup eda eda-smoke task1-download task1-data task1-baselines task1-train-tfidf

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

# --- Task 1 (sentiment) ---------------------------------------------------------------
T1 := experiments/task1_sentiment

task1-download:   ## Metacritic (Kaggle davutb/metacritic-games) + Steam (Zenodo 1000885)
	mkdir -p data/raw/metacritic data/raw/steam
	curl -fSL --retry 10 --retry-all-errors --speed-limit 50000 --speed-time 60 -C - -o data/raw/metacritic/metacritic-games.zip \
	  https://www.kaggle.com/api/v1/datasets/download/davutb/metacritic-games
	curl -fSL --retry 10 --retry-all-errors --speed-limit 50000 --speed-time 60 -C - -o data/raw/steam/steam.csv.bz2 \
	  https://zenodo.org/api/records/1000885/files/steam.csv.bz2/content

task1-data:       ## label, select games, split (needs task1-download)
	cd $(T1) && ../../$(PYTHON) prepare_metacritic.py && ../../$(PYTHON) prepare_steam.py

task1-baselines:  ## S0 VADER + S2 zero-shot (no training)
	$(PYTHON) $(T1)/evaluate.py --system vader
	$(PYTHON) $(T1)/evaluate.py --system transformer --name S2-zeroshot

task1-train-tfidf: ## S1 TF-IDF + LR (trains, then evaluates)
	$(PYTHON) $(T1)/train_tfidf.py
	$(PYTHON) $(T1)/evaluate.py --system tfidf --model artifacts/models/task1/s1_tfidf.pkl
