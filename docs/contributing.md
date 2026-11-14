# Contributing

## Ownership

| Area | Owner | Paths |
|---|---|---|
| Task 1: sentiment | Farrel | `src/.../sentiment/`, `experiments/task1_sentiment/`, `configs/task1/` |
| Task 2: aspects | Mahesa | `src/.../aspect/`, `experiments/task2_aspect/`, `configs/task2/` |
| Task 3: summarization | Lukas | `src/.../summarization/`, `experiments/task3_summarization/`, `configs/task3/` |
| Shared contract | everyone | `src/.../core/`, `pipeline.py`, `configs/pipeline.yaml` |

`.github/CODEOWNERS` encodes this (replace the placeholder handles).

## Branches and PRs

- `main` is always green. No direct pushes; everything goes through a PR.
- Branch names: `<task>/<short-topic>`, e.g. `task2/bertopic-tagger`, `core/add-confidence`.
- Commits: Conventional Commits (`feat(aspect): ...`, `fix(core): ...`, `docs: ...`).
- Before opening a PR: `make format && make lint && make test`. CI runs the same checks.
- PRs that only touch your own task folder need 1 approval. PRs touching shared code need
  all three owners (CODEOWNERS enforces this once branch protection is on).
- Keep research code in `experiments/`; move it into `src/` only once it implements the
  task Protocol and has a test.
- Never commit data, model weights, or `.env` (see `data/README.md`).

## Changing the shared schema (`core/schemas.py`)

The field names are an agreed JSON contract. To change them:

1. Open an issue or message the group describing the change and why.
2. Prefer **additive, optional** fields (with a default) over renames or removals.
3. Update `schemas.py`, every implementation that produces/consumes the field, the tests,
   the JSON example in `README.md`, and `docs/architecture.md` in the **same PR**.
4. Get approval from all three owners.

## Adding a real implementation

1. Implement the Protocol from `core/interfaces.py` in your task package, e.g.
   `src/steam_review_translator/aspect/bertopic_tagger.py`. Import heavy libraries inside
   that module only (never from `core/`).
2. Register it in your registry in `pipeline.py`:
   `ASPECT_REGISTRY["bertopic"] = "steam_review_translator.aspect.bertopic_tagger:BertopicTagger"`.
3. Select it in `configs/pipeline.yaml` (`impl: bertopic`, constructor kwargs in `params`).
4. Add tests. Tests that need the heavy extras must skip cleanly without them:
   `pytest.importorskip("bertopic")`, because CI installs only `.[dev]`.
