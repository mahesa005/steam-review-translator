# CLAUDE.md

Instructions for Claude Code (and any AI assistant) working in this repository.

## Git and commit rules

- **Never add a `Co-Authored-By: Claude ...` trailer** (or any AI/tool attribution) to commit
  messages. The commit author and committer must be the repository owner only.
- **Never commit on behalf of another person** and never create commits with a different
  `user.name` / `user.email`. Use the identity already configured in git
  (`rlukassa <19623069@std.stei.itb.ac.id>`); do not override it.
- Commits must be authored by the human owner. If an AI session is asked to commit, the
  resulting commit must look exactly like a human commit: correct author, correct
  committer, no bot trailers, no "Generated with ..." lines.
- Conventional Commits are required (`feat(summarization): ...`, `fix(core): ...`,
  `docs: ...`) — see `docs/contributing.md`.
- Branch naming: `<task>/<short-topic>` (e.g. `task3/meta-review-v2`). Task ownership is
  fixed: Task 1 sentiment = Farrel, Task 2 aspects = Mahesa, Task 3 summarization = Lukas.
- `main` is protected: never push directly to it. Work on a branch and open a PR.
- Never commit data, model weights, `.env`, or derived datasets (GameWikiSum and friends).
  See `data/README.md`.

## Before finishing a change

Run `make format && make lint && make test` (or `ruff` + `mypy` + `pytest` directly) and
make sure CI-equivalent checks pass. Tests that need heavy extras must use
`pytest.importorskip(...)` so CI stays green without them.

## Project layout

| Path | What |
|---|---|
| `src/steam_review_translator/core/` | Shared contract: schemas, Protocols, config, logging |
| `src/steam_review_translator/sentiment/` | Task 1 |
| `src/steam_review_translator/aspect/` | Task 2 |
| `src/steam_review_translator/summarization/` | Task 3 |
| `src/steam_review_translator/pipeline.py` | Orchestration + implementation registries |
| `configs/` | `pipeline.yaml` plus per-task configs |
| `experiments/` | Research scripts, not part of the library |
| `tests/` | Unit + integration tests, fake fixtures |

Heavy dependencies (torch, transformers, bertopic, ...) belong only in per-task extras and
must be imported inside the module that needs them, never at `core/` import time.
