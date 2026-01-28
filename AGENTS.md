# AGENTS.md

Guidance for AI agents (and humans) working in this repo.

## Project basics

- **Language/runtime**: Python **3.11+**
- **Packaging/layout**: **src-layout**
  - Library code: `src/algorithms/`
  - Tests: `tests/`
- **Tooling**: Poetry, pytest, ruff, mypy, pre-commit

## Setup (local dev)

Install dependencies (including dev tools):

```bash
poetry install -E dev
```

Optional (recommended) git hooks:

```bash
poetry run pre-commit install
```

## Commands you should run

- **Tests**:

```bash
poetry run pytest
```

- **Lint (and autofix) + format**:

```bash
poetry run ruff check --fix .
poetry run ruff format .
```

- **Typecheck**:

```bash
poetry run mypy
```

- **All checks (suggested order)**:

```bash
poetry run ruff check --fix . && poetry run ruff format . && poetry run mypy && poetry run pytest
```

## Coding expectations

- **Type hints**: Required for new/modified code. Keep `mypy` strictness in mind.
- **Style**: Follow `ruff` formatting; do not hand-format against it.
- **Imports**: Keep imports sorted/grouped (ruff is configured with first-party `algorithms`).
- **Public API**: Prefer small, well-named functions/classes; add docstrings when behavior is non-obvious.

## Adding new code

- **Algorithms / library code**: Add modules under `src/algorithms/` and export from `src/algorithms/__init__.py` when appropriate.
- **Tests**: Add/extend tests in `tests/` using pytest. Prefer small, deterministic unit tests.
- **Examples**: Keep runnable snippets in `src/algorithms/example.py` or add a new example module if needed.

## When changing behavior

- Update or add tests that demonstrate the change.
- Run (at minimum): `ruff check`, `ruff format`, `mypy`, `pytest`.

