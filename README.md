# algorithms

Python repo scaffold (src-layout) with pytest + ruff + mypy.

## Quickstart

Install dependencies with Poetry (including dev tools):

```bash
poetry install -E dev
```

Run tests:

```bash
poetry run pytest
```

Lint + format:

```bash
poetry run ruff check --fix .
poetry run ruff format .
```

Typecheck:

```bash
poetry run mypy
```

Pre-commit (optional):

```bash
poetry run pre-commit install
poetry run pre-commit run --all-files
```

