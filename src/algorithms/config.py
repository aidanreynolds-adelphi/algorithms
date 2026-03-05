from __future__ import annotations

from pathlib import Path

"""Application-level configuration for the project."""


def get_project_root() -> Path:
    """Return the repository root directory.

    This file lives at ``src/algorithms/config.py``, so the repository root is
    two directories above this file (the parent of ``src/``).
    """
    return Path(__file__).resolve().parents[2]


# Project (application) configuration
PROJECT_ROOT: Path = get_project_root()
DATA_DIR: Path = PROJECT_ROOT / "data"
REPORT_DIR: Path = PROJECT_ROOT / "report"

# Number of rows in the dataset that correspond to actual (non-synthetic) data.
ACTUAL_DATA_ROWS: int = 498
