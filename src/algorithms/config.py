from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

"""Application-level configuration for the project."""

if TYPE_CHECKING:
    import pandas as pd

# Canonical order for NObeyesdad (underweight → normal → overweight I/II → obesity I/II/III).
# Use this for encoding so "class index" is consistent across logreg, NN, XGBoost, and viz.
OBESITY_LEVEL_ORDER: tuple[str, ...] = (
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
)


class ObesityLabelEncoder:
    """Encoder that maps string labels to 0..n-1 using OBESITY_LEVEL_ORDER."""

    def __init__(self, classes: Sequence[str]) -> None:
        self.classes_ = list(classes)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return np.array([self.classes_[int(i)] for i in y], dtype=object)


def encode_obesity_labels(
    y_raw: np.ndarray | pd.Series[str],
) -> tuple[np.ndarray, ObesityLabelEncoder]:
    """Encode NObeyesdad string labels to 0..n-1 using canonical OBESITY_LEVEL_ORDER.
    Returns (y_encoded, encoder). encoder.classes_ and encoder.inverse_transform match.
    """
    arr = np.asarray(y_raw)
    present = [lbl for lbl in OBESITY_LEVEL_ORDER if lbl in set(arr.tolist())]
    label_to_idx = {lbl: i for i, lbl in enumerate(present)}
    y = np.array([label_to_idx[lbl] for lbl in arr], dtype=np.intp)
    return y, ObesityLabelEncoder(present)


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
