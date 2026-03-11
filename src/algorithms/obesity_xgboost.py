from __future__ import annotations

from typing import NamedTuple

import numpy as np  # type: ignore[import-untyped,unused-ignore]
import pandas as pd  # type: ignore[import-untyped,unused-ignore]
from sklearn.metrics import accuracy_score, classification_report  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from xgboost import XGBClassifier

from algorithms.config import DATA_DIR, ObesityLabelEncoder, encode_obesity_labels


class TrainResult(NamedTuple):
    """Result of training XGBoost: model, encoder, test labels, predictions, and feature names."""

    model: XGBClassifier
    label_encoder: ObesityLabelEncoder
    y_test: np.ndarray
    y_pred: np.ndarray
    feature_names: list[str]


#
# Obesity dataset configuration
dataset_name: str = "ObesityDataSet_raw_and_data_sinthetic.csv"
dataset_target_column: str = "NObeyesdad"

# train_test_split hyperparameters
random_state: int = 42
test_size: float = 0.1
stratify = None

# XGBoost hyperparameters
is_multiclass: bool = True
n_estimators: int = 300
learning_rate: float = 0.1
max_depth: int = 5
subsample: float = 0.8
colsample_bytree: float = 0.8
gamma: float = 0.0
min_child_weight: int = 1
objective: str = "multi:softprob" if is_multiclass else "binary:logistic"
eval_metric: str = "mlogloss" if is_multiclass else "logloss"
tree_method: str = "hist"

def load_obesity_data() -> pd.DataFrame:
    """Load the obesity CSV data from the project ``data`` directory."""
    data_path = DATA_DIR / dataset_name
    if not data_path.is_file():
        msg = f"Could not find data file at {data_path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(data_path)


def train_and_predict(df: pd.DataFrame, test_size_override: float | None = None) -> TrainResult:
    """Train XGBoost and return model, encoder, test labels, predictions, feature names."""
    if dataset_target_column not in df.columns:
        msg = f"Target column {dataset_target_column!r} not found in columns: {list(df.columns)}"
        raise ValueError(msg)

    split_frac = test_size_override if test_size_override is not None else test_size
    y_raw = df[dataset_target_column]
    x = df.drop(columns=[dataset_target_column])
    x_encoded = pd.get_dummies(x, drop_first=True)
    feature_names = x_encoded.columns.tolist()

    y, label_encoder = encode_obesity_labels(y_raw)

    x_train, x_temp, y_train, y_temp = train_test_split(
        x_encoded,
        y,
        test_size=split_frac,
        train_size=1.0 - split_frac,
        random_state=random_state,
        stratify=stratify,
    )
    # Split temp into validation and test sets (e.g., 50/50 split of the remaining data)
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5,
        random_state=random_state, stratify=stratify
    )

    eval_set = [(x_train, y_train), (x_valid, y_valid)]

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective=objective,
        eval_metric=eval_metric,
        tree_method=tree_method,
        gamma=gamma,
        min_child_weight=min_child_weight,
        random_state=random_state,
        early_stopping_rounds=10 # Stop if no improvement in 10 rounds
    )
    model.fit(
        x_train,
        y_train,
        eval_set=eval_set,
        verbose=True,
    )
    y_pred = model.predict(x_test)

    return TrainResult(
        model=model,
        label_encoder=label_encoder,
        y_test=y_test,
        y_pred=y_pred,
        feature_names=feature_names,
    )


def train_xgboost_classifier(df: pd.DataFrame) -> XGBClassifier:
    """Train an XGBoost classifier on the provided obesity dataframe."""
    result = train_and_predict(df)
    acc = accuracy_score(result.y_test, result.y_pred)

    print("Label mapping (class index -> label):")
    for idx, label in enumerate(result.label_encoder.classes_):
        print(f"  {idx}: {label}")

    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            result.label_encoder.inverse_transform(result.y_test),
            result.label_encoder.inverse_transform(result.y_pred),
        ),
    )

    return result.model


def main() -> None:
    """Entry point for running XGBoost on the obesity dataset."""
    df = load_obesity_data()
    train_xgboost_classifier(df)


if __name__ == "__main__":
    main()
