from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score, classification_report  # type: ignore[import-not-found]
from sklearn.model_selection import train_test_split  # type: ignore[import-not-found]
from sklearn.preprocessing import LabelEncoder  # type: ignore[import-not-found]
from xgboost import XGBClassifier  # type: ignore[import-not-found]

from algorithms.config import DATA_DIR

#
# Obesity dataset configuration
dataset_name: str = "ObesityDataSet_raw_and_data_sinthetic.csv"
dataset_target_column: str = "NObeyesdad"

# train_test_split hyperparameters
test_size: float = 0.2
train_size: float = 1.0 - test_size # must sum to 1.0
random_state: int = 42
stratify = None

# XGBoost hyperparameters
is_multiclass: bool = True
n_estimators: int = 300
learning_rate: float = 0.1
max_depth: int = 5
subsample: float = 0.8
colsample_bytree: float = 0.8
objective: str = "multi:softprob" if is_multiclass else "binary:logistic"
eval_metric: str = "mlogloss" if is_multiclass else "logloss"
tree_method: str = "hist"
use_label_encoder: bool = False

def load_obesity_data() -> pd.DataFrame:
    """Load the obesity CSV data from the project ``data`` directory."""
    data_path = DATA_DIR / dataset_name
    if not data_path.is_file():
        msg = f"Could not find data file at {data_path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(data_path)


def train_xgboost_classifier(df: pd.DataFrame) -> XGBClassifier:
    """Train an XGBoost classifier on the provided obesity dataframe."""
    if dataset_target_column not in df.columns:
        msg = f"Target column {dataset_target_column!r} not found in columns: {list(df.columns)}"
        raise ValueError(msg)

    # By convention, X is the feature matrix and y is the labels (target variables).

    y_raw = df[dataset_target_column]
    X = df.drop(columns=[dataset_target_column])

    # One-hot encode categorical features; keep numeric columns as-is.
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Encode string labels into integer classes 0..n-1 for XGBoost.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
        stratify=stratify,
    )

    is_multiclass = len(label_encoder.classes_) > 2
    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective=objective,
        eval_metric=eval_metric,
        tree_method=tree_method,
        use_label_encoder=use_label_encoder,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Label mapping (class index -> label):")
    for idx, label in enumerate(label_encoder.classes_):
        print(f"  {idx}: {label}")

    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            label_encoder.inverse_transform(y_test),
            label_encoder.inverse_transform(y_pred),
        ),
    )

    return model


def main() -> None:
    """Entry point for running XGBoost on the obesity dataset."""
    df = load_obesity_data()
    train_xgboost_classifier(df)


if __name__ == "__main__":
    main()

