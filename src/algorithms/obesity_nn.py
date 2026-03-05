from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score, classification_report  # type: ignore[import-not-found]
from sklearn.model_selection import train_test_split  # type: ignore[import-not-found]
from sklearn.neural_network import MLPClassifier  # type: ignore[import-not-found]
from sklearn.preprocessing import LabelEncoder  # type: ignore[import-not-found]

from algorithms.config import DATA_DIR

dataset_name: str = "ObesityDataSet_raw_and_data_sinthetic.csv"
dataset_target_column: str = "NObeyesdad"

test_size: float = 0.2
train_size: float = 1.0 - test_size
random_state: int = 42
stratify = None

# Basic MLP hyperparameters
hidden_layer_sizes: tuple[int, int] = (128, 64)
activation: str = "relu"
solver: str = "adam"
alpha: float = 1e-4
batch_size: int = 64
learning_rate_init: float = 0.001
max_iter: int = 200


def load_obesity_data() -> pd.DataFrame:
    """Load the obesity CSV data from the project ``data`` directory."""
    data_path = DATA_DIR / dataset_name
    if not data_path.is_file():
        msg = f"Could not find data file at {data_path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(data_path)


def train_mlp_classifier(df: pd.DataFrame) -> MLPClassifier:
    """Train an MLP classifier (neural network) on the obesity dataframe."""
    if dataset_target_column not in df.columns:
        msg = f"Target column {dataset_target_column!r} not found in columns: {list(df.columns)}"
        raise ValueError(msg)

    y_raw = df[dataset_target_column]
    X = df.drop(columns=[dataset_target_column])

    # One-hot encode categorical features; keep numeric columns as-is.
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Encode string labels into integer classes 0..n-1.
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

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
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
    """Entry point for running an MLP on the obesity dataset."""
    df = load_obesity_data()
    train_mlp_classifier(df)


if __name__ == "__main__":
    main()

