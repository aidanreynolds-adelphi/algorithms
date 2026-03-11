from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from algorithms.config import DATA_DIR, OBESITY_LEVEL_ORDER, encode_obesity_labels

dataset_name: str = "ObesityDataSet_raw_and_data_sinthetic.csv"
dataset_target_column: str = "NObeyesdad"

random_state: int = 42
test_size: float = 0.2
stratify = None

# MLP hyperparameters (single-run training; tune via run_nn_gridsearch)
hidden_layer_sizes: tuple[int, int] = (512, 128)
activation: str = "tanh"
solver: str = "adam"
alpha: float = 5e-05
batch_size: int = 64
learning_rate_init: float = 0.001
max_iter: int = 1000


def load_obesity_data() -> pd.DataFrame:
    """Load the obesity CSV data from the project ``data`` directory."""
    data_path = DATA_DIR / dataset_name
    if not data_path.is_file():
        msg = f"Could not find data file at {data_path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(data_path)


def get_mlp_architecture() -> tuple[int, tuple[int, ...], int, list[str], list[str]]:
    """
    Return (n_input_features, hidden_layer_sizes, n_classes, class_names, feature_names)
    for the obesity MLP, using the same preprocessing as training (no model fit).
    """
    df = load_obesity_data()
    features = df.drop(columns=[dataset_target_column])
    encoded = pd.get_dummies(features, drop_first=True)
    n_input = encoded.shape[1]
    feature_names = encoded.columns.astype(str).tolist()
    y_raw = df[dataset_target_column]
    present = set(y_raw.unique().tolist())
    # Same order as in obesity_viz diagrams (ordinal: insufficient → obesity type III)
    class_names = [lvl for lvl in OBESITY_LEVEL_ORDER if lvl in present]
    n_classes = len(class_names)
    return (n_input, hidden_layer_sizes, n_classes, class_names, feature_names)


def train_mlp_classifier(df: pd.DataFrame) -> MLPClassifier:
    """Train an MLP classifier (neural network) on the obesity dataframe."""
    if dataset_target_column not in df.columns:
        msg = f"Target column {dataset_target_column!r} not found in columns: {list(df.columns)}"
        raise ValueError(msg)

    y_raw = df[dataset_target_column]
    x = df.drop(columns=[dataset_target_column])

    # One-hot encode categorical features; keep numeric columns as-is.
    x_encoded = pd.get_dummies(x, drop_first=True)

    # Encode string labels into integer classes 0..n-1 (canonical OBESITY_LEVEL_ORDER).
    y, label_encoder = encode_obesity_labels(y_raw)

    x_train, x_test, y_train, y_test = train_test_split(
        x_encoded,
        y,
        test_size=test_size,
        train_size=1.0 - test_size,
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

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
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
    """Entry point: train a single MLP with fixed hyperparameters and print metrics."""
    df = load_obesity_data()
    train_mlp_classifier(df)


if __name__ == "__main__":
    main()
