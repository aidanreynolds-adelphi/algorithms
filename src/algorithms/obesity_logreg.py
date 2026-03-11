from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from algorithms.config import DATA_DIR, encode_obesity_labels

#
# Obesity dataset configuration
dataset_name: str = "ObesityDataSet_raw_and_data_sinthetic.csv"
dataset_target_column: str = "NObeyesdad"

# train_test_split hyperparameters
random_state: int = 42
test_size: float = 0.2
stratify = None

# Logistic Regression hyperparameters
# Best params: {'C': 10, 'class_weight': 'balanced', 'fit_intercept': False, 'intercept_scaling': 1, 'l1_ratio': 0, 'max_iter': 1000, 'solver': 'newton-cholesky', 'tol': 0.01}
solver: str = "newton-cholesky"
max_iter: int = 10000
C: float = 10
class_weight: str | None = "balanced"
fit_intercept: bool = False
intercept_scaling: float = 1
l1_ratio: float = 0
tol: float = 0.01

def load_obesity_data() -> pd.DataFrame:
    """Load the obesity CSV data from the project ``data`` directory."""
    data_path = DATA_DIR / dataset_name
    if not data_path.is_file():
        msg = f"Could not find data file at {data_path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(data_path)


def train_logistic_regression(df: pd.DataFrame) -> LogisticRegression:
    """Train a logistic regression classifier on the obesity dataframe."""
    if dataset_target_column not in df.columns:
        msg = f"Target column {dataset_target_column!r} not found in columns: {list(df.columns)}"
        raise ValueError(msg)

    y_raw = df[dataset_target_column]
    x = df.drop(columns=[dataset_target_column])

    # One-hot encode categorical features; keep numeric columns as-is.
    x_encoded = pd.get_dummies(x, drop_first=True)
    feature_names = x_encoded.columns.astype(str).tolist()

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

    model = LogisticRegression(
        solver=solver,
        max_iter=max_iter,
        C=C,
        class_weight=class_weight,
        fit_intercept=fit_intercept,
        intercept_scaling=intercept_scaling,
        l1_ratio=l1_ratio,
        tol=tol,
    )

    model.fit(x_train, y_train)

    # Report optimizer progress so we can see how close we were to max_iter.
    n_iter = model.n_iter_
    max_n_iter = int(max(n_iter)) if hasattr(n_iter, "__len__") else int(n_iter)

    print(f"\nSolver: {solver}, max_iter={max_iter}, C={C}")
    print(f"Number of iterations run (n_iter_): {n_iter} (max across classes: {max_n_iter})")

    # ------------------------------------------------------------------
    # Feature contribution section
    # ------------------------------------------------------------------
    coef = model.coef_
    class_names = [str(c) for c in label_encoder.classes_]

    # For multiclass, coef_ has shape (n_classes, n_features).
    coef_df = pd.DataFrame(coef, columns=feature_names, index=class_names)

    # Aggregate absolute coefficient magnitudes across classes as a simple
    # measure of overall contribution strength.
    importance = coef_df.abs().sum(axis=0).sort_values(ascending=False)

    print("\nFeature contribution summary (logistic regression coefficients)")
    print("Top features by absolute coefficient magnitude (summed over classes):")
    for feature, score in importance.head(20).items():
        print(f"  {feature}: {score:.4f}")

    print("\nFull coefficient table (rows = class, columns = feature):")
    # Transpose so that each row is a feature for readability in the report.
    print(coef_df.T.to_string())

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
    """Entry point for running logistic regression on the obesity dataset."""
    df = load_obesity_data()
    train_logistic_regression(df)


if __name__ == "__main__":
    main()
