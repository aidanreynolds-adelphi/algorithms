"""XGBoost hyperparameter grid search for the obesity dataset."""

from __future__ import annotations

import csv

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from algorithms.config import DATA_DIR, REPORT_DIR, encode_obesity_labels

dataset_name: str = "ObesityDataSet_raw_and_data_sinthetic.csv"
dataset_target_column: str = "NObeyesdad"

# train_test_split hyperparameters (kept consistent with other models)
random_state: int = 42
test_size: float = 0.2
stratify = None


def load_obesity_data() -> pd.DataFrame:
    """Load the obesity CSV data from the project ``data`` directory."""
    data_path = DATA_DIR / dataset_name
    if not data_path.is_file():
        msg = f"Could not find data file at {data_path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(data_path)


# Parameter grid for GridSearchCV (objective/eval_metric fixed via base estimator)
XGB_PARAM_GRID: dict[str, object] = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.5, 0.8],
    "min_child_weight": [1, 5],
    "gamma": [0.0, 1.0],
}
CV_FOLDS: int = 5


def run_xgb_grid_search(df: pd.DataFrame) -> GridSearchCV:
    """
    Run GridSearchCV to find best XGBoost hyperparameters for the obesity dataset.
    Returns the fitted GridSearchCV object (best_estimator_ available).
    """
    if dataset_target_column not in df.columns:
        msg = f"Target column {dataset_target_column!r} not found in columns: {list(df.columns)}"
        raise ValueError(msg)

    y_raw = df[dataset_target_column]
    x = df.drop(columns=[dataset_target_column])
    x_encoded = pd.get_dummies(x, drop_first=True)
    y, label_encoder = encode_obesity_labels(y_raw)

    x_train, x_test, y_train, y_test = train_test_split(
        x_encoded,
        y,
        test_size=test_size,
        train_size=1.0 - test_size,
        random_state=random_state,
        stratify=stratify,
    )

    base = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )
    cv_splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        base,
        param_grid=XGB_PARAM_GRID,
        cv=cv_splitter,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(x_train, y_train)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Write summary report
    report_path = REPORT_DIR / "xgboost_gridsearch_report.txt"
    lines = [
        "XGBoost GridSearchCV results (obesity dataset)",
        "=" * 50,
        f"CV folds: {CV_FOLDS}",
        f"Best CV score (accuracy): {grid.best_score_:.4f}",
        "",
        "Best parameters:",
    ]
    for k, v in grid.best_params_.items():
        lines.append(f"  {k}: {v}")
    lines.extend(
        [
            "",
            "Test set evaluation (best estimator):",
            f"  Accuracy: {accuracy_score(y_test, grid.predict(x_test)):.4f}",
            "",
            "Classification report (test set):",
            classification_report(
                label_encoder.inverse_transform(y_test),
                label_encoder.inverse_transform(grid.predict(x_test)),
            ),
        ],
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Grid search report written to {report_path}")

    # Write full CV results to CSV
    cv_path = REPORT_DIR / "xgboost_gridsearch_cv_results.csv"
    if hasattr(grid, "cv_results_") and grid.cv_results_:
        params_list = grid.cv_results_["params"]
        fieldnames = ["rank", "mean_test_score", "std_test_score"]
        if params_list:
            fieldnames.extend(sorted(params_list[0].keys()))
        with open(cv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for i, params in enumerate(params_list):
                row: dict[str, str | int | float] = {
                    "rank": i + 1,
                    "mean_test_score": grid.cv_results_["mean_test_score"][i],
                    "std_test_score": grid.cv_results_["std_test_score"][i],
                    **{k: str(v) for k, v in params.items()},
                }
                writer.writerow(row)
        print(f"CV results written to {cv_path}")

    return grid


def main() -> None:
    """Run GridSearchCV for XGBoost hyperparameters and write results."""
    df = load_obesity_data()
    grid = run_xgb_grid_search(df)
    print(f"\nBest CV accuracy: {grid.best_score_:.4f}")
    print("Best params:", grid.best_params_)


if __name__ == "__main__":
    main()

