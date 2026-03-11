from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from algorithms import obesity_logreg
from algorithms.config import ACTUAL_DATA_ROWS, OBESITY_LEVEL_ORDER, encode_obesity_labels

DATA_FILENAME = "ObesityDataSet_raw_and_data_sinthetic.csv"


def get_default_dataset_path() -> Path:
    """Return the default path to the obesity CSV dataset."""
    return Path(__file__).resolve().parents[2] / "data" / DATA_FILENAME


def load_obesity_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load the obesity dataset from the given path or the default location.

    Parameters
    ----------
    csv_path:
        Optional path to the CSV file. If not provided, the default dataset path is used.
    """
    path = Path(csv_path) if csv_path is not None else get_default_dataset_path()
    return pd.read_csv(path)


def plot_numeric_distributions(
    df: pd.DataFrame,
    *,
    columns: Iterable[str] | None = None,
) -> Figure:
    """Create histograms for numeric columns in the dataset."""
    if columns is None:
        columns = df.select_dtypes(include="number").columns

    numeric_cols = [col for col in columns if col in df.columns]
    if not numeric_cols:
        msg = "No numeric columns found to plot."
        raise ValueError(msg)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        nrows=len(numeric_cols),
        ncols=1,
        figsize=(8, 3 * len(numeric_cols)),
        constrained_layout=True,
    )

    if len(numeric_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_cols, strict=False):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {_display_name(col)}")

    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> Figure:
    """Create a correlation heatmap for numeric features."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        msg = "Dataset has no numeric columns to correlate."
        raise ValueError(msg)

    corr = numeric_df.corr(numeric_only=True)
    # Use display names for axis labels where defined
    corr_display = corr.rename(
        index=FEATURE_DISPLAY_NAMES,
        columns=FEATURE_DISPLAY_NAMES,
    )

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_display,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0.0,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Correlation heatmap of numeric features")

    return fig


def plot_correlation_scatter(df: pd.DataFrame) -> Figure:
    """Create a grid of scatter plots: each pair of numeric features is one subplot.

    Each subplot shows the actual data (one point per row) for that variable pair,
    matching the same variable set as the correlation heatmap.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        msg = "Dataset has no numeric columns to plot."
        raise ValueError(msg)

    cols = list(numeric_df.columns)
    n = len(cols)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        nrows=n,
        ncols=n,
        figsize=(2.5 * n, 2.5 * n),
        squeeze=False,
    )

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            ax.scatter(
                numeric_df.iloc[:, j],
                numeric_df.iloc[:, i],
                alpha=0.4,
                s=8,
                rasterized=(n > 4),
            )
            ax.set_xlabel(_display_name(cols[j]), fontsize=8)
            ax.set_ylabel(_display_name(cols[i]), fontsize=8)
            ax.tick_params(axis="both", labelsize=6)
            if i == 0:
                ax.set_title(_display_name(cols[j]), fontsize=8)
            if j == 0:
                ax.set_ylabel(_display_name(cols[i]), fontsize=8)

    fig.suptitle("Pairwise scatter plots of numeric features (actual data)", y=1.01)
    fig.tight_layout()
    return fig


def plot_obesity_levels_count(df: pd.DataFrame) -> Figure:
    """Plot the count of samples per obesity level."""
    target_col = "NObeyesdad"
    if target_col not in df.columns:
        msg = f"Target column '{target_col}' not found in dataset."
        raise ValueError(msg)

    sns.set_theme(style="whitegrid")
    present_levels = set(df[target_col].unique())
    order = [level for level in OBESITY_LEVEL_ORDER if level in present_levels]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x=target_col, order=order, ax=ax)
    ax.set_title("Count of samples per obesity level")
    ax.set_xlabel("Obesity level")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    fig.subplots_adjust(bottom=0.30)

    return fig


def plot_obesity_level_by_gender(df: pd.DataFrame) -> Figure:
    """Plot obesity level distribution by gender (grouped bar chart).

    X-axis is gender; bars show count per obesity level for each gender.
    """
    target_col = "NObeyesdad"
    gender_col = "Gender"
    for col in (target_col, gender_col):
        if col not in df.columns:
            msg = f"Column '{col}' not found in dataset."
            raise ValueError(msg)

    sns.set_theme(style="whitegrid")
    present_levels = set(df[target_col].unique())
    order_levels = [level for level in OBESITY_LEVEL_ORDER if level in present_levels]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        data=df,
        x=gender_col,
        hue=target_col,
        hue_order=order_levels,
        ax=ax,
    )
    ax.set_title("Obesity level by gender")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Count")
    ax.legend(title="Obesity level", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    return fig


def _ordinal_obesity_series(df: pd.DataFrame, target_col: str = "NObeyesdad") -> pd.Series:
    """Map NObeyesdad labels to ordinal numbers 1..7 for correlation."""
    return (
        df[target_col]
        .map({label: i + 1 for i, label in enumerate(OBESITY_LEVEL_ORDER)})
        .astype(float)
    )


def plot_point_biserial_gender_obesity(df: pd.DataFrame) -> Figure:
    """Plot point-biserial correlation between Gender and obesity level, with statistics.

    Encodes Gender as binary (Female=0, Male=1) and NObeyesdad as ordinal 1..7,
    then computes point-biserial r and displays it with a visual (mean ordinal
    score by gender and correlation annotation).
    """
    target_col = "NObeyesdad"
    gender_col = "Gender"
    for col in (target_col, gender_col):
        if col not in df.columns:
            msg = f"Column '{col}' not found in dataset."
            raise ValueError(msg)

    # Encode: Female=0, Male=1; obesity level 1..7
    gender_binary = (df[gender_col] == "Male").astype(int).to_numpy(dtype=np.float64)
    obesity_ordinal = _ordinal_obesity_series(df).to_numpy(dtype=np.float64)
    valid = ~(np.isnan(gender_binary) | np.isnan(obesity_ordinal))
    x = gender_binary[valid]
    y = obesity_ordinal[valid]
    n = int(np.sum(valid))

    r_pb, p_value = stats.pointbiserialr(x, y)
    r_pb = float(r_pb)
    p_value = float(p_value)

    # Summary stats by gender for the plot
    female_mask = x == 0
    male_mask = x == 1
    mean_f = float(np.mean(y[female_mask]))
    mean_m = float(np.mean(y[male_mask]))
    count_f = int(np.sum(female_mask))
    count_m = int(np.sum(male_mask))

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    genders = ["Female", "Male"]
    means = [mean_f, mean_m]
    counts = [count_f, count_m]
    x_pos = [0, 1]
    bars = ax.bar(x_pos, means, color=["#e74c3c", "#3498db"], edgecolor="black", linewidth=1.2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(genders)
    ax.set_ylabel("Mean obesity level (1–7 ordinal)")
    ax.set_xlabel("Gender")
    ax.set_ylim(0, max(means) * 1.25 if means else 8)
    ax.set_title("Point-biserial correlation: Gender vs. obesity level")

    for bar, m, c in zip(bars, means, counts, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{m:.2f}\nn={c}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    # Annotation box with r and p (bottom-left to avoid bar labels)
    textstr = f"Point-biserial r = {r_pb:.3f}\np-value = {p_value:.4f}\nN = {n}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.9)
    ax.text(
        0.02,
        0.02,
        textstr,
        transform=ax.transAxes,
        fontsize=11,
        va="bottom",
        ha="left",
        bbox=props,
    )
    fig.tight_layout()
    return fig


def plot_pairplot_by_obesity_level(
    df: pd.DataFrame,
    *,
    vars: Iterable[str] | None = None,
) -> sns.axisgrid.PairGrid:
    """Create a Seaborn pairplot colored by obesity level."""
    target_col = "NObeyesdad"
    if target_col not in df.columns:
        msg = f"Target column '{target_col}' not found in dataset."
        raise ValueError(msg)

    if vars is None:
        vars = ["Age", "Height", "Weight"]

    selected_vars = [col for col in vars if col in df.columns]
    if not selected_vars:
        msg = "No valid variables provided for pairplot."
        raise ValueError(msg)

    sns.set_theme(style="whitegrid")
    present_levels = set(df[target_col].unique())
    hue_order = [level for level in OBESITY_LEVEL_ORDER if level in present_levels]
    grid = sns.pairplot(
        df,
        vars=selected_vars,
        hue=target_col,
        hue_order=hue_order,
        corner=True,
        diag_kind="kde",
    )
    grid.fig.suptitle("Pairplot of selected features by obesity level", y=1.02)

    # Ensure axis labels are visible and clear (use display names where defined).
    for idx, var in enumerate(selected_vars):
        label = _display_name(var)
        if grid.axes[-1][idx] is not None:
            grid.axes[-1][idx].set_xlabel(label)
        if grid.axes[idx][0] is not None:
            grid.axes[idx][0].set_ylabel(label)

    return grid


def plot_feature_importance(
    model: object,
    feature_names: list[str],
    *,
    top_n: int = 20,
) -> Figure:
    """Create a horizontal bar chart of XGBoost (or tree) feature importances.

    Parameters
    ----------
    model
        Fitted model with a ``feature_importances_`` attribute (e.g. XGBClassifier).
    feature_names
        List of feature names in the same order as the model's features.
    top_n
        Number of top features to show (default 20).
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        msg = "Model has no feature_importances_ attribute."
        raise ValueError(msg)
    n = min(top_n, len(feature_names), len(importances))
    indices = (-importances).argsort()[:n]
    names = [_display_name(feature_names[i]) for i in indices]
    values = [float(importances[i]) for i in indices]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, max(5, n * 0.35)))
    ax.barh(range(n), values, align="center")
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance")
    ax.set_title(f"Top {n} XGBoost feature importances")
    fig.tight_layout()
    return fig


def plot_logreg_feature_contributions(
    df: pd.DataFrame,
    *,
    target_col: str = "NObeyesdad",
    top_n: int = 20,
) -> Figure:
    """Plot top logistic regression feature contributions (absolute coefficients).

    The model is fit on the full dataset using one-hot encoded features and the
    same obesity label encoding as the classifiers. For multiclass logistic
    regression, contributions are computed as the sum of absolute coefficients
    across classes for each feature.
    """
    if target_col not in df.columns:
        msg = f"Target column '{target_col}' not found in dataset."
        raise ValueError(msg)

    y_raw = df[target_col]
    x = df.drop(columns=[target_col])
    x_encoded = pd.get_dummies(x, drop_first=True)
    feature_names = x_encoded.columns.astype(str).tolist()

    y, _ = encode_obesity_labels(y_raw)

    # Use the same hyperparameters as obesity_logreg (grid-search tuned).
    model = LogisticRegression(
        solver=obesity_logreg.solver,
        max_iter=obesity_logreg.max_iter,
        C=obesity_logreg.C,
        class_weight=obesity_logreg.class_weight,
        fit_intercept=obesity_logreg.fit_intercept,
        intercept_scaling=obesity_logreg.intercept_scaling,
        l1_ratio=obesity_logreg.l1_ratio,
        tol=obesity_logreg.tol,
    )
    model.fit(x_encoded, y)

    coef = getattr(model, "coef_", None)
    if coef is None:
        msg = "Trained logistic regression model has no coef_ attribute."
        raise ValueError(msg)

    coef_arr = np.asarray(coef)
    contrib = np.abs(coef_arr) if coef_arr.ndim == 1 else np.abs(coef_arr).sum(axis=0)

    if contrib.shape[0] != len(feature_names):
        msg = "Mismatch between number of coefficients and feature names."
        raise ValueError(msg)

    importance = pd.Series(contrib, index=feature_names).sort_values(ascending=False)
    n = min(top_n, len(importance))
    top_features = importance.head(n)

    names = [_display_name(name) for name in top_features.index]
    values = [float(v) for v in top_features.values]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.35)))
    ax.barh(range(n), values, align="center")
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Contribution (|logistic regression coefficient|, summed over classes)")
    ax.set_title(f"Top {n} logistic regression feature contributions")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: Iterable[int] | np.ndarray,
    y_pred: Iterable[int] | np.ndarray,
    class_names: Iterable[str],
    *,
    normalize: bool = False,
    display_order: Iterable[str] | None = None,
) -> Figure:
    """Create a heatmap of the confusion matrix for multiclass predictions.

    Parameters
    ----------
    y_true
        True labels (integer indices or labels).
    y_pred
        Predicted labels (same encoding as y_true).
    class_names
        Display names for each class, in index order (index i -> class_names[i]).
    normalize
        If True, show proportions per true class (row-normalized).
    display_order
        Optional order for rows/columns (e.g. OBESITY_LEVEL_ORDER). Only classes
        present in class_names are included; order is preserved.
    """
    names = list(class_names)
    # When y_true/y_pred are integer-encoded, do not pass string labels to confusion_matrix.
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if np.issubdtype(y_true_arr.dtype, np.integer):
        cm = confusion_matrix(y_true_arr, y_pred_arr)
        unique_labels = sorted(set(np.ravel(y_true_arr)) | set(np.ravel(y_pred_arr)))
        idx_to_name = {i: names[i] for i in unique_labels if i < len(names)}
        if display_order is not None:
            order_names = [n for n in display_order if n in idx_to_name.values()]
            name_to_idx = {name: idx for idx, name in idx_to_name.items()}
            desired_indices = [name_to_idx[n] for n in order_names]
            cm = cm[np.ix_(desired_indices, desired_indices)]
            tick_labels = order_names
        else:
            tick_labels = [names[i] for i in unique_labels if i < len(names)]
    else:
        cm = confusion_matrix(y_true, y_pred, labels=names)
        tick_labels = names
    if normalize and cm.sum() > 0:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        ax=ax,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix" + (" (row-normalized)" if normalize else ""))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    return fig


# Display names for feature columns (used in plot titles, legends, and feature importance).
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "FAVC": "High Caloric Food",
    "FCVC": "Vegetables",
    "NCP": "Meals Per Day",
    "CAEC": "Eating Between Meals",
    "MTRANS": "Main Transportation",
    "Mtrans": "Main Transportation",
}


def _display_name(col: str) -> str:
    """Return human-readable display name for a column (e.g. FAVC -> High Caloric Food)."""
    return FEATURE_DISPLAY_NAMES.get(col, col.replace("_", " ").title())


# Default categorical columns to plot vs obesity level (lifestyle/demographics).
DEFAULT_CATEGORICAL_FOR_OBESITY: list[str] = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "MTRANS",
]


def plot_categorical_by_obesity(
    df: pd.DataFrame,
    *,
    columns: Iterable[str] | None = None,
    target_col: str = "NObeyesdad",
) -> Figure:
    """Create grouped bar charts of categorical feature counts by obesity level.

    One subplot per categorical column: x-axis = obesity level, bars = counts
    per category (grouped by category).
    """
    if target_col not in df.columns:
        msg = f"Target column '{target_col}' not found in dataset."
        raise ValueError(msg)
    cols = list(columns) if columns is not None else DEFAULT_CATEGORICAL_FOR_OBESITY
    valid = [c for c in cols if c in df.columns and c != target_col]
    if not valid:
        msg = "No valid categorical columns to plot."
        raise ValueError(msg)

    n = len(valid)
    nrows = (n + 2) // 3
    ncols = min(3, n)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False,
    )
    present_levels = set(df[target_col].unique())
    order = [level for level in OBESITY_LEVEL_ORDER if level in present_levels]

    for idx, col in enumerate(valid):
        ax = axes.flat[idx]
        cross = pd.crosstab(df[target_col], df[col]).reindex(index=order, fill_value=0)
        cross.plot(kind="bar", ax=ax, width=0.8)
        label = _display_name(col)
        ax.set_title(label)
        ax.set_xlabel("Obesity level")
        ax.set_ylabel("Count")
        ax.legend(title=label, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for j in range(len(valid), nrows * ncols):
        axes.flat[j].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.suptitle("Categorical predictors by obesity level", y=0.98)
    return fig


def generate_all_figures(output_dir: str | Path | None = None) -> None:
    """Generate a set of standard figures from the dataset and save them to disk.

    Numeric feature distributions are saved as one file per column.
    """
    df = load_obesity_data()

    if output_dir is None:
        repo_root = Path(__file__).resolve().parents[2]
        output_dir_path = repo_root / "figures"
    else:
        output_dir_path = Path(output_dir)

    output_dir_path.mkdir(parents=True, exist_ok=True)

    # One file per numeric column for distributions.
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        fig = plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {_display_name(col)}")
        fig.savefig(output_dir_path / f"numeric_distribution_{col}.png", dpi=150)
        plt.close(fig)

    # Distribution of age for the first ACTUAL_DATA_ROWS rows only.
    age_col = "Age"
    if age_col in df.columns:
        age_subset = df.head(ACTUAL_DATA_ROWS)
        fig = plt.figure(figsize=(8, 4))
        sns.histplot(age_subset[age_col], kde=True)
        plt.title("Distribution of Age (Actual Data)")
        fig.savefig(output_dir_path / "numeric_distribution_Age_actual.png", dpi=150)
        plt.close(fig)

    heatmap_fig = plot_correlation_heatmap(df)
    heatmap_fig.savefig(output_dir_path / "correlation_heatmap.png", dpi=150)
    plt.close(heatmap_fig)

    scatter_fig = plot_correlation_scatter(df)
    scatter_fig.savefig(output_dir_path / "correlation_scatter.png", dpi=150)
    plt.close(scatter_fig)

    counts_fig = plot_obesity_levels_count(df)
    counts_fig.savefig(output_dir_path / "obesity_level_counts.png", dpi=150)
    plt.close(counts_fig)

    gender_fig = plot_obesity_level_by_gender(df)
    gender_fig.savefig(output_dir_path / "obesity_level_by_gender.png", dpi=150)
    plt.close(gender_fig)

    pb_fig = plot_point_biserial_gender_obesity(df)
    pb_fig.savefig(output_dir_path / "point_biserial_gender_obesity.png", dpi=150)
    plt.close(pb_fig)

    # Obesity level counts for actual (non-synthetic) data only.
    if len(df) >= ACTUAL_DATA_ROWS:
        df_actual = df.head(ACTUAL_DATA_ROWS)
        counts_actual_fig = plot_obesity_levels_count(df_actual)
        if counts_actual_fig.axes:
            counts_actual_fig.axes[0].set_title(
                "Count of samples per obesity level (actual data only)",
            )
        counts_actual_fig.savefig(
            output_dir_path / "obesity_level_counts_actual.png",
            dpi=150,
        )
        plt.close(counts_actual_fig)

    pair_grid = plot_pairplot_by_obesity_level(df)
    pair_grid.savefig(output_dir_path / "pairplot_obesity_levels.png", dpi=150)
    plt.close(pair_grid.fig)

    # Categorical predictors by obesity level (data-only).
    cat_fig = plot_categorical_by_obesity(df)
    cat_fig.savefig(output_dir_path / "categorical_by_obesity.png", dpi=150)
    plt.close(cat_fig)

    # XGBoost feature importance and confusion matrix (using model default test_size).
    _generate_model_figures(output_dir_path)

    # Logistic regression feature contributions (trained on full dataset).
    logreg_fig = plot_logreg_feature_contributions(df)
    logreg_fig.savefig(output_dir_path / "logreg_feature_contributions.png", dpi=150)
    plt.close(logreg_fig)


def _generate_model_figures(output_dir: str | Path) -> None:
    """Generate XGBoost feature importance and confusion matrix; save under output_dir."""
    from algorithms.obesity_xgboost import load_obesity_data as load_xgb_data
    from algorithms.obesity_xgboost import train_and_predict

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    df_xgb = load_xgb_data()
    result = train_and_predict(df_xgb)  # uses model default test_size
    # Class names in encoder index order for confusion matrix tick labels.
    class_names = list(result.label_encoder.classes_)

    imp_fig = plot_feature_importance(
        result.model,
        result.feature_names,
        top_n=20,
    )
    imp_fig.savefig(output_dir_path / "xgboost_feature_importance.png", dpi=150)
    plt.close(imp_fig)

    cm_fig = plot_confusion_matrix(
        result.y_test,
        result.y_pred,
        class_names,
        normalize=False,
        display_order=OBESITY_LEVEL_ORDER,
    )
    cm_fig.savefig(output_dir_path / "confusion_matrix.png", dpi=150)
    plt.close(cm_fig)


if __name__ == "__main__":
    generate_all_figures()
