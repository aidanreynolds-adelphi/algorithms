from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd  # type: ignore[import-untyped]
import seaborn as sns  # type: ignore[import-untyped]
from matplotlib import pyplot as plt  # type: ignore[import-not-found]
from matplotlib.figure import Figure  # type: ignore[import-not-found]

from algorithms.config import ACTUAL_DATA_ROWS


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
        ax.set_title(f"Distribution of {col}")

    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> Figure:
    """Create a correlation heatmap for numeric features."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        msg = "Dataset has no numeric columns to correlate."
        raise ValueError(msg)

    corr = numeric_df.corr(numeric_only=True)

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
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


def plot_obesity_levels_count(df: pd.DataFrame) -> Figure:
    """Plot the count of samples per obesity level."""
    target_col = "NObeyesdad"
    if target_col not in df.columns:
        msg = f"Target column '{target_col}' not found in dataset."
        raise ValueError(msg)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x=target_col, order=sorted(df[target_col].unique()), ax=ax)
    ax.set_title("Count of samples per obesity level")
    ax.set_xlabel("Obesity level")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)

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
    grid = sns.pairplot(
        df,
        vars=selected_vars,
        hue=target_col,
        corner=True,
        diag_kind="kde",
    )
    grid.fig.suptitle("Pairplot of selected features by obesity level", y=1.02)

    # Ensure axis labels are visible and clear.
    for idx, var in enumerate(selected_vars):
        if grid.axes[-1][idx] is not None:
            grid.axes[-1][idx].set_xlabel(var)
        if grid.axes[idx][0] is not None:
            grid.axes[idx][0].set_ylabel(var)

    return grid


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
        plt.title(f"Distribution of {col}")
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

    counts_fig = plot_obesity_levels_count(df)
    counts_fig.savefig(output_dir_path / "obesity_level_counts.png", dpi=150)
    plt.close(counts_fig)

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


if __name__ == "__main__":
    generate_all_figures()

