"""Univariate analysis for numeric and target columns."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .visualization import finalize_and_save


def _safe_name(column_name: str) -> str:
    return column_name.replace(" ", "_").lower()


def run_univariate_analysis(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
    target_column: str,
    output_dir: Path,
    dpi: int,
) -> list[str]:
    """Create univariate plots and return data-driven observations."""
    insights: list[str] = []

    for column in numeric_columns:
        safe_col = _safe_name(column)

        plt.figure(figsize=(8, 5))
        sns.histplot(dataframe[column], bins=25, kde=False, color="#2a9d8f")
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        finalize_and_save(output_dir / f"{safe_col}_histogram.png", dpi=dpi)

        plt.figure(figsize=(8, 5))
        sns.histplot(dataframe[column], bins=25, kde=True, color="#264653")
        plt.title(f"Distribution Plot of {column}")
        plt.xlabel(column)
        plt.ylabel("Density")
        finalize_and_save(output_dir / f"{safe_col}_distribution.png", dpi=dpi)

        plt.figure(figsize=(8, 5))
        sns.boxplot(x=dataframe[column], color="#e9c46a")
        plt.title(f"Box Plot of {column}")
        plt.xlabel(column)
        finalize_and_save(output_dir / f"{safe_col}_boxplot.png", dpi=dpi)

        plt.figure(figsize=(8, 5))
        sns.violinplot(x=dataframe[column], color="#f4a261")
        plt.title(f"Violin Plot of {column}")
        plt.xlabel(column)
        finalize_and_save(output_dir / f"{safe_col}_violin.png", dpi=dpi)

        plt.figure(figsize=(8, 5))
        sns.kdeplot(dataframe[column], fill=True, color="#e76f51")
        plt.title(f"Density Plot of {column}")
        plt.xlabel(column)
        plt.ylabel("Density")
        finalize_and_save(output_dir / f"{safe_col}_density.png", dpi=dpi)

        skew_value = dataframe[column].skew()
        if skew_value > 0.5:
            insights.append(f"{column} distribution is right-skewed (skew={skew_value:.2f}).")
        elif skew_value < -0.5:
            insights.append(f"{column} distribution is left-skewed (skew={skew_value:.2f}).")
        else:
            insights.append(f"{column} distribution is approximately symmetric (skew={skew_value:.2f}).")

    label_counts = dataframe[target_column].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    sns.countplot(data=dataframe, x=target_column, order=label_counts.index, palette="viridis")
    plt.title(f"Count Plot of {target_column}")
    plt.xlabel(target_column)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    finalize_and_save(output_dir / f"{target_column}_countplot.png", dpi=dpi)

    plt.figure(figsize=(10, 10))
    plt.pie(
        label_counts.values,
        labels=label_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.85,
    )
    plt.title(f"Pie Chart of {target_column}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{target_column}_piechart.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    imbalance_ratio = label_counts.max() / label_counts.min()
    if imbalance_ratio <= 1.5:
        insights.append(
            f"Crop classes are relatively balanced (max/min count ratio={imbalance_ratio:.2f})."
        )
    else:
        insights.append(
            f"Crop classes show imbalance (max/min count ratio={imbalance_ratio:.2f})."
        )

    return insights
