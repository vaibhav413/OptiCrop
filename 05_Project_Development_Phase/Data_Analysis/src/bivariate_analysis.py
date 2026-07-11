"""Bivariate analysis against crop label."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .visualization import finalize_and_save


def _safe_name(column_name: str) -> str:
    return column_name.replace(" ", "_").lower()


def run_bivariate_analysis(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    output_dir: Path,
    dpi: int,
) -> list[str]:
    """Create bivariate plots and produce data-backed observations."""
    insights: list[str] = []

    for feature in feature_columns:
        safe_col = _safe_name(feature)

        plt.figure(figsize=(14, 6))
        sns.scatterplot(
            data=dataframe,
            x=feature,
            y=target_column,
            hue=target_column,
            legend=False,
            alpha=0.7,
        )
        plt.title(f"Scatter Plot: {feature} vs {target_column}")
        plt.xlabel(feature)
        plt.ylabel(target_column)
        finalize_and_save(output_dir / f"{safe_col}_vs_{target_column}_scatter.png", dpi=dpi)

        plt.figure(figsize=(14, 6))
        sns.boxplot(data=dataframe, x=target_column, y=feature, palette="Set2")
        plt.title(f"Box Plot: {feature} vs {target_column}")
        plt.xlabel(target_column)
        plt.ylabel(feature)
        plt.xticks(rotation=45, ha="right")
        finalize_and_save(output_dir / f"{safe_col}_vs_{target_column}_box.png", dpi=dpi)

        plt.figure(figsize=(14, 6))
        sns.violinplot(data=dataframe, x=target_column, y=feature, palette="Set3", cut=0)
        plt.title(f"Violin Plot: {feature} vs {target_column}")
        plt.xlabel(target_column)
        plt.ylabel(feature)
        plt.xticks(rotation=45, ha="right")
        finalize_and_save(output_dir / f"{safe_col}_vs_{target_column}_violin.png", dpi=dpi)

        plt.figure(figsize=(14, 6))
        feature_means = dataframe.groupby(target_column, observed=False)[feature].mean().sort_values(ascending=False)
        sns.barplot(x=feature_means.index, y=feature_means.values, palette="magma")
        plt.title(f"Bar Plot: Mean {feature} by {target_column}")
        plt.xlabel(target_column)
        plt.ylabel(f"Mean {feature}")
        plt.xticks(rotation=45, ha="right")
        finalize_and_save(output_dir / f"{safe_col}_vs_{target_column}_bar.png", dpi=dpi)

        max_label = feature_means.idxmax()
        min_label = feature_means.idxmin()
        insight = (
            f"{max_label} has the highest average {feature} ({feature_means.loc[max_label]:.2f}), "
            f"while {min_label} has the lowest ({feature_means.loc[min_label]:.2f})."
        )
        insights.append(insight)

    return insights
