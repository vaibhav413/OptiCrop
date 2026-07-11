"""Multivariate analysis utilities for relationship mining."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import parallel_coordinates
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .visualization import finalize_and_save


def run_multivariate_analysis(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
    target_column: str,
    output_dir: Path,
    random_state: int,
    dpi: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Create multivariate plots and return matrices with observations."""
    insights: list[str] = []

    numeric_df = dataframe[numeric_columns]
    correlation_matrix = numeric_df.corr(method="pearson")
    covariance_matrix = numeric_df.cov()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    finalize_and_save(output_dir / "correlation_heatmap.png", dpi=dpi)

    pairplot_df = dataframe[numeric_columns + [target_column]].copy()
    pair_plot = sns.pairplot(pairplot_df, hue=target_column, corner=True, diag_kind="kde")
    pair_plot.fig.suptitle("Pair Plot of Numerical Features by Crop Label", y=1.02)
    pair_plot.fig.savefig(output_dir / "pairplot.png", dpi=dpi, bbox_inches="tight")
    plt.close(pair_plot.fig)

    pair_grid = sns.PairGrid(pairplot_df, hue=target_column, corner=True)
    pair_grid.map_lower(sns.scatterplot, alpha=0.5)
    pair_grid.map_diag(sns.kdeplot, fill=True)
    pair_grid.add_legend()
    pair_grid.fig.suptitle("Pair Grid of Numerical Features", y=1.02)
    pair_grid.fig.savefig(output_dir / "pairgrid.png", dpi=dpi, bbox_inches="tight")
    plt.close(pair_grid.fig)

    cluster_map = sns.clustermap(correlation_matrix, cmap="viridis", annot=True, fmt=".2f")
    cluster_map.fig.suptitle("Cluster Map of Correlation Matrix", y=1.02)
    cluster_map.fig.savefig(output_dir / "cluster_map.png", dpi=dpi, bbox_inches="tight")
    plt.close(cluster_map.fig)

    parallel_df = dataframe[numeric_columns + [target_column]].copy()
    plt.figure(figsize=(14, 6))
    parallel_coordinates(parallel_df, class_column=target_column, colormap="tab20", alpha=0.35)
    plt.title("Parallel Coordinates Plot")
    plt.xlabel("Features")
    plt.ylabel("Scaled Feature Value")
    plt.xticks(rotation=45, ha="right")
    finalize_and_save(output_dir / "parallel_coordinates.png", dpi=dpi)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_df)

    pca = PCA(n_components=3, random_state=random_state)
    pca_components = pca.fit_transform(scaled_features)
    pca_df = pd.DataFrame(pca_components, columns=["PC1", "PC2", "PC3"])
    pca_df[target_column] = dataframe[target_column].values

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        pca_df["PC3"],
        c=pd.factorize(pca_df[target_column])[0],
        cmap="tab20",
        alpha=0.7,
    )
    ax.set_title("3D Scatter Plot (PCA Components)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.colorbar(scatter, ax=ax, shrink=0.6, label="Encoded Crop Label")
    plt.tight_layout()
    plt.savefig(output_dir / "3d_scatter_pca.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    explained = pca.explained_variance_ratio_.sum()
    insights.append(f"First 3 PCA components explain {explained * 100:.2f}% of total variance.")

    kmeans = KMeans(n_clusters=dataframe[target_column].nunique(), random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=pca_df["PC1"], y=pca_df["PC2"], hue=cluster_labels, palette="tab20", s=60)
    plt.title("KMeans Clusters in PCA Space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    finalize_and_save(output_dir / "kmeans_pca_clusters.png", dpi=dpi)

    strongest_pair = correlation_matrix.where(~correlation_matrix.eq(1.0)).abs().stack().idxmax()
    strongest_value = correlation_matrix.loc[strongest_pair[0], strongest_pair[1]]
    insights.append(
        f"Strongest linear relationship is between {strongest_pair[0]} and {strongest_pair[1]} "
        f"(r={strongest_value:.2f})."
    )

    return correlation_matrix, covariance_matrix, insights
