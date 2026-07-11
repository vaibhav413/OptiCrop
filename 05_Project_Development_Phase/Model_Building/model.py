"""Train and evaluate OptiCrop crop recommendation models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

plt.style.use("fivethirtyeight")


FEATURE_COLUMNS = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]
TARGET_COLUMN = "label"
RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelConfig:
    """Runtime configuration for model training."""

    project_root: Path
    test_size: float = 0.2
    random_state: int = RANDOM_STATE

    @property
    def module_root(self) -> Path:
        return self.project_root / "05_Model_Building"

    @property
    def outputs_dir(self) -> Path:
        return self.module_root / "outputs"

    @property
    def plots_dir(self) -> Path:
        return self.module_root / "plots"

    @property
    def model_path(self) -> Path:
        return self.module_root / "crop_model.pkl"

    @property
    def cleaned_dataset_path(self) -> Path:
        return self.project_root / "04_Preprocessing" / "outputs" / "cleaned_dataset.csv"

    @property
    def original_dataset_path(self) -> Path:
        return self.project_root / "03_Data_Analysis" / "Crop_recommendation.csv"


def ensure_directories(config: ModelConfig) -> None:
    """Create model output folders if they do not exist."""
    config.module_root.mkdir(parents=True, exist_ok=True)
    config.outputs_dir.mkdir(parents=True, exist_ok=True)
    config.plots_dir.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV file and validate the expected schema."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)
    missing_columns = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Dataset at {path} is missing required columns: {missing_columns}")
    return dataframe


def choose_dataset(config: ModelConfig) -> tuple[pd.DataFrame, Path, list[str]]:
    """Prefer the cleaned dataset unless it drops crop classes."""
    candidates: list[tuple[Path, pd.DataFrame]] = []
    for path in (config.cleaned_dataset_path, config.original_dataset_path):
        if path.exists():
            candidates.append((path, load_dataset(path)))

    if not candidates:
        raise FileNotFoundError("No crop dataset was found in preprocessing or analysis folders.")

    label_counts = [(frame[TARGET_COLUMN].nunique(), -frame.shape[0], path, frame) for path, frame in candidates]
    _, _, selected_path, selected_frame = max(label_counts, key=lambda item: (item[0], item[1]))

    selected_labels = sorted(selected_frame[TARGET_COLUMN].unique())
    return selected_frame.copy(), selected_path, selected_labels


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Train a scaled multinomial logistic regression model."""
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=4000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_kmeans(X_train: pd.DataFrame, n_clusters: int) -> Pipeline:
    """Train a scaled K-Means clustering pipeline."""
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clusterer",
                KMeans(
                    n_clusters=n_clusters,
                    random_state=RANDOM_STATE,
                    n_init=20,
                ),
            ),
        ]
    )
    model.fit(X_train)
    return model


def map_clusters_to_labels(cluster_ids: np.ndarray, y_reference: pd.Series) -> dict[int, str]:
    """Map each cluster to the majority label inside the training data."""
    mapping: dict[int, str] = {}
    reference = pd.DataFrame({"cluster": cluster_ids, "label": y_reference.values})
    for cluster_id, group in reference.groupby("cluster"):
        majority_label = group["label"].value_counts().idxmax()
        mapping[int(cluster_id)] = str(majority_label)
    return mapping


def predict_kmeans_labels(model: Pipeline, X: pd.DataFrame, cluster_to_label: dict[int, str]) -> np.ndarray:
    """Convert cluster assignments into crop labels using the majority mapping."""
    cluster_ids = model.predict(X)
    return np.array([cluster_to_label.get(int(cluster_id), "unknown") for cluster_id in cluster_ids])


def save_confusion_matrix_plot(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[str],
    save_path: Path,
) -> None:
    """Save a labeled confusion matrix plot."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(14, 11))
    sns.heatmap(
        matrix,
        annot=False,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
    )
    plt.title("Logistic Regression Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_kmeans_cluster_plot(
    X_test: pd.DataFrame,
    cluster_ids: np.ndarray,
    save_path: Path,
) -> None:
    """Plot K-Means clusters in PCA space."""
    scaled = StandardScaler().fit_transform(X_test)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(scaled)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=cluster_ids, cmap="tab20", s=24, alpha=0.85)
    plt.title("K-Means Clusters in PCA Space")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_model_summary(
    dataset_path: Path,
    data_shape: tuple[int, int],
    labels: list[str],
    logistic_accuracy: float,
    kmeans_accuracy: float,
    best_model_name: str,
) -> str:
    """Create a human-readable summary of the model build."""
    lines = [
        "OptiCrop Model Summary",
        "=" * 22,
        "",
        f"Dataset used: {dataset_path}",
        f"Dataset shape: {data_shape}",
        f"Labels trained: {len(labels)}",
        f"Label set: {', '.join(labels)}",
        "",
        f"Logistic Regression Accuracy: {logistic_accuracy:.4f}",
        f"K-Means Pseudo-Label Accuracy: {kmeans_accuracy:.4f}",
        f"Best Model: {best_model_name}",
        "",
        "Notes:",
        "- Logistic Regression is used as the deployable classifier.",
        "- K-Means is retained for exploratory cluster analysis and visualization.",
    ]
    return "\n".join(lines)


def run_model_building(config: ModelConfig) -> dict[str, Any]:
    """Train, evaluate, and persist the OptiCrop model artifacts."""
    ensure_directories(config)

    dataset, dataset_path, labels = choose_dataset(config)

    X = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    logistic_model = train_logistic_regression(X_train, y_train)
    logistic_predictions = logistic_model.predict(X_test)
    logistic_accuracy = accuracy_score(y_test, logistic_predictions)
    logistic_report = classification_report(y_test, logistic_predictions, zero_division=0)

    kmeans_model = train_kmeans(X_train, n_clusters=len(labels))
    train_clusters = kmeans_model.predict(X_train)
    cluster_to_label = map_clusters_to_labels(train_clusters, y_train)
    kmeans_predictions = predict_kmeans_labels(kmeans_model, X_test, cluster_to_label)
    kmeans_accuracy = accuracy_score(y_test, kmeans_predictions)

    best_model_name = "Logistic Regression"
    best_model = logistic_model

    save_confusion_matrix_plot(
        y_true=y_test,
        y_pred=pd.Series(logistic_predictions, index=y_test.index),
        labels=labels,
        save_path=config.outputs_dir / "confusion_matrix.png",
    )

    save_kmeans_cluster_plot(
        X_test=X_test,
        cluster_ids=kmeans_model.predict(X_test),
        save_path=config.plots_dir / "kmeans_clusters.png",
    )

    joblib.dump(
        {
            "model": best_model,
            "feature_columns": FEATURE_COLUMNS,
            "labels": labels,
            "dataset_path": str(dataset_path),
            "best_model_name": best_model_name,
            "logistic_accuracy": logistic_accuracy,
            "kmeans_accuracy": kmeans_accuracy,
        },
        config.model_path,
    )

    (config.outputs_dir / "accuracy.txt").write_text(
        f"Logistic Regression Accuracy: {logistic_accuracy:.4f}\nK-Means Pseudo-Label Accuracy: {kmeans_accuracy:.4f}\n",
        encoding="utf-8",
    )
    (config.outputs_dir / "classification_report.txt").write_text(logistic_report, encoding="utf-8")
    (config.outputs_dir / "model_summary.txt").write_text(
        build_model_summary(
            dataset_path=dataset_path,
            data_shape=dataset.shape,
            labels=labels,
            logistic_accuracy=logistic_accuracy,
            kmeans_accuracy=kmeans_accuracy,
            best_model_name=best_model_name,
        ),
        encoding="utf-8",
    )

    return {
        "dataset_path": dataset_path,
        "logistic_accuracy": logistic_accuracy,
        "kmeans_accuracy": kmeans_accuracy,
        "labels": labels,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train OptiCrop crop recommendation models.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the repository root.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = ModelConfig(project_root=args.project_root)
    results = run_model_building(config)
    print(f"Dataset used: {results['dataset_path']}")
    print(f"Logistic Regression Accuracy: {results['logistic_accuracy']:.4f}")
    print(f"K-Means Pseudo-Label Accuracy: {results['kmeans_accuracy']:.4f}")
    print(f"Saved model: {config.model_path}")


if __name__ == "__main__":
    main()