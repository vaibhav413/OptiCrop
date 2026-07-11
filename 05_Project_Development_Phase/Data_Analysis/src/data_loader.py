"""Dataset loading and validation utilities."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd


class DatasetLoadError(Exception):
    """Raised when dataset loading fails."""


def load_dataset(path: Path) -> pd.DataFrame:
    """Load CSV dataset from disk."""
    if not path.exists():
        raise DatasetLoadError(f"Dataset file not found: {path}")

    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise DatasetLoadError(f"Failed to load dataset from {path}: {exc}") from exc


def dataset_preview(dataframe: pd.DataFrame, sample_size: int = 5) -> dict[str, Any]:
    """Generate standard preview artifacts for quick inspection."""
    info_buffer = StringIO()
    dataframe.info(buf=info_buffer)

    return {
        "head": dataframe.head(sample_size),
        "tail": dataframe.tail(sample_size),
        "sample": dataframe.sample(min(sample_size, len(dataframe)), random_state=42),
        "shape": dataframe.shape,
        "columns": dataframe.columns.tolist(),
        "dtypes": dataframe.dtypes,
        "memory_usage": dataframe.memory_usage(deep=True),
        "info": info_buffer.getvalue(),
    }


def validate_dataset(dataframe: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """Run structural and quality checks on the dataset."""
    numerical_cols = dataframe.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = dataframe.select_dtypes(exclude=["number"]).columns.tolist()

    report: dict[str, Any] = {
        "missing_values": dataframe.isna().sum(),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "unique_labels": sorted(dataframe[target_column].dropna().unique().tolist()),
        "feature_names": dataframe.columns.tolist(),
        "n_records": int(dataframe.shape[0]),
        "n_features": int(dataframe.shape[1]),
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
    }
    return report
