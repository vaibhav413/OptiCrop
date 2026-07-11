"""Utility helpers for EDA pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


def ensure_directories(paths: Iterable[Path]) -> None:
    """Create directories if they do not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def configure_logging(log_file: Path) -> logging.Logger:
    """Configure and return a module-level logger."""
    logger = logging.getLogger("opticrop_eda")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def save_dataframe(dataframe: pd.DataFrame, path: Path, index: bool = True) -> None:
    """Persist a DataFrame as CSV."""
    dataframe.to_csv(path, index=index)


def save_text(content: str, path: Path) -> None:
    """Persist plain text content to a file."""
    path.write_text(content, encoding="utf-8")


def split_column_types(dataframe: pd.DataFrame) -> Tuple[list[str], list[str]]:
    """Return numerical and categorical columns."""
    numerical_cols = dataframe.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = dataframe.select_dtypes(exclude=["number"]).columns.tolist()
    return numerical_cols, categorical_cols


def print_section(title: str) -> None:
    """Print a formatted section header."""
    border = "=" * len(title)
    print(f"\n{title}\n{border}")
