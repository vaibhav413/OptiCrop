"""Descriptive statistics calculations for OptiCrop dataset."""

from __future__ import annotations

import pandas as pd
from scipy import stats


def build_descriptive_statistics(dataframe: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """Build an extended descriptive statistics table for numeric features."""
    numeric_df = dataframe[numeric_columns]

    mode_df = numeric_df.mode(dropna=True)
    first_mode = mode_df.iloc[0] if not mode_df.empty else pd.Series(index=numeric_columns, dtype=float)

    descriptive = pd.DataFrame(
        {
            "count": numeric_df.count(),
            "mean": numeric_df.mean(),
            "median": numeric_df.median(),
            "mode": first_mode,
            "variance": numeric_df.var(),
            "std_dev": numeric_df.std(),
            "min": numeric_df.min(),
            "25%": numeric_df.quantile(0.25),
            "50%": numeric_df.quantile(0.50),
            "75%": numeric_df.quantile(0.75),
            "max": numeric_df.max(),
            "range": numeric_df.max() - numeric_df.min(),
            "skewness": numeric_df.apply(stats.skew, nan_policy="omit"),
            "kurtosis": numeric_df.apply(stats.kurtosis, nan_policy="omit"),
        }
    )

    return descriptive.sort_index()
