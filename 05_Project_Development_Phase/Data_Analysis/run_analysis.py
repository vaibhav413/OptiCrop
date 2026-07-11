"""Entry point for OptiCrop exploratory data analysis pipeline."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy  # noqa: F401
import seaborn as sns  # noqa: F401
import sklearn  # noqa: F401

from src.bivariate_analysis import run_bivariate_analysis
from src.config import AnalysisConfig
from src.data_loader import DatasetLoadError, dataset_preview, load_dataset, validate_dataset
from src.descriptive_analysis import build_descriptive_statistics
from src.multivariate_analysis import run_multivariate_analysis
from src.univariate_analysis import run_univariate_analysis
from src.utils import configure_logging, ensure_directories, print_section, save_dataframe, save_text

warnings.filterwarnings("ignore")
plt.style.use("fivethirtyeight")


def _format_validation_text(validation_report: dict[str, object]) -> str:
    """Format validation report as human-readable text."""
    lines = ["Dataset Validation Report", "=" * 24]
    lines.append(f"Number of records: {validation_report['n_records']}")
    lines.append(f"Number of features: {validation_report['n_features']}")
    lines.append(f"Feature names: {validation_report['feature_names']}")
    lines.append(f"Numerical columns: {validation_report['numerical_columns']}")
    lines.append(f"Categorical columns: {validation_report['categorical_columns']}")
    lines.append(f"Duplicate rows: {validation_report['duplicate_rows']}")
    lines.append(f"Unique labels: {validation_report['unique_labels']}")
    return "\n".join(lines)


def _build_dataset_summary(dataframe: pd.DataFrame, validation_report: dict[str, object]) -> str:
    """Create compact dataset summary text."""
    n_records = validation_report["n_records"]
    n_features = validation_report["n_features"]
    n_labels = len(validation_report["unique_labels"])
    missing_total = int(dataframe.isna().sum().sum())
    duplicates = int(validation_report["duplicate_rows"])

    summary = [
        "OptiCrop Dataset Summary",
        "=" * 23,
        f"Records: {n_records}",
        f"Features: {n_features}",
        f"Crop classes: {n_labels}",
        f"Missing values: {missing_total}",
        f"Duplicate rows: {duplicates}",
    ]
    return "\n".join(summary)


def run_pipeline(config: AnalysisConfig) -> None:
    """Execute complete EDA workflow and save all outputs."""
    ensure_directories(
        [
            config.analysis_root,
            config.outputs_dir,
            config.logs_dir,
            config.univariate_plots_dir,
            config.bivariate_plots_dir,
            config.multivariate_plots_dir,
        ]
    )

    logger = configure_logging(config.log_file)
    logger.info("Starting OptiCrop EDA pipeline")
    logger.info("Using numpy=%s pandas=%s sklearn=%s", np.__version__, pd.__version__, sklearn.__version__)
    logger.info("Current working directory: %s", os.getcwd())

    try:
        print_section("STEP 2: LOAD DATASET")
        dataframe = load_dataset(config.dataset_path)
        preview = dataset_preview(dataframe, sample_size=config.sample_size)

        print("Head:\n", preview["head"])
        print("\nTail:\n", preview["tail"])
        print("\nSample:\n", preview["sample"])
        print(f"\nShape: {preview['shape']}")
        print("Columns:", preview["columns"])
        print("\nData types:\n", preview["dtypes"])
        print("\nMemory usage (bytes):\n", preview["memory_usage"])
        print("\nDataset information:\n", preview["info"])

        save_text(preview["info"], config.outputs_dir / "dataset_info.txt")

        print_section("STEP 3: DATA VALIDATION")
        validation_report = validate_dataset(dataframe, target_column=config.target_column)
        missing_df = validation_report["missing_values"].rename("missing_count").to_frame()
        duplicate_df = pd.DataFrame({"duplicate_rows": [validation_report["duplicate_rows"]]})

        print(missing_df)
        print(f"Duplicate rows: {validation_report['duplicate_rows']}")
        print(f"Unique labels ({len(validation_report['unique_labels'])}): {validation_report['unique_labels']}")

        save_dataframe(missing_df, config.outputs_dir / "missing_values.csv")
        save_dataframe(duplicate_df, config.outputs_dir / "duplicate_report.csv", index=False)
        save_text(_format_validation_text(validation_report), config.outputs_dir / "validation_report.txt")
        save_text(_build_dataset_summary(dataframe, validation_report), config.outputs_dir / "dataset_summary.txt")

        if int(missing_df["missing_count"].sum()) == 0:
            print("Observation: The dataset contains no missing values.")
        else:
            print("Observation: The dataset contains missing values that require handling.")

        if int(validation_report["duplicate_rows"]) == 0:
            print("Observation: No duplicate rows are present.")
        else:
            print("Observation: Duplicate rows are present and may require deduplication.")

        numeric_columns = [col for col in dataframe.select_dtypes(include=["number"]).columns if col != config.target_column]
        feature_columns = [col for col in dataframe.columns if col != config.target_column]

        print_section("STEP 4: DESCRIPTIVE ANALYSIS")
        descriptive_stats = build_descriptive_statistics(dataframe, numeric_columns)
        print(descriptive_stats.round(4).to_string())
        save_dataframe(descriptive_stats, config.outputs_dir / "descriptive_statistics.csv")

        print_section("STEP 5: UNIVARIATE ANALYSIS")
        univariate_insights = run_univariate_analysis(
            dataframe=dataframe,
            numeric_columns=numeric_columns,
            target_column=config.target_column,
            output_dir=config.univariate_plots_dir,
            dpi=config.dpi,
        )
        for insight in univariate_insights:
            print(f"Observation: {insight}")

        print_section("STEP 6: BIVARIATE ANALYSIS")
        bivariate_insights = run_bivariate_analysis(
            dataframe=dataframe,
            feature_columns=feature_columns,
            target_column=config.target_column,
            output_dir=config.bivariate_plots_dir,
            dpi=config.dpi,
        )
        for insight in bivariate_insights:
            print(f"Observation: {insight}")

        print_section("STEP 7 & 8: MULTIVARIATE + CORRELATION ANALYSIS")
        correlation_matrix, covariance_matrix, multivariate_insights = run_multivariate_analysis(
            dataframe=dataframe,
            numeric_columns=numeric_columns,
            target_column=config.target_column,
            output_dir=config.multivariate_plots_dir,
            random_state=config.random_state,
            dpi=config.dpi,
        )

        save_dataframe(correlation_matrix, config.outputs_dir / "correlation_matrix.csv")
        save_dataframe(covariance_matrix, config.outputs_dir / "covariance_matrix.csv")

        for insight in multivariate_insights:
            print(f"Observation: {insight}")

        logger.info("EDA pipeline completed successfully")
        print_section("STEP 10: EXPORT COMPLETE")
        print(f"Saved outputs to: {config.outputs_dir}")
        print(f"Saved plots to: {config.plots_dir}")

    except DatasetLoadError as exc:
        logger.exception("Dataset loading failed")
        raise SystemExit(str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected pipeline error")
        raise SystemExit(f"Pipeline failed: {exc}") from exc


def main() -> None:
    """Program entry point."""
    project_root = Path(__file__).resolve().parents[1]
    config = AnalysisConfig(project_root=project_root)
    run_pipeline(config)


if __name__ == "__main__":
    main()
