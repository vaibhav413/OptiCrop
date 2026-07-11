"""Configuration objects for the EDA pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalysisConfig:
    """Centralized configuration for file paths and analysis settings."""

    project_root: Path
    dataset_relative_path: Path = Path("03_Data_Analysis/Crop_recommendation.csv")
    target_column: str = "label"
    dpi: int = 300
    random_state: int = 42
    sample_size: int = 5

    @property
    def analysis_root(self) -> Path:
        return self.project_root / "03_Data_Analysis"

    @property
    def src_dir(self) -> Path:
        return self.analysis_root / "src"

    @property
    def outputs_dir(self) -> Path:
        return self.analysis_root / "outputs"

    @property
    def plots_dir(self) -> Path:
        return self.analysis_root / "plots"

    @property
    def logs_dir(self) -> Path:
        return self.analysis_root / "logs"

    @property
    def univariate_plots_dir(self) -> Path:
        return self.plots_dir / "univariate"

    @property
    def bivariate_plots_dir(self) -> Path:
        return self.plots_dir / "bivariate"

    @property
    def multivariate_plots_dir(self) -> Path:
        return self.plots_dir / "multivariate"

    @property
    def dataset_path(self) -> Path:
        return self.project_root / self.dataset_relative_path

    @property
    def log_file(self) -> Path:
        return self.logs_dir / "analysis.log"
