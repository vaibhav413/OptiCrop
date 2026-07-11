# OptiCrop Data Analysis Module

## Project Objective
This module performs production-quality exploratory data analysis (EDA) for OptiCrop's crop recommendation dataset. It validates dataset quality, computes descriptive statistics, generates univariate/bivariate/multivariate visualizations, and exports analysis artifacts for downstream ML work.

## Dataset Description
- Dataset file: `Data-Analysis/Crop_recommendation.csv`
- Target column: `label`
- Features:
  - `N` (Nitrogen)
  - `P` (Phosphorous)
  - `K` (Potassium)
  - `temperature`
  - `humidity`
  - `ph`
  - `rainfall`

## Analysis Workflow
1. Import libraries and configure plotting/logging.
2. Load dataset and print structural preview.
3. Validate data quality (missing values, duplicates, schema checks).
4. Compute comprehensive descriptive statistics.
5. Generate univariate analysis plots.
6. Generate bivariate feature-vs-label visualizations.
7. Generate multivariate plots (heatmap, pairplot, pairgrid, clustermap, PCA, KMeans).
8. Export correlation and covariance matrices.
9. Print data-driven observations from computed statistics.
10. Save all outputs into organized folders.

## Visualizations Generated
- Univariate:
  - Histogram
  - Distribution plot
  - Box plot
  - Violin plot
  - Density plot
  - Label count plot
  - Label pie chart
- Bivariate (for each feature vs label):
  - Scatter plot
  - Box plot
  - Violin plot
  - Mean bar plot
- Multivariate:
  - Correlation heatmap
  - Pair plot
  - Pair grid
  - Cluster map
  - Parallel coordinates
  - 3D PCA scatter
  - KMeans clusters in PCA space

## Folder Structure
```text
04_Data_Analysis/
├── README.md
├── run_analysis.py
├── notebooks/
│   └── EDA.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── descriptive_analysis.py
│   ├── univariate_analysis.py
│   ├── bivariate_analysis.py
│   ├── multivariate_analysis.py
│   ├── visualization.py
│   └── utils.py
├── outputs/
├── plots/
│   ├── univariate/
│   ├── bivariate/
│   └── multivariate/
└── logs/
```

## How To Execute
From repository root:

```powershell
python 04_Data_Analysis/run_analysis.py
```

## Expected Outputs
Generated automatically after execution:
- `outputs/descriptive_statistics.csv`
- `outputs/dataset_info.txt`
- `outputs/missing_values.csv`
- `outputs/duplicate_report.csv`
- `outputs/validation_report.txt`
- `outputs/dataset_summary.txt`
- `outputs/correlation_matrix.csv`
- `outputs/covariance_matrix.csv`
- All plots inside `plots/univariate`, `plots/bivariate`, and `plots/multivariate`
- Runtime log file in `logs/analysis.log`

## Future Improvements
- Add automated outlier detection reports using robust statistics.
- Add interactive dashboards (Plotly/Altair) for stakeholder reporting.
- Add statistical significance tests across crop groups.
- Add automated EDA regression tests for CI/CD integration.
