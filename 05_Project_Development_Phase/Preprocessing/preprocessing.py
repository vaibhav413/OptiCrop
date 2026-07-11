"""
OptiCrop - Data Preprocessing Module (Epic 3)
This script prepares the Crop Recommendation dataset for machine learning.
It performs data cleaning, missing value handling, duplicate removal,
outlier treatment, feature engineering, and train-test splitting.
"""
import os
import sys
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# Step 1: Ignore warnings and set plot style
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def resolve_paths() -> tuple[Path, Path, Path]:
    """
    Resolves the paths for the dataset, outputs, and plots.
    Ensures that output and plot directories exist.
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Look for dataset in 03_Data_Analysis first, then fallback to current directory
    dataset_path = project_root / "03_Data_Analysis" / "Crop_recommendation.csv"
    if not dataset_path.exists():
        dataset_path = Path("Crop_recommendation.csv")

    outputs_dir = script_dir / "outputs"
    plots_dir = script_dir / "plots"

    # Create directories if they do not exist
    outputs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    return dataset_path, outputs_dir, plots_dir


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """
    Step 2: Read Dataset
    Loads the CSV dataset and displays basic metadata.
    """
    print("========================================= ")
    print("STEP 2: Reading Dataset                   ")
    print("========================================= ")
    try:
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at: {dataset_path.absolute()}"
            )

        df = pd.read_csv(dataset_path)
        print(f"Dataset successfully loaded from: {dataset_path}\n")

        # Display basic information
        print("--- First 5 Rows (head) ---")
        print(df.head())
        print("\n--- Dataset Shape ---")
        print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        print("\n--- Columns ---")
        print(df.columns.tolist())
        print("\n--- Data Types ---")
        print(df.dtypes)
        print("-" * 50)

        return df

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(
            "Please ensure Crop_recommendation.csv is in the '04_Data_Analysis' folder or the working directory."
        )
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading the dataset: {e}")
        sys.exit(1)


def check_missing_values(df: pd.DataFrame, plots_dir: Path) -> tuple[pd.Series, pd.Series]:
    """
    Step 3: Check Missing Values
    Counts and displays missing values and percentages.
    Generates a heatmap if missing values exist.
    """
    print("\n========================================= ")
    print("STEP 3: Checking Missing Values           ")
    print("========================================= ")

    missing_counts = df.isnull().sum()
    missing_percentage = (missing_counts / len(df)) * 100

    missing_df = pd.DataFrame({
        "Missing Count": missing_counts,
        "Percentage (%)": missing_percentage
    })

    print("--- Missing Values Summary ---")
    print(missing_df)

    total_missing = missing_counts.sum()
    plot_path = plots_dir / "missing_values.png"
    
    if total_missing > 0:
        print(f"\nDetected {total_missing} missing values in total. Generating heatmap...")
        # Save missing values heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
        plt.title("Heatmap of Missing Values", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Heatmap saved to: {plot_path}")
    else:
        print("\nNo missing values found.")
        # Save a bar plot showing 0 missing values for all columns
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing_counts.index, y=missing_counts.values, color="#2a9d8f")
        plt.title("Missing Values Count by Feature (All Zero)", fontsize=14, fontweight="bold")
        plt.xlabel("Features", fontsize=12)
        plt.ylabel("Missing Count", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 10)  # set a small limit since they are all 0
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Missing values status plot saved to: {plot_path}")

    print("-" * 50)
    return missing_counts, missing_percentage


def handle_missing_values(df: pd.DataFrame, missing_counts: pd.Series) -> pd.DataFrame:
    """
    Step 4: Handle Missing Values
    Imputes missing values in numerical columns using the median.
    """
    print("\n========================================= ")
    print("STEP 4: Handling Missing Values           ")
    print("========================================= ")

    df_imputed = df.copy()
    total_missing = missing_counts.sum()

    if total_missing > 0:
        numerical_cols = df_imputed.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if missing_counts[col] > 0:
                median_val = df_imputed[col].median()
                df_imputed[col].fillna(median_val, inplace=True)
                print(f"Imputed missing values in '{col}' with median: {median_val}")
        print("Missing values imputation completed.")
    else:
        print("No missing values to handle. Continuing without modification.")

    print("-" * 50)
    return df_imputed


def check_and_remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Step 5: Check and Remove Duplicate Rows
    Identifies and removes duplicate records.
    """
    print("\n========================================= ")
    print("STEP 5: Checking Duplicate Rows          ")
    print("========================================= ")

    duplicate_mask = df.duplicated()
    total_duplicates = duplicate_mask.sum()

    print(f"Total duplicate rows: {total_duplicates}")

    if total_duplicates > 0:
        df_cleaned = df.drop_duplicates().reset_index(drop=True)
        print(f"Removed {total_duplicates} duplicate rows. New shape: {df_cleaned.shape}")
    else:
        df_cleaned = df.copy()
        print("No duplicate rows found.")

    print("-" * 50)
    return df_cleaned, total_duplicates


def detect_and_treat_outliers(
    df: pd.DataFrame, plots_dir: Path
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Step 6 & 7: Detect and Treat Outliers using the IQR Method
    Generates boxplots before and after treatment and saves them to outliers.png.
    Removes outliers using the IQR method and prints the counts.
    """
    print("\n========================================= ")
    print("STEPS 6 & 7: Detecting and Treating Outliers")
    print("========================================= ")

    numeric_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    outliers_indices = set()
    outlier_counts = {}

    # Calculate outliers for each numerical column
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Identify indices of outliers
        col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outlier_counts[col] = len(col_outliers)
        outliers_indices.update(col_outliers)

    print("Number of outliers detected and removed for each feature:")
    for col, count in outlier_counts.items():
        print(f"  - {col}: {count}")

    total_outliers_removed = len(outliers_indices)
    print(f"Total unique rows containing outliers: {total_outliers_removed}")

    # Create the cleaned dataframe by dropping outlier indices
    df_no_outliers = df.drop(index=outliers_indices).reset_index(drop=True)

    # Generate Boxplots (Before vs After Treatment)
    print("\nGenerating boxplot comparison (Before vs After Treatment)...")
    
    # Melt dataframes for easy plotting with Seaborn
    df_before_melt = df[numeric_cols].melt(var_name="Feature", value_name="Value")
    df_after_melt = df_no_outliers[numeric_cols].melt(var_name="Feature", value_name="Value")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    
    # Before treatment plot
    sns.boxplot(ax=axes[0], data=df_before_melt, x="Feature", y="Value", palette="Set2")
    axes[0].set_title("Outlier Visualization - Before Treatment", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Value")
    axes[0].set_xlabel("")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # After treatment plot
    sns.boxplot(ax=axes[1], data=df_after_melt, x="Feature", y="Value", palette="Set2")
    axes[1].set_title("Outlier Visualization - After Treatment (IQR Method)", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Value")
    axes[1].set_xlabel("Features")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    outliers_plot_path = plots_dir / "outliers.png"
    plt.savefig(outliers_plot_path, dpi=300)
    plt.close()
    print(f"Outliers comparison plot saved to: {outliers_plot_path}")

    print("-" * 50)
    return df_no_outliers, outlier_counts


def extract_seasonal_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 8: Extract Seasonal Crop Information
    Creates a 'Season' column based on the 'temperature' column:
      - Temperature < 20: Winter
      - Temperature between 20 and 30 (inclusive): Monsoon
      - Temperature > 30: Summer
    """
    print("\n========================================= ")
    print("STEP 8: Extracting Seasonal Crop Info     ")
    print("========================================= ")

    df_seasonal = df.copy()

    def get_season(temp: float) -> str:
        if temp < 20.0:
            return "Winter"
        elif 20.0 <= temp <= 30.0:
            return "Monsoon"
        else:  # temp > 30.0
            return "Summer"

    df_seasonal["Season"] = df_seasonal["temperature"].apply(get_season)
    print("Successfully added 'Season' column to the dataset.")
    print("Season distribution:")
    print(df_seasonal["Season"].value_counts())
    print("-" * 50)
    return df_seasonal


def split_and_save(df: pd.DataFrame, outputs_dir: Path) -> tuple[int, int]:
    """
    Step 9 & 10: Prepare Features and Train-Test Split
    Separates Features (X) and Target (y), splits the dataset, and saves train/test files.
    """
    print("\n========================================= ")
    print("STEPS 9 & 10: Features Separation & Split ")
    print("========================================= ")

    # Step 9: Separate Features (X) and Target (y)
    # Features as specified: N, P, K, temperature, humidity, ph, rainfall
    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    target_col = "label"

    X = df[feature_cols]
    y = df[target_col]

    print(f"Features (X) columns: {X.columns.tolist()}")
    print(f"Target (y) column: {y.name}")

    # Step 10: Train-Test Split (80% Train, 20% Test)
    # We split the dataframe itself to preserve all columns (including Season and label)
    # but we can also split X and y. To ensure we save complete files with labels,
    # we split the entire cleaned dataframe.
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[target_col]
    )

    train_path = outputs_dir / "train.csv"
    test_path = outputs_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train dataset saved to: {train_path} (Shape: {train_df.shape})")
    print(f"Test dataset saved to: {test_path} (Shape: {test_df.shape})")
    print("-" * 50)

    return train_df.shape[0], test_df.shape[0]


def save_cleaned_dataset(df: pd.DataFrame, outputs_dir: Path) -> Path:
    """
    Step 11: Save Clean Dataset
    Saves the fully processed dataset to outputs/cleaned_dataset.csv.
    """
    print("\n========================================= ")
    print("STEP 11: Saving Cleaned Dataset           ")
    print("========================================= ")

    clean_path = outputs_dir / "cleaned_dataset.csv"
    df.to_csv(clean_path, index=False)
    print(f"Cleaned dataset saved to: {clean_path} (Shape: {df.shape})")
    print("-" * 50)
    return clean_path


def generate_report(
    original_shape: tuple[int, int],
    cleaned_shape: tuple[int, int],
    total_missing: int,
    duplicate_count: int,
    outlier_counts: dict[str, int],
    train_size: int,
    test_size: int,
    outputs_dir: Path,
) -> None:
    """
    Step 12: Generate Report
    Creates a preprocessing_report.txt summary file.
    """
    print("\n========================================= ")
    print("STEP 12: Generating Preprocessing Report  ")
    print("========================================= ")

    report_path = outputs_dir / "preprocessing_report.txt"
    total_outliers_removed = original_shape[0] - duplicate_count - cleaned_shape[0]

    report_content = f"""OptiCrop Data Preprocessing Report
======================================
1. Summary of Records:
   - Total Original Records: {original_shape[0]}
   - Total Cleaned Records : {cleaned_shape[0]}
   - Total Features        : {cleaned_shape[1]}

2. Missing Values:
   - Total Missing Values Identified: {total_missing}
   - Action Taken: Imputed with column median (for numerical columns if any)

3. Duplicate Rows:
   - Total Duplicate Rows Found: {duplicate_count}
   - Action Taken: Removed duplicate records

4. Outliers Removed (IQR Method):
   - Outliers detected per feature:
"""
    for col, count in outlier_counts.items():
        report_content += f"     * {col}: {count}\n"
    
    report_content += f"   - Total Unique Rows Removed as Outliers: {total_outliers_removed}\n"

    report_content += f"""
5. New Features Added:
   - Feature Name: Season
   - Details: Created from 'temperature' column:
     * Temperature < 20: Winter
     * Temperature between 20 and 30: Monsoon
     * Temperature > 30: Summer

6. Dataset Splits:
   - Training Set Size (80%): {train_size} records
   - Testing Set Size (20%) : {test_size} records
   - Split Parameters: test_size = 0.2, random_state = 42
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Preprocessing report written to: {report_path}")
    print("\n--- Report Content Preview ---")
    print(report_content)
    print("=" * 50)


def main() -> None:
    """
    Main execution function.
    Coordinates all preprocessing steps.
    """
    print("Starting OptiCrop Preprocessing Pipeline...\n")

    # Step 0: Resolve and create necessary folders
    dataset_path, outputs_dir, plots_dir = resolve_paths()

    # Step 2: Load Dataset
    df = load_dataset(dataset_path)
    original_shape = df.shape

    # Step 3: Check Missing Values
    missing_counts, _ = check_missing_values(df, plots_dir)
    total_missing = missing_counts.sum()

    # Step 4: Handle Missing Values
    df_imputed = handle_missing_values(df, missing_counts)

    # Step 5: Check Duplicate Rows
    df_no_duplicates, duplicate_count = check_and_remove_duplicates(df_imputed)

    # Steps 6 & 7: Detect & Treat Outliers
    df_no_outliers, outlier_counts = detect_and_treat_outliers(df_no_duplicates, plots_dir)

    # Step 8: Extract Seasonal Crop Information
    df_with_season = extract_seasonal_info(df_no_outliers)

    # Steps 9 & 10: Prepare Features & Split Dataset
    train_size, test_size = split_and_save(df_with_season, outputs_dir)

    # Step 11: Save Cleaned Dataset
    save_cleaned_dataset(df_with_season, outputs_dir)

    # Step 12: Generate Preprocessing Report
    generate_report(
        original_shape=original_shape,
        cleaned_shape=df_with_season.shape,
        total_missing=total_missing,
        duplicate_count=duplicate_count,
        outlier_counts=outlier_counts,
        train_size=train_size,
        test_size=test_size,
        outputs_dir=outputs_dir,
    )

    print("\nOptiCrop Preprocessing Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()
