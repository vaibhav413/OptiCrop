# Model Building

## Objective

Build and evaluate machine learning models that recommend the most suitable crop from soil and environmental parameters.

## Models Used

- Logistic Regression
- K-Means Clustering

## Workflow

Dataset -> Load dataset -> Split into train and test -> Train K-Means -> Train Logistic Regression -> Evaluate -> Save best model -> Predict crop

## Inputs

- Preferred source: `05_Preprocessing/cleaned_dataset.csv`
- Fallback source: `04_Data_Analysis/Crop_recommendation.csv`
- Target column: `label`
- Features: `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`

## Outputs

- `crop_model.pkl`
- `outputs/accuracy.txt`
- `outputs/classification_report.txt`
- `outputs/confusion_matrix.png`
- `outputs/model_summary.txt`
- `plots/kmeans_clusters.png`

## How To Run

From repository root:

```powershell
python Model_Building/model.py
```

## Predict

```powershell
python Model_Building/predict.py --N 90 --P 42 --K 43 --temperature 20.88 --humidity 82.00 --ph 6.50 --rainfall 202.93
```

## Notes

- The training script prefers the cleaned dataset, but if the cleaned file removes crop labels it falls back to the original dataset so the deployed recommender can still predict the full crop set.
- Logistic Regression is saved as the deployable model because K-Means is unsupervised and is better suited for cluster exploration.