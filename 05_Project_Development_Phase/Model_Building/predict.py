"""Make crop predictions with the trained OptiCrop model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


FEATURE_COLUMNS = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for a single crop prediction."""
    parser = argparse.ArgumentParser(description="Predict the most suitable crop.")
    parser.add_argument("--model-path", type=Path, default=Path(__file__).resolve().parent / "crop_model.pkl")
    parser.add_argument("--N", type=float, required=True)
    parser.add_argument("--P", type=float, required=True)
    parser.add_argument("--K", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--humidity", type=float, required=True)
    parser.add_argument("--ph", type=float, required=True)
    parser.add_argument("--rainfall", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    """Load the trained model and print a crop recommendation."""
    args = parse_args()
    bundle = joblib.load(args.model_path)
    model = bundle["model"]

    sample = pd.DataFrame(
        [
            {
                "N": args.N,
                "P": args.P,
                "K": args.K,
                "temperature": args.temperature,
                "humidity": args.humidity,
                "ph": args.ph,
                "rainfall": args.rainfall,
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    prediction = model.predict(sample)[0]
    print(prediction)

    classifier = model[-1]
    if hasattr(classifier, "predict_proba"):
        probabilities = model.predict_proba(sample)[0]
        classes = model.classes_
        ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)[:5]
        print("Top predictions:")
        for label, probability in ranked:
            print(f"{label}: {probability:.4f}")


if __name__ == "__main__":
    main()