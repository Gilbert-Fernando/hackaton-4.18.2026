"""Train a simple classifier from CSV feature vectors.

Input CSV format (from collect_data.py):
- first column: label
- remaining columns: 63 float features

Output:
- model/asl_model.joblib

Default model: Logistic Regression (fast, reliable, gives probabilities)
Alternative: KNN (sometimes better for small datasets)
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def load_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column")

    y = df["label"].astype(str).to_numpy()
    x = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    return x, y


def build_model(model_name: str) -> Pipeline:
    if model_name == "logreg":
        clf = LogisticRegression(
            max_iter=2000,
            n_jobs=None,
        )
    elif model_name == "knn":
        clf = KNeighborsClassifier(n_neighbors=7, weights="distance")
    else:
        raise ValueError("model must be one of: logreg, knn")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.join(os.path.dirname(__file__), "data", "asl_landmarks.csv"),
        help="Path to CSV created by collect_data.py",
    )
    parser.add_argument(
        "--model",
        default="logreg",
        choices=["logreg", "knn"],
        help="Classifier type",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "model", "asl_model.joblib"),
        help="Output model path",
    )
    args = parser.parse_args()

    x, y = load_dataset(args.data)

    if len(y) < 20:
        raise RuntimeError("Not enough samples yet. Collect more data (aim: 50+ per label).")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    model = build_model(args.model)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(model, args.out)
    print(f"Saved model to: {args.out}")


if __name__ == "__main__":
    main()
