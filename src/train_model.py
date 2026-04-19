from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def main() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "asl_data.csv"),
    )
    parser.add_argument(
        "--model-out",
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "models", "asl_model.h5"
        ),
    )
    parser.add_argument(
        "--label-encoder-out",
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "models", "label_encoder.pkl"
        ),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column")

    y_raw = df["label"].astype(str).to_numpy()
    x = df.drop(columns=["label"]).to_numpy(dtype=np.float32)

    if len(y_raw) < 20:
        raise RuntimeError("Not enough samples yet. Collect more data (aim: 50+ per label).")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(model, args.model_out)
    joblib.dump(encoder, args.label_encoder_out)

    print(f"Saved model to: {args.model_out}")
    print(f"Saved label encoder to: {args.label_encoder_out}")


if __name__ == "__main__":
    main()
