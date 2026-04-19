from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import cv2
import joblib
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.hand_tracking import HandTracker
from utils.visualization import draw_landmark_points


@dataclass
class SmoothedPrediction:
    label: str
    confidence: float


class PredictionSmoother:
    def __init__(self, window_size: int = 10, min_count: int = 5) -> None:
        self._items: Deque[Tuple[str, float]] = deque(maxlen=window_size)
        self._min_count = min_count

    def push(self, label: str, confidence: float) -> None:
        self._items.append((label, float(confidence)))

    def get(self) -> Optional[SmoothedPrediction]:
        if len(self._items) < self._min_count:
            return None

        labels = [lbl for (lbl, _conf) in self._items]
        winner, winner_count = Counter(labels).most_common(1)[0]
        if winner_count < self._min_count:
            return None

        winner_confs = [conf for (lbl, conf) in self._items if lbl == winner]
        return SmoothedPrediction(label=winner, confidence=float(np.mean(winner_confs)))


def predict_label(model, encoder, features: np.ndarray) -> Tuple[str, float]:
    x = features.reshape(1, -1)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        label = str(encoder.inverse_transform([idx])[0])
        return label, float(proba[idx])

    idx = int(model.predict(x)[0])
    label = str(encoder.inverse_transform([idx])[0])
    return label, 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "asl_model.h5"),
    )
    parser.add_argument(
        "--label-encoder",
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "models", "label_encoder.pkl"
        ),
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--min-count", type=int, default=5)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}. Run: python src/train_model.py")
    if not os.path.exists(args.label_encoder):
        raise FileNotFoundError(
            f"Label encoder not found: {args.label_encoder}. Run: python src/train_model.py"
        )

    model = joblib.load(args.model)
    encoder = joblib.load(args.label_encoder)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam (index {args.camera})")

    tracker = HandTracker(max_num_hands=1)
    smoother = PredictionSmoother(window_size=args.window, min_count=args.min_count)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        hand = tracker.detect(frame)

        overlay = frame.copy()

        if hand is None:
            cv2.putText(
                overlay,
                "No hand detected",
                (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
        else:
            draw_landmark_points(overlay, hand.landmarks_xy)

            label, conf = predict_label(model, encoder, hand.feature_vector)
            smoother.push(label, conf)

            smoothed = smoother.get()
            if smoothed is None:
                display_label = label
                display_conf = conf
                color = (255, 255, 0)
            else:
                display_label = smoothed.label
                display_conf = smoothed.confidence
                color = (0, 255, 0)

            cv2.putText(
                overlay,
                f"{display_label}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                color,
                4,
            )
            cv2.putText(
                overlay,
                f"conf: {display_conf:.2f}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )

        cv2.putText(
            overlay,
            "q = quit",
            (10, overlay.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("ASL Real-time", overlay)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
