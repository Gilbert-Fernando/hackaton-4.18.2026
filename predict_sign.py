"""Live ASL sign/letter prediction from webcam.

Pipeline:
webcam -> MediaPipe hand landmarks -> feature vector -> classifier -> on-screen overlay

Features:
- draws hand landmarks
- shows predicted label + confidence
- simple smoothing to reduce flicker (majority vote over recent frames)

Usage:
  python predict_sign.py --model model/asl_model.joblib

Exit:
  press q
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import cv2
import joblib
import numpy as np

from hand_detection import HandLandmarkDetector




@dataclass
class SmoothedPrediction:
    label: str
    confidence: float


class PredictionSmoother:
    """Majority vote over last N labels; confidence = mean over chosen label."""

    def __init__(self, window_size: int = 10, min_count: int = 5) -> None:
        self.window_size = window_size
        self.min_count = min_count
        self._items: Deque[Tuple[str, float]] = deque(maxlen=window_size)

    def push(self, label: str, confidence: float) -> None:
        self._items.append((label, float(confidence)))

    def get(self) -> Optional[SmoothedPrediction]:
        if len(self._items) < self.min_count:
            return None

        labels = [lbl for (lbl, _conf) in self._items]
        winner, winner_count = Counter(labels).most_common(1)[0]
        if winner_count < self.min_count:
            return None

        winner_confs = [conf for (lbl, conf) in self._items if lbl == winner]
        return SmoothedPrediction(label=winner, confidence=float(np.mean(winner_confs)))


def predict_with_confidence(model, features: np.ndarray) -> Tuple[str, float]:
    features_2d = features.reshape(1, -1)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_2d)[0]
        idx = int(np.argmax(proba))
        label = str(model.classes_[idx])
        conf = float(proba[idx])
        return label, conf

    label = str(model.predict(features_2d)[0])
    return label, 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.path.join(os.path.dirname(__file__), "model", "asl_model.joblib"),
        help="Path to trained joblib model",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--window", type=int, default=10, help="Smoothing window size")
    parser.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Minimum occurrences in the window to display a label",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Model not found: {args.model}. Train one with: python train_model.py"
        )

    model = joblib.load(args.model)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam (index {args.camera})")

    detector = HandLandmarkDetector(max_num_hands=1)
    smoother = PredictionSmoother(window_size=args.window, min_count=args.min_count)
    previous_display_label = ""

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        hand = detector.detect(frame)

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
            cv2.imshow("ASL Live Prediction", overlay)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            continue

        detector.draw(overlay, hand)

        label, conf = predict_with_confidence(model, hand.feature_vector)
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

        if previous_display_label and display_label != previous_display_label and display_conf > 0.80:
            print(f"New word: {display_label} (confidence: {display_conf:.2f})")
        previous_display_label = display_label

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

        cv2.imshow("ASL Live Prediction", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
