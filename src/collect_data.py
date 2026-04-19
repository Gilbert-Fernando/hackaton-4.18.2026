from __future__ import annotations

import csv
import os
import sys
from typing import Dict, Optional

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.hand_tracking import HandTracker
from utils.visualization import draw_landmark_points


KEY_TO_LABEL: Dict[str, str] = {
    **{chr(ord("a") + i): chr(ord("A") + i) for i in range(26)},
    "1": "hello",
    "2": "yes",
    "3": "no",
    "4": "help",
    "5": "stop",
    "6": "thank_you",
    "7": "please",
    "8": "water",
    "9": "bathroom",
    "0": "emergency",
}


def ensure_csv_header(csv_path: str, feature_dim: int) -> None:
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        return

    header = ["label"] + [f"f{i}" for i in range(feature_dim)]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)


def append_row(csv_path: str, label: str, features: np.ndarray) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([label, *features.tolist()])


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0)")

    tracker = HandTracker(max_num_hands=1)

    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "asl_data.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    feature_dim = 63
    ensure_csv_header(csv_path, feature_dim)

    current_label: Optional[str] = None
    saved_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        hand = tracker.detect(frame)

        overlay = frame.copy()

        status = "HAND: OK" if hand is not None else "HAND: NONE"
        if hand is not None:
            draw_landmark_points(overlay, hand.landmarks_xy)

        label_text = current_label if current_label else "(no label selected)"

        cv2.putText(
            overlay,
            f"{status} | label: {label_text} | saved: {saved_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            overlay,
            "Mapped key = set label | SPACE = save | q = quit",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("ASL Data Collector", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key == 32:
            if hand is None or not current_label:
                continue
            append_row(csv_path, current_label, hand.feature_vector)
            saved_count += 1
            continue

        if key != 255:
            ch = chr(key).lower()
            if ch in KEY_TO_LABEL:
                current_label = KEY_TO_LABEL[ch]

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
