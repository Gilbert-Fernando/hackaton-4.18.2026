"""Collect labeled hand landmark feature vectors into a CSV.

Controls (default):
- Press a mapped key to set the current label
- Press SPACE to save one sample with that label
- Press q to quit

Hackathon note: keyboard-to-label mapping is the fastest way to collect data.
Edit KEY_TO_LABEL for your label set.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Dict, Optional

import cv2
import numpy as np

from hand_detection import HandLandmarkDetector


# Edit this mapping for your hackathon label set.
# Letters are easy: map 'a'->'A', etc.
# For phrases, map digits or other unused keys.
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
        writer = csv.writer(f)
        writer.writerow(header)


def append_row(csv_path: str, label: str, features: np.ndarray) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([label, *features.tolist()])


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0)")

    detector = HandLandmarkDetector(max_num_hands=1)

    out_path = os.environ.get(
        "ASL_DATA_CSV",
        os.path.join(os.path.dirname(__file__), "data", "asl_landmarks.csv"),
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    feature_dim = 63
    ensure_csv_header(out_path, feature_dim)

    current_label: Optional[str] = None
    saved_count = 0

    print("Data collection running")
    print(f"Saving to: {out_path}")
    print("Press a key to select a label, SPACE to save a sample, q to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        hand = detector.detect(frame)

        overlay = frame.copy()

        if hand is not None:
            detector.draw(overlay, hand)
            status = "HAND: OK"
        else:
            status = "HAND: NONE"

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
            "Keys: mapped label key = set label | SPACE = save | q = quit",
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

        if key == 32:  # SPACE
            if hand is None:
                continue
            if not current_label:
                continue
            append_row(out_path, current_label, hand.feature_vector)
            saved_count += 1
            continue

        if key != 255:
            ch = chr(key).lower()
            if ch in KEY_TO_LABEL:
                current_label = KEY_TO_LABEL[ch]

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

    print(f"Done. Saved {saved_count} samples.")


if __name__ == "__main__":
    main()
