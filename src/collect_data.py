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
    "!": "UTRGV",
    "]": "SPACE",
    "(": "ENTER",
    ")": "is",
    "@": "the",

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

    tracker = HandTracker(max_num_hands=2)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)

    letters_csv = os.path.join(data_dir, "asl_letters.csv")
    phrases_csv = os.path.join(data_dir, "asl_phrases.csv")

    ensure_csv_header(letters_csv, 63)
    ensure_csv_header(phrases_csv, 126)

    current_label: Optional[str] = None
    saved_letters = 0
    saved_phrases = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        hands = tracker.detect_hands(frame)

        overlay = frame.copy()

        status = f"HANDS: {len(hands)}"
        if len(hands) > 0:
            for hnd in hands:
                draw_landmark_points(overlay, hnd.landmarks_xy)

        label_text = current_label if current_label else "(no label selected)"

        cv2.putText(
            overlay,
            f"{status} | label: {label_text} | letters: {saved_letters} | phrases: {saved_phrases}",
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
            if not current_label:
                continue

            is_letter = len(current_label) == 1 and current_label.isalpha()

            if is_letter:
                if len(hands) != 1:
                    continue
                append_row(letters_csv, current_label, hands[0].feature_vector)
                saved_letters += 1
            else:
                if len(hands) != 2:
                    continue
                features_2hand = np.concatenate([hands[0].feature_vector, hands[1].feature_vector])
                append_row(phrases_csv, current_label, features_2hand)
                saved_phrases += 1

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
