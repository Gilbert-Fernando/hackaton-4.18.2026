from __future__ import annotations

import argparse
import os
import sys
import time
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
    def __init__(self, window_size: int = 10, dominance: float = 0.7) -> None:
        self._items: Deque[Tuple[str, float]] = deque(maxlen=window_size)
        self._dominance = dominance

    def push(self, label: str, confidence: float) -> None:
        self._items.append((label, float(confidence)))

    def get(self) -> Optional[SmoothedPrediction]:
        if len(self._items) < self._items.maxlen:
            return None

        labels = [lbl for (lbl, _) in self._items]
        winner, winner_count = Counter(labels).most_common(1)[0]

        if winner_count / len(self._items) < self._dominance:
            return None

        winner_confs = [c for (lbl, c) in self._items if lbl == winner]
        return SmoothedPrediction(label=winner, confidence=float(np.mean(winner_confs)))


def predict_label(model, encoder, features: np.ndarray) -> Tuple[str, float]:
    x = features.reshape(1, -1)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        return str(encoder.inverse_transform([idx])[0]), float(proba[idx])
    idx = int(model.predict(x)[0])
    return str(encoder.inverse_transform([idx])[0]), 1.0


CONFIDENCE_THRESHOLD = 0.70
STABLE_SECONDS       = 1.2
COOLDOWN_SECONDS     = 1.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--letters-model",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "letters_model.pkl"))
    parser.add_argument("--letters-encoder",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "letters_encoder.pkl"))
    parser.add_argument("--phrases-model",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "phrases_model.pkl"))
    parser.add_argument("--phrases-encoder",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "phrases_encoder.pkl"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--dominance", type=float, default=0.7)
    args = parser.parse_args()

    for path, name in [
        (args.letters_model,   "Letters model"),
        (args.letters_encoder, "Letters encoder"),
        (args.phrases_model,   "Phrases model"),
        (args.phrases_encoder, "Phrases encoder"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    letters_model   = joblib.load(args.letters_model)
    letters_encoder = joblib.load(args.letters_encoder)
    phrases_model   = joblib.load(args.phrases_model)
    phrases_encoder = joblib.load(args.phrases_encoder)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam (index {args.camera})")

    tracker          = HandTracker(max_num_hands=2)
    letters_smoother = PredictionSmoother(window_size=args.window, dominance=args.dominance)
    phrases_smoother = PredictionSmoother(window_size=args.window, dominance=args.dominance)

    pending_label: str               = ""
    label_stable_since: Optional[float] = None
    last_printed_label: str          = ""
    last_print_time: float           = 0.0
    last_mode: str                   = ""

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame   = cv2.flip(frame, 1)
        hands   = tracker.detect_hands(frame)
        overlay = frame.copy()

        if len(hands) == 0:
            pending_label      = ""
            label_stable_since = None
            cv2.putText(overlay, "No hand detected", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            for hnd in hands:
                draw_landmark_points(overlay, hnd.landmarks_xy)

            mode_text = ""
            if len(hands) == 1:
                mode_text = "LETTERS"
                label, conf = predict_label(letters_model, letters_encoder, hands[0].feature_vector)
                letters_smoother.push(label, conf)
                smoothed = letters_smoother.get()
            elif len(hands) == 2:
                mode_text = "PHRASES"
                feats = np.concatenate([hands[0].feature_vector, hands[1].feature_vector])
                label, conf = predict_label(phrases_model, phrases_encoder, feats)
                phrases_smoother.push(label, conf)
                smoothed = phrases_smoother.get()
            else:
                smoothed = None
                label, conf = "", 0.0

            if smoothed is None:
                display_label, display_conf, color = label, conf, (255, 255, 0)
            else:
                display_label, display_conf, color = smoothed.label, smoothed.confidence, (0, 255, 0)

            # --- Stability-gated + cooldown print logic ---
            now = time.monotonic()
            in_cooldown = (display_label == last_printed_label and
                           now - last_print_time < COOLDOWN_SECONDS)

            if display_label and display_conf >= CONFIDENCE_THRESHOLD and not in_cooldown:
                if display_label != pending_label:
                    pending_label      = display_label
                    label_stable_since = now
                elif label_stable_since is not None:
                    elapsed = now - label_stable_since
                    if elapsed >= STABLE_SECONDS:
                        if display_label == "SPACE":
                            sys.stdout.write("\n")
                            last_mode = ""
                        elif mode_text == "LETTERS":
                            sys.stdout.write(display_label)
                            last_mode = "LETTERS"
                        else:
                            # If the last mode was LETTERS, add a leading space
                            if last_mode == "LETTERS":
                                sys.stdout.write(" " + display_label + " ")
                            else:
                                sys.stdout.write(display_label + " ")
                            last_mode = "PHRASES"
                        sys.stdout.flush()
                        last_printed_label = display_label
                        last_print_time    = now
                        label_stable_since = None

                    # Progress bar
                    if label_stable_since is not None:
                        progress     = min(elapsed / STABLE_SECONDS, 1.0)
                        bar_w        = int(overlay.shape[1] * 0.6)
                        bar_x, bar_y = 10, 130
                        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14),
                                      (60, 60, 60), -1)
                        cv2.rectangle(overlay, (bar_x, bar_y),
                                      (bar_x + int(bar_w * progress), bar_y + 14),
                                      (0, 220, 0), -1)
            else:
                if not in_cooldown:
                    pending_label      = ""
                    label_stable_since = None

            cv2.putText(overlay, f"{display_label}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
            cv2.putText(overlay, f"conf: {display_conf:.2f}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.putText(overlay, "q = quit", (10, overlay.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("ASL Real-time", overlay)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()