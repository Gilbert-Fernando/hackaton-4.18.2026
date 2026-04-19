from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
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

    def reset(self) -> None:
        self._items.clear()


def predict_label(model, encoder, features: np.ndarray) -> Tuple[str, float]:
    x = features.reshape(1, -1)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        return str(encoder.inverse_transform([idx])[0]), float(proba[idx])
    idx = int(model.predict(x)[0])
    return str(encoder.inverse_transform([idx])[0]), 1.0


class TTS:
    def __init__(self, rate: int = 0, volume: float = 1.0) -> None:
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._rate = int(rate)
        self._volume = float(volume)

        try:
            import win32com.client  # type: ignore
            import pythoncom  # type: ignore

            self._win32 = win32com.client
            self._pythoncom = pythoncom
        except Exception:
            self._win32 = None
            self._pythoncom = None
            print("Warning: TTS disabled. Install: pip install pywin32")
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def say(self, text: str) -> None:
        if self._win32 is None:
            return
        text = (text or "").strip()
        if not text:
            return
        self._q.put(text)

    def close(self) -> None:
        if self._win32 is None or self._thread is None:
            return
        self._q.put(None)
        self._thread.join(timeout=3.0)

    def _run(self) -> None:
        self._pythoncom.CoInitialize()
        try:
            voice = self._win32.Dispatch("SAPI.SpVoice")
            if self._rate:
                voice.Rate = self._rate
            try:
                voice.Volume = int(max(0, min(100, self._volume * 100)))
            except Exception:
                pass

            while True:
                text = self._q.get()
                if text is None:
                    break
                try:
                    voice.Speak(text)
                except Exception as e:
                    print(f"TTS failed (SAPI Speak): {e}")
        finally:
            self._pythoncom.CoUninitialize()


CONFIDENCE_THRESHOLD = 0.60
STABLE_SECONDS = 1.2
COOLDOWN_SECONDS = 1.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--letters-model",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "letters_model.pkl"),
    )
    parser.add_argument(
        "--letters-encoder",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "letters_encoder.pkl"),
    )
    parser.add_argument(
        "--phrases-model",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "phrases_model.pkl"),
    )
    parser.add_argument(
        "--phrases-encoder",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "phrases_encoder.pkl"),
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--dominance", type=float, default=0.7)
    args = parser.parse_args()

    for path, name in [
        (args.letters_model, "Letters model"),
        (args.letters_encoder, "Letters encoder"),
        (args.phrases_model, "Phrases model"),
        (args.phrases_encoder, "Phrases encoder"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    letters_model = joblib.load(args.letters_model)
    letters_encoder = joblib.load(args.letters_encoder)
    phrases_model = joblib.load(args.phrases_model)
    phrases_encoder = joblib.load(args.phrases_encoder)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam (index {args.camera})")

    tracker = HandTracker(max_num_hands=2)
    letters_smoother = PredictionSmoother(window_size=args.window, dominance=args.dominance)
    phrases_smoother = PredictionSmoother(window_size=args.window, dominance=args.dominance)

    pending_label: str = ""
    label_stable_since: Optional[float] = None
    last_printed_label: str = ""
    last_print_time: float = 0.0

    current_line = ""
    enter_armed = True
    enter_release_since: Optional[float] = None

    tts = TTS()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            hands = tracker.detect_hands(frame)
            overlay = frame.copy()

            if len(hands) == 0:
                pending_label = ""
                label_stable_since = None
                enter_armed = True
                enter_release_since = None
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
                for hnd in hands:
                    draw_landmark_points(overlay, hnd.landmarks_xy)

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
                    mode_text = ""
                    smoothed = None
                    label, conf = "", 0.0

                raw_label_upper = label.strip().upper() if label else ""
                raw_conf = conf

                if smoothed is None:
                    display_label, display_conf, color = label, conf, (255, 255, 0)
                else:
                    display_label, display_conf, color = smoothed.label, smoothed.confidence, (0, 255, 0)

                now = time.monotonic()
                label_upper = display_label.strip().upper() if display_label else ""

                # Re-arm ENTER based on RAW prediction (not smoothed), so the smoother
                # "sticking" on ENTER doesn't prevent future presses.
                if raw_label_upper != "ENTER" or raw_conf < CONFIDENCE_THRESHOLD:
                    if enter_release_since is None:
                        enter_release_since = now
                    if now - enter_release_since >= 0.25:
                        enter_armed = True
                else:
                    enter_release_since = None

                in_cooldown = (
                    label_upper not in ("ENTER", "SPACE")
                    and display_label == last_printed_label
                    and now - last_print_time < COOLDOWN_SECONDS
                )

                if display_label and display_conf >= CONFIDENCE_THRESHOLD and not in_cooldown:
                    if display_label != pending_label:
                        pending_label = display_label
                        label_stable_since = now
                    elif label_stable_since is not None:
                        elapsed = now - label_stable_since
                        if elapsed >= STABLE_SECONDS:
                            if label_upper == "ENTER":
                                if enter_armed:
                                    sys.stdout.write("\n")
                                    sys.stdout.flush()
                                    if current_line.strip():
                                        tts.say(current_line.strip())
                                    current_line = ""
                                    enter_armed = False
                                    enter_release_since = None
                                    letters_smoother.reset()
                                    phrases_smoother.reset()

                            elif label_upper == "SPACE":
                                sys.stdout.write(" ")
                                sys.stdout.flush()
                                current_line += " "

                            else:
                                if mode_text == "PHRASES":
                                    if current_line and not current_line.endswith(" "):
                                        sys.stdout.write(" ")
                                        current_line += " "
                                sys.stdout.write(display_label)
                                sys.stdout.flush()
                                current_line += display_label

                            last_printed_label = display_label
                            last_print_time = now
                            label_stable_since = None
                            pending_label = ""

                        if label_stable_since is not None:
                            progress = min(elapsed / STABLE_SECONDS, 1.0)
                            bar_w = int(overlay.shape[1] * 0.6)
                            bar_x, bar_y = 10, 130
                            cv2.rectangle(
                                overlay,
                                (bar_x, bar_y),
                                (bar_x + bar_w, bar_y + 14),
                                (60, 60, 60),
                                -1,
                            )
                            cv2.rectangle(
                                overlay,
                                (bar_x, bar_y),
                                (bar_x + int(bar_w * progress), bar_y + 14),
                                (0, 220, 0),
                                -1,
                            )
                else:
                    if not in_cooldown:
                        pending_label = ""
                        label_stable_since = None

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

    finally:
        tts.close()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
