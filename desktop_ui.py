from __future__ import annotations

import os
import sys
import time
from typing import Optional

import cv2
import joblib
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.hand_tracking import HandTracker
from src.predict_realtime import (
    COOLDOWN_SECONDS,
    CONFIDENCE_THRESHOLD,
    STABLE_SECONDS,
    PredictionSmoother,
    TTS,
    predict_from_hands,
)
from utils.visualization import draw_landmark_points


def _wrap_lines(text: str, max_chars: int) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    for raw in text.split("\n"):
        s = raw
        while len(s) > max_chars:
            cut = s.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            out.append(s[:cut].rstrip())
            s = s[cut:].lstrip()
        out.append(s)
    return out


def _rounded_rect(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], radius: int, color: tuple[int, int, int], thickness: int) -> None:
    x1, y1 = p1
    x2, y2 = p2
    r = max(0, min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2))

    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
        return

    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def _glass_panel(canvas: np.ndarray, x1: int, y1: int, x2: int, y2: int, *, alpha: float = 0.82) -> None:
    overlay = canvas.copy()
    _rounded_rect(overlay, (x1, y1), (x2, y2), 16, (22, 24, 30), -1)
    canvas[:] = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)
    _rounded_rect(canvas, (x1, y1), (x2, y2), 16, (55, 60, 70), 1)


def _badge(canvas: np.ndarray, x: int, y: int, text: str, *, bg: tuple[int, int, int], fg: tuple[int, int, int]) -> int:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    w = tw + 18
    h = th + 16
    _rounded_rect(canvas, (x, y), (x + w, y + h), 12, bg, -1)
    cv2.putText(canvas, text, (x + 9, y + h - 6), font, scale, fg, 1, cv2.LINE_AA)
    return w


def _draw_scanlines(img: np.ndarray, *, strength: float = 0.12, step: int = 3) -> None:
    h, w = img.shape[:2]
    overlay = img.copy()
    for y in range(0, h, step):
        cv2.line(overlay, (0, y), (w, y), (0, 0, 0), 1)
    img[:] = cv2.addWeighted(overlay, strength, img, 1 - strength, 0)


class _Particles:
    def __init__(self, width: int, height: int, count: int = 90) -> None:
        rng = np.random.default_rng(7)
        self.w = width
        self.h = height
        self.x = rng.random(count) * width
        self.y = rng.random(count) * height
        self.vx = (rng.random(count) - 0.5) * 0.45
        self.vy = (rng.random(count) - 0.5) * 0.35
        self.r = 1.6 + rng.random(count) * 1.8

    def step(self) -> None:
        self.x += self.vx
        self.y += self.vy
        self.x = np.where(self.x < 0, self.x + self.w, self.x)
        self.x = np.where(self.x > self.w, self.x - self.w, self.x)
        self.y = np.where(self.y < 0, self.y + self.h, self.y)
        self.y = np.where(self.y > self.h, self.y - self.h, self.y)

    def draw(self, img: np.ndarray) -> None:
        overlay = img.copy()
        n = len(self.x)
        for i in range(n):
            cv2.circle(
                overlay,
                (int(self.x[i]), int(self.y[i])),
                int(self.r[i]),
                (180, 110, 70),
                -1,
                cv2.LINE_AA,
            )
        img[:] = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)

        overlay2 = img.copy()
        max_d2 = 150.0 * 150.0
        for i in range(n):
            xi, yi = float(self.x[i]), float(self.y[i])
            for j in range(i + 1, min(n, i + 16)):
                dx = xi - float(self.x[j])
                dy = yi - float(self.y[j])
                d2 = dx * dx + dy * dy
                if d2 > max_d2:
                    continue
                alpha = max(0.0, 1.0 - (d2 / max_d2)) * 0.16
                if alpha <= 0.01:
                    continue
                cv2.line(
                    overlay2,
                    (int(xi), int(yi)),
                    (int(self.x[j]), int(self.y[j])),
                    (160, 120, 90),
                    1,
                    cv2.LINE_AA,
                )
                img[:] = cv2.addWeighted(overlay2, alpha, img, 1 - alpha, 0)


def main() -> None:
    models_dir = os.path.join(PROJECT_ROOT, "models")

    letters_model = joblib.load(os.path.join(models_dir, "letters_model.pkl"))
    letters_encoder = joblib.load(os.path.join(models_dir, "letters_encoder.pkl"))
    phrases_model = joblib.load(os.path.join(models_dir, "phrases_model.pkl"))
    phrases_encoder = joblib.load(os.path.join(models_dir, "phrases_encoder.pkl"))

    camera_index = int(os.environ.get("ASL_CAMERA", "0"))
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam (index {camera_index})")

    tracker = HandTracker(max_num_hands=2)
    letters_smoother = PredictionSmoother(window_size=10, dominance=0.7)
    phrases_smoother = PredictionSmoother(window_size=10, dominance=0.7)

    pending_label: str = ""
    label_stable_since: Optional[float] = None
    last_committed_label: str = ""
    last_commit_time: float = 0.0

    current_line = ""
    enter_armed = True
    enter_release_since: Optional[float] = None

    commit_progress: Optional[float] = None
    commit_left: Optional[float] = None

    tts = TTS()
    tts_enabled = False

    sidebar_particles = _Particles(width=460, height=720, count=90)

    last_frame_t = time.monotonic()
    fps = 0.0
    ema_alpha = 0.15

    window_name = "ASL Desktop"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            now_t = time.monotonic()
            inst_fps = 1.0 / max(1e-6, now_t - last_frame_t)
            last_frame_t = now_t
            fps = (1 - ema_alpha) * fps + ema_alpha * inst_fps

            frame = cv2.flip(frame, 1)
            hands = tracker.detect_hands(frame)
            overlay = frame.copy()

            target_h = 540
            if overlay.shape[0] != target_h:
                target_w = int(overlay.shape[1] * (target_h / overlay.shape[0]))
                overlay = cv2.resize(overlay, (target_w, target_h), interpolation=cv2.INTER_AREA)

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

                payload = predict_from_hands(
                    hands,
                    letters_model=letters_model,
                    letters_encoder=letters_encoder,
                    phrases_model=phrases_model,
                    phrases_encoder=phrases_encoder,
                    letters_smoother=letters_smoother,
                    phrases_smoother=phrases_smoother,
                )

                mode_text = payload["mode"]
                raw_label = str(payload["raw_label"])
                raw_conf = float(payload["raw_confidence"])
                raw_label_upper = raw_label.strip().upper() if raw_label else ""

                display_label = str(payload["label"])
                display_conf = float(payload["confidence"])
                color = (0, 255, 0) if payload["is_smoothed"] else (255, 255, 0)

                label_upper = display_label.strip().upper() if display_label else ""

                if raw_label_upper != "ENTER" or raw_conf < CONFIDENCE_THRESHOLD:
                    if enter_release_since is None:
                        enter_release_since = now_t
                    if now_t - enter_release_since >= 0.25:
                        enter_armed = True
                else:
                    enter_release_since = None

                in_cooldown = (
                    label_upper not in ("ENTER", "SPACE")
                    and display_label == last_committed_label
                    and now_t - last_commit_time < COOLDOWN_SECONDS
                )

                commit_progress = None
                commit_left = None

                if display_label and display_conf >= CONFIDENCE_THRESHOLD and not in_cooldown:
                    if display_label != pending_label:
                        pending_label = display_label
                        label_stable_since = now_t
                    elif label_stable_since is not None:
                        elapsed = now_t - label_stable_since
                        commit_progress = min(elapsed / STABLE_SECONDS, 1.0)
                        commit_left = max(0.0, STABLE_SECONDS - elapsed)
                        if elapsed >= STABLE_SECONDS:
                            if label_upper == "ENTER":
                                if enter_armed:
                                    if current_line.strip() and tts_enabled:
                                        tts.say(current_line.strip())
                                    current_line += "\n"
                                    enter_armed = False
                                    enter_release_since = None
                                    letters_smoother.reset()
                                    phrases_smoother.reset()
                            elif label_upper == "SPACE":
                                current_line += " "
                            else:
                                if mode_text == "PHRASES":
                                    if current_line and not current_line.endswith(" "):
                                        current_line += " "
                                current_line += display_label

                            last_committed_label = display_label
                            last_commit_time = now_t
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
                        commit_progress = None
                        commit_left = None

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

            panel_w = 520
            h, w = overlay.shape[:2]
            canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
            canvas[:, :w] = overlay

            sidebar = np.zeros((h, panel_w, 3), dtype=np.uint8)
            for i in range(h):
                t = i / max(1, h - 1)
                sidebar[i, :, :] = (
                    int(14 + 8 * t),
                    int(14 + 10 * t),
                    int(18 + 16 * t),
                )

            sidebar_particles.w = panel_w
            sidebar_particles.h = h
            sidebar_particles.step()
            sidebar_particles.draw(sidebar)
            canvas[:, w:] = sidebar

            px1, py1 = w + 16, 16
            px2, py2 = w + panel_w - 16, h - 16
            _glass_panel(canvas, px1, py1, px2, py2)

            x0 = px1 + 18
            y = py1 + 36

            cv2.putText(
                canvas,
                "ASL Desktop",
                (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )

            y += 22
            cv2.putText(
                canvas,
                "Camera + MediaPipe + scikit-learn",
                (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (175, 175, 175),
                1,
                cv2.LINE_AA,
            )

            y += 18
            bx = x0
            bx += _badge(canvas, bx, y, f"FPS {fps:.1f}", bg=(45, 45, 45), fg=(235, 235, 235)) + 8
            bx += _badge(canvas, bx, y, "TTS ON" if tts_enabled else "TTS OFF", bg=(20, 120, 60) if tts_enabled else (80, 55, 25), fg=(245, 245, 245)) + 8
            _badge(canvas, bx, y, "q quit", bg=(45, 45, 45), fg=(235, 235, 235))

            y += 54
            cv2.putText(canvas, "Transcript", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)

            hist_y1 = y + 14
            hist_x1 = x0
            hist_x2 = px2 - 18
            hist_y2 = hist_y1 + 210
            _rounded_rect(canvas, (hist_x1, hist_y1), (hist_x2, hist_y2), 14, (8, 8, 10), -1)
            _rounded_rect(canvas, (hist_x1, hist_y1), (hist_x2, hist_y2), 14, (55, 60, 70), 1)

            transcript_lines = _wrap_lines(current_line.strip("\n"), max_chars=48)
            yy = hist_y1 + 30
            for line in transcript_lines[-11:]:
                cv2.putText(
                    canvas,
                    line if line else " ",
                    (hist_x1 + 12, yy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
                yy += 22

            y = hist_y2 + 34
            cv2.putText(canvas, "Phrase Builder", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
            y += 18
            cv2.putText(canvas, "Appends when prediction stays stable", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (175, 175, 175), 1, cv2.LINE_AA)

            y += 10
            box_x1 = x0
            box_y1 = y + 12
            box_x2 = px2 - 18
            box_y2 = py2 - 120
            _rounded_rect(canvas, (box_x1, box_y1), (box_x2, box_y2), 14, (8, 8, 10), -1)
            _rounded_rect(canvas, (box_x1, box_y1), (box_x2, box_y2), 14, (55, 60, 70), 1)

            if not current_line.strip():
                cv2.putText(
                    canvas,
                    "Waiting for commits…",
                    (box_x1 + 12, box_y1 + 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (145, 145, 145),
                    1,
                    cv2.LINE_AA,
                )

            bar_y = box_y2 + 16
            cv2.putText(canvas, "Commit in", (box_x1, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (175, 175, 175), 1, cv2.LINE_AA)
            if commit_left is None:
                cv2.putText(canvas, "-", (box_x2 - 14, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (175, 175, 175), 1, cv2.LINE_AA)
            else:
                cv2.putText(canvas, f"{commit_left:.1f}s", (box_x2 - 60, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (175, 175, 175), 1, cv2.LINE_AA)

            bar_x1 = box_x1
            bar_x2 = box_x2
            bar_y1 = bar_y + 10
            bar_y2 = bar_y1 + 12
            _rounded_rect(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), 8, (6, 6, 8), -1)
            _rounded_rect(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), 8, (55, 60, 70), 1)
            if commit_progress is not None:
                _rounded_rect(
                    canvas,
                    (bar_x1 + 1, bar_y1 + 1),
                    (bar_x1 + 1 + int((bar_x2 - bar_x1 - 2) * commit_progress), bar_y2 - 1),
                    7,
                    (40, 200, 120),
                    -1,
                )

            _draw_scanlines(canvas[:, :w], strength=0.10, step=3)

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("t"):
                tts_enabled = not tts_enabled
            if key == ord("c"):
                current_line = ""
            if key == ord("r"):
                current_line = ""
                pending_label = ""
                label_stable_since = None
                last_committed_label = ""
                last_commit_time = 0.0
                enter_armed = True
                enter_release_since = None
                commit_progress = None
                commit_left = None
            if key == ord("b"):
                if current_line:
                    current_line = current_line[:-1]

    finally:
        tts.close()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
