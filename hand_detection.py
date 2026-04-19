"""Hand detection + landmark/feature extraction.

This module is the core of the pipeline:
webcam frame -> MediaPipe Hands -> 21 landmarks -> normalized feature vector.

Hackathon goals:
- reliable single-hand tracking
- simple, consistent feature vector
- easy to reuse in data collection + live prediction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class HandResult:
    landmarks_xy: List[Tuple[int, int]]
    feature_vector: np.ndarray  # shape: (63,)


class HandLandmarkDetector:
    """Wrapper around MediaPipe Hands with a small, stable API."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "mediapipe is not installed. Install with: pip install mediapipe"
            ) from exc

        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._drawing = mp.solutions.drawing_utils
        self._drawing_styles = mp.solutions.drawing_styles

    def close(self) -> None:
        self._hands.close()

    def detect(self, frame_bgr: np.ndarray) -> Optional[HandResult]:
        """Return first hand's landmarks + normalized feature vector, or None."""

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        h, w = frame_bgr.shape[:2]

        # Convert normalized landmarks to pixel coords (for drawing/UI)
        landmarks_xy: List[Tuple[int, int]] = []
        for lm in hand_landmarks.landmark:
            landmarks_xy.append((int(lm.x * w), int(lm.y * h)))

        feature_vector = extract_features_wrist_relative(hand_landmarks)
        return HandResult(landmarks_xy=landmarks_xy, feature_vector=feature_vector)

    def draw(self, frame_bgr: np.ndarray, results: "HandResult") -> None:
        """Draw landmark dots (lightweight)."""

        for (x, y) in results.landmarks_xy:
            cv2.circle(frame_bgr, (x, y), 3, (0, 255, 0), -1)


def extract_features_wrist_relative(hand_landmarks) -> np.ndarray:
    """Flatten (x,y,z) landmarks normalized relative to wrist + scaled.

    Normalization used (simple and effective for hackathon MVP):
    - translate so wrist (landmark 0) becomes the origin
    - scale by distance between wrist (0) and middle finger MCP (9)

    Output: np.ndarray of shape (63,), dtype float32.
    """

    lms = hand_landmarks.landmark

    wrist = np.array([lms[0].x, lms[0].y, lms[0].z], dtype=np.float32)
    middle_mcp = np.array([lms[9].x, lms[9].y, lms[9].z], dtype=np.float32)

    scale = float(np.linalg.norm(middle_mcp - wrist))
    if scale < 1e-6:
        scale = 1.0

    features: List[float] = []
    for lm in lms:
        v = (np.array([lm.x, lm.y, lm.z], dtype=np.float32) - wrist) / scale
        features.extend([float(v[0]), float(v[1]), float(v[2])])

    return np.array(features, dtype=np.float32)
