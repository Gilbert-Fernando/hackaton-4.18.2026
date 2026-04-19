from __future__ import annotations

from dataclasses import dataclass
import os
import sys
import time
import urllib.request
from typing import List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.preprocess import landmarks_to_feature_vector


@dataclass
class HandResult:
    landmarks_xy: List[Tuple[int, int]]
    feature_vector: np.ndarray


class HandTracker:
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
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
        self._impl = "solutions" if hasattr(mp, "solutions") else "tasks"

        self._hands = None
        self._landmarker = None

        if self._impl == "solutions":
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            model_path = os.environ.get(
                "MP_HAND_LANDMARKER_PATH",
                os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task"),
            )
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            if not os.path.exists(model_path):
                self._download_hand_landmarker(model_path)

            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
            from mediapipe.tasks.python.vision import RunningMode

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.VIDEO,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._landmarker = HandLandmarker.create_from_options(options)

    def _download_hand_landmarker(self, out_path: str) -> None:
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, out_path)

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
        if self._landmarker is not None:
            self._landmarker.close()

    def detect_hands(self, frame_bgr: np.ndarray) -> list[HandResult]:
        if self._impl == "solutions":
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = self._hands.process(frame_rgb)
            frame_rgb.flags.writeable = True

            if not results.multi_hand_landmarks:
                return []
            
            hands_out: list[HandResult] = []
            h, w = frame_bgr.shape[:2]

            for hand_landmarks in results.multi_hand_landmarks[:2]:
                landmarks_xy: list[tuple[int, int]] = []
                for lm in hand_landmarks.landmark:
                    landmarks_xy.append((int(lm.x * w),int(lm.y * h)))

                features = landmarks_to_feature_vector(hand_landmarks)
                hands_out.append(HandResult(landmarks_xy=landmarks_xy, feature_vector=features))

            return hands_out
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(time.time()* 1000)
        result = self._landmarker.detect_for_video(image, timestamp_ms)

        if not result.hand_landmarks:
            return []
        hands_out: list[HandResult] = []
        h, w = frame_bgr.shape[:2]
        
        for lms in result.hand_landmarks[:2]:
            landmarks_xy = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

            class _Tmp:
                landmark = lms

            features = landmarks_to_feature_vector(_Tmp)
            hands_out.append(HandResult(landmarks_xy=landmarks_xy, feature_vector=features))

        return hands_out

    def detect(self, frame_bgr: np.ndarray) -> Optional[HandResult]:
        hands = self.detect_hands(frame_bgr)
        return hands[0] if hands else None