from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np


def draw_landmark_points(frame_bgr: np.ndarray, points_xy: Iterable[Tuple[int, int]]) -> None:
    for (x, y) in points_xy:
        cv2.circle(frame_bgr, (x, y), 3, (0, 255, 0), -1)
