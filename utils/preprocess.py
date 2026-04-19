from __future__ import annotations

from typing import List

import numpy as np


def landmarks_to_feature_vector(hand_landmarks) -> np.ndarray:
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
