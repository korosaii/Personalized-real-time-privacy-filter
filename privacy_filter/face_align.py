from __future__ import annotations

import cv2
import numpy as np


ARCFACE_DESTINATION = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def estimate_similarity_transform(keypoints: np.ndarray) -> np.ndarray:
    source = np.asarray(keypoints, dtype=np.float32)
    if source.shape != (5, 2):
        raise ValueError(f"Expected five 2D keypoints, got {source.shape}")
    if not np.isfinite(source).all():
        raise ValueError("Face keypoints contain non-finite values")

    matrix, _ = cv2.estimateAffinePartial2D(
        source,
        ARCFACE_DESTINATION,
        method=cv2.LMEDS,
    )
    if matrix is None or matrix.shape != (2, 3):
        raise ValueError("Could not estimate a stable face-alignment transform")
    return matrix.astype(np.float32, copy=False)


def align_face(frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR image with shape HxWx3")
    matrix = estimate_similarity_transform(keypoints)
    return cv2.warpAffine(
        frame,
        matrix,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
