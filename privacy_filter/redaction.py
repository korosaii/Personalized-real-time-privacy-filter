from __future__ import annotations

import cv2
import numpy as np


def expand_box(
    box: np.ndarray,
    image_width: int,
    image_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(value) for value in box[:4])
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    x1 -= width * padding
    x2 += width * padding
    y1 -= height * padding
    y2 += height * padding

    return (
        max(0, min(image_width, int(np.floor(x1)))),
        max(0, min(image_height, int(np.floor(y1)))),
        max(0, min(image_width, int(np.ceil(x2)))),
        max(0, min(image_height, int(np.ceil(y2)))),
    )


def _strong_blur(region: np.ndarray) -> np.ndarray:
    height, width = region.shape[:2]
    if height < 2 or width < 2:
        return region


    small_width = max(2, min(16, width // 12))
    small_height = max(2, min(16, height // 12))
    reduced = cv2.resize(region, (small_width, small_height), interpolation=cv2.INTER_AREA)
    redacted = cv2.resize(reduced, (width, height), interpolation=cv2.INTER_LINEAR)

    min_side = min(width, height)
    kernel = min(31, min_side if min_side % 2 == 1 else min_side - 1)
    if kernel >= 3:
        redacted = cv2.GaussianBlur(redacted, (kernel, kernel), sigmaX=0)
    return redacted


def blur_faces(
    frame: np.ndarray,
    detections: np.ndarray,
    padding: float = 0.18,
) -> np.ndarray:
    output = frame.copy()
    image_height, image_width = output.shape[:2]

    for detection in detections:
        x1, y1, x2, y2 = expand_box(
            detection,
            image_width=image_width,
            image_height=image_height,
            padding=padding,
        )
        if x2 <= x1 or y2 <= y1:
            continue
        output[y1:y2, x1:x2] = _strong_blur(output[y1:y2, x1:x2])

    return output


def redact_entire_frame(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    reduced = cv2.resize(frame, (16, 16), interpolation=cv2.INTER_AREA)
    return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_NEAREST)
