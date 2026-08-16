from __future__ import annotations

import cv2
import numpy as np


def _clip_box(
    box: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(value) for value in box[:4])
    return (
        max(0, min(image_width, int(np.floor(x1)))),
        max(0, min(image_height, int(np.floor(y1)))),
        max(0, min(image_width, int(np.ceil(x2)))),
        max(0, min(image_height, int(np.ceil(y2)))),
    )


def _pixelate(region: np.ndarray) -> np.ndarray:
    height, width = region.shape[:2]
    if height < 2 or width < 2:
        return region

    small_width = max(2, min(8, width // 16))
    small_height = max(2, min(8, height // 16))
    reduced = cv2.resize(region, (small_width, small_height), interpolation=cv2.INTER_AREA)
    return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_NEAREST)


def pixelate_faces(
    frame: np.ndarray,
    detections: np.ndarray,
) -> np.ndarray:
    output = frame.copy()
    image_height, image_width = output.shape[:2]

    for detection in detections:
        x1, y1, x2, y2 = _clip_box(
            detection,
            image_width=image_width,
            image_height=image_height,
        )
        if x2 <= x1 or y2 <= y1:
            continue

        region = output[y1:y2, x1:x2]
        pixelated = _pixelate(region)
        region_height, region_width = region.shape[:2]
        mask = np.zeros((region_height, region_width), dtype=np.uint8)
        cv2.ellipse(
            mask,
            (region_width // 2, region_height // 2),
            (max(1, region_width // 2), max(1, region_height // 2)),
            0.0,
            0.0,
            360.0,
            255,
            thickness=-1,
            lineType=cv2.LINE_8,
        )
        cv2.copyTo(pixelated, mask, region)

    return output


def redact_entire_frame(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    reduced = cv2.resize(frame, (16, 16), interpolation=cv2.INTER_AREA)
    return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_NEAREST)
