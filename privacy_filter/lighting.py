from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class LightingMode(str, Enum):
    NORMAL = "NORMAL"
    LOW_LIGHT = "LOW_LIGHT"
    OVEREXPOSED = "OVEREXPOSED"


@dataclass(frozen=True)
class LightingMetrics:
    ambient_median: float
    ambient_contrast: float
    ambient_black_ratio: float
    face_median: float
    face_p10: float
    face_p90: float
    face_contrast: float
    face_black_ratio: float
    face_white_ratio: float


def _pixel_metrics(
    pixels: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    values = np.asarray(pixels, dtype=np.uint8).reshape(-1)
    if not values.size:
        raise ValueError("Lighting region is empty")
    p10, median, p90 = np.percentile(values, (10, 50, 90))
    black_ratio = float(np.mean(values < 12))
    white_ratio = float(np.mean(values > 243))
    return (
        float(p10),
        float(median),
        float(p90),
        float(p90 - p10),
        black_ratio,
        white_ratio,
    )


def measure_lighting(
    frame: np.ndarray,
    detection: np.ndarray,
    padding: float = 0.25,
) -> LightingMetrics:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR image with shape HxWx3")
    if padding <= 0.0:
        raise ValueError("lighting padding must be positive")

    height, width = frame.shape[:2]
    box = np.asarray(detection, dtype=np.float32).reshape(-1)
    if box.size < 4 or not np.isfinite(box[:4]).all():
        raise ValueError("detection must contain a finite bbox")
    x1 = max(0, min(width, math.floor(float(box[0]))))
    y1 = max(0, min(height, math.floor(float(box[1]))))
    x2 = max(0, min(width, math.ceil(float(box[2]))))
    y2 = max(0, min(height, math.ceil(float(box[3]))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError("face bbox is too small for lighting analysis")

    box_width = x2 - x1
    box_height = y2 - y1
    outer_x1 = max(0, math.floor(x1 - box_width * padding))
    outer_y1 = max(0, math.floor(y1 - box_height * padding))
    outer_x2 = min(width, math.ceil(x2 + box_width * padding))
    outer_y2 = min(height, math.ceil(y2 + box_height * padding))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_pixels = gray[y1:y2, x1:x2]
    outer = gray[outer_y1:outer_y2, outer_x1:outer_x2]
    ring_mask = np.ones(outer.shape, dtype=bool)
    ring_mask[
        y1 - outer_y1 : y2 - outer_y1,
        x1 - outer_x1 : x2 - outer_x1,
    ] = False
    ring_pixels = outer[ring_mask]
    if ring_pixels.size < 64:
        ring_pixels = gray.reshape(-1)

    (
        _,
        ambient_median,
        _,
        ambient_contrast,
        ambient_black_ratio,
        _,
    ) = _pixel_metrics(ring_pixels)
    (
        face_p10,
        face_median,
        face_p90,
        face_contrast,
        face_black_ratio,
        face_white_ratio,
    ) = _pixel_metrics(face_pixels)
    return LightingMetrics(
        ambient_median=ambient_median,
        ambient_contrast=ambient_contrast,
        ambient_black_ratio=ambient_black_ratio,
        face_median=face_median,
        face_p10=face_p10,
        face_p90=face_p90,
        face_contrast=face_contrast,
        face_black_ratio=face_black_ratio,
        face_white_ratio=face_white_ratio,
    )


def classify_lighting(
    ambient_median: float,
    face_p10: float,
    face_p90: float,
    face_black_ratio: float,
    face_white_ratio: float,
) -> LightingMode:
    # Prefer information loss over face mean; absolute face brightness can
    # otherwise become an unintended proxy for skin tone.
    if face_p10 > 210.0 or face_white_ratio >= 0.45:
        return LightingMode.OVEREXPOSED
    if face_p90 < 45.0 or face_black_ratio >= 0.70:
        return LightingMode.LOW_LIGHT
    if (
        face_p90 < 90.0
        or face_black_ratio >= 0.35
        or (ambient_median < 35.0 and face_p90 < 120.0)
    ):
        return LightingMode.LOW_LIGHT
    return LightingMode.NORMAL
