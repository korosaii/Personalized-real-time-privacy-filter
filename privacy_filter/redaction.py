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


def pixelate_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    block_size: int = 16,
) -> np.ndarray:
    """Pixelate only pixels selected by a full-frame boolean mask."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if mask.shape != frame.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape} does not match frame shape {frame.shape[:2]}"
        )

    selected = np.asarray(mask, dtype=bool)
    if not np.any(selected):
        return frame.copy()

    height, width = frame.shape[:2]
    small_width = max(1, (width + block_size - 1) // block_size)
    small_height = max(1, (height + block_size - 1) // block_size)
    reduced = cv2.resize(
        frame,
        (small_width, small_height),
        interpolation=cv2.INTER_AREA,
    )
    pixelated = cv2.resize(
        reduced,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    output = frame.copy()
    output[selected] = pixelated[selected]
    return output


def blur_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    kernel_size: int = 51,
) -> np.ndarray:
    """Apply a true Gaussian blur only to pixels selected by a full-frame mask."""
    if kernel_size <= 1:
        raise ValueError("kernel_size must be greater than 1")
    if mask.shape != frame.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape} does not match frame shape {frame.shape[:2]}"
        )
    selected = np.asarray(mask, dtype=bool)
    if not np.any(selected):
        return frame.copy()

    maximum = min(frame.shape[:2])
    maximum = maximum if maximum % 2 == 1 else maximum - 1
    effective_size = min(kernel_size, maximum)
    effective_size = effective_size if effective_size % 2 == 1 else effective_size + 1
    effective_size = max(3, effective_size)
    blurred = cv2.GaussianBlur(
        frame,
        (effective_size, effective_size),
        sigmaX=0.0,
        sigmaY=0.0,
    )
    output = frame.copy()
    output[selected] = blurred[selected]
    return output


def blur_entire_frame(frame: np.ndarray, kernel_size: int = 51) -> np.ndarray:
    mask = np.ones(frame.shape[:2], dtype=bool)
    return blur_mask(frame, mask, kernel_size)


def redact_entire_frame(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    reduced = cv2.resize(frame, (16, 16), interpolation=cv2.INTER_AREA)
    return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_NEAREST)
