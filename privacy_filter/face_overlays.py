from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .lighting import LightingMode
from .tracking import FaceState, FaceTrack


def draw_track_overlay(
    frame: np.ndarray,
    track: FaceTrack,
    confirmations: int,
    minimum_face_size: float,
    *,
    draw_bbox: bool,
    draw_statistics: bool,
    detection_override: np.ndarray | None = None,
) -> None:
    detection = track.detection if detection_override is None else detection_override
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, int(detection[0])))
    y1 = max(0, min(height - 1, int(detection[1])))
    x2 = max(0, min(width - 1, int(detection[2])))
    y2 = max(0, min(height - 1, int(detection[3])))
    if track.state is FaceState.AUTHORIZED:
        color = (70, 230, 70)
        label = f"#{track.track_id} {track.identity_name or 'OWNER'}"
    elif track.recognition_block_reason == "face_near_frame_edge":
        color = (40, 210, 255)
        label = f"#{track.track_id} WAIT FULL FACE"
    elif track.recognition_block_reason == "track_stabilizing":
        color = (40, 210, 255)
        label = f"#{track.track_id} STABILIZING"
    elif track.overlap_uncertain:
        color = (40, 70, 255)
        label = f"#{track.track_id} TRACK UNCERTAIN"
    elif track.face_size < minimum_face_size:
        color = (40, 210, 255)
        label = f"#{track.track_id} TOO SMALL {track.face_size:.0f}px"
    elif track.state is FaceState.PENDING:
        color = (40, 210, 255)
        label = f"#{track.track_id} PENDING {track.positive_streak}/{confirmations}"
    else:
        color = (40, 70, 255)
        label = f"#{track.track_id} UNKNOWN"
    if track.score is not None:
        label += f" {track.score:.3f}"
    if draw_bbox:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if not draw_statistics:
        return
    text_y = max(22, y1 - 8)
    cv2.putText(
        frame,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_runtime_metrics(
    frame: np.ndarray,
    rolling_ms: deque[float],
    detector_ms: float,
    recognition_ms: float,
    recognition_calls: int,
    visible_tracks: int,
    lighting_modes: tuple[str, ...],
    threshold: float,
    difficult_lighting_threshold: float,
    confirmations: int,
    authorized_interval: int,
    minimum_face_size: float,
) -> None:
    fps = 1000.0 / float(np.mean(rolling_ms)) if rolling_ms else 0.0
    lighting_counts = {
        mode.value: lighting_modes.count(mode.value) for mode in LightingMode
    }
    lines = (
        f"FPS {fps:5.1f}",
        (
            f"Detector {detector_ms:5.1f} ms  Recognition "
            f"{recognition_ms:5.1f} ms ({recognition_calls} calls)"
        ),
        (
            f"Tracks {visible_tracks}  threshold normal:{threshold:.3f} "
            f"difficult:{difficult_lighting_threshold:.3f}  "
            f"confirm {confirmations}  min-face {minimum_face_size:.0f}px  "
            f"recheck {authorized_interval}"
        ),
        (
            "Lighting "
            f"NORMAL:{lighting_counts[LightingMode.NORMAL.value]}  "
            f"LOW_LIGHT:{lighting_counts[LightingMode.LOW_LIGHT.value]}  "
            f"OVEREXPOSED:{lighting_counts[LightingMode.OVEREXPOSED.value]}"
        ),
    )
    y = 28
    for line in lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 230, 80),
            2,
            cv2.LINE_AA,
        )
        y += 27


def statistics_canvas(frame: np.ndarray, mirror_input: bool) -> np.ndarray:
    """Return a text canvas that remains readable after Zoom mirrors the view."""
    return frame if mirror_input else cv2.flip(frame, 1)


def commit_statistics_canvas(
    output: np.ndarray,
    canvas: np.ndarray,
    mirror_input: bool,
) -> None:
    if not mirror_input:
        output[:] = cv2.flip(canvas, 1)
