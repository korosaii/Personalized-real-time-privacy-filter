from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class FaceState(str, Enum):
    PENDING = "PENDING"
    AUTHORIZED = "AUTHORIZED"
    UNKNOWN = "UNKNOWN"


def recognition_interval_for_state(
    state: FaceState,
    face_count: int,
    single_face_interval: int,
    authorized_interval: int,
    crowded_unknown_interval: int,
) -> int:
    if face_count > 1:
        return crowded_unknown_interval if state is FaceState.UNKNOWN else 1
    if state is FaceState.AUTHORIZED:
        return authorized_interval
    return single_face_interval


def intersection_over_union(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2] - first[0])) * max(
        0.0, float(first[3] - first[1])
    )
    second_area = max(0.0, float(second[2] - second[0])) * max(
        0.0, float(second[3] - second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass
class FaceTrack:
    track_id: int
    detection: np.ndarray
    keypoints: np.ndarray | None
    created_frame: int
    state: FaceState = FaceState.PENDING
    score: float | None = None
    positive_streak: int = 0
    last_recognition_frame: int | None = None
    last_positive_frame: int | None = None
    missed_frames: int = 0
    visible_frames: int = 1
    last_match_iou: float = 1.0

    @property
    def authorized(self) -> bool:
        return self.state is FaceState.AUTHORIZED

    def revoke(self, state: FaceState = FaceState.PENDING) -> None:
        self.state = state
        self.score = None
        self.positive_streak = 0
        self.last_positive_frame = None

    def update_geometry(
        self,
        detection: np.ndarray,
        keypoints: np.ndarray | None,
        match_iou: float,
        authorization_iou_threshold: float,
    ) -> bool:
        was_authorized = self.authorized
        if was_authorized and match_iou < authorization_iou_threshold:
            self.revoke()
        self.detection = np.asarray(detection, dtype=np.float32).copy()
        self.keypoints = (
            None if keypoints is None else np.asarray(keypoints, dtype=np.float32).copy()
        )
        self.last_match_iou = float(match_iou)
        self.missed_frames = 0
        self.visible_frames += 1
        return was_authorized and not self.authorized

    def mark_missed(self) -> bool:
        was_authorized = self.authorized
        self.missed_frames += 1
        self.revoke()
        return was_authorized

    def expire_authorization(self, frame_index: int, ttl_frames: int) -> bool:
        if not self.authorized or self.last_positive_frame is None:
            return False
        if frame_index - self.last_positive_frame <= ttl_frames:
            return False
        self.revoke()
        return True

    def should_recognize(self, frame_index: int, interval_frames: int) -> bool:
        return (
            self.last_recognition_frame is None
            or frame_index - self.last_recognition_frame >= interval_frames
        )

    def record_recognition(
        self,
        score: float | None,
        threshold: float,
        confirmations: int,
        frame_index: int,
    ) -> tuple[FaceState, FaceState]:
        previous = self.state
        self.last_recognition_frame = frame_index
        self.score = None if score is None else float(score)
        if score is None or score < threshold:
            self.state = FaceState.UNKNOWN
            self.positive_streak = 0
            self.last_positive_frame = None
            return previous, self.state

        self.last_positive_frame = frame_index
        self.positive_streak = min(confirmations, self.positive_streak + 1)
        if self.positive_streak >= confirmations:
            self.state = FaceState.AUTHORIZED
        elif not self.authorized:
            self.state = FaceState.PENDING
        return previous, self.state


class FaceTracker:
    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_missed_frames: int = 8,
        authorization_iou_threshold: float = 0.40,
    ) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames cannot be negative")
        if not iou_threshold <= authorization_iou_threshold <= 1.0:
            raise ValueError(
                "authorization_iou_threshold must be between iou_threshold and 1"
            )
        self.iou_threshold = float(iou_threshold)
        self.max_missed_frames = int(max_missed_frames)
        self.authorization_iou_threshold = float(authorization_iou_threshold)
        self.tracks: dict[int, FaceTrack] = {}
        self.next_track_id = 1
        self.created_tracks = 0
        self.expired_tracks = 0
        self.revocations = 0

    @staticmethod
    def _keypoints_at(keypoints: np.ndarray | None, index: int) -> np.ndarray | None:
        if keypoints is None or index >= len(keypoints):
            return None
        return keypoints[index]

    def _new_track(
        self,
        detection: np.ndarray,
        keypoints: np.ndarray | None,
        frame_index: int,
    ) -> FaceTrack:
        track = FaceTrack(
            track_id=self.next_track_id,
            detection=np.asarray(detection, dtype=np.float32).copy(),
            keypoints=(
                None
                if keypoints is None
                else np.asarray(keypoints, dtype=np.float32).copy()
            ),
            created_frame=frame_index,
        )
        self.tracks[track.track_id] = track
        self.next_track_id += 1
        self.created_tracks += 1
        return track

    def update(
        self,
        detections: np.ndarray,
        keypoints: np.ndarray | None,
        frame_index: int,
    ) -> list[FaceTrack]:
        boxes = np.asarray(detections, dtype=np.float32)
        if boxes.size == 0:
            boxes = np.empty((0, 5), dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[1] < 4:
            raise ValueError(f"Expected Nx4+ detections, got {boxes.shape}")

        existing = list(self.tracks.values())
        matches: dict[int, tuple[int, float]] = {}
        if existing and len(boxes):
            overlaps = np.asarray(
                [
                    [intersection_over_union(track.detection, box) for box in boxes]
                    for track in existing
                ],
                dtype=np.float32,
            )
            best_detection_for_track = overlaps.argmax(axis=1)
            best_track_for_detection = overlaps.argmax(axis=0)
            for track_index, detection_index in enumerate(best_detection_for_track):
                detection_index = int(detection_index)
                overlap = float(overlaps[track_index, detection_index])
                if (
                    overlap >= self.iou_threshold
                    and int(best_track_for_detection[detection_index]) == track_index
                ):
                    matches[track_index] = (detection_index, overlap)

        visible_by_detection: dict[int, FaceTrack] = {}
        matched_track_indexes = set(matches)
        matched_detection_indexes = {item[0] for item in matches.values()}
        for track_index, track in enumerate(existing):
            if track_index not in matched_track_indexes:
                if track.mark_missed():
                    self.revocations += 1
                continue
            detection_index, overlap = matches[track_index]
            if track.update_geometry(
                boxes[detection_index],
                self._keypoints_at(keypoints, detection_index),
                overlap,
                self.authorization_iou_threshold,
            ):
                self.revocations += 1
            visible_by_detection[detection_index] = track

        for detection_index, detection in enumerate(boxes):
            if detection_index in matched_detection_indexes:
                continue
            visible_by_detection[detection_index] = self._new_track(
                detection,
                self._keypoints_at(keypoints, detection_index),
                frame_index,
            )

        expired_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missed_frames > self.max_missed_frames
        ]
        for track_id in expired_ids:
            del self.tracks[track_id]
            self.expired_tracks += 1

        return [visible_by_detection[index] for index in sorted(visible_by_detection)]
