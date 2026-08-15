from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class FaceState(str, Enum):
    PENDING = "PENDING"
    AUTHORIZED = "AUTHORIZED"
    UNKNOWN = "UNKNOWN"


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
    created_frame: int
    state: FaceState = FaceState.PENDING
    score: float | None = None
    positive_streak: int = 0
    last_recognition_frame: int | None = None
    last_positive_frame: int | None = None
    missed_frames: int = 0
    visible_frames: int = 1
    last_match_iou: float = 1.0
    last_recognition_size: float | None = None
    last_recognition_detection: np.ndarray | None = None
    tracking_confident: bool = False
    verification_required: bool = True
    stable_matches: int = 0
    overlap_uncertain: bool = False
    recognition_block_reason: str | None = None
    lighting_mode: str = "UNKNOWN"
    lighting_ambient_median: float | None = None
    lighting_face_p10: float | None = None
    lighting_face_p90: float | None = None
    lighting_face_black_ratio: float | None = None
    lighting_face_white_ratio: float | None = None
    lighting_effective_threshold: float | None = None
    velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float32)
    )

    @property
    def authorized(self) -> bool:
        return self.state is FaceState.AUTHORIZED

    @property
    def face_size(self) -> float:
        width = max(0.0, float(self.detection[2] - self.detection[0]))
        height = max(0.0, float(self.detection[3] - self.detection[1]))
        return min(width, height)

    def revoke(self, state: FaceState = FaceState.PENDING) -> None:
        self.state = state
        self.score = None
        self.positive_streak = 0
        self.last_positive_frame = None

    def mark_uncertain(self) -> bool:
        was_authorized = self.authorized
        self.revoke()
        self.tracking_confident = False
        self.verification_required = True
        self.stable_matches = 0
        return was_authorized

    def predicted_detection(self) -> np.ndarray:
        predicted = self.detection.copy()
        predicted[:4] += self.velocity
        return predicted

    def update_geometry(
        self,
        detection: np.ndarray,
        match_iou: float,
        authorization_iou_threshold: float,
        ambiguous: bool,
    ) -> bool:
        previous_detection = self.detection.copy()
        self.overlap_uncertain = False
        next_detection = np.asarray(detection, dtype=np.float32).copy()
        previous_width = max(1.0, float(previous_detection[2] - previous_detection[0]))
        previous_height = max(1.0, float(previous_detection[3] - previous_detection[1]))
        next_width = max(1.0, float(next_detection[2] - next_detection[0]))
        next_height = max(1.0, float(next_detection[3] - next_detection[1]))
        previous_center = np.asarray(
            [
                (previous_detection[0] + previous_detection[2]) / 2.0,
                (previous_detection[1] + previous_detection[3]) / 2.0,
            ],
            dtype=np.float32,
        )
        next_center = np.asarray(
            [
                (next_detection[0] + next_detection[2]) / 2.0,
                (next_detection[1] + next_detection[3]) / 2.0,
            ],
            dtype=np.float32,
        )
        center_shift = float(
            np.linalg.norm(next_center - previous_center)
            / np.hypot(previous_width, previous_height)
        )
        scale_ratio = max(
            next_width / previous_width,
            previous_width / next_width,
            next_height / previous_height,
            previous_height / next_height,
        )
        confidence = float(next_detection[4]) if next_detection.size > 4 else 1.0
        reliable = (
            not ambiguous
            and match_iou >= authorization_iou_threshold
            and center_shift <= 0.35
            and scale_ratio <= 1.35
            and confidence >= 0.25
        )
        was_authorized = self.authorized
        if not reliable:
            self.mark_uncertain()
        else:
            self.tracking_confident = True
            self.stable_matches += 1
        displacement = next_detection[:4] - previous_detection[:4]
        self.velocity = 0.6 * self.velocity + 0.4 * displacement
        self.detection = next_detection
        self.last_match_iou = float(match_iou)
        self.missed_frames = 0
        self.visible_frames += 1
        return was_authorized and not self.authorized

    def mark_missed(self) -> bool:
        self.missed_frames += 1
        self.velocity *= 0.5
        return self.mark_uncertain()

    def should_recognize(
        self,
        frame_index: int,
        minimum_face_size: float,
        unknown_retry_growth: float,
        unknown_retry_movement: float,
        unknown_retry_cooldown: int,
        authorized_recheck_interval: int,
    ) -> bool:
        return self.recognition_reason(
            frame_index,
            minimum_face_size,
            unknown_retry_growth,
            unknown_retry_movement,
            unknown_retry_cooldown,
            authorized_recheck_interval,
        ) is not None

    def recognition_reason(
        self,
        frame_index: int,
        minimum_face_size: float,
        unknown_retry_growth: float,
        unknown_retry_movement: float,
        unknown_retry_cooldown: int,
        authorized_recheck_interval: int,
    ) -> str | None:
        if self.overlap_uncertain:
            return None
        if self.face_size < minimum_face_size:
            return None
        if self.last_recognition_frame is None:
            return "new_track"
        if self.verification_required:
            return "tracker_uncertain"
        if self.state is FaceState.PENDING and self.positive_streak > 0:
            return "confirmation"
        if self.state is FaceState.UNKNOWN:
            if (
                self.last_recognition_frame is not None
                and frame_index - self.last_recognition_frame < unknown_retry_cooldown
            ):
                return None
            if (
                self.last_recognition_size is None
                or self.face_size >= self.last_recognition_size * unknown_retry_growth
            ):
                return "face_grew"
            if self.last_recognition_detection is not None:
                previous = self.last_recognition_detection
                previous_center = np.asarray(
                    [
                        (previous[0] + previous[2]) / 2.0,
                        (previous[1] + previous[3]) / 2.0,
                    ],
                    dtype=np.float32,
                )
                current_center = np.asarray(
                    [
                        (self.detection[0] + self.detection[2]) / 2.0,
                        (self.detection[1] + self.detection[3]) / 2.0,
                    ],
                    dtype=np.float32,
                )
                movement = float(
                    np.linalg.norm(current_center - previous_center)
                    / max(self.last_recognition_size or 1.0, 1.0)
                )
                if movement >= unknown_retry_movement:
                    return "face_moved"
            return None
        if (
            authorized_recheck_interval > 0
            and self.last_positive_frame is not None
            and frame_index - self.last_positive_frame >= authorized_recheck_interval
        ):
            return "safety_recheck"
        return None

    def record_recognition(
        self,
        score: float | None,
        threshold: float,
        confirmations: int,
        frame_index: int,
    ) -> tuple[FaceState, FaceState]:
        previous = self.state
        self.last_recognition_frame = frame_index
        self.last_recognition_size = self.face_size
        self.last_recognition_detection = self.detection.copy()
        self.verification_required = False
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
        overlap_uncertainty_threshold: float = 0.05,
    ) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames cannot be negative")
        if not iou_threshold <= authorization_iou_threshold <= 1.0:
            raise ValueError(
                "authorization_iou_threshold must be between iou_threshold and 1"
            )
        if not 0.0 <= overlap_uncertainty_threshold <= 1.0:
            raise ValueError("overlap_uncertainty_threshold must be between 0 and 1")
        self.iou_threshold = float(iou_threshold)
        self.max_missed_frames = int(max_missed_frames)
        self.authorization_iou_threshold = float(authorization_iou_threshold)
        self.overlap_uncertainty_threshold = float(overlap_uncertainty_threshold)
        self.tracks: dict[int, FaceTrack] = {}
        self.next_track_id = 1
        self.created_tracks = 0
        self.expired_tracks = 0
        self.revocations = 0

    def _new_track(
        self,
        detection: np.ndarray,
        frame_index: int,
    ) -> FaceTrack:
        track = FaceTrack(
            track_id=self.next_track_id,
            detection=np.asarray(detection, dtype=np.float32).copy(),
            created_frame=frame_index,
        )
        self.tracks[track.track_id] = track
        self.next_track_id += 1
        self.created_tracks += 1
        return track

    def update(
        self,
        detections: np.ndarray,
        frame_index: int,
    ) -> list[FaceTrack]:
        boxes = np.asarray(detections, dtype=np.float32)
        if boxes.size == 0:
            boxes = np.empty((0, 5), dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[1] < 4:
            raise ValueError(f"Expected Nx4+ detections, got {boxes.shape}")

        existing = list(self.tracks.values())
        matches: dict[int, tuple[int, float, bool]] = {}
        if existing and len(boxes):
            overlaps = np.asarray(
                [
                    [intersection_over_union(track.predicted_detection(), box) for box in boxes]
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
                    row = np.sort(overlaps[track_index])[::-1]
                    column = np.sort(overlaps[:, detection_index])[::-1]
                    row_ambiguous = (
                        len(row) > 1
                        and row[1] >= self.iou_threshold
                        and row[0] - row[1] <= 0.10
                    )
                    column_ambiguous = (
                        len(column) > 1
                        and column[1] >= self.iou_threshold
                        and column[0] - column[1] <= 0.10
                    )
                    matches[track_index] = (
                        detection_index,
                        overlap,
                        bool(row_ambiguous or column_ambiguous),
                    )

        visible_by_detection: dict[int, FaceTrack] = {}
        matched_track_indexes = set(matches)
        matched_detection_indexes = {item[0] for item in matches.values()}
        for track_index, track in enumerate(existing):
            if track_index not in matched_track_indexes:
                if track.mark_missed():
                    self.revocations += 1
                continue
            detection_index, overlap, ambiguous = matches[track_index]
            if track.update_geometry(
                boxes[detection_index],
                overlap,
                self.authorization_iou_threshold,
                ambiguous,
            ):
                self.revocations += 1
            visible_by_detection[detection_index] = track

        for detection_index, detection in enumerate(boxes):
            if detection_index in matched_detection_indexes:
                continue
            visible_by_detection[detection_index] = self._new_track(
                detection,
                frame_index,
            )

        visible_tracks = [
            visible_by_detection[index] for index in sorted(visible_by_detection)
        ]
        for first_index, first in enumerate(visible_tracks):
            for second in visible_tracks[first_index + 1 :]:
                if (
                    intersection_over_union(first.detection, second.detection)
                    <= self.overlap_uncertainty_threshold
                ):
                    continue
                if first.mark_uncertain():
                    self.revocations += 1
                if second.mark_uncertain():
                    self.revocations += 1
                first.overlap_uncertain = True
                second.overlap_uncertain = True

        expired_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missed_frames > self.max_missed_frames
        ]
        for track_id in expired_ids:
            del self.tracks[track_id]
            self.expired_tracks += 1

        return visible_tracks
