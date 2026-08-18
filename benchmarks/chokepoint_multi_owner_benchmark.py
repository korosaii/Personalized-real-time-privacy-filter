from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.chokepoint_privacy_benchmark import (  # noqa: E402
    GroundTruthFace,
    SequenceInput,
    discover_sequences,
    parse_sequence_groups,
)
from privacy_filter.enrollment import (  # noqa: E402
    EnrollmentGallery,
    build_template,
    save_template,
)
from privacy_filter.model_setup import prepare_runtime_models  # noqa: E402
from privacy_filter.recognition import (  # noqa: E402
    FACE_PREPROCESSING,
    LANDMARK_FACE_PREPROCESSING,
    FaceEmbedder,
)
from privacy_filter.redaction import pixelate_faces  # noqa: E402
from privacy_filter.tracking import FaceTrack, create_face_tracker  # noqa: E402
from privacy_filter.yolo import YOLOFaceDetector  # noqa: E402


@dataclass(frozen=True)
class PassDefinition:
    name: str
    enrollment_camera: str
    evaluation_camera: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Two-pass cross-camera ChokePoint benchmark for multiple authorized owners"
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--sequence",
        dest="sequence_values",
        action="append",
        default=None,
        metavar="GROUP[,GROUP...]",
        help=(
            "ChokePoint group(s) to test; repeat the option or separate groups "
            "with commas (default: P1E_S1)"
        ),
    )
    parser.add_argument(
        "--owners",
        default="0001,0003,0006",
        help="Comma-separated ChokePoint person IDs treated as authorized owners",
    )
    parser.add_argument("--enrollment-samples", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--confirmations", type=int, default=3)
    parser.add_argument(
        "--minimum-recognition-face-size",
        "--minimum-owner-face-size",
        dest="minimum_recognition_face_size",
        type=float,
        default=80.0,
        help=(
            "Minimum detected bbox side before an owner recognition attempt; "
            "smaller faces remain redacted"
        ),
    )
    parser.add_argument("--detector", default="yolo11-pose-roll90")
    parser.add_argument("--recognition-model", default="r50-webface600k")
    parser.add_argument("--provider", default="auto")
    parser.add_argument("--detector-threshold", type=float, default=0.25)
    parser.add_argument(
        "--tracker",
        choices=("none", "iou", "bytetrack", "botsort"),
        default="none",
        help="Tracking backend; none preserves the original confirmation simulation",
    )
    parser.add_argument("--tracker-buffer", type=int, default=30)
    parser.add_argument("--track-iou-threshold", type=float, default=0.25)
    parser.add_argument("--authorization-iou-threshold", type=float, default=0.40)
    parser.add_argument("--track-max-missed", type=int, default=8)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("benchmarks/chokepoint_multi_owner"),
    )
    parser.add_argument(
        "--enrollment-output",
        type=Path,
        default=Path("data/enrollments/chokepoint_multi_owner"),
    )
    args = parser.parse_args()
    try:
        args.sequence_groups = parse_sequence_groups(args.sequence_values)
    except ValueError as error:
        parser.error(str(error))
    args.owner_ids = tuple(
        dict.fromkeys(value.strip() for value in args.owners.split(",") if value.strip())
    )
    if len(args.owner_ids) < 2:
        parser.error("--owners must contain at least two distinct person IDs")
    if args.enrollment_samples < 1:
        parser.error("--enrollment-samples must be positive")
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.confirmations < 1:
        parser.error("--confirmations must be positive")
    if args.minimum_recognition_face_size <= 0.0:
        parser.error("--minimum-recognition-face-size must be positive")
    if not 0.0 < args.detector_threshold < 1.0:
        parser.error("--detector-threshold must be between 0 and 1")
    if args.tracker_buffer < 1:
        parser.error("--tracker-buffer must be positive")
    if not 0.0 < args.track_iou_threshold <= 1.0:
        parser.error("--track-iou-threshold must be in (0, 1]")
    if not args.track_iou_threshold <= args.authorization_iou_threshold <= 1.0:
        parser.error(
            "--authorization-iou-threshold must be between "
            "--track-iou-threshold and 1"
        )
    if args.track_max_missed < 0:
        parser.error("--track-max-missed cannot be negative")
    if args.target_fps <= 0.0:
        parser.error("--target-fps must be positive")
    return args


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _camera_sequence(
    sequences: tuple[SequenceInput, ...], camera_suffix: str
) -> SequenceInput:
    matches = [sequence for sequence in sequences if sequence.name.endswith(camera_suffix)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one sequence ending in {camera_suffix}, found {len(matches)}"
        )
    return matches[0]


def _inside_detection(point: tuple[float, float], detection: np.ndarray) -> bool:
    x1, y1, x2, y2 = (float(value) for value in detection[:4])
    radius_x = (x2 - x1) / 2.0
    radius_y = (y2 - y1) / 2.0
    if radius_x <= 0.0 or radius_y <= 0.0:
        return False
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    return (
        ((point[0] - center_x) / radius_x) ** 2
        + ((point[1] - center_y) / radius_y) ** 2
        <= 1.0
    )


def _face_detection(
    face: GroundTruthFace, detections: np.ndarray
) -> np.ndarray | None:
    candidates = [
        detection
        for detection in detections
        if _inside_detection(face.left_eye, detection)
        and _inside_detection(face.right_eye, detection)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda detection: float(detection[4]))


def _face_track(face: GroundTruthFace, tracks: list[FaceTrack]) -> FaceTrack | None:
    candidates = [
        track
        for track in tracks
        if _inside_detection(face.left_eye, track.detection)
        and _inside_detection(face.right_eye, track.detection)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda track: float(track.detection[4]))


def _diverse_candidates(
    sequence: SequenceInput,
    owner_id: str,
) -> list[tuple[str, GroundTruthFace]]:
    candidates = [
        (number, face)
        for number, face in sequence.frames
        if face is not None and face.person_id == owner_id
    ]
    return sorted(candidates, key=lambda item: item[1].eye_distance, reverse=True)


def build_gallery(
    definition: PassDefinition,
    sequence: SequenceInput,
    owner_ids: tuple[str, ...],
    detector: YOLOFaceDetector,
    embedder: FaceEmbedder,
    model_sha256: str,
    threshold: float,
    samples_per_owner: int,
    output_root: Path,
) -> tuple[EnrollmentGallery, dict[str, list[int]]]:
    templates = []
    paths = []
    enrollment_frames: dict[str, list[int]] = {}
    preprocessing = (
        LANDMARK_FACE_PREPROCESSING if detector.has_landmarks else FACE_PREPROCESSING
    )
    for owner_id in owner_ids:
        accepted: list[np.ndarray] = []
        accepted_frames: list[int] = []
        for frame_number, face in _diverse_candidates(sequence, owner_id):
            numeric_frame = int(frame_number)
            if accepted_frames and min(
                abs(numeric_frame - previous) for previous in accepted_frames
            ) < 4:
                continue
            frame_path = sequence.frame_directory / f"{frame_number}.jpg"
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            detected = detector.detect(frame)
            detection = _face_detection(face, detected.detections)
            if detection is None:
                continue
            try:
                accepted.append(embedder.embed_bbox(frame, detection).embedding)
            except ValueError:
                continue
            accepted_frames.append(numeric_frame)
            if len(accepted) >= samples_per_owner:
                break
        if len(accepted) < samples_per_owner:
            raise RuntimeError(
                f"Could only enroll {len(accepted)}/{samples_per_owner} samples "
                f"for person {owner_id} from {sequence.name}"
            )
        identity_name = f"person_{owner_id}"
        template = build_template(
            identity_name,
            accepted,
            model_sha256=model_sha256,
            threshold=threshold,
            minimum_samples=samples_per_owner,
            source=f"ChokePoint {sequence.name} cross-camera benchmark",
            face_preprocessing=preprocessing,
            rotation_angles=(0,),
        )
        output_path = (
            output_root.expanduser().resolve()
            / definition.name
            / f"{identity_name}.npz"
        )
        save_template(template, output_path)
        templates.append(template)
        paths.append(output_path)
        enrollment_frames[owner_id] = accepted_frames
    return EnrollmentGallery(tuple(templates), tuple(paths)), enrollment_frames


def evaluate_pass(
    definition: PassDefinition,
    sequence: SequenceInput,
    gallery: EnrollmentGallery,
    owner_ids: tuple[str, ...],
    detector: YOLOFaceDetector,
    embedder: FaceEmbedder,
    threshold: float,
    confirmations: int,
    minimum_face_size: float,
    target_fps: float,
    tracker_backend: str,
    tracker_buffer: int,
    track_iou_threshold: float,
    authorization_iou_threshold: float,
    track_max_missed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    detector_latencies: list[float] = []
    tracker_latencies: list[float] = []
    recognition_latencies: list[float] = []
    redaction_latencies: list[float] = []
    pipeline_latencies: list[float] = []
    streak_identity: str | None = None
    streak = 0
    confirmed_identity: str | None = None
    previous_person: str | None = None
    previous_frame: int | None = None
    processed_faces = 0
    tracker = (
        None
        if tracker_backend == "none"
        else create_face_tracker(
            backend_name=tracker_backend,
            iou_threshold=track_iou_threshold,
            max_missed_frames=track_max_missed,
            authorization_iou_threshold=authorization_iou_threshold,
            track_buffer=tracker_buffer,
        )
    )

    for frame_number_text, face in sequence.frames:
        if face is None:
            if tracker is not None:
                tracker.tracks.clear()
            previous_person = None
            previous_frame = None
            streak_identity = None
            streak = 0
            confirmed_identity = None
            continue
        frame_number = int(frame_number_text)
        if (
            face.person_id != previous_person
            or previous_frame is None
            or frame_number != previous_frame + 1
        ):
            if tracker is not None:
                tracker.tracks.clear()
            streak_identity = None
            streak = 0
            confirmed_identity = None
        previous_person = face.person_id
        previous_frame = frame_number

        frame_path = sequence.frame_directory / f"{frame_number_text}.jpg"
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Could not decode evaluation frame: {frame_path}")
        detected = detector.detect(frame)
        if tracker is None:
            visible_tracks: list[FaceTrack] = []
            selected_track = None
            detection = _face_detection(face, detected.detections)
            tracker_ms = 0.0
        else:
            tracker_started = perf_counter()
            visible_tracks = tracker.update(detected.detections, frame_number, frame)
            selected_track = _face_track(face, visible_tracks)
            detection = selected_track.detection if selected_track is not None else None
            tracker_ms = (perf_counter() - tracker_started) * 1000.0
        face_size = (
            min(float(detection[2] - detection[0]), float(detection[3] - detection[1]))
            if detection is not None
            else 0.0
        )
        recognition_ms = 0.0
        matching_ms = 0.0
        score: float | None = None
        raw_identity: str | None = None
        recognition_attempted = detection is not None and face_size >= minimum_face_size
        if recognition_attempted:
            embedded = embedder.embed_bbox(frame, detection)
            recognition_ms = embedded.latency_ms
            matching_started = perf_counter()
            match = gallery.best_match(embedded.embedding, threshold)
            matching_ms = (perf_counter() - matching_started) * 1000.0
            score = match.score
            if score >= match.threshold:
                raw_identity = match.identity_name.removeprefix("person_")

        if tracker is None:
            if raw_identity is None:
                streak_identity = None
                streak = 0
            elif raw_identity == streak_identity:
                streak += 1
            else:
                streak_identity = raw_identity
                streak = 1
            if confirmed_identity is None and streak >= confirmations:
                confirmed_identity = streak_identity
        elif selected_track is not None:
            if recognition_attempted:
                selected_track.record_recognition(
                    score,
                    threshold,
                    confirmations,
                    frame_number,
                    identity_name=raw_identity,
                )
            confirmed_identity = (
                selected_track.identity_name if selected_track.authorized else None
            )
        else:
            confirmed_identity = None

        redaction_started = perf_counter()
        if tracker is None:
            if confirmed_identity is None:
                pixelate_faces(frame, detected.detections)
        else:
            authorized_detection_indexes = {
                track.detection_index
                for track in visible_tracks
                if track.authorized and track.detection_index is not None
            }
            unauthorized = np.asarray(
                [
                    detection_row
                    for detection_index, detection_row in enumerate(detected.detections)
                    if detection_index not in authorized_detection_indexes
                ],
                dtype=np.float32,
            )
            if len(unauthorized):
                pixelate_faces(frame, unauthorized)
        redaction_ms = (perf_counter() - redaction_started) * 1000.0
        pipeline_ms = (
            detected.latency_ms
            + tracker_ms
            + recognition_ms
            + matching_ms
            + redaction_ms
        )
        detector_latencies.append(detected.latency_ms)
        tracker_latencies.append(tracker_ms)
        if recognition_attempted:
            recognition_latencies.append(recognition_ms)
        redaction_latencies.append(redaction_ms)
        pipeline_latencies.append(pipeline_ms)

        is_owner = face.person_id in owner_ids
        detector_covers_eyes = detection is not None
        raw_correct_owner = is_owner and raw_identity == face.person_id
        confirmed_correct_owner = is_owner and confirmed_identity == face.person_id
        raw_false_authorization = not is_owner and raw_identity is not None
        confirmed_false_authorization = (
            not is_owner and confirmed_identity is not None
        )
        stranger_privacy_safe = (
            is_owner
            or (
                detector_covers_eyes
                and confirmed_identity is None
            )
        )
        if is_owner:
            if confirmed_correct_owner:
                outcome = "owner_revealed_correctly"
            elif confirmed_identity is not None:
                outcome = "owner_wrong_identity"
            elif not detector_covers_eyes:
                outcome = "owner_detector_miss"
            elif not recognition_attempted:
                outcome = "owner_too_small"
            else:
                outcome = "owner_hidden"
        elif confirmed_false_authorization:
            outcome = "stranger_false_authorized"
        elif not detector_covers_eyes:
            outcome = "stranger_detector_leak"
        else:
            outcome = "stranger_redacted"

        rows.append(
            {
                "pass": definition.name,
                "enrollment_camera": definition.enrollment_camera,
                "evaluation_camera": definition.evaluation_camera,
                "sequence": sequence.name,
                "frame_number": frame_number,
                "person_id": face.person_id,
                "is_owner": is_owner,
                "track_id": selected_track.track_id if selected_track is not None else "",
                "eye_distance_px": round(face.eye_distance, 4),
                "detector_covers_eyes": detector_covers_eyes,
                "face_size_px": round(face_size, 4),
                "recognition_attempted": recognition_attempted,
                "candidate_identity": raw_identity or "",
                "score": round(score, 6) if score is not None else "",
                "raw_correct_owner": raw_correct_owner,
                "raw_false_authorization": raw_false_authorization,
                "confirmed_identity": confirmed_identity or "",
                "confirmed_correct_owner": confirmed_correct_owner,
                "confirmed_false_authorization": confirmed_false_authorization,
                "stranger_privacy_safe": stranger_privacy_safe,
                "outcome": outcome,
                "detector_ms": round(detected.latency_ms, 4),
                "tracker_ms": round(tracker_ms, 4),
                "recognition_ms": round(recognition_ms, 4),
                "matching_ms": round(matching_ms, 4),
                "redaction_ms": round(redaction_ms, 4),
                "pipeline_ms": round(pipeline_ms, 4),
            }
        )
        processed_faces += 1
        if processed_faces % 250 == 0:
            print(
                f"{definition.name}: evaluated {processed_faces} GT face frames...",
                flush=True,
            )

    owner_rows = [row for row in rows if row["is_owner"]]
    stranger_rows = [row for row in rows if not row["is_owner"]]
    owner_attempted = [row for row in owner_rows if row["recognition_attempted"]]
    owner_size_gated = [
        row
        for row in owner_rows
        if row["detector_covers_eyes"] and not row["recognition_attempted"]
    ]
    raw_owner_correct = [row for row in owner_rows if row["raw_correct_owner"]]
    confirmed_owner_correct = [
        row for row in owner_rows if row["confirmed_correct_owner"]
    ]
    raw_false_authorized = [
        row for row in stranger_rows if row["raw_false_authorization"]
    ]
    confirmed_false_authorized = [
        row for row in stranger_rows if row["confirmed_false_authorization"]
    ]
    privacy_safe = [row for row in stranger_rows if row["stranger_privacy_safe"]]
    wrong_owner = [
        row
        for row in owner_rows
        if row["confirmed_identity"]
        and row["confirmed_identity"] != row["person_id"]
    ]
    per_owner = {}
    for owner_id in owner_ids:
        selected = [row for row in owner_rows if row["person_id"] == owner_id]
        per_owner[owner_id] = {
            "frames": len(selected),
            "recognition_attempted_frames": sum(
                bool(row["recognition_attempted"]) for row in selected
            ),
            "raw_identification_recall": (
                sum(bool(row["raw_correct_owner"]) for row in selected) / len(selected)
                if selected
                else None
            ),
            "confirmed_reveal_recall": (
                sum(bool(row["confirmed_correct_owner"]) for row in selected)
                / len(selected)
                if selected
                else None
            ),
        }
    budget_ms = 1000.0 / target_fps
    pipeline_p95 = float(np.percentile(pipeline_latencies, 95))
    metrics: dict[str, object] = {
        "pass": definition.name,
        "enrollment_camera": definition.enrollment_camera,
        "evaluation_camera": definition.evaluation_camera,
        "frames_with_gt_faces": len(rows),
        "owner_frames": len(owner_rows),
        "owner_recognition_attempted_frames": len(owner_attempted),
        "owner_size_gated_frames": len(owner_size_gated),
        "owner_size_gated_rate": (
            len(owner_size_gated) / len(owner_rows) if owner_rows else 0.0
        ),
        "stranger_frames": len(stranger_rows),
        "owner_raw_identification_recall": (
            len(raw_owner_correct) / len(owner_rows) if owner_rows else 0.0
        ),
        "owner_raw_identification_recall_when_attempted": (
            len(raw_owner_correct) / len(owner_attempted) if owner_attempted else 0.0
        ),
        "owner_confirmed_reveal_recall": (
            len(confirmed_owner_correct) / len(owner_rows) if owner_rows else 0.0
        ),
        "owner_wrong_identity_rate": (
            len(wrong_owner) / len(owner_rows) if owner_rows else 0.0
        ),
        "stranger_raw_false_authorization_rate": (
            len(raw_false_authorized) / len(stranger_rows) if stranger_rows else 0.0
        ),
        "stranger_confirmed_false_authorization_rate": (
            len(confirmed_false_authorized) / len(stranger_rows)
            if stranger_rows
            else 0.0
        ),
        "stranger_privacy_recall": (
            len(privacy_safe) / len(stranger_rows) if stranger_rows else 0.0
        ),
        "per_owner": per_owner,
        "performance": {
            "frame_budget_ms": budget_ms,
            "detector_ms": _distribution(detector_latencies),
            "tracker_ms": _distribution(tracker_latencies),
            "recognition_call_ms": _distribution(recognition_latencies),
            "redaction_ms": _distribution(redaction_latencies),
            "face_active_pipeline_ms": _distribution(pipeline_latencies),
            "estimated_face_active_fps": 1000.0 / mean(pipeline_latencies),
            "p95_meets_target_fps": pipeline_p95 <= budget_ms,
        },
    }
    return metrics, rows


def run(args: argparse.Namespace) -> dict[str, object]:
    grouped_sequences = tuple(
        (group, discover_sequences(args.data_root, group))
        for group in args.sequence_groups
    )
    base_definitions = (
        PassDefinition("pass_1", "C1", "C2"),
        PassDefinition("pass_2", "C2", "C1"),
    )
    for group, sequences in grouped_sequences:
        available_ids = {
            face.person_id
            for sequence in sequences
            for _, face in sequence.frames
            if face is not None
        }
        missing_owners = sorted(set(args.owner_ids).difference(available_ids))
        if missing_owners:
            raise ValueError(
                f"Owner IDs absent from group {group}: " + ", ".join(missing_owners)
            )

    models = prepare_runtime_models(
        args.detector,
        args.recognition_model,
        args.provider,
    )
    detector = YOLOFaceDetector(
        models.detector_runtime,
        threshold=args.detector_threshold,
        provider=args.provider,
    )
    embedder = FaceEmbedder(models.recognition_runtime, provider=args.provider)
    blank = np.zeros((detector.input_size[1], detector.input_size[0], 3), dtype=np.uint8)
    detector.detect(blank)
    detector.detect(blank)
    embedder.warmup(2)

    pass_metrics = []
    all_rows: list[dict[str, object]] = []
    enrollment_details = {}
    multiple_groups = len(grouped_sequences) > 1
    for group, sequences in grouped_sequences:
        for base_definition in base_definitions:
            definition = (
                PassDefinition(
                    f"{group}_{base_definition.name}",
                    base_definition.enrollment_camera,
                    base_definition.evaluation_camera,
                )
                if multiple_groups
                else base_definition
            )
            print(
                f"{definition.name}: enroll {definition.enrollment_camera}, "
                f"evaluate {definition.evaluation_camera}",
                flush=True,
            )
            enrollment_sequence = _camera_sequence(
                sequences, f"_{definition.enrollment_camera}"
            )
            evaluation_sequence = _camera_sequence(
                sequences, f"_{definition.evaluation_camera}"
            )
            gallery, enrollment_frames = build_gallery(
                definition,
                enrollment_sequence,
                args.owner_ids,
                detector,
                embedder,
                models.recognition_source_sha256,
                args.threshold,
                args.enrollment_samples,
                args.enrollment_output,
            )
            print(
                f"{definition.name}: enrolled {', '.join(gallery.names)}",
                flush=True,
            )
            metrics, rows = evaluate_pass(
                definition,
                evaluation_sequence,
                gallery,
                args.owner_ids,
                detector,
                embedder,
                args.threshold,
                args.confirmations,
                args.minimum_recognition_face_size,
                args.target_fps,
                args.tracker,
                args.tracker_buffer,
                args.track_iou_threshold,
                args.authorization_iou_threshold,
                args.track_max_missed,
            )
            pass_metrics.append(metrics)
            all_rows.extend(rows)
            enrollment_details[definition.name] = {
                "templates": [str(path) for path in gallery.paths],
                "frames_by_owner": enrollment_frames,
            }

    owner_frames = sum(int(item["owner_frames"]) for item in pass_metrics)
    owner_attempted_frames = sum(
        int(item["owner_recognition_attempted_frames"]) for item in pass_metrics
    )
    owner_size_gated_frames = sum(
        int(item["owner_size_gated_frames"]) for item in pass_metrics
    )
    stranger_frames = sum(int(item["stranger_frames"]) for item in pass_metrics)
    weighted = lambda key, denominator_key: (  # noqa: E731
        sum(
            float(item[key]) * int(item[denominator_key])
            for item in pass_metrics
        )
        / sum(int(item[denominator_key]) for item in pass_metrics)
    )
    combined = {
        "owner_frames": owner_frames,
        "owner_recognition_attempted_frames": owner_attempted_frames,
        "owner_size_gated_frames": owner_size_gated_frames,
        "owner_size_gated_rate": owner_size_gated_frames / owner_frames,
        "stranger_frames": stranger_frames,
        "owner_raw_identification_recall": weighted(
            "owner_raw_identification_recall", "owner_frames"
        ),
        "owner_raw_identification_recall_when_attempted": (
            sum(
                float(item["owner_raw_identification_recall_when_attempted"])
                * int(item["owner_recognition_attempted_frames"])
                for item in pass_metrics
            )
            / owner_attempted_frames
        ),
        "owner_confirmed_reveal_recall": weighted(
            "owner_confirmed_reveal_recall", "owner_frames"
        ),
        "owner_wrong_identity_rate": weighted(
            "owner_wrong_identity_rate", "owner_frames"
        ),
        "stranger_raw_false_authorization_rate": weighted(
            "stranger_raw_false_authorization_rate", "stranger_frames"
        ),
        "stranger_confirmed_false_authorization_rate": weighted(
            "stranger_confirmed_false_authorization_rate", "stranger_frames"
        ),
        "stranger_privacy_recall": weighted(
            "stranger_privacy_recall", "stranger_frames"
        ),
    }
    checks = {
        "owner_confirmed_reveal_recall_at_least_80pct": (
            combined["owner_confirmed_reveal_recall"] >= 0.80
        ),
        "owner_wrong_identity_rate_at_most_1pct": (
            combined["owner_wrong_identity_rate"] <= 0.01
        ),
        "stranger_privacy_recall_at_least_99pct": (
            combined["stranger_privacy_recall"] >= 0.99
        ),
        "stranger_confirmed_false_authorization_at_most_0_1pct": (
            combined["stranger_confirmed_false_authorization_rate"] <= 0.001
        ),
        "both_passes_meet_target_fps": all(
            bool(item["performance"]["p95_meets_target_fps"])
            for item in pass_metrics
        ),
    }
    output_prefix = args.output_prefix.expanduser().resolve()
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary: dict[str, object] = {
        "benchmark": "chokepoint_multi_owner_two_pass",
        "passed": all(checks.values()),
        "checks": checks,
        "configuration": {
            "data_root": str(args.data_root.expanduser().resolve()),
            "sequence": (
                args.sequence_groups[0] if len(args.sequence_groups) == 1 else None
            ),
            "sequence_groups": list(args.sequence_groups),
            "owner_ids": list(args.owner_ids),
            "enrollment_samples_per_owner": args.enrollment_samples,
            "threshold": args.threshold,
            "confirmations": args.confirmations,
            "minimum_recognition_face_size": args.minimum_recognition_face_size,
            "minimum_owner_face_size": args.minimum_recognition_face_size,
            "detector": args.detector,
            "recognition_model": args.recognition_model,
            "tracker": args.tracker,
            "tracker_buffer": args.tracker_buffer,
            "track_iou_threshold": args.track_iou_threshold,
            "authorization_iou_threshold": args.authorization_iou_threshold,
            "track_max_missed": args.track_max_missed,
            "detector_providers": detector.providers,
            "recognition_providers": embedder.providers,
            "target_fps": args.target_fps,
        },
        "protocol": {
            "pass_1": "enroll C1, evaluate C2 within each selected group",
            "pass_2": "enroll C2, evaluate C1 within each selected group",
            "group_isolation": "Enrollment templates are rebuilt independently per group.",
            "gt_usage": [
                "XML frame number selects the matching source JPG without changing it.",
                "person id defines owner versus stranger and the expected owner identity.",
                "leftEye/rightEye select the detector box belonging to the annotated face.",
                "GT is not used to calculate embeddings, scores, or tune the fixed threshold.",
            ],
            "confirmation_simulation": (
                f"Reveal after {args.confirmations} consecutive matches to the same owner; "
                "reset on a person or frame gap."
            ),
            "scope": (
                "Classifier and redaction decision on every GT face frame; "
                + (
                    "tracker association is replayed, but UNKNOWN retry scheduling is not."
                    if args.tracker != "none"
                    else "tracker association and UNKNOWN retry scheduling are not replayed."
                )
            ),
        },
        "enrollment": enrollment_details,
        "passes": pass_metrics,
        "combined": combined,
        "artifacts": {"json": str(json_path), "frame_csv": str(csv_path)},
    }
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(f"Result: {'PASS' if summary['passed'] else 'FAIL'}")
    for item in summary["passes"]:
        performance = item["performance"]["face_active_pipeline_ms"]
        print(
            f"{item['pass']} ({item['enrollment_camera']}->{item['evaluation_camera']}): "
            f"owner reveal {item['owner_confirmed_reveal_recall']:.2%}, "
            f"eligible owner ID {item['owner_raw_identification_recall_when_attempted']:.2%}, "
            f"stranger privacy {item['stranger_privacy_recall']:.2%}, "
            f"false auth {item['stranger_confirmed_false_authorization_rate']:.2%}, "
            f"p95 {performance['p95']:.2f} ms"
        )
    combined = summary["combined"]
    print(
        "Combined: "
        f"owner reveal {combined['owner_confirmed_reveal_recall']:.2%}, "
        f"eligible owner ID {combined['owner_raw_identification_recall_when_attempted']:.2%}, "
        f"stranger privacy {combined['stranger_privacy_recall']:.2%}, "
        f"wrong owner {combined['owner_wrong_identity_rate']:.2%}"
    )
    print(f"JSON: {summary['artifacts']['json']}")
    print(f"CSV: {summary['artifacts']['frame_csv']}")
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
