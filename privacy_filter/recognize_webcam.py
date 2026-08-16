from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

from .camera import Camera, VideoFile
from .enrollment import load_template, sha256_file
from .lighting import LightingMode, classify_lighting, measure_lighting
from .model_setup import (
    detector_model_help,
    prepare_runtime_models,
    recognition_model_help,
)
from .ort_session import PROVIDER_CHOICES
from .recognition import (
    FACE_PREPROCESSING,
    FACE_ROTATION_ANGLES,
    LANDMARK_FACE_PREPROCESSING,
    FaceEmbedder,
)
from .redaction import pixelate_faces, redact_entire_frame
from .tracking import FaceState, FaceTrack, create_face_tracker
from .yolo import YOLOFaceDetector


WINDOW_TITLE = "Personalized Privacy Filter (Q/Esc to quit)"
LIGHTING_SEVERITY = {
    LightingMode.NORMAL.value: 0,
    LightingMode.LOW_LIGHT.value: 1,
    LightingMode.OVEREXPOSED.value: 1,
}


@dataclass(frozen=True)
class ProcessedFrame:
    output: np.ndarray
    detector_ms: float
    recognition_ms: float
    recognition_calls: int
    visible_tracks: int
    lighting_modes: tuple[str, ...]
    processing_ms: float


def _parse_video_output_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width = int(width_text)
        height = int(height_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "video output size must have the form WIDTHxHEIGHT, for example 1920x1080"
        ) from error
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("video output width and height must be positive")
    return width, height


def _create_video_writer(
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> tuple[cv2.VideoWriter, Path]:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f"{output_path.stem}.part{output_path.suffix or '.mp4'}"
    )
    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not create output video: {output_path}")
    return writer, temporary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-time personalized face privacy filter."
    )
    parser.add_argument("--template", default="data/enrollments/owner.npz")
    parser.add_argument(
        "--detector-model",
        "--detector",
        default="yolo11",
        help=detector_model_help(),
    )
    parser.add_argument(
        "--recognition-model",
        "--model",
        default="r34-glint360k",
        help=recognition_model_help(),
    )
    parser.add_argument("--provider", choices=PROVIDER_CHOICES, default="auto")
    parser.add_argument("--threshold", type=float, default=None, help="Override template threshold")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument(
        "--offline-video",
        action="store_true",
        help="Process a video file offline instead of using the camera",
    )
    parser.add_argument(
        "--realtime-video",
        action="store_true",
        help="Process a video file with the real-time face-recognition pipeline",
    )
    parser.add_argument(
        "--offline-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="PyTorch device for offline processing; auto selects CUDA when available",
    )
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument(
        "--video-prompt",
        type=str,
        action="append",
        default=None,
        help=(
            "Open-vocabulary concept prompt; repeat it to redact multiple concepts "
            "in one invocation"
        ),
    )
    parser.add_argument("--video-output", type=Path, default=None)
    parser.add_argument(
        "--video-output-size",
        type=_parse_video_output_size,
        default=None,
        metavar="WIDTHxHEIGHT",
        help="Resize the redacted output video, for example 1920x1080",
    )
    parser.add_argument(
        "--grounding-model",
        default="IDEA-Research/grounding-dino-tiny",
    )
    parser.add_argument("--grounding-box-threshold", type=float, default=0.20)
    parser.add_argument("--grounding-text-threshold", type=float, default=0.20)
    parser.add_argument(
        "--grounding-redetect-interval",
        type=int,
        default=25,
        help="Run Grounding DINO every N frames; SAM 2.1 tracks between runs",
    )
    parser.add_argument(
        "--video-inference-max-side",
        type=int,
        default=1280,
        help="Resize model input so its longest side is this size; 0 keeps source size",
    )
    parser.add_argument(
        "--sam2-model",
        default="facebook/sam2.1-hiera-small",
    )
    parser.add_argument("--sam2-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--sam2-model-config",
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
    )
    parser.add_argument("--video-pixel-block-size", type=int, default=16)
    parser.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Mirror frames; enabled by default for the webcam and disabled for files",
    )
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the processed stream in a window; enabled by default",
    )
    parser.add_argument(
        "--rotations",
        action="store_true",
        help="Use a template enrolled with 30/90/180/270/330-degree rotations",
    )
    parser.add_argument(
        "--authorized-recheck-interval",
        type=int,
        default=0,
        help=(
            "Optional safety recheck for a stable authorized track in frames; "
            "disabled by default"
        ),
    )
    parser.add_argument(
        "--minimum-recognition-face-size",
        type=float,
        default=80.0,
        help="Minimum bbox side in pixels before recognition is attempted",
    )
    parser.add_argument(
        "--minimum-authorized-face-size",
        type=float,
        default=56.0,
        help="Minimum bbox side in pixels for keeping an authorized track visible",
    )
    parser.add_argument(
        "--unknown-retry-interval",
        type=int,
        default=30,
        help="Initial UNKNOWN retry delay in frames before exponential backoff",
    )
    parser.add_argument("--recognition-stable-frames", type=int, default=3)
    parser.add_argument("--recognition-edge-margin", type=float, default=0.05)
    parser.add_argument(
        "--confirmations",
        type=int,
        default=3,
        help="Consecutive positive recognition checks required before reveal",
    )
    parser.add_argument(
        "--detector-threshold",
        type=float,
        default=None,
        help="Detector threshold; defaults to 0.10 for Ultralytics trackers and 0.25 for IoU",
    )
    parser.add_argument(
        "--tracker",
        choices=("bytetrack", "botsort", "iou"),
        default="bytetrack",
        help="Tracking backend; ByteTrack and BoT-SORT are provided by Ultralytics",
    )
    parser.add_argument("--tracker-buffer", type=int, default=30)
    parser.add_argument(
        "--lighting-padding",
        type=float,
        default=0.25,
        help="Padding around the face bbox used as an ambient-light ring",
    )
    parser.add_argument(
        "--lighting-ema-alpha",
        type=float,
        default=0.20,
        help="EMA weight for current-frame lighting measurements",
    )
    parser.add_argument(
        "--enrollment-has-difficult-lighting",
        action="store_true",
        help=(
            "Enrollment photos include dark or overexposed examples; use the "
            "normal authorization threshold in degraded lighting"
        ),
    )
    parser.add_argument(
        "--difficult-lighting-threshold-increase",
        type=float,
        default=0.10,
        help=(
            "Threshold increase in LOW_LIGHT and OVEREXPOSED when enrollment "
            "does not contain difficult-lighting photos"
        ),
    )
    parser.add_argument("--track-iou-threshold", type=float, default=0.25)
    parser.add_argument("--authorization-iou-threshold", type=float, default=0.40)
    parser.add_argument("--track-max-missed", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--benchmark-out", default="benchmarks/latest.json")
    return parser


def _distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(array.mean()), 3),
        "median": round(float(np.median(array)), 3),
        "p95": round(float(np.percentile(array, 95)), 3),
        "p99": round(float(np.percentile(array, 99)), 3),
        "min": round(float(array.min()), 3),
        "max": round(float(array.max()), 3),
    }


def _update_ema(previous: float | None, current: float, alpha: float) -> float:
    return current if previous is None else (1.0 - alpha) * previous + alpha * current


def _is_near_frame_edge(
    track: FaceTrack,
    frame_width: int,
    frame_height: int,
    edge_margin_ratio: float,
) -> bool:
    detection = track.detection
    margin = max(2.0, track.face_size * edge_margin_ratio)
    return bool(
        detection[0] <= margin
        or detection[1] <= margin
        or frame_width - detection[2] <= margin
        or frame_height - detection[3] <= margin
    )


def _unmirror_detection(detection: np.ndarray, frame_width: int) -> np.ndarray:
    restored = np.asarray(detection, dtype=np.float32).copy()
    restored[0] = frame_width - detection[2]
    restored[2] = frame_width - detection[0]
    if restored.size >= 20:
        landmarks = restored[5:20].reshape(5, 3).copy()
        landmarks[:, 0] = frame_width - 1 - landmarks[:, 0]
        landmarks = landmarks[[1, 0, 2, 4, 3]]
        restored[5:20] = landmarks.reshape(-1)
    return restored


def _recognition_gate(
    track: FaceTrack,
    frame_width: int,
    frame_height: int,
    minimum_face_size: float,
    stable_frames: int,
    edge_margin_ratio: float,
) -> str | None:
    if track.overlap_uncertain:
        return "overlap_uncertain"
    if _is_near_frame_edge(
        track,
        frame_width,
        frame_height,
        edge_margin_ratio,
    ):
        return "face_near_frame_edge"
    if track.face_size < minimum_face_size:
        return "face_too_small"
    required_matches = max(0, stable_frames - 1)
    if not track.tracking_confident or track.stable_matches < required_matches:
        return "track_stabilizing"
    return None


def _authorization_size_requires_revoke(
    track: FaceTrack,
    minimum_authorized_face_size: float,
    near_frame_edge: bool,
) -> bool:
    return (
        track.authorized
        and track.face_size < minimum_authorized_face_size
        and not near_frame_edge
    )


def _draw_label(
    frame: np.ndarray,
    track: FaceTrack,
    identity: str,
    confirmations: int,
    minimum_face_size: float,
) -> None:
    detection = track.detection
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, int(detection[0])))
    y1 = max(0, min(height - 1, int(detection[1])))
    x2 = max(0, min(width - 1, int(detection[2])))
    y2 = max(0, min(height - 1, int(detection[3])))
    if track.state is FaceState.AUTHORIZED:
        color = (70, 230, 70)
        label = f"#{track.track_id} {identity}"
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
    if track.matching_centroid_index is not None:
        label += f" IDX:{track.matching_centroid_index}"
        if track.matching_rotation_angle is not None:
            label += f" R:{track.matching_rotation_angle}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text_y = max(22, y1 - 8)
    cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _draw_metrics(
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
        f"Detector {detector_ms:5.1f} ms  Recognition {recognition_ms:5.1f} ms ({recognition_calls} calls)",
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
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 230, 80), 2, cv2.LINE_AA)
        y += 27


def _validate_args(args: argparse.Namespace, threshold: float) -> None:
    if not 0.0 < threshold < 1.0:
        raise ValueError("Authorization threshold must be between 0 and 1")
    if args.authorized_recheck_interval < 0:
        raise ValueError("--authorized-recheck-interval cannot be negative")
    if args.lighting_padding <= 0.0:
        raise ValueError("--lighting-padding must be positive")
    if not 0.0 < args.lighting_ema_alpha <= 1.0:
        raise ValueError("--lighting-ema-alpha must be in (0, 1]")
    if args.difficult_lighting_threshold_increase < 0.0:
        raise ValueError(
            "--difficult-lighting-threshold-increase cannot be negative"
        )
    if (
        not args.enrollment_has_difficult_lighting
        and threshold + args.difficult_lighting_threshold_increase >= 1.0
    ):
        raise ValueError("Difficult-lighting authorization threshold must be below 1")
    if args.minimum_recognition_face_size <= 0:
        raise ValueError("--minimum-recognition-face-size must be positive")
    if not 0 < args.minimum_authorized_face_size <= args.minimum_recognition_face_size:
        raise ValueError(
            "--minimum-authorized-face-size must be positive and no greater than "
            "--minimum-recognition-face-size"
        )
    if args.unknown_retry_interval < 1:
        raise ValueError("--unknown-retry-interval must be at least 1")
    if args.recognition_stable_frames < 1:
        raise ValueError("--recognition-stable-frames must be at least 1")
    if not 0.0 <= args.recognition_edge_margin < 0.5:
        raise ValueError("--recognition-edge-margin must be between 0 and 0.5")
    if not 0.0 < args.detector_threshold < 1.0:
        raise ValueError("--detector-threshold must be between 0 and 1")
    if args.confirmations < 1:
        raise ValueError("--confirmations must be at least 1")
    if args.track_max_missed < 0:
        raise ValueError("--track-max-missed cannot be negative")
    if args.tracker_buffer < 1:
        raise ValueError("--tracker-buffer must be at least 1")


def run(args: argparse.Namespace) -> dict[str, object]:
    template = load_template(args.template)
    models = prepare_runtime_models(
        args.detector_model,
        args.recognition_model,
        args.provider,
    )
    if models.generated:
        print("Preparing optimized runtime model cache:")
        for path in models.generated:
            print(f"  {path}")
    recognition_matches = template.model_sha256 == models.recognition_source_sha256
    if not recognition_matches and models.recognition_runtime != models.recognition_source:
        recognition_matches = template.model_sha256 == sha256_file(
            models.recognition_runtime
        )
    if not recognition_matches:
        raise ValueError(
            "Enrollment was created with a different recognition model. "
            "Re-run privacy-enroll with the current model."
        )
    template_angles = tuple(
        int(value) for value in template.metadata.get("rotation_angles", [0])
    )
    expected_angles = FACE_ROTATION_ANGLES if args.rotations else (0,)
    if template_angles != expected_angles:
        mode = "with --rotations" if args.rotations else "without --rotations"
        raise ValueError(
            f"Enrollment rotation mode does not match this launch. "
            f"Create and select a template enrolled {mode}."
        )
    threshold = template.threshold if args.threshold is None else args.threshold
    if args.detector_threshold is None:
        args.detector_threshold = 0.25 if args.tracker == "iou" else 0.10
    _validate_args(args, threshold)
    difficult_lighting_threshold = (
        threshold
        if args.enrollment_has_difficult_lighting
        else threshold + args.difficult_lighting_threshold_increase
    )

    detector = YOLOFaceDetector(
        models.detector_runtime,
        threshold=args.detector_threshold,
        provider=args.provider,
    )
    face_preprocessing = (
        LANDMARK_FACE_PREPROCESSING
        if detector.has_landmarks
        else FACE_PREPROCESSING
    )
    if template.metadata.get("face_preprocessing") != face_preprocessing:
        mode = "yolo11-pose" if detector.has_landmarks else "yolo11"
        raise ValueError(
            "Enrollment preprocessing does not match the selected detector. "
            f"Re-run privacy-enroll with --detector {mode}."
        )
    embedder = FaceEmbedder(models.recognition_runtime, provider=args.provider)
    tracker = create_face_tracker(
        backend_name=args.tracker,
        iou_threshold=args.track_iou_threshold,
        max_missed_frames=args.track_max_missed,
        authorization_iou_threshold=args.authorization_iou_threshold,
        track_buffer=args.tracker_buffer,
    )
    blank = np.zeros((detector.input_size[1], detector.input_size[0], 3), dtype=np.uint8)
    detector.detect(blank)
    detector.detect(blank)
    recognition_warmup = embedder.warmup(2)
    print(f"Identity: {template.name}")
    print(f"Recognition model: {models.recognition_name}")
    print(f"Detector model: {models.detector_name}")
    print(
        "Face preprocessing: "
        f"{'5-point alignment' if detector.has_landmarks else 'bbox crop'}"
    )
    print(f"Template source photos: {template.metadata.get('source_photos', 'unknown')}")
    print(f"Template embeddings: {len(template.embeddings)}")
    print(f"Rotation centroids: {len(template.rotation_centroids)}")
    print(f"Rotation mode: {'enabled' if args.rotations else 'disabled'}")
    print("Template matching: maximum similarity across per-rotation centroids.")
    print(f"Authorization threshold: {threshold:.3f}")
    print(
        "Difficult-lighting enrollment photos: "
        f"{'yes' if args.enrollment_has_difficult_lighting else 'no'}"
    )
    print(
        "LOW_LIGHT/OVEREXPOSED threshold: "
        f"{difficult_lighting_threshold:.3f}"
    )
    print(f"Detector providers: {detector.providers}")
    print(f"Recognition providers: {embedder.providers}")
    print(
        f"Tracker: {tracker.backend_name} "
        f"({tracker.backend_version})"
    )
    for warning in (detector.provider_warning, embedder.provider_warning):
        if warning:
            print(f"Provider warning: {warning}", file=sys.stderr)
    print(f"Recognition warmup: {[round(value, 2) for value in recognition_warmup]} ms")
    print(
        f"Event recognition: minimum face {args.minimum_recognition_face_size:.0f}px, "
        f"{args.recognition_stable_frames} stable frames, {args.confirmations} "
        "confirmations."
    )
    retry_schedule = [
        min(300, args.unknown_retry_interval * (2 ** exponent))
        for exponent in range(5)
    ]
    print(f"UNKNOWN retry backoff: {retry_schedule} frames, then every 300 frames.")
    if args.authorized_recheck_interval:
        print(
            "Stable AUTHORIZED tracks are rechecked every "
            f"{args.authorized_recheck_interval} frames."
        )
    else:
        print("Stable AUTHORIZED tracks are not rechecked while tracking is confident.")
    print("AUTHORIZED tracks are immediately hidden when tracking becomes uncertain.")
    source_kind = "video file" if args.realtime_video else "camera"
    print(f"Pipeline: {source_kind}/UI on main thread, inference on one worker, queue depth 1.")
    print("Privacy rule: PENDING, UNKNOWN, stale, lost, or failed recognition => pixelated.")

    camera = (
        VideoFile(args.video_path)
        if args.realtime_video
        else Camera(args.camera, args.width, args.height, args.camera_fps)
    )
    print(
        f"Input: {camera.info.width}x{camera.info.height} at reported "
        f"{camera.info.fps:.1f} FPS ({camera.info.backend})"
    )
    print(f"Mirror: {'enabled' if args.mirror else 'disabled'}")
    output_path = args.video_output.expanduser().resolve() if args.video_output else None
    output_size = args.video_output_size or (camera.info.width, camera.info.height)
    writer: cv2.VideoWriter | None = None
    temporary_output_path: Path | None = None
    if output_path is not None:
        writer, temporary_output_path = _create_video_writer(
            output_path,
            camera.info.fps,
            output_size,
        )
        print(f"Recording: {output_path} ({output_size[0]}x{output_size[1]})")
    else:
        print("Recording: disabled")
    started = perf_counter()
    recording_started = started
    rolling_ms: deque[float] = deque(maxlen=120)
    detector_latencies: list[float] = []
    recognition_frame_latencies: list[float] = []
    recognition_call_latencies: list[float] = []
    processing_latencies: list[float] = []
    loop_latencies: list[float] = []
    camera_read_latencies: list[float] = []
    inference_wait_latencies: list[float] = []
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    frames = 0
    failures = 0
    recognition_calls = 0
    recognition_skips = 0
    recognition_failures = 0
    recognition_reasons: dict[str, int] = {}
    recognition_skip_reasons: dict[str, int] = {}
    lighting_mode_observations: dict[str, int] = {}
    authorization_grants = 0
    state_revocations = 0
    authorized_observations = 0
    pending_observations = 0
    unknown_observations = 0
    crowded_frames = 0
    interrupted = False
    recorded_frames = 0
    recorded_source_frames = 0

    def process_frame(frame: np.ndarray, frame_index: int) -> ProcessedFrame:
        nonlocal failures
        nonlocal recognition_calls, recognition_skips, recognition_failures
        nonlocal authorization_grants, state_revocations
        nonlocal authorized_observations, pending_observations, unknown_observations
        nonlocal crowded_frames, lighting_mode_observations

        processing_started = perf_counter()
        recognition_frame = frame
        if args.mirror:
            frame = cv2.flip(frame, 1)
        detector_ms = 0.0
        recognition_ms = 0.0
        frame_recognition_calls = 0
        visible_tracks: list[FaceTrack] = []
        try:
            detected = detector.detect(frame)
            detector_ms = detected.latency_ms
            visible_tracks = tracker.update(detected.detections, frame_index, frame)
            if len(visible_tracks) > 1:
                crowded_frames += 1

            frame_height, frame_width = frame.shape[:2]
            for track in visible_tracks:
                lighting = measure_lighting(
                    frame,
                    track.detection,
                    padding=args.lighting_padding,
                )
                previous_lighting_mode = track.lighting_mode
                track.lighting_ambient_median = _update_ema(
                    track.lighting_ambient_median,
                    lighting.ambient_median,
                    args.lighting_ema_alpha,
                )
                track.lighting_face_p90 = _update_ema(
                    track.lighting_face_p90,
                    lighting.face_p90,
                    args.lighting_ema_alpha,
                )
                track.lighting_face_p10 = _update_ema(
                    track.lighting_face_p10,
                    lighting.face_p10,
                    args.lighting_ema_alpha,
                )
                track.lighting_face_black_ratio = _update_ema(
                    track.lighting_face_black_ratio,
                    lighting.face_black_ratio,
                    args.lighting_ema_alpha,
                )
                track.lighting_face_white_ratio = _update_ema(
                    track.lighting_face_white_ratio,
                    lighting.face_white_ratio,
                    args.lighting_ema_alpha,
                )
                lighting_mode = classify_lighting(
                    track.lighting_ambient_median,
                    track.lighting_face_p10,
                    track.lighting_face_p90,
                    track.lighting_face_black_ratio,
                    track.lighting_face_white_ratio,
                )
                track.lighting_mode = lighting_mode.value
                lighting_mode_observations[lighting_mode.value] = (
                    lighting_mode_observations.get(lighting_mode.value, 0) + 1
                )
                effective_threshold = threshold
                if lighting_mode is not LightingMode.NORMAL:
                    effective_threshold = difficult_lighting_threshold
                track.lighting_effective_threshold = effective_threshold

                if previous_lighting_mode not in ("UNKNOWN", lighting_mode.value):
                    if (
                        LIGHTING_SEVERITY[lighting_mode.value]
                        > LIGHTING_SEVERITY[previous_lighting_mode]
                    ):
                        if track.mark_uncertain():
                            state_revocations += 1
                    else:
                        track.verification_required = True

                near_frame_edge = _is_near_frame_edge(
                    track,
                    frame_width,
                    frame_height,
                    args.recognition_edge_margin,
                )
                if (
                    _authorization_size_requires_revoke(
                        track,
                        args.minimum_authorized_face_size,
                        near_frame_edge,
                    )
                    and track.mark_uncertain()
                ):
                    state_revocations += 1
                gate_reason = _recognition_gate(
                    track,
                    frame_width,
                    frame_height,
                    args.minimum_recognition_face_size,
                    args.recognition_stable_frames,
                    args.recognition_edge_margin,
                )
                track.recognition_block_reason = gate_reason
                if gate_reason is not None:
                    recognition_skips += 1
                    recognition_skip_reasons[gate_reason] = (
                        recognition_skip_reasons.get(gate_reason, 0) + 1
                    )
                    continue
                recognition_reason = track.recognition_reason(
                    frame_index,
                    args.minimum_recognition_face_size,
                    args.unknown_retry_interval,
                    args.authorized_recheck_interval,
                )
                if recognition_reason is None:
                    recognition_skips += 1
                    if track.state is FaceState.UNKNOWN:
                        skip_reason = "unknown_waiting_for_change"
                    else:
                        skip_reason = "stable_track"
                    recognition_skip_reasons[skip_reason] = (
                        recognition_skip_reasons.get(skip_reason, 0) + 1
                    )
                    continue

                recognition_calls += 1
                recognition_reasons[recognition_reason] = (
                    recognition_reasons.get(recognition_reason, 0) + 1
                )
                frame_recognition_calls += 1
                score: float | None = None
                matching_centroid_index: int | None = None
                matching_rotation_angle: int | None = None
                try:
                    recognition_detection = (
                        _unmirror_detection(track.detection, frame_width)
                        if args.mirror
                        else track.detection
                    )
                    result = embedder.embed_bbox(
                        recognition_frame,
                        recognition_detection,
                    )
                    recognition_ms += result.latency_ms
                    recognition_call_latencies.append(result.latency_ms)
                    (
                        score,
                        matching_centroid_index,
                        matching_rotation_angle,
                    ) = template.best_rotation_match(result.embedding)
                    if score >= effective_threshold:
                        positive_scores.append(score)
                    else:
                        negative_scores.append(score)
                except Exception as error:
                    recognition_failures += 1
                    print(
                        f"Recognition warning for track #{track.track_id}: {error}",
                        file=sys.stderr,
                    )

                previous, current = track.record_recognition(
                    score,
                    effective_threshold,
                    args.confirmations,
                    frame_index,
                    matching_centroid_index,
                    matching_rotation_angle,
                )
                if previous is not FaceState.AUTHORIZED and current is FaceState.AUTHORIZED:
                    authorization_grants += 1
                elif previous is FaceState.AUTHORIZED and current is not FaceState.AUTHORIZED:
                    state_revocations += 1

            authorized_detection_indexes = {
                track.detection_index
                for track in visible_tracks
                if track.authorized and track.detection_index is not None
            }
            unauthorized = [
                detection
                for detection_index, detection in enumerate(detected.detections)
                if detection_index not in authorized_detection_indexes
            ]
            output = (
                pixelate_faces(frame, np.asarray(unauthorized))
                if unauthorized
                else frame.copy()
            )
        except Exception as error:
            failures += 1
            for track in tracker.tracks.values():
                if track.mark_missed():
                    tracker.revocations += 1
            output = redact_entire_frame(frame)
            visible_tracks = []
            cv2.putText(
                output,
                f"FAIL-CLOSED: {type(error).__name__}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (30, 30, 255),
                2,
                cv2.LINE_AA,
            )

        for track in visible_tracks:
            if track.state is FaceState.AUTHORIZED:
                authorized_observations += 1
            elif track.state is FaceState.PENDING:
                pending_observations += 1
            else:
                unknown_observations += 1
            _draw_label(
                output,
                track,
                template.name,
                args.confirmations,
                args.minimum_recognition_face_size,
            )

        processing_ms = (perf_counter() - processing_started) * 1000.0
        detector_latencies.append(detector_ms)
        recognition_frame_latencies.append(recognition_ms)
        processing_latencies.append(processing_ms)
        return ProcessedFrame(
            output=output,
            detector_ms=detector_ms,
            recognition_ms=recognition_ms,
            recognition_calls=frame_recognition_calls,
            visible_tracks=len(visible_tracks),
            lighting_modes=tuple(track.lighting_mode for track in visible_tracks),
            processing_ms=processing_ms,
        )

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="privacy-inference")
    pending: Future[ProcessedFrame] | None = None
    submitted_frames = 0
    try:
        camera_started = perf_counter()
        first_frame = camera.read()
        camera_read_latencies.append((perf_counter() - camera_started) * 1000.0)
        if first_frame is None:
            raise RuntimeError("Input did not provide any video frames")
        submitted_frames = 1
        pending = executor.submit(process_frame, first_frame, submitted_frames)

        while pending is not None:
            loop_started = perf_counter()
            next_frame: np.ndarray | None = None
            if args.max_frames <= 0 or submitted_frames < args.max_frames:
                camera_started = perf_counter()
                next_frame = camera.read()
                camera_read_latencies.append((perf_counter() - camera_started) * 1000.0)

            inference_wait_started = perf_counter()
            processed = pending.result()
            inference_wait_latencies.append(
                (perf_counter() - inference_wait_started) * 1000.0
            )
            _draw_metrics(
                processed.output,
                rolling_ms,
                processed.detector_ms,
                processed.recognition_ms,
                processed.recognition_calls,
                processed.visible_tracks,
                processed.lighting_modes,
                threshold,
                difficult_lighting_threshold,
                args.confirmations,
                args.authorized_recheck_interval,
                args.minimum_recognition_face_size,
            )
            if writer is not None:
                recorded = processed.output
                if (recorded.shape[1], recorded.shape[0]) != output_size:
                    recorded = cv2.resize(recorded, output_size, interpolation=cv2.INTER_AREA)
                if args.realtime_video:
                    writer.write(recorded)
                    recorded_frames += 1
                else:
                    target_count = max(
                        recorded_frames + 1,
                        int(round((perf_counter() - recording_started) * camera.info.fps)),
                    )
                    while recorded_frames < target_count:
                        writer.write(recorded)
                        recorded_frames += 1
                recorded_source_frames += 1
            key = -1
            if args.preview:
                cv2.imshow(WINDOW_TITLE, processed.output)
                key = cv2.waitKey(1) & 0xFF
            loop_ms = (perf_counter() - loop_started) * 1000.0
            rolling_ms.append(loop_ms)
            loop_latencies.append(loop_ms)
            frames += 1
            if key in (ord("q"), 27):
                pending = None
                break
            if next_frame is None:
                pending = None
                break
            submitted_frames += 1
            pending = executor.submit(process_frame, next_frame, submitted_frames)
    except KeyboardInterrupt:
        interrupted = True
        print("\nStopping; saving the benchmark...")
    finally:
        if pending is not None:
            pending.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        camera.close()
        if writer is not None:
            writer.release()
            if temporary_output_path is not None and output_path is not None:
                os.replace(temporary_output_path, output_path)
        cv2.destroyAllWindows()

    elapsed = perf_counter() - started
    summary: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": 19,
        "identity": template.name,
        "template_samples": len(template.embeddings),
        "template_source_photos": template.metadata.get("source_photos"),
        "template_rotation_centroids": len(template.rotation_centroids),
        "settings": {
            "threshold": threshold,
            "confirmations": args.confirmations,
            "authorized_recheck_interval": args.authorized_recheck_interval,
            "minimum_recognition_face_size": args.minimum_recognition_face_size,
            "minimum_authorized_face_size": args.minimum_authorized_face_size,
            "unknown_retry_interval": args.unknown_retry_interval,
            "unknown_retry_policy": "exponential_backoff_x2_cap_300",
            "recognition_stable_frames": args.recognition_stable_frames,
            "recognition_edge_margin": args.recognition_edge_margin,
            "detector_threshold": args.detector_threshold,
            "lighting_padding": args.lighting_padding,
            "lighting_ema_alpha": args.lighting_ema_alpha,
            "enrollment_has_difficult_lighting": (
                args.enrollment_has_difficult_lighting
            ),
            "difficult_lighting_threshold_increase": (
                args.difficult_lighting_threshold_increase
            ),
            "difficult_lighting_threshold": difficult_lighting_threshold,
            "lighting_policy": "enrollment-aware_conservative-threshold",
            "track_iou_threshold": args.track_iou_threshold,
            "authorization_iou_threshold": args.authorization_iou_threshold,
            "track_max_missed": args.track_max_missed,
            "tracker_backend": tracker.backend_name,
            "tracker_version": tracker.backend_version,
            "tracker_buffer": args.tracker_buffer,
            "tracker_settings": tracker.backend_settings,
            "mirror": args.mirror,
            "preview": args.preview,
            "realtime_video": args.realtime_video,
            "rotations": args.rotations,
            "rotation_angles": list(template_angles),
            "template_matching_policy": "per_rotation_centroid_max",
            "recognition_policy": "size-gated_event-driven_tracker-uncertainty",
            "face_preprocessing": face_preprocessing,
            "pipeline": f"main-thread-{source_kind.replace(' ', '-')}_single-worker-inference",
            "pipeline_queue_depth": 1,
        },
        "detector_providers": detector.providers,
        "recognition_providers": embedder.providers,
        "provider_warnings": [
            warning
            for warning in (detector.provider_warning, embedder.provider_warning)
            if warning
        ],
        "models": {
            "detector_source": str(models.detector_source),
            "detector_runtime": str(models.detector_runtime),
            "detector_name": models.detector_name,
            "detector_landmarks": detector.has_landmarks,
            "recognition_name": models.recognition_name,
            "recognition_source": str(models.recognition_source),
            "recognition_runtime": str(models.recognition_runtime),
            "preferred_execution_provider": models.preferred_execution_provider,
        },
        "frames": frames,
        "input": {
            "kind": source_kind,
            "path": str(args.video_path.resolve()) if args.video_path else None,
            "width": camera.info.width,
            "height": camera.info.height,
            "fps": camera.info.fps,
        },
        "output": {
            "path": str(output_path) if output_path else None,
            "width": output_size[0] if output_path else None,
            "height": output_size[1] if output_path else None,
            "recorded_frames": recorded_frames,
            "recorded_source_frames": recorded_source_frames,
            "recording_timing": (
                "source_fps_one_output_per_input"
                if args.realtime_video
                else "wall_clock_frame_duplication"
            ) if output_path else None,
            "audio_preserved": False,
        },
        "failures": failures,
        "interrupted": interrupted,
        "elapsed_seconds": round(elapsed, 3),
        "effective_fps": round(frames / elapsed, 3) if elapsed else 0.0,
        "detector_latency_ms": _distribution(detector_latencies),
        "recognition_latency_per_frame_ms": _distribution(recognition_frame_latencies),
        "recognition_latency_per_call_ms": _distribution(recognition_call_latencies),
        "processing_latency_ms": _distribution(processing_latencies),
        "camera_read_latency_ms": _distribution(camera_read_latencies),
        "pipeline_inference_wait_ms": _distribution(inference_wait_latencies),
        "loop_latency_ms": _distribution(loop_latencies),
        "recognition_calls": recognition_calls,
        "recognition_skips": recognition_skips,
        "recognition_failures": recognition_failures,
        "recognition_reasons": recognition_reasons,
        "recognition_skip_reasons": recognition_skip_reasons,
        "lighting_mode_observations": lighting_mode_observations,
        "crowded_frames": crowded_frames,
        "state_observations": {
            "authorized": authorized_observations,
            "pending": pending_observations,
            "unknown": unknown_observations,
        },
        "state_transitions": {
            "authorization_grants": authorization_grants,
            "authorization_revocations": tracker.revocations + state_revocations,
        },
        "tracker": {
            "backend": tracker.backend_name,
            "version": tracker.backend_version,
            "created_tracks": tracker.created_tracks,
            "expired_tracks": tracker.expired_tracks,
            "active_tracks_at_exit": len(tracker.tracks),
        },
        "positive_scores": _distribution(positive_scores),
        "negative_scores": _distribution(negative_scores),
    }
    benchmark_path = Path(args.benchmark_out)
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    args = build_parser().parse_args()
    if args.max_frames < 0:
        raise SystemExit("--max-frames cannot be negative")
    try:
        if args.offline_video and args.realtime_video:
            raise ValueError("--offline-video and --realtime-video are mutually exclusive")
        if args.video_output_size is not None and args.video_output is None:
            raise ValueError("--video-output-size requires --video-output")
        if args.offline_video:
            if args.video_path is None:
                raise ValueError("--video-path is required with --offline-video")
            if not args.video_prompt:
                raise ValueError("--video-prompt is required with --offline-video")
            from .grounded_sam2_video import process_video_with_grounded_sam2

            process_video_with_grounded_sam2(
                video_path=args.video_path,
                prompt=args.video_prompt,
                output_path=args.video_output,
                grounding_model_id=args.grounding_model,
                sam2_model_id=args.sam2_model,
                sam2_checkpoint_path=args.sam2_checkpoint,
                sam2_model_config=args.sam2_model_config,
                box_threshold=args.grounding_box_threshold,
                text_threshold=args.grounding_text_threshold,
                redetect_interval=args.grounding_redetect_interval,
                inference_max_side=args.video_inference_max_side,
                max_frames=args.max_frames,
                device=args.offline_device,
                pixel_block_size=args.video_pixel_block_size,
                output_size=args.video_output_size,
            )
            return
        if args.realtime_video:
            if args.video_path is None:
                raise ValueError("--video-path is required with --realtime-video")
            if args.video_output is None:
                raise ValueError("--video-output is required with --realtime-video")
            if args.video_prompt:
                raise ValueError("--video-prompt is only used with --offline-video")
        elif args.video_path is not None:
            raise ValueError("--video-path requires --offline-video or --realtime-video")
        elif args.video_prompt or args.sam2_checkpoint is not None:
            raise ValueError("Grounded SAM 2 options require --offline-video")
        if args.mirror is None:
            args.mirror = not args.realtime_video
        run(args)
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
