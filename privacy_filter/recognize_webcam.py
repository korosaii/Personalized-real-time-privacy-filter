from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

from .camera import Camera
from .enrollment import load_template, sha256_file
from .model_setup import prepare_runtime_models
from .ort_session import PROVIDER_CHOICES
from .recognition import FaceEmbedder
from .redaction import blur_faces, redact_entire_frame
from .scrfd import SCRFDDetector
from .tracking import (
    FaceState,
    FaceTrack,
    FaceTracker,
    recognition_interval_for_state,
)


WINDOW_TITLE = "Personalized Privacy Filter (Q/Esc to quit)"


@dataclass(frozen=True)
class ProcessedFrame:
    output: np.ndarray
    detector_ms: float
    recognition_ms: float
    recognition_calls: int
    visible_tracks: int
    processing_ms: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-time personalized face privacy filter."
    )
    parser.add_argument("--template", default="data/enrollments/owner.npz")
    parser.add_argument("--detector-model", type=Path, default=None)
    parser.add_argument("--recognition-model", type=Path, default=None)
    parser.add_argument("--provider", choices=PROVIDER_CHOICES, default="auto")
    parser.add_argument("--threshold", type=float, default=None, help="Override template threshold")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--blur-padding", type=float, default=0.18)
    parser.add_argument(
        "--recognition-interval",
        type=int,
        default=3,
        help="Recognize a single pending/unknown face every N frames",
    )
    parser.add_argument(
        "--authorized-recheck-interval",
        type=int,
        default=2,
        help="Recheck an authorized single face every N frames",
    )
    parser.add_argument(
        "--crowded-unknown-interval",
        type=int,
        default=5,
        help="Recheck an already rejected track every N crowded frames",
    )
    parser.add_argument(
        "--confirmations",
        type=int,
        default=3,
        help="Consecutive positive recognition checks required before reveal",
    )
    parser.add_argument(
        "--authorization-ttl",
        type=int,
        default=4,
        help="Maximum frames since the last positive check",
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


def _draw_label(frame: np.ndarray, track: FaceTrack, identity: str, confirmations: int) -> None:
    detection = track.detection
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, int(detection[0])))
    y1 = max(0, min(height - 1, int(detection[1])))
    x2 = max(0, min(width - 1, int(detection[2])))
    y2 = max(0, min(height - 1, int(detection[3])))
    if track.state is FaceState.AUTHORIZED:
        color = (70, 230, 70)
        label = f"#{track.track_id} {identity}"
    elif track.state is FaceState.PENDING:
        color = (40, 210, 255)
        label = f"#{track.track_id} PENDING {track.positive_streak}/{confirmations}"
    else:
        color = (40, 70, 255)
        label = f"#{track.track_id} UNKNOWN"
    if track.score is not None:
        label += f" {track.score:.3f}"
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
    threshold: float,
    confirmations: int,
    interval: int,
    authorized_interval: int,
) -> None:
    fps = 1000.0 / float(np.mean(rolling_ms)) if rolling_ms else 0.0
    lines = (
        f"FPS {fps:5.1f}",
        f"Detector {detector_ms:5.1f} ms  Recognition {recognition_ms:5.1f} ms ({recognition_calls} calls)",
        (
            f"Tracks {visible_tracks}  threshold {threshold:.3f}  "
            f"confirm {confirmations}  intervals {interval}/{authorized_interval}"
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
    if args.recognition_interval < 1:
        raise ValueError("--recognition-interval must be at least 1")
    if args.authorized_recheck_interval < 1:
        raise ValueError("--authorized-recheck-interval must be at least 1")
    if args.crowded_unknown_interval < 1:
        raise ValueError("--crowded-unknown-interval must be at least 1")
    if args.confirmations < 1:
        raise ValueError("--confirmations must be at least 1")
    if args.authorization_ttl < args.authorized_recheck_interval:
        raise ValueError(
            "--authorization-ttl must be at least --authorized-recheck-interval"
        )
    if args.track_max_missed < 0:
        raise ValueError("--track-max-missed cannot be negative")


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
    threshold = template.threshold if args.threshold is None else args.threshold
    _validate_args(args, threshold)

    detector = SCRFDDetector(
        models.detector_runtime,
        input_size=(512, 512),
        threshold=0.40,
        provider=args.provider,
    )
    embedder = FaceEmbedder(models.recognition_runtime, provider=args.provider)
    tracker = FaceTracker(
        iou_threshold=args.track_iou_threshold,
        max_missed_frames=args.track_max_missed,
        authorization_iou_threshold=args.authorization_iou_threshold,
    )
    blank = np.zeros((512, 512, 3), dtype=np.uint8)
    detector.detect(blank)
    detector.detect(blank)
    recognition_warmup = embedder.warmup(2)
    print(f"Identity: {template.name}")
    print(f"Template samples: {len(template.embeddings)}")
    print(f"Authorization threshold: {threshold:.3f}")
    print(f"Detector providers: {detector.providers}")
    print(f"Recognition providers: {embedder.providers}")
    for warning in (detector.provider_warning, embedder.provider_warning):
        if warning:
            print(f"Provider warning: {warning}", file=sys.stderr)
    print(f"Recognition warmup: {[round(value, 2) for value in recognition_warmup]} ms")
    print(
        f"Safe state: {args.confirmations} confirmations, recognition every "
        f"{args.recognition_interval} frames while blurred, recheck every "
        f"{args.authorized_recheck_interval} frames after reveal, authorization TTL "
        f"{args.authorization_ttl} frames."
    )
    print(
        "Crowded schedule: AUTHORIZED/PENDING every frame, confirmed UNKNOWN every "
        f"{args.crowded_unknown_interval} frames."
    )
    print("Pipeline: camera/UI on main thread, inference on one worker, queue depth 1.")
    print("Privacy rule: PENDING, UNKNOWN, stale, lost, or failed recognition => blurred.")

    camera = Camera(args.camera, args.width, args.height, args.camera_fps)
    print(
        f"Camera: {camera.info.width}x{camera.info.height} at reported "
        f"{camera.info.fps:.1f} FPS ({camera.info.backend})"
    )
    started = perf_counter()
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
    authorization_grants = 0
    state_revocations = 0
    authorized_observations = 0
    pending_observations = 0
    unknown_observations = 0
    crowded_frames = 0
    interrupted = False

    def process_frame(frame: np.ndarray, frame_index: int) -> ProcessedFrame:
        nonlocal failures
        nonlocal recognition_calls, recognition_skips, recognition_failures
        nonlocal authorization_grants, state_revocations
        nonlocal authorized_observations, pending_observations, unknown_observations
        nonlocal crowded_frames

        processing_started = perf_counter()
        detector_ms = 0.0
        recognition_ms = 0.0
        frame_recognition_calls = 0
        visible_tracks: list[FaceTrack] = []
        try:
            detected = detector.detect(frame)
            detector_ms = detected.latency_ms
            visible_tracks = tracker.update(
                detected.detections,
                detected.keypoints,
                frame_index,
            )
            if len(visible_tracks) > 1:
                crowded_frames += 1

            for track in visible_tracks:
                if track.expire_authorization(frame_index, args.authorization_ttl):
                    state_revocations += 1
                recognition_interval = recognition_interval_for_state(
                    track.state,
                    len(visible_tracks),
                    args.recognition_interval,
                    args.authorized_recheck_interval,
                    args.crowded_unknown_interval,
                )
                if not track.should_recognize(frame_index, recognition_interval):
                    recognition_skips += 1
                    continue

                recognition_calls += 1
                frame_recognition_calls += 1
                score: float | None = None
                try:
                    if track.keypoints is None:
                        raise ValueError("Missing five-point landmarks")
                    result = embedder.embed(frame, track.keypoints)
                    recognition_ms += result.latency_ms
                    recognition_call_latencies.append(result.latency_ms)
                    score = template.score(result.embedding)
                    if score >= threshold:
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
                    threshold,
                    args.confirmations,
                    frame_index,
                )
                if previous is not FaceState.AUTHORIZED and current is FaceState.AUTHORIZED:
                    authorization_grants += 1
                elif previous is FaceState.AUTHORIZED and current is not FaceState.AUTHORIZED:
                    state_revocations += 1

            unauthorized = [
                track.detection for track in visible_tracks if not track.authorized
            ]
            output = (
                blur_faces(frame, np.asarray(unauthorized), padding=args.blur_padding)
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
            _draw_label(output, track, template.name, args.confirmations)

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
            processing_ms=processing_ms,
        )

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="privacy-inference")
    pending: Future[ProcessedFrame] | None = None
    submitted_frames = 0
    try:
        camera_started = perf_counter()
        first_frame = camera.read()
        camera_read_latencies.append((perf_counter() - camera_started) * 1000.0)
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
                threshold,
                args.confirmations,
                args.recognition_interval,
                args.authorized_recheck_interval,
            )
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
        cv2.destroyAllWindows()

    elapsed = perf_counter() - started
    summary: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": 1,
        "identity": template.name,
        "template_samples": len(template.embeddings),
        "settings": {
            "threshold": threshold,
            "confirmations": args.confirmations,
            "recognition_interval": args.recognition_interval,
            "authorized_recheck_interval": args.authorized_recheck_interval,
            "crowded_pending_interval": 1,
            "crowded_authorized_interval": 1,
            "crowded_unknown_interval": args.crowded_unknown_interval,
            "authorization_ttl": args.authorization_ttl,
            "track_iou_threshold": args.track_iou_threshold,
            "authorization_iou_threshold": args.authorization_iou_threshold,
            "track_max_missed": args.track_max_missed,
            "pipeline": "main-thread-camera_single-worker-inference",
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
            "recognition_source": str(models.recognition_source),
            "recognition_runtime": str(models.recognition_runtime),
            "preferred_execution_provider": models.preferred_execution_provider,
        },
        "frames": frames,
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
        run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
