from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Iterable

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from privacy_filter.redaction import pixelate_faces  # noqa: E402
from privacy_filter.yolo import YOLOFaceDetector  # noqa: E402


@dataclass(frozen=True)
class GroundTruthFace:
    person_id: str
    left_eye: tuple[float, float]
    right_eye: tuple[float, float]

    @property
    def eye_distance(self) -> float:
        return math.dist(self.left_eye, self.right_eye)


@dataclass(frozen=True)
class SequenceInput:
    name: str
    frame_directory: Path
    groundtruth_path: Path
    frames: tuple[tuple[str, GroundTruthFace | None], ...]


@dataclass(frozen=True)
class FaceEvaluation:
    both_eyes_covered: bool
    zone_coverage: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark face redaction on ChokePoint frames using eye annotations. "
            "The command writes a JSON summary, a frame-level CSV, and a failure montage."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--sequence",
        default="P1E_S1",
        help="ChokePoint group to test, for example P1E_S1 (all available cameras)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/detector/yolov11n-face.onnx"),
    )
    parser.add_argument(
        "--provider",
        choices=("auto", "cpu", "coreml", "directml", "cuda"),
        default="auto",
    )
    parser.add_argument("--detector-threshold", type=float, default=0.25)
    parser.add_argument(
        "--minimum-eye-distance",
        type=float,
        default=10.0,
        help="Grade faces with at least this inter-eye distance; smaller faces are reported separately",
    )
    parser.add_argument(
        "--minimum-zone-coverage",
        type=float,
        default=0.0,
        help=(
            "Optional minimum fraction of the approximate eye-derived face zone; "
            "0 disables this heuristic and grades only the exact eye annotations"
        ),
    )
    parser.add_argument("--minimum-privacy-recall", type=float, default=0.99)
    parser.add_argument("--maximum-leak-streak", type=int, default=2)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Evaluate every Nth XML frame; use 1 for the real benchmark",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames across all cameras; 0 processes everything",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("benchmarks/chokepoint_privacy"),
    )
    parser.add_argument("--failure-images", type=int, default=12)
    args = parser.parse_args()
    if not 0.0 < args.detector_threshold < 1.0:
        parser.error("--detector-threshold must be between 0 and 1")
    if args.minimum_eye_distance < 0.0:
        parser.error("--minimum-eye-distance cannot be negative")
    if not 0.0 <= args.minimum_zone_coverage <= 1.0:
        parser.error("--minimum-zone-coverage must be between 0 and 1")
    if not 0.0 <= args.minimum_privacy_recall <= 1.0:
        parser.error("--minimum-privacy-recall must be between 0 and 1")
    if args.maximum_leak_streak < 0:
        parser.error("--maximum-leak-streak cannot be negative")
    if args.target_fps <= 0.0:
        parser.error("--target-fps must be positive")
    if args.frame_step <= 0:
        parser.error("--frame-step must be positive")
    if args.max_frames < 0:
        parser.error("--max-frames cannot be negative")
    if args.failure_images < 0:
        parser.error("--failure-images cannot be negative")
    return args


def _groundtruth_root(data_root: Path) -> Path:
    candidates = (data_root / "groundtruth" / "groundtruth", data_root / "groundtruth")
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.xml")):
            return candidate
    raise FileNotFoundError(
        f"Could not find ChokePoint XML files below {data_root / 'groundtruth'}"
    )


def _frame_directory(group_root: Path, sequence_name: str) -> Path:
    direct = group_root / sequence_name
    candidates = [direct, direct / sequence_name]
    candidates.extend(
        candidate
        for candidate in group_root.rglob(sequence_name)
        if candidate.is_dir()
    )
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.jpg")):
            return candidate
    raise FileNotFoundError(
        f"Could not find extracted JPG frames for {sequence_name} below {group_root}"
    )


def _parse_groundtruth(path: Path) -> tuple[tuple[str, GroundTruthFace | None], ...]:
    root = ET.parse(path).getroot()
    parsed: list[tuple[str, GroundTruthFace | None]] = []
    for frame in root.findall("frame"):
        number = frame.attrib["number"]
        person = frame.find("person")
        if person is None:
            parsed.append((number, None))
            continue
        left = person.find("leftEye")
        right = person.find("rightEye")
        if left is None or right is None:
            raise ValueError(f"Incomplete eye annotation in {path}, frame {number}")
        parsed.append(
            (
                number,
                GroundTruthFace(
                    person_id=person.attrib.get("id", "unknown"),
                    left_eye=(float(left.attrib["x"]), float(left.attrib["y"])),
                    right_eye=(float(right.attrib["x"]), float(right.attrib["y"])),
                ),
            )
        )
    return tuple(parsed)


def discover_sequences(data_root: Path, group: str) -> tuple[SequenceInput, ...]:
    resolved_data = data_root.expanduser().resolve()
    group_root = resolved_data / group
    if not group_root.is_dir():
        raise FileNotFoundError(f"ChokePoint group directory not found: {group_root}")
    groundtruth_root = _groundtruth_root(resolved_data)
    xml_paths = sorted(groundtruth_root.glob(f"{group}_C*.xml"))
    if not xml_paths:
        raise FileNotFoundError(
            f"No ground truth matching {group}_C*.xml in {groundtruth_root}"
        )
    sequences = []
    for xml_path in xml_paths:
        name = xml_path.stem
        sequences.append(
            SequenceInput(
                name=name,
                frame_directory=_frame_directory(group_root, name),
                groundtruth_path=xml_path,
                frames=_parse_groundtruth(xml_path),
            )
        )
    return tuple(sequences)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": float(mean(values)) if values else None,
        "p50": _percentile(values, 50.0),
        "p95": _percentile(values, 95.0),
        "p99": _percentile(values, 99.0),
        "max": max(values) if values else None,
    }


def _inside_redaction_ellipse(point: tuple[float, float], detection: np.ndarray) -> bool:
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


def evaluate_face(face: GroundTruthFace, detections: np.ndarray) -> FaceEvaluation:
    both_eyes_covered = any(
        _inside_redaction_ellipse(face.left_eye, detection)
        and _inside_redaction_ellipse(face.right_eye, detection)
        for detection in detections
    )

    # ChokePoint provides eye points rather than a face mask. This conservative,
    # deterministic ellipse approximates the privacy-sensitive face zone. The
    # exact eye-point recall above remains available without this approximation.
    distance = max(face.eye_distance, 1.0)
    midpoint_x = (face.left_eye[0] + face.right_eye[0]) / 2.0
    midpoint_y = (face.left_eye[1] + face.right_eye[1]) / 2.0
    center_x = midpoint_x
    center_y = midpoint_y + 0.50 * distance
    radius_x = 1.35 * distance
    radius_y = 1.70 * distance

    axis = np.linspace(-1.0, 1.0, 41, dtype=np.float32)
    normalized_x, normalized_y = np.meshgrid(axis, axis)
    inside = normalized_x**2 + normalized_y**2 <= 1.0
    sample_x = center_x + normalized_x[inside] * radius_x
    sample_y = center_y + normalized_y[inside] * radius_y
    covered = np.zeros(sample_x.shape, dtype=bool)
    for detection in detections:
        x1, y1, x2, y2 = (float(value) for value in detection[:4])
        detection_radius_x = max((x2 - x1) / 2.0, 1e-6)
        detection_radius_y = max((y2 - y1) / 2.0, 1e-6)
        detection_center_x = (x1 + x2) / 2.0
        detection_center_y = (y1 + y2) / 2.0
        covered |= (
            ((sample_x - detection_center_x) / detection_radius_x) ** 2
            + ((sample_y - detection_center_y) / detection_radius_y) ** 2
            <= 1.0
        )
    return FaceEvaluation(
        both_eyes_covered=both_eyes_covered,
        zone_coverage=float(np.mean(covered)) if covered.size else 0.0,
    )


def _redacted_area_fraction(detections: np.ndarray, width: int, height: int) -> float:
    if not len(detections):
        return 0.0
    area = 0.0
    for detection in detections:
        x1, y1, x2, y2 = (float(value) for value in detection[:4])
        area += math.pi * max(0.0, x2 - x1) * max(0.0, y2 - y1) / 4.0
    return min(1.0, area / float(width * height))


def _draw_groundtruth(
    image: np.ndarray,
    face: GroundTruthFace,
    coverage: float,
    passed: bool,
) -> None:
    distance = max(face.eye_distance, 1.0)
    midpoint = (
        int(round((face.left_eye[0] + face.right_eye[0]) / 2.0)),
        int(round((face.left_eye[1] + face.right_eye[1]) / 2.0 + 0.50 * distance)),
    )
    color = (30, 200, 30) if passed else (20, 20, 240)
    cv2.ellipse(
        image,
        midpoint,
        (int(round(1.35 * distance)), int(round(1.70 * distance))),
        0.0,
        0.0,
        360.0,
        color,
        2,
        cv2.LINE_AA,
    )
    for point in (face.left_eye, face.right_eye):
        cv2.circle(image, (int(point[0]), int(point[1])), 3, color, -1, cv2.LINE_AA)
    cv2.putText(
        image,
        f"GT coverage {coverage:.1%}",
        (max(4, midpoint[0] - 70), max(18, midpoint[1] - int(1.70 * distance) - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def _failure_montage(
    failures: list[dict[str, object]],
    output_path: Path,
    minimum_coverage: float,
) -> None:
    if not failures:
        output_path.unlink(missing_ok=True)
        return
    tiles: list[np.ndarray] = []
    for failure in failures:
        frame = cv2.imread(str(failure["path"]), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        detections = np.asarray(failure["detections"], dtype=np.float32)
        face = failure["face"]
        assert isinstance(face, GroundTruthFace)
        redacted = pixelate_faces(frame, detections)
        for detection in detections:
            x1, y1, x2, y2 = (int(round(value)) for value in detection[:4])
            cv2.rectangle(redacted, (x1, y1), (x2, y2), (255, 180, 0), 1)
        coverage = float(failure["coverage"])
        _draw_groundtruth(redacted, face, coverage, coverage >= minimum_coverage)
        cv2.putText(
            redacted,
            f"{failure['sequence']} frame {failure['frame_number']}",
            (8, redacted.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        target_width = 400
        target_height = int(round(redacted.shape[0] * target_width / redacted.shape[1]))
        tiles.append(cv2.resize(redacted, (target_width, target_height)))
    if not tiles:
        return
    columns = min(3, len(tiles))
    rows = math.ceil(len(tiles) / columns)
    tile_height, tile_width = tiles[0].shape[:2]
    canvas = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        canvas[
            row * tile_height : (row + 1) * tile_height,
            column * tile_width : (column + 1) * tile_width,
        ] = tile
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Could not write failure montage: {output_path}")


def _longest_leak_streak(rows: Iterable[dict[str, object]]) -> int:
    longest = 0
    current = 0
    previous_sequence: str | None = None
    previous_frame: int | None = None
    for row in rows:
        if not row["graded_face"]:
            current = 0
            previous_sequence = str(row["sequence"])
            previous_frame = int(row["frame_number"])
            continue
        sequence = str(row["sequence"])
        frame_number = int(row["frame_number"])
        consecutive = sequence == previous_sequence and (
            previous_frame is not None and frame_number == previous_frame + 1
        )
        if not consecutive:
            current = 0
        current = current + 1 if row["privacy_leak"] else 0
        longest = max(longest, current)
        previous_sequence = sequence
        previous_frame = frame_number
    return longest


def run(args: argparse.Namespace) -> dict[str, object]:
    sequences = discover_sequences(args.data_root, args.sequence)
    model_path = args.model.expanduser().resolve()
    detector = YOLOFaceDetector(
        model_path,
        threshold=args.detector_threshold,
        provider=args.provider,
    )
    blank = np.zeros((detector.input_size[1], detector.input_size[0], 3), dtype=np.uint8)
    detector.detect(blank)
    detector.detect(blank)

    rows: list[dict[str, object]] = []
    detector_latencies: list[float] = []
    redaction_latencies: list[float] = []
    pipeline_latencies: list[float] = []
    read_latencies: list[float] = []
    failure_candidates: list[dict[str, object]] = []
    missing_frames: list[str] = []
    processed = 0

    for sequence in sequences:
        for ordinal, (frame_number_text, face) in enumerate(sequence.frames):
            if ordinal % args.frame_step:
                continue
            if args.max_frames and processed >= args.max_frames:
                break
            frame_path = sequence.frame_directory / f"{frame_number_text}.jpg"
            read_started = perf_counter()
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            read_ms = (perf_counter() - read_started) * 1000.0
            if frame is None:
                missing_frames.append(str(frame_path))
                continue

            detected = detector.detect(frame)
            redaction_started = perf_counter()
            pixelate_faces(frame, detected.detections)
            redaction_ms = (perf_counter() - redaction_started) * 1000.0
            pipeline_ms = detected.latency_ms + redaction_ms
            read_latencies.append(read_ms)
            detector_latencies.append(detected.latency_ms)
            redaction_latencies.append(redaction_ms)
            pipeline_latencies.append(pipeline_ms)

            eye_distance = face.eye_distance if face is not None else None
            graded_face = face is not None and eye_distance >= args.minimum_eye_distance
            evaluation = evaluate_face(face, detected.detections) if face is not None else None
            privacy_leak = bool(
                graded_face
                and evaluation is not None
                and (
                    not evaluation.both_eyes_covered
                    or (
                        args.minimum_zone_coverage > 0.0
                        and evaluation.zone_coverage < args.minimum_zone_coverage
                    )
                )
            )
            row: dict[str, object] = {
                "sequence": sequence.name,
                "frame_number": int(frame_number_text),
                "frame_path": str(frame_path),
                "person_id": face.person_id if face is not None else "",
                "has_groundtruth_face": face is not None,
                "eye_distance_px": round(eye_distance, 4) if eye_distance is not None else "",
                "graded_face": graded_face,
                "detections": len(detected.detections),
                "both_eyes_covered": (
                    evaluation.both_eyes_covered if evaluation is not None else ""
                ),
                "face_zone_coverage": (
                    round(evaluation.zone_coverage, 6) if evaluation is not None else ""
                ),
                "privacy_leak": privacy_leak,
                "false_positive_proxy": face is None and len(detected.detections) > 0,
                "redacted_area_fraction": round(
                    _redacted_area_fraction(
                        detected.detections, frame.shape[1], frame.shape[0]
                    ),
                    6,
                ),
                "read_ms": round(read_ms, 4),
                "detector_ms": round(detected.latency_ms, 4),
                "redaction_ms": round(redaction_ms, 4),
                "pipeline_ms": round(pipeline_ms, 4),
            }
            rows.append(row)
            if privacy_leak and face is not None and evaluation is not None:
                failure_candidates.append(
                    {
                        "coverage": evaluation.zone_coverage,
                        "eyes_covered": evaluation.both_eyes_covered,
                        "sequence": sequence.name,
                        "frame_number": frame_number_text,
                        "path": frame_path,
                        "detections": detected.detections.tolist(),
                        "face": face,
                    }
                )
            processed += 1
        if args.max_frames and processed >= args.max_frames:
            break

    graded = [row for row in rows if row["graded_face"]]
    annotated = [row for row in rows if row["has_groundtruth_face"]]
    small = [
        row
        for row in annotated
        if isinstance(row["eye_distance_px"], float)
        and row["eye_distance_px"] < args.minimum_eye_distance
    ]
    no_face = [row for row in rows if not row["has_groundtruth_face"]]
    private = [row for row in graded if not row["privacy_leak"]]
    eye_covered = [row for row in graded if row["both_eyes_covered"]]
    leaks = [row for row in graded if row["privacy_leak"]]
    false_positive_frames = [row for row in no_face if row["false_positive_proxy"]]
    privacy_recall = len(private) / len(graded) if graded else 0.0
    eye_recall = len(eye_covered) / len(graded) if graded else 0.0
    max_leak_streak = _longest_leak_streak(rows)
    budget_ms = 1000.0 / args.target_fps
    pipeline_p95 = _percentile(pipeline_latencies, 95.0)
    speed_pass = pipeline_p95 is not None and pipeline_p95 <= budget_ms
    checks = {
        "dataset_integrity": not missing_frames and bool(rows) and bool(graded),
        "privacy_recall": privacy_recall >= args.minimum_privacy_recall,
        "temporal_leak_streak": max_leak_streak <= args.maximum_leak_streak,
        "target_fps_p95": speed_pass,
    }

    output_prefix = args.output_prefix.expanduser().resolve()
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    montage_path = output_prefix.with_name(f"{output_prefix.name}_failures.jpg")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    worst_failures = sorted(
        failure_candidates,
        key=lambda item: (bool(item["eyes_covered"]), float(item["coverage"])),
    )[: args.failure_images]
    _failure_montage(worst_failures, montage_path, args.minimum_zone_coverage)

    summary: dict[str, object] = {
        "benchmark": "chokepoint_face_privacy",
        "passed": all(checks.values()),
        "checks": checks,
        "configuration": {
            "data_root": str(args.data_root.expanduser().resolve()),
            "sequence_group": args.sequence,
            "sequences": [sequence.name for sequence in sequences],
            "model": str(model_path),
            "provider_requested": args.provider,
            "providers_active": detector.providers,
            "detector_threshold": args.detector_threshold,
            "redaction": "project pixelate_faces ellipse",
            "minimum_eye_distance_px": args.minimum_eye_distance,
            "minimum_zone_coverage": args.minimum_zone_coverage,
            "minimum_privacy_recall": args.minimum_privacy_recall,
            "maximum_leak_streak_frames": args.maximum_leak_streak,
            "target_fps": args.target_fps,
            "frame_step": args.frame_step,
            "max_frames": args.max_frames,
        },
        "dataset": {
            "frames_processed": len(rows),
            "annotated_face_frames": len(annotated),
            "graded_face_frames": len(graded),
            "small_face_frames_excluded_from_grade": len(small),
            "no_face_frames": len(no_face),
            "missing_frame_count": len(missing_frames),
            "missing_frames": missing_frames[:20],
        },
        "privacy": {
            "private_frames": len(private),
            "leak_frames": len(leaks),
            "privacy_recall": privacy_recall,
            "both_eyes_covered_recall": eye_recall,
            "mean_face_zone_coverage": (
                float(mean(float(row["face_zone_coverage"]) for row in graded))
                if graded
                else None
            ),
            "maximum_consecutive_leak_frames": max_leak_streak,
            "maximum_consecutive_leak_ms_at_target_fps": (
                max_leak_streak * budget_ms
            ),
            "false_positive_proxy_frames": len(false_positive_frames),
            "false_positive_proxy_rate_on_unannotated_frames": (
                len(false_positive_frames) / len(no_face) if no_face else None
            ),
        },
        "performance": {
            "frame_budget_ms": budget_ms,
            "estimated_sequential_fps_from_mean_pipeline_ms": (
                1000.0 / mean(pipeline_latencies) if pipeline_latencies else None
            ),
            "image_read_ms": _distribution(read_latencies),
            "detector_ms": _distribution(detector_latencies),
            "redaction_ms": _distribution(redaction_latencies),
            "pipeline_ms": _distribution(pipeline_latencies),
        },
        "artifacts": {
            "json": str(json_path),
            "frame_csv": str(csv_path),
            "failure_montage": str(montage_path) if worst_failures else None,
        },
        "notes": [
            "ChokePoint ground truth contains eye points, not segmentation masks.",
            "Face-zone coverage uses a documented eye-derived ellipse; eye recall is exact.",
            "Approximate face-zone coverage is diagnostic unless minimum_zone_coverage is greater than zero.",
            "False positives are a proxy because unannotated profile faces may still be visible.",
            "Performance measures detector plus the project's face redaction, excluding disk read and report metrics.",
        ],
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = run(args)
    privacy = summary["privacy"]
    performance = summary["performance"]
    assert isinstance(privacy, dict)
    assert isinstance(performance, dict)
    pipeline = performance["pipeline_ms"]
    assert isinstance(pipeline, dict)
    print(f"Result: {'PASS' if summary['passed'] else 'FAIL'}")
    print(
        "Privacy recall: "
        f"{float(privacy['privacy_recall']):.2%}; "
        f"eye recall: {float(privacy['both_eyes_covered_recall']):.2%}; "
        f"leaks: {privacy['leak_frames']}"
    )
    print(
        "Longest leak streak: "
        f"{privacy['maximum_consecutive_leak_frames']} frames "
        f"({float(privacy['maximum_consecutive_leak_ms_at_target_fps']):.1f} ms)"
    )
    print(
        "Pipeline latency: "
        f"mean {float(pipeline['mean']):.2f} ms; "
        f"p95 {float(pipeline['p95']):.2f} ms; "
        f"estimated {float(performance['estimated_sequential_fps_from_mean_pipeline_ms']):.1f} FPS"
    )
    artifacts = summary["artifacts"]
    assert isinstance(artifacts, dict)
    print(f"JSON: {artifacts['json']}")
    print(f"CSV: {artifacts['frame_csv']}")
    if artifacts["failure_montage"]:
        print(f"Failures: {artifacts['failure_montage']}")
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
