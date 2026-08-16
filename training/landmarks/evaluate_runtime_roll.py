from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from privacy_filter.enrollment import load_template
from privacy_filter.model_setup import prepare_runtime_models
from privacy_filter.recognition import FaceEmbedder
from privacy_filter.yolo import YOLOFaceDetector


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--detector", default="yolo11-pose")
    parser.add_argument("--model", default="r34-glint360k")
    parser.add_argument("--provider", default="auto")
    parser.add_argument("--angles", default="-90,-75,-60,-45,-30,0,30,45,60,75,90")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--detector-threshold", type=float, default=0.10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def parse_angles(value: str) -> tuple[float, ...]:
    angles = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not angles or any(abs(angle) > 180.0 for angle in angles):
        raise ValueError("--angles must contain comma-separated values in [-180, 180]")
    return angles


def image_paths(root: Path) -> list[Path]:
    resolved = root.expanduser().resolve()
    if resolved.is_file():
        return [resolved]
    if not resolved.is_dir():
        raise FileNotFoundError(f"Photo path not found: {resolved}")
    return sorted(
        path
        for path in resolved.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def expanded_rotation(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), angle, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    output_width = int(np.ceil(height * sine + width * cosine))
    output_height = int(np.ceil(height * cosine + width * sine))
    matrix[0, 2] += (output_width - width) / 2.0
    matrix[1, 2] += (output_height - height) / 2.0
    return cv2.warpAffine(
        image,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114),
    )


def distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def main() -> None:
    args = parse_args()
    photos = image_paths(args.photos)
    if not photos:
        raise SystemExit("No photographs found")
    template = load_template(args.template)
    threshold = template.threshold if args.threshold is None else args.threshold
    models = prepare_runtime_models(
        args.detector,
        args.model,
        args.provider,
    )
    detector = YOLOFaceDetector(
        models.detector_runtime,
        threshold=args.detector_threshold,
        provider=args.provider,
    )
    if not detector.has_landmarks:
        raise SystemExit("Runtime roll evaluation requires a detector with landmarks")
    embedder = FaceEmbedder(models.recognition_runtime, provider=args.provider)
    detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
    embedder.warmup(1)
    summary: dict[str, object] = {
        "photos": [str(path) for path in photos],
        "template": str(args.template.expanduser().resolve()),
        "detector": models.detector_name,
        "recognition_model": models.recognition_name,
        "threshold": threshold,
        "angles": {},
    }
    angle_results = summary["angles"]
    for angle in parse_angles(args.angles):
        scores: list[float] = []
        detector_misses = 0
        alignment_failures = 0
        for path in photos:
            encoded = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                detector_misses += 1
                continue
            rotated = expanded_rotation(image, angle)
            detections = detector.detect(rotated).detections
            if not len(detections):
                detector_misses += 1
                continue
            detection = max(
                detections,
                key=lambda row: float((row[2] - row[0]) * (row[3] - row[1])),
            )
            try:
                result = embedder.embed_bbox(rotated, detection)
            except (ValueError, cv2.error):
                alignment_failures += 1
                continue
            scores.append(template.score(result.embedding))
        authorized = sum(score >= threshold for score in scores)
        attempted = len(photos)
        angle_results[str(int(angle) if angle.is_integer() else angle)] = {
            "attempted": attempted,
            "scored": len(scores),
            "detector_misses": detector_misses,
            "alignment_failures": alignment_failures,
            "authorized": authorized,
            "authorization_recall": authorized / attempted,
            "similarity": distribution(scores),
        }
        score_text = distribution(scores)
        median = "none" if score_text is None else f"{score_text['median']:.3f}"
        print(
            f"angle={angle:+.0f}: authorized={authorized}/{attempted} "
            f"median-score={median}"
        )
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Summary: {output}")
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
