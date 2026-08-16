from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from dataset_common import find_image, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed/pose"))
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--angles", default="-90,-75,-60,-45,-30,0,30,45,60,75,90")
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def parse_angles(value: str) -> tuple[float, ...]:
    angles = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not angles or any(abs(angle) > 180.0 for angle in angles):
        raise ValueError("--angles must contain comma-separated values in [-180, 180]")
    return angles


def label_rows(path: Path) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = np.asarray([float(value) for value in line.split()], dtype=np.float32)
        if values.size != 20 or values[0] != 0:
            raise ValueError(f"Invalid YOLO Pose label: {path}")
        if np.count_nonzero(values[7:20:3]) < 5:
            continue
        rows.append(values)
    return rows


def absolute_target(row: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    center_x, center_y, box_width, box_height = row[1:5]
    box = np.asarray(
        [
            (center_x - box_width * 0.5) * width,
            (center_y - box_height * 0.5) * height,
            (center_x + box_width * 0.5) * width,
            (center_y + box_height * 0.5) * height,
        ],
        dtype=np.float32,
    )
    landmarks = row[5:20].reshape(5, 3).copy()
    landmarks[:, 0] *= width
    landmarks[:, 1] *= height
    return box, landmarks


def expanded_rotation(width: int, height: int, angle: float) -> tuple[np.ndarray, tuple[int, int]]:
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), angle, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    output_width = int(np.ceil(height * sine + width * cosine))
    output_height = int(np.ceil(height * cosine + width * sine))
    matrix[0, 2] += (output_width - width) / 2.0
    matrix[1, 2] += (output_height - height) / 2.0
    return matrix.astype(np.float32), (output_width, output_height)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack(
        (points.astype(np.float32), np.ones(len(points), dtype=np.float32))
    )
    return homogeneous @ matrix.T


def transform_box(box: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    corners = np.asarray(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )
    transformed = transform_points(corners, matrix)
    return np.asarray(
        [
            transformed[:, 0].min(),
            transformed[:, 1].min(),
            transformed[:, 0].max(),
            transformed[:, 1].max(),
        ],
        dtype=np.float32,
    )


def intersection_over_union(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def eye_line_angle(landmarks: np.ndarray) -> float:
    delta = landmarks[1, :2] - landmarks[0, :2]
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def angle_error(first: float, second: float) -> float:
    difference = (first - second + 180.0) % 360.0 - 180.0
    return abs(difference)


def distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def samples(root: Path, split: str, limit: int, seed: int) -> list[tuple[Path, np.ndarray]]:
    labels_root = root / "labels" / split
    images_root = root / "images" / split
    selected: list[tuple[Path, np.ndarray]] = []
    for label_path in sorted(labels_root.rglob("*.txt")):
        rows = label_rows(label_path)
        if not rows:
            continue
        relative = label_path.relative_to(labels_root).with_suffix("")
        try:
            image_path = find_image(images_root, str(relative))
        except FileNotFoundError:
            continue
        largest = max(rows, key=lambda row: float(row[3] * row[4]))
        selected.append((image_path, largest))
    random.Random(seed).shuffle(selected)
    return selected[:limit]


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    angles = parse_angles(args.angles)
    device = resolve_device(args.device)
    root = args.dataset_root.expanduser().resolve()
    selected = samples(root, args.split, args.limit, args.seed)
    if not selected:
        raise SystemExit(f"No samples found under {root}")
    model = YOLO(str(args.weights.expanduser().resolve()))
    summary: dict[str, object] = {
        "weights": str(args.weights.expanduser().resolve()),
        "dataset_root": str(root),
        "split": args.split,
        "device": device,
        "requested_samples": args.limit,
        "source_samples": len(selected),
        "angles": {},
    }
    angle_summaries = summary["angles"]
    for angle in angles:
        roll_errors: list[float] = []
        nmes: list[float] = []
        confidences: list[float] = []
        matched = 0
        for image_path, row in selected:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            target_box, target_landmarks = absolute_target(row, width, height)
            matrix, output_size = expanded_rotation(width, height, angle)
            rotated = cv2.warpAffine(
                image,
                matrix,
                output_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(114, 114, 114),
            )
            target_box = transform_box(target_box, matrix)
            target_landmarks[:, :2] = transform_points(target_landmarks[:, :2], matrix)
            result = model.predict(
                source=rotated,
                imgsz=args.imgsz,
                conf=args.confidence,
                device=device,
                verbose=False,
            )[0]
            if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
                continue
            predicted_boxes = result.boxes.xyxy.detach().cpu().numpy()
            overlaps = np.asarray(
                [intersection_over_union(target_box, box) for box in predicted_boxes],
                dtype=np.float32,
            )
            index = int(np.argmax(overlaps))
            if overlaps[index] < args.iou_threshold:
                continue
            keypoints = result.keypoints.data[index].detach().cpu().numpy()
            if keypoints.shape != (5, 3):
                raise ValueError(f"Expected 5x3 keypoints, got {keypoints.shape}")
            matched += 1
            face_scale = max(
                1.0,
                float(np.hypot(target_box[2] - target_box[0], target_box[3] - target_box[1])),
            )
            nmes.append(
                float(
                    np.mean(
                        np.linalg.norm(
                            keypoints[:, :2] - target_landmarks[:, :2],
                            axis=1,
                        )
                    )
                    / face_scale
                )
            )
            roll_errors.append(
                angle_error(
                    eye_line_angle(keypoints),
                    eye_line_angle(target_landmarks),
                )
            )
            confidences.append(float(np.min(keypoints[:, 2])))
        angle_summaries[str(int(angle) if angle.is_integer() else angle)] = {
            "attempted": len(selected),
            "matched": matched,
            "matched_recall": matched / len(selected),
            "roll_error_degrees": distribution(roll_errors),
            "five_point_nme_by_bbox_diagonal": distribution(nmes),
            "minimum_keypoint_confidence": distribution(confidences),
        }
        roll_text = distribution(roll_errors)
        print(
            f"angle={angle:+.0f}: matched={matched}/{len(selected)} "
            f"median-roll-error={roll_text['median']:.2f}deg"
            if roll_text is not None
            else f"angle={angle:+.0f}: matched=0/{len(selected)}"
        )
    if args.output is not None:
        write_json(args.output.expanduser().resolve(), summary)
        print(f"Summary: {args.output.expanduser().resolve()}")
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
