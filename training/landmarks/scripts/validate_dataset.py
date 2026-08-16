from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2

from dataset_common import IMAGE_SUFFIXES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/processed/pose"))
    parser.add_argument("--render", type=int, default=12)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/dataset_samples"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def find_matching_image(root: Path, split: str, relative_label: Path) -> Path | None:
    stem = relative_label.with_suffix("")
    for suffix in IMAGE_SUFFIXES:
        candidate = root / "images" / split / stem.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def parse_label(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 20:
            raise ValueError(f"{path}:{line_number}: expected 20 values, found {len(fields)}")
        values = [float(value) for value in fields]
        if values[0] != 0:
            raise ValueError(f"{path}:{line_number}: only class 0 is allowed")
        if any(value < 0.0 or value > 1.0 for value in values[1:5]):
            raise ValueError(f"{path}:{line_number}: bbox values must be normalized")
        for index in range(5, 20, 3):
            x, y, visibility = values[index : index + 3]
            if visibility not in (0.0, 1.0, 2.0):
                raise ValueError(f"{path}:{line_number}: visibility must be 0, 1, or 2")
            if visibility > 0 and not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"{path}:{line_number}: visible landmark is outside the image")
        rows.append(values)
    return rows


def render_sample(image_path: Path, rows: list[list[float]], target: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not read {image_path}")
    height, width = image.shape[:2]
    colors = ((255, 80, 80), (80, 255, 80), (80, 160, 255), (255, 80, 255), (80, 255, 255))
    for values in rows:
        center_x, center_y, box_width, box_height = values[1:5]
        x1 = int((center_x - box_width * 0.5) * width)
        y1 = int((center_y - box_height * 0.5) * height)
        x2 = int((center_x + box_width * 0.5) * width)
        y2 = int((center_y + box_height * 0.5) * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for point_index, offset in enumerate(range(5, 20, 3)):
            x, y, visibility = values[offset : offset + 3]
            if visibility > 0:
                cv2.circle(image, (int(x * width), int(y * height)), 3, colors[point_index], -1)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image):
        raise ValueError(f"Could not write {target}")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    artifacts = args.artifacts.expanduser().resolve()
    samples: list[tuple[Path, list[list[float]], str, Path]] = []
    stats = {
        "root": str(root),
        "images": 0,
        "labels": 0,
        "faces": 0,
        "faces_with_five_landmarks": 0,
        "faces_with_partial_landmarks": 0,
        "faces_without_landmarks": 0,
        "splits": {},
    }
    errors: list[str] = []
    for split in ("train", "val"):
        labels_root = root / "labels" / split
        split_labels = sorted(labels_root.rglob("*.txt")) if labels_root.is_dir() else []
        split_stats = {"images": 0, "labels": len(split_labels), "faces": 0}
        for label_path in split_labels:
            relative_label = label_path.relative_to(labels_root)
            image_path = find_matching_image(root, split, relative_label)
            if image_path is None:
                errors.append(f"Missing image for {label_path}")
                continue
            try:
                rows = parse_label(label_path)
            except ValueError as error:
                errors.append(str(error))
                continue
            stats["labels"] += 1
            stats["images"] += 1
            split_stats["images"] += 1
            split_stats["faces"] += len(rows)
            stats["faces"] += len(rows)
            for values in rows:
                visible = sum(1 for offset in range(7, 20, 3) if values[offset] > 0)
                if visible == 5:
                    stats["faces_with_five_landmarks"] += 1
                elif visible:
                    stats["faces_with_partial_landmarks"] += 1
                else:
                    stats["faces_without_landmarks"] += 1
            samples.append((image_path, rows, split, relative_label))
        stats["splits"][split] = split_stats
    if errors:
        preview = "\n".join(errors[:30])
        raise SystemExit(f"Dataset validation failed with {len(errors)} error(s):\n{preview}")
    if not samples:
        raise SystemExit(f"No YOLO Pose labels were found under {root}")
    random.Random(args.seed).shuffle(samples)
    for index, (image_path, rows, split, relative_label) in enumerate(samples[: max(args.render, 0)]):
        target = artifacts / f"{index:03d}_{split}_{relative_label.stem}.jpg"
        render_sample(image_path, rows, target)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if args.render > 0:
        print(f"Rendered samples: {artifacts}")


if __name__ == "__main__":
    main()
