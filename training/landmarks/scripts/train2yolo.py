from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from dataset_common import (
    detection_dataset_yaml,
    detection_line,
    find_image,
    find_landmark_file,
    image_size,
    materialize_image,
    normalized_bbox,
    normalized_landmarks,
    pose_dataset_yaml,
    pose_line,
    write_json,
    write_lines,
    write_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-images", action="store_true")
    return parser.parse_args()


def annotation_records(path: Path):
    relative: str | None = None
    faces: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if relative is not None:
                    yield relative, faces
                relative = line[1:].strip()
                faces = []
                continue
            if relative is None:
                continue
            try:
                faces.append([float(value) for value in line.split()])
            except ValueError:
                continue
    if relative is not None:
        yield relative, faces


def choose_split(relative: str, seed: int, val_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{relative}".encode()).digest()
    sample = int.from_bytes(digest[:8], "big") / float(2**64)
    return "val" if sample < val_fraction else "train"


def main() -> None:
    args = parse_args()
    if not 0.01 <= args.val_fraction <= 0.5:
        raise SystemExit("--val-fraction must be between 0.01 and 0.5")
    images_root = args.images.expanduser().resolve()
    annotations = find_landmark_file(args.annotations, "train")
    output = args.output.expanduser().resolve()
    pose_root = output / "pose"
    detection_root = output / "detection_pose_val"
    stats = {
        "annotation_file": str(annotations),
        "images_root": str(images_root),
        "train_images": 0,
        "val_images": 0,
        "faces": 0,
        "faces_with_five_landmarks": 0,
        "faces_with_partial_landmarks": 0,
        "faces_without_landmarks": 0,
        "invalid_faces": 0,
        "missing_images": 0,
    }
    for relative, faces in annotation_records(annotations):
        try:
            source = find_image(images_root, relative)
            width, height = image_size(source)
        except (FileNotFoundError, OSError):
            stats["missing_images"] += 1
            continue
        split = choose_split(relative, args.seed, args.val_fraction)
        relative_image = Path(relative)
        pose_image = pose_root / "images" / split / relative_image
        pose_label = pose_root / "labels" / split / relative_image.with_suffix(".txt")
        materialize_image(source, pose_image, args.copy_images)
        pose_lines: list[str] = []
        detection_lines: list[str] = []
        for values in faces:
            result = normalized_bbox(values, width, height)
            if result is None:
                stats["invalid_faces"] += 1
                continue
            box, _ = result
            landmarks, visible = normalized_landmarks(values, width, height)
            pose_lines.append(pose_line(box, landmarks))
            detection_lines.append(detection_line(box))
            stats["faces"] += 1
            if visible == 5:
                stats["faces_with_five_landmarks"] += 1
            elif visible:
                stats["faces_with_partial_landmarks"] += 1
            else:
                stats["faces_without_landmarks"] += 1
        write_lines(pose_label, pose_lines)
        stats[f"{split}_images"] += 1
        if split == "val":
            detection_image = detection_root / "images" / "val" / relative_image
            detection_label = detection_root / "labels" / "val" / relative_image.with_suffix(".txt")
            materialize_image(source, detection_image, args.copy_images)
            write_lines(detection_label, detection_lines)
    if stats["train_images"] == 0 or stats["val_images"] == 0:
        raise SystemExit(f"No usable train/val split was produced: {stats}")
    write_yaml(
        output / "widerface_pose.yaml",
        pose_dataset_yaml(pose_root, "images/train", "images/val"),
    )
    write_yaml(
        output / "widerface_detection_pose_val.yaml",
        detection_dataset_yaml(detection_root, "images/val", "images/val"),
    )
    write_json(output / "train_conversion.json", stats)
    print(f"Converted {stats['train_images']} train and {stats['val_images']} validation images")
    print(f"Converted {stats['faces']} faces; {stats['faces_with_five_landmarks']} have all 5 landmarks")
    print(f"Pose dataset: {output / 'widerface_pose.yaml'}")
    print(f"Detection baseline dataset: {output / 'widerface_detection_pose_val.yaml'}")
    if stats["missing_images"]:
        print(f"Warning: {stats['missing_images']} annotation images were not found", file=sys.stderr)


if __name__ == "__main__":
    main()
