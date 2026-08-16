from __future__ import annotations

import argparse
from pathlib import Path

from dataset_common import (
    detection_dataset_yaml,
    detection_line,
    find_image,
    image_size,
    materialize_image,
    normalized_bbox,
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
    parser.add_argument("--copy-images", action="store_true")
    return parser.parse_args()


def validation_records(path: Path):
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    index = 0
    while index < len(lines):
        relative = lines[index]
        index += 1
        if index >= len(lines):
            raise ValueError(f"Missing face count after {relative}")
        count = int(lines[index])
        index += 1
        faces: list[list[float]] = []
        for _ in range(count):
            if index >= len(lines):
                raise ValueError(f"Annotation ended inside {relative}")
            faces.append([float(value) for value in lines[index].split()])
            index += 1
        yield relative, faces


def main() -> None:
    args = parse_args()
    images_root = args.images.expanduser().resolve()
    annotations = args.annotations.expanduser().resolve()
    output = args.output.expanduser().resolve()
    detection_root = output / "detection_official_val"
    pose_root = output / "pose_official_val"
    stats = {
        "annotation_file": str(annotations),
        "images_root": str(images_root),
        "images": 0,
        "faces": 0,
        "invalid_faces": 0,
        "missing_images": 0,
    }
    empty_landmarks = [0.0] * 15
    for relative, faces in validation_records(annotations):
        try:
            source = find_image(images_root, relative)
            width, height = image_size(source)
        except (FileNotFoundError, OSError):
            stats["missing_images"] += 1
            continue
        relative_image = Path(relative)
        detection_image = detection_root / "images" / "val" / relative_image
        pose_image = pose_root / "images" / "val" / relative_image
        detection_label = detection_root / "labels" / "val" / relative_image.with_suffix(".txt")
        pose_label = pose_root / "labels" / "val" / relative_image.with_suffix(".txt")
        materialize_image(source, detection_image, args.copy_images)
        materialize_image(source, pose_image, args.copy_images)
        detection_lines: list[str] = []
        pose_lines: list[str] = []
        for values in faces:
            result = normalized_bbox(values, width, height)
            if result is None:
                stats["invalid_faces"] += 1
                continue
            box, _ = result
            detection_lines.append(detection_line(box))
            pose_lines.append(pose_line(box, empty_landmarks))
            stats["faces"] += 1
        write_lines(detection_label, detection_lines)
        write_lines(pose_label, pose_lines)
        stats["images"] += 1
    if stats["images"] == 0:
        raise SystemExit(f"No validation images were converted: {stats}")
    write_yaml(
        output / "widerface_detection_official_val.yaml",
        detection_dataset_yaml(detection_root, "images/val", "images/val"),
    )
    write_yaml(
        output / "widerface_pose_official_val.yaml",
        pose_dataset_yaml(pose_root, "images/val", "images/val"),
    )
    write_json(output / "val_conversion.json", stats)
    print(f"Converted {stats['images']} official validation images and {stats['faces']} faces")
    print(f"Detection dataset: {output / 'widerface_detection_official_val.yaml'}")
    print(f"Pose-compatible dataset: {output / 'widerface_pose_official_val.yaml'}")


if __name__ == "__main__":
    main()
