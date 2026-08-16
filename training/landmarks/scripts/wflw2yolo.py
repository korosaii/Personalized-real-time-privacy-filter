from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from dataset_common import (
    combined_pose_dataset_yaml,
    image_manifest_lines,
    materialize_image,
    normalized_bbox,
    pose_dataset_yaml,
    pose_line,
    write_json,
    write_lines,
    write_yaml,
)


WFLW_POINT_COUNT = 98
WFLW_ATTRIBUTE_NAMES = (
    "pose",
    "expression",
    "illumination",
    "makeup",
    "occlusion",
    "blur",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--wider-pose-root",
        type=Path,
        default=Path("data/processed/pose"),
    )
    parser.add_argument("--crop-scale", type=float, default=1.35)
    parser.add_argument("--hard-pose-repeats", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_annotation_file(root: Path, split: str) -> Path:
    name = f"list_98pt_rect_attr_{split}.txt"
    direct = root / name
    if direct.is_file():
        return direct.resolve()
    matches = sorted(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} under {root}")
    return matches[0].resolve()


def find_wflw_image(root: Path, relative: str) -> Path:
    candidates = (
        root / relative,
        root / "WFLW_images" / relative,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = list(root.rglob(Path(relative).name))
    if len(matches) == 1:
        return matches[0].resolve()
    raise FileNotFoundError(f"WFLW image not found: {relative}")


def annotation_records(path: Path):
    expected = WFLW_POINT_COUNT * 2 + 4 + len(WFLW_ATTRIBUTE_NAMES) + 1
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fields = raw_line.strip().split()
            if not fields:
                continue
            if len(fields) != expected:
                raise ValueError(
                    f"{path}:{line_number}: expected {expected} fields, got {len(fields)}"
                )
            coordinates = np.asarray(
                [float(value) for value in fields[: WFLW_POINT_COUNT * 2]],
                dtype=np.float32,
            ).reshape(WFLW_POINT_COUNT, 2)
            rectangle = [
                float(value)
                for value in fields[WFLW_POINT_COUNT * 2 : WFLW_POINT_COUNT * 2 + 4]
            ]
            attributes = {
                name: int(value)
                for name, value in zip(
                    WFLW_ATTRIBUTE_NAMES,
                    fields[WFLW_POINT_COUNT * 2 + 4 : -1],
                    strict=True,
                )
            }
            yield fields[-1], coordinates, rectangle, attributes


def five_landmarks(points: np.ndarray) -> np.ndarray:
    first_eye = points[60:68].mean(axis=0)
    second_eye = points[68:76].mean(axis=0)
    eyes = sorted((first_eye, second_eye), key=lambda point: float(point[0]))
    mouth = sorted((points[76], points[82]), key=lambda point: float(point[0]))
    return np.asarray(
        [eyes[0], eyes[1], points[54], mouth[0], mouth[1]],
        dtype=np.float32,
    )


def normalized_five_landmarks(
    points: np.ndarray,
    width: int,
    height: int,
) -> list[float] | None:
    if not np.isfinite(points).all():
        return None
    if np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width):
        return None
    if np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height):
        return None
    output: list[float] = []
    for x, y in points:
        output.extend((float(x) / width, float(y) / height, 2.0))
    return output


def crop_face(
    image: np.ndarray,
    rectangle: list[float],
    landmarks: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, list[float], list[float]] | None:
    x1, y1, x2, y2 = rectangle
    width = x2 - x1
    height = y2 - y1
    if width < 2.0 or height < 2.0:
        return None
    side = max(16, int(np.ceil(max(width, height) * scale)))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    left = center_x - side / 2.0
    top = center_y - side / 2.0
    transform = np.asarray(
        [[1.0, 0.0, -left], [0.0, 1.0, -top]],
        dtype=np.float32,
    )
    crop = cv2.warpAffine(
        image,
        transform,
        (side, side),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114),
    )
    box_result = normalized_bbox(
        [x1 - left, y1 - top, width, height],
        side,
        side,
    )
    shifted_landmarks = landmarks - np.asarray([left, top], dtype=np.float32)
    normalized = normalized_five_landmarks(shifted_landmarks, side, side)
    if box_result is None or normalized is None:
        return None
    return crop, box_result[0], normalized


def write_crop(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise ValueError(f"Could not encode {path}")
    encoded.tofile(path)


def convert_split(
    split: str,
    images_root: Path,
    annotation_file: Path,
    pose_root: Path,
    hard_pose_root: Path,
    crop_scale: float,
) -> dict[str, object]:
    stats: dict[str, object] = {
        "annotation_file": str(annotation_file),
        "images": 0,
        "source_images": 0,
        "faces": 0,
        "invalid": 0,
        "missing_images": 0,
        "hard_pose_images": 0,
        "attributes": {name: 0 for name in WFLW_ATTRIBUTE_NAMES},
    }
    attribute_counts: dict[str, int] = stats["attributes"]
    grouped: dict[str, list[tuple[np.ndarray, list[float], dict[str, int]]]] = {}
    for relative, points, rectangle, attributes in annotation_records(annotation_file):
        grouped.setdefault(relative, []).append((points, rectangle, attributes))
    for relative, records in grouped.items():
        try:
            source = find_wflw_image(images_root, relative)
            encoded = np.fromfile(source, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"OpenCV could not read {source}")
        except (FileNotFoundError, OSError):
            stats["missing_images"] = int(stats["missing_images"]) + 1
            continue
        stats["source_images"] = int(stats["source_images"]) + 1
        relative_source = Path(relative)
        for record_index, (points, rectangle, attributes) in enumerate(records):
            sample = crop_face(
                image,
                rectangle,
                five_landmarks(points),
                crop_scale,
            )
            if sample is None:
                stats["invalid"] = int(stats["invalid"]) + 1
                continue
            crop, box, landmarks = sample
            relative_image = relative_source.with_name(
                f"{relative_source.stem}__wflw_{record_index:02d}.jpg"
            )
            target_image = pose_root / "images" / split / relative_image
            target_label = pose_root / "labels" / split / relative_image.with_suffix(".txt")
            write_crop(target_image, crop)
            write_lines(target_label, [pose_line(box, landmarks)])
            if attributes["pose"]:
                hard_image = hard_pose_root / "images" / split / relative_image
                hard_label = hard_pose_root / "labels" / split / relative_image.with_suffix(".txt")
                materialize_image(target_image, hard_image, False)
                write_lines(hard_label, [pose_line(box, landmarks)])
                stats["hard_pose_images"] = int(stats["hard_pose_images"]) + 1
            stats["images"] = int(stats["images"]) + 1
            stats["faces"] = int(stats["faces"]) + 1
            for name, enabled in attributes.items():
                if enabled:
                    attribute_counts[name] = int(attribute_counts[name]) + 1
    return stats


def main() -> None:
    args = parse_args()
    images_root = args.images.expanduser().resolve()
    annotations_root = args.annotations.expanduser().resolve()
    output = args.output.expanduser().resolve()
    pose_root = output / "wflw_pose"
    hard_pose_root = output / "wflw_pose_hard"
    if args.crop_scale < 1.0:
        raise SystemExit("--crop-scale must be at least 1.0")
    if args.hard_pose_repeats < 0:
        raise SystemExit("--hard-pose-repeats must be non-negative")
    existing = [path for path in (pose_root, hard_pose_root) if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "WFLW output already exists. Pass --overwrite to rebuild: "
            + ", ".join(str(path) for path in existing)
        )
    for path in existing:
        shutil.rmtree(path)
    train_file = find_annotation_file(annotations_root, "train")
    test_file = find_annotation_file(annotations_root, "test")
    train_stats = convert_split(
        "train",
        images_root,
        train_file,
        pose_root,
        hard_pose_root,
        args.crop_scale,
    )
    val_stats = convert_split(
        "val",
        images_root,
        test_file,
        pose_root,
        hard_pose_root,
        args.crop_scale,
    )
    if not train_stats["images"] or not val_stats["images"]:
        raise SystemExit(
            f"WFLW conversion did not produce train and validation data: "
            f"{train_stats}, {val_stats}"
        )
    wider_pose_root = args.wider_pose_root.expanduser().resolve()
    write_yaml(
        output / "wflw_pose.yaml",
        pose_dataset_yaml(pose_root, "images/train", "images/val"),
    )
    write_yaml(
        output / "wflw_pose_hard.yaml",
        pose_dataset_yaml(hard_pose_root, "images/train", "images/val"),
    )
    train_manifest = output / "widerface_wflw_train.txt"
    val_manifest = output / "widerface_wflw_val.txt"
    train_roots = [
        wider_pose_root / "images" / "train",
        pose_root / "images" / "train",
        *([hard_pose_root / "images" / "train"] * args.hard_pose_repeats),
    ]
    val_roots = [
        wider_pose_root / "images" / "val",
        pose_root / "images" / "val",
    ]
    train_images = image_manifest_lines(train_roots)
    val_images = image_manifest_lines(val_roots)
    write_lines(train_manifest, train_images)
    write_lines(val_manifest, val_images)
    write_yaml(
        output / "widerface_wflw_pose.yaml",
        combined_pose_dataset_yaml(
            train_manifest,
            val_manifest,
        ),
    )
    summary = {
        "images_root": str(images_root),
        "pose_root": str(pose_root),
        "hard_pose_root": str(hard_pose_root),
        "crop_scale": args.crop_scale,
        "hard_pose_repeats": args.hard_pose_repeats,
        "combined_dataset": {
            "train_images": len(train_images),
            "val_images": len(val_images),
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest),
        },
        "five_point_mapping": {
            "left_eye": "mean(60:68), sorted by image x",
            "right_eye": "mean(68:76), sorted by image x",
            "nose": 54,
            "left_mouth": "76/82, sorted by image x",
            "right_mouth": "76/82, sorted by image x",
        },
        "train": train_stats,
        "val": val_stats,
    }
    write_json(output / "wflw_conversion.json", summary)
    print(
        f"Converted WFLW: {train_stats['images']} train and "
        f"{val_stats['images']} validation images"
    )
    print(f"WFLW pose root: {pose_root}")
    print(f"Combined dataset: {output / 'widerface_wflw_pose.yaml'}")


if __name__ == "__main__":
    main()
