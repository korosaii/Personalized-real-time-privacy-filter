from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
LANDMARK_OFFSETS = (4, 7, 10, 13, 16)


def find_landmark_file(path: Path, split: str) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    candidates = (
        path / split / "label.txt",
        path / f"{split}_label.txt",
        path / "label.txt",
        path / "retinaface_gt_v1.1" / split / "label.txt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(path.rglob("label.txt"))
    for candidate in matches:
        if split.lower() in {part.lower() for part in candidate.parts}:
            return candidate
    raise FileNotFoundError(f"Could not find {split}/label.txt under {path}")


def find_image(images_root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    candidates = [images_root / relative_path]
    if not relative_path.suffix:
        candidates.extend((images_root / relative_path).with_suffix(suffix) for suffix in IMAGE_SUFFIXES)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Image not found: {relative}")


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def materialize_image(source: Path, target: Path, copy_images: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    if copy_images:
        shutil.copy2(source, target)
        return
    try:
        target.symlink_to(source.resolve())
    except OSError:
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def normalized_bbox(values: list[float], width: int, height: int) -> tuple[list[float], list[float]] | None:
    if len(values) < 4 or width <= 0 or height <= 0:
        return None
    x, y, box_width, box_height = values[:4]
    x1 = min(max(x, 0.0), float(width))
    y1 = min(max(y, 0.0), float(height))
    x2 = min(max(x + box_width, 0.0), float(width))
    y2 = min(max(y + box_height, 0.0), float(height))
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None
    normalized = [
        ((x1 + x2) * 0.5) / width,
        ((y1 + y2) * 0.5) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    ]
    return normalized, [x1, y1, x2, y2]


def normalized_landmarks(values: list[float], width: int, height: int) -> tuple[list[float], int]:
    points: list[float] = []
    visible = 0
    for offset in LANDMARK_OFFSETS:
        if len(values) <= offset + 1:
            points.extend((0.0, 0.0, 0.0))
            continue
        x, y = values[offset], values[offset + 1]
        if 0.0 <= x < width and 0.0 <= y < height:
            points.extend((x / width, y / height, 2.0))
            visible += 1
        else:
            points.extend((0.0, 0.0, 0.0))
    return points, visible


def pose_line(box: list[float], landmarks: list[float]) -> str:
    fields = ["0", *(f"{value:.8f}" for value in box)]
    for index in range(0, len(landmarks), 3):
        fields.extend((f"{landmarks[index]:.8f}", f"{landmarks[index + 1]:.8f}", str(int(landmarks[index + 2]))))
    return " ".join(fields)


def detection_line(box: list[float]) -> str:
    return " ".join(["0", *(f"{value:.8f}" for value in box)])


def pose_dataset_yaml(root: Path, train: str, val: str) -> dict:
    return {
        "path": str(root.resolve()),
        "train": train,
        "val": val,
        "names": {0: "face"},
        "kpt_shape": [5, 3],
        "flip_idx": [1, 0, 2, 4, 3],
        "kpt_names": {0: ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]},
    }


def combined_pose_dataset_yaml(
    train_manifest: Path,
    val_manifest: Path,
) -> dict:
    return {
        "path": "/",
        "train": str(train_manifest.resolve()),
        "val": str(val_manifest.resolve()),
        "names": {0: "face"},
        "kpt_shape": [5, 3],
        "flip_idx": [1, 0, 2, 4, 3],
        "kpt_names": {0: ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]},
    }


def image_manifest_lines(roots: list[Path]) -> list[str]:
    images: list[str] = []
    for root in roots:
        root = root.expanduser().absolute()
        if not root.is_dir():
            raise FileNotFoundError(f"Image directory not found: {root}")
        images.extend(
            str(path.absolute())
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    return images


def detection_dataset_yaml(root: Path, train: str, val: str) -> dict:
    return {
        "path": str(root.resolve()),
        "train": train,
        "val": val,
        "names": {0: "face"},
    }
