from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DAVIS = ROOT / "data" / "DAVIS" / "DAVIS"


def evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    if len(paths) <= count:
        return paths
    indices = np.linspace(0, len(paths) - 1, count, dtype=np.int64)
    return [paths[int(index)] for index in np.unique(indices)]


def segmentation_lines(mask_path: Path) -> list[str]:
    # PIL preserves the integer object IDs of DAVIS palette PNGs. OpenCV
    # expands their palette to BGR, which loses those IDs.
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    height, width = mask.shape
    lines: list[str] = []
    for object_id in sorted(int(value) for value in np.unique(mask) if value):
        contours, _hierarchy = cv2.findContours(
            (mask == object_id).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            if len(contour) < 3 or cv2.contourArea(contour) < 4:
                continue
            points = contour.reshape(-1, 2).astype(np.float64)
            points[:, 0] /= width
            points[:, 1] /= height
            coordinates = " ".join(f"{value:.6f}" for value in points.reshape(-1))
            lines.append(f"0 {coordinates}")
    return lines


def link_or_copy(source: Path, destination: Path) -> str:
    if destination.exists():
        return "existing"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--davis", type=Path, default=DEFAULT_DAVIS)
    parser.add_argument("--frames-per-sequence", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "int8_calibration_davis_train",
    )
    args = parser.parse_args()

    davis = args.davis.resolve()
    output = args.output.resolve()
    image_output = output / "images" / "calibration"
    label_output = output / "labels" / "calibration"
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    (label_output.parent / "calibration.cache").unlink(missing_ok=True)

    sequences = [
        line.strip()
        for line in (davis / "ImageSets" / "2017" / "train.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if "blackswan" in sequences:
        raise RuntimeError("DAVIS validation sequence blackswan leaked into train split")

    manifest = []
    link_modes: dict[str, int] = {}
    for sequence in sequences:
        frames = sorted((davis / "JPEGImages" / "480p" / sequence).glob("*.jpg"))
        for frame_path in evenly_spaced(frames, args.frames_per_sequence):
            mask_path = davis / "Annotations" / "480p" / sequence / f"{frame_path.stem}.png"
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)
            name = f"{sequence}__{frame_path.name}"
            destination = image_output / name
            mode = link_or_copy(frame_path, destination)
            link_modes[mode] = link_modes.get(mode, 0) + 1
            label_path = label_output / f"{Path(name).stem}.txt"
            label_path.write_text(
                "\n".join(segmentation_lines(mask_path)) + "\n", encoding="utf-8"
            )
            manifest.append(
                {
                    "sequence": sequence,
                    "image": str(frame_path),
                    "mask": str(mask_path),
                    "calibration_image": str(destination),
                }
            )

    yaml_path = output / "davis_train_calibration.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output.as_posix()}",
                "train: images/calibration",
                "val: images/calibration",
                "names:",
                "  0: foreground_object",
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary = {
        "source": str(davis),
        "split": "DAVIS 2017 train",
        "excluded_validation_sequence": "blackswan",
        "sequences": len(sequences),
        "frames_per_sequence": args.frames_per_sequence,
        "frames": len(manifest),
        "link_modes": link_modes,
        "yaml": str(yaml_path),
        "items": manifest,
    }
    (output / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "items"}, indent=2))


if __name__ == "__main__":
    main()
