from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--format", choices=("onnx", "coreml", "both"), default="onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise SystemExit(f"Weights not found: {weights}")
    model = YOLO(str(weights))
    if args.format in ("onnx", "both"):
        output = model.export(
            format="onnx",
            imgsz=args.imgsz,
            dynamic=False,
            simplify=True,
            opset=args.opset,
            nms=False,
        )
        print(f"ONNX: {output}")
    if args.format in ("coreml", "both"):
        try:
            output = model.export(
                format="coreml",
                imgsz=args.imgsz,
                dynamic=False,
                nms=False,
            )
        except ModuleNotFoundError as error:
            raise SystemExit("Core ML export requires: python -m pip install coremltools") from error
        print(f"Core ML: {output}")


if __name__ == "__main__":
    main()
