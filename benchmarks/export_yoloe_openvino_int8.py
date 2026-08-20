from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from privacy_filter.image_prompt_video import YoloESamPipeline

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    staged_weights = output / args.weights.name
    if not staged_weights.exists():
        shutil.copy2(args.weights.resolve(), staged_weights)

    setup_started = perf_counter()
    pipeline = YoloESamPipeline(
        reference_groups=[(args.reference.resolve(),)],
        yolo_model_id=str(staged_weights),
        yolo_onnx=False,
        device="cpu",
        precision="fp32",
        yolo_imgsz=args.imgsz,
        yolo_reference_imgsz=args.imgsz,
        reference_size=1280,
        reference_sam_model_id="facebook/sam2.1-hiera-tiny",
        reference_sam_points=8,
        reference_sam_min_area_ratio=0.01,
        reference_sam_max_area_ratio=0.98,
        reference_sam_mask_output_directory=output / "reference_masks",
        yolo_confidence=0.10,
        yolo_iou=0.50,
        min_mask_area=64,
        max_mask_area_ratio=0.98,
        max_objects=20,
        mask_dilation=5,
        iou_threshold=0.30,
        iou_max_missed=0,
    )
    setup_ms = (perf_counter() - setup_started) * 1000.0

    fp32_started = perf_counter()
    fp32_path = Path(
        pipeline.yolo.export(
            format="openvino",
            imgsz=args.imgsz,
            batch=1,
            dynamic=False,
            quantize=32,
            device="cpu",
        )
    ).resolve()
    fp32_export_ms = (perf_counter() - fp32_started) * 1000.0

    int8_started = perf_counter()
    int8_path = Path(
        pipeline.yolo.export(
            format="openvino",
            imgsz=args.imgsz,
            batch=1,
            dynamic=False,
            quantize=8,
            data=str(args.calibration_data.resolve()),
            fraction=1.0,
            device="cpu",
        )
    ).resolve()
    int8_export_ms = (perf_counter() - int8_started) * 1000.0

    result = {
        "weights": str(args.weights.resolve()),
        "reference": str(args.reference.resolve()),
        "calibration_data": str(args.calibration_data.resolve()),
        "imgsz": args.imgsz,
        "setup_ms": setup_ms,
        "fp32": {"path": str(fp32_path), "export_ms": fp32_export_ms},
        "int8": {"path": str(int8_path), "export_ms": int8_export_ms},
    }
    (output / "export_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
