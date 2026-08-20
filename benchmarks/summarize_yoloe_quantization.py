from __future__ import annotations

import argparse
import csv
import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def mask_agreement(first: Path, second: Path) -> dict[str, float]:
    scores: list[float] = []
    for first_path in sorted(first.glob("*.png")):
        second_path = second / first_path.name
        a = cv2.imread(str(first_path), cv2.IMREAD_GRAYSCALE) > 0
        b = cv2.imread(str(second_path), cv2.IMREAD_GRAYSCALE) > 0
        union = int(np.count_nonzero(a | b))
        intersection = int(np.count_nonzero(a & b))
        scores.append(intersection / union if union else 1.0)
    return {
        "mean_iou": float(np.mean(scores)),
        "min_iou": min(scores),
        "min_frame": int(np.argmin(scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    paths = {
        "PyTorch FP32": root / "pytorch_fp32_50",
        "OpenVINO FP32": root / "openvino_fp32_50",
        "OpenVINO INT8": root / "openvino_int8_50",
    }
    summaries = {name: read_json(path / "summary.json") for name, path in paths.items()}
    fp32 = summaries["OpenVINO FP32"]
    int8 = summaries["OpenVINO INT8"]
    pytorch = summaries["PyTorch FP32"]

    rows = []
    for name, summary in summaries.items():
        rows.append(
            {
                "runtime": name,
                "fps": summary["processing_fps"],
                "mean_ms": summary["latency_ms"]["mean"],
                "p50_ms": summary["latency_ms"]["p50"],
                "p95_ms": summary["latency_ms"]["p95"],
                "iou_j": summary["metrics"]["iou"],
                "boundary_f": summary["metrics"]["boundary_f"],
                "j_and_f": summary["metrics"]["j_and_f"],
                "privacy_recall": summary["metrics"]["privacy_recall"],
                "leakage": summary["metrics"]["leakage"],
                "over_redaction": summary["metrics"]["over_redaction"],
                "worst_leakage": summary["worst_frame"]["leakage"]["leakage"],
                "worst_leakage_frame": summary["worst_frame"]["leakage"]["frame"],
            }
        )

    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fp32_model = root / "export" / "yoloe-26n-seg_openvino_model"
    int8_model = root / "export" / "yoloe-26n-seg_int8_openvino_model"
    report = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "ultralytics": version("ultralytics"),
            "torch": version("torch"),
            "openvino": version("openvino"),
            "nncf": version("nncf"),
        },
        "protocol": {
            "sequence": "DAVIS 2017 validation / blackswan",
            "frames": 50,
            "resolution": "854x480",
            "input_size": 640,
            "confidence": 0.10,
            "warmup_frames": 1,
            "device": "CPU",
            "calibration": "600 evenly sampled frames from 60 DAVIS 2017 train sequences",
            "validation_excluded_from_calibration": True,
        },
        "results": rows,
        "int8_vs_openvino_fp32": {
            "fps_multiplier": int8["processing_fps"] / fp32["processing_fps"],
            "fps_percent": (int8["processing_fps"] / fp32["processing_fps"] - 1.0) * 100.0,
            "latency_percent": (int8["latency_ms"]["mean"] / fp32["latency_ms"]["mean"] - 1.0) * 100.0,
            "iou_delta": int8["metrics"]["iou"] - fp32["metrics"]["iou"],
            "boundary_f_delta": int8["metrics"]["boundary_f"] - fp32["metrics"]["boundary_f"],
            "j_and_f_delta": int8["metrics"]["j_and_f"] - fp32["metrics"]["j_and_f"],
            "leakage_delta": int8["metrics"]["leakage"] - fp32["metrics"]["leakage"],
        },
        "int8_vs_pytorch_fp32": {
            "fps_multiplier": int8["processing_fps"] / pytorch["processing_fps"],
            "fps_percent": (int8["processing_fps"] / pytorch["processing_fps"] - 1.0) * 100.0,
            "latency_percent": (int8["latency_ms"]["mean"] / pytorch["latency_ms"]["mean"] - 1.0) * 100.0,
            "iou_delta": int8["metrics"]["iou"] - pytorch["metrics"]["iou"],
            "j_and_f_delta": int8["metrics"]["j_and_f"] - pytorch["metrics"]["j_and_f"],
            "leakage_delta": int8["metrics"]["leakage"] - pytorch["metrics"]["leakage"],
        },
        "mask_agreement_int8_vs_openvino_fp32": mask_agreement(
            paths["OpenVINO INT8"] / "masks", paths["OpenVINO FP32"] / "masks"
        ),
        "artifacts": {
            "openvino_fp32_bytes": directory_size(fp32_model),
            "openvino_int8_bytes": directory_size(int8_model),
            "size_reduction_percent": (1.0 - directory_size(int8_model) / directory_size(fp32_model)) * 100.0,
        },
    }
    (root / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    def percentage(value: float) -> str:
        return f"{value * 100:.2f}%"

    markdown = [
        "# YOLOE-26n: квантизация OpenVINO INT8",
        "",
        "## Протокол",
        "",
        "- CPU: Intel Core i5-12450H; вход модели 640x640.",
        "- Точность: все 50 кадров DAVIS 2017 `blackswan`, исходное разрешение 854x480.",
        "- Калибровка: 600 кадров из 60 последовательностей официального train split; `blackswan` исключён.",
        "- Один warm-up кадр; затем измеряется весь вызов inference и производственной постобработки маски.",
        "- Во всех вариантах одинаковы conf=0.10, IoU/NMS=0.50, max_det=20, фильтры площади и dilation=5.",
        "",
        "## Результаты",
        "",
        "| Runtime | FPS | mean, ms | p95, ms | J/IoU | F | J&F | recall | leakage | over-redaction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['runtime']} | {row['fps']:.2f} | {row['mean_ms']:.2f} | "
            f"{row['p95_ms']:.2f} | {row['iou_j']:.4f} | {row['boundary_f']:.4f} | "
            f"{row['j_and_f']:.4f} | {row['privacy_recall']:.4f} | "
            f"{row['leakage']:.4f} | {row['over_redaction']:.4f} |"
        )
    ov_delta = report["int8_vs_openvino_fp32"]
    pt_delta = report["int8_vs_pytorch_fp32"]
    agreement = report["mask_agreement_int8_vs_openvino_fp32"]
    markdown.extend(
        [
            "",
            "## Вывод",
            "",
            f"- INT8 против OpenVINO FP32: {ov_delta['fps_multiplier']:.2f}x по FPS; "
            f"J&F {ov_delta['j_and_f_delta'] * 100:+.2f} п.п.; leakage {ov_delta['leakage_delta'] * 100:+.2f} п.п.",
            f"- INT8 против текущего PyTorch FP32: FPS {pt_delta['fps_percent']:+.1f}%, "
            f"средняя задержка {pt_delta['latency_percent']:+.1f}%; J&F {pt_delta['j_and_f_delta'] * 100:+.2f} п.п.",
            f"- Размер артефакта уменьшен на {report['artifacts']['size_reduction_percent']:.1f}%.",
            f"- Согласие масок INT8 и OpenVINO FP32: mean IoU {agreement['mean_iou']:.4f}; "
            f"минимум {agreement['min_iou']:.4f} на кадре {agreement['min_frame']}.",
            f"- Худший leakage INT8: {percentage(rows[2]['worst_leakage'])} на кадре "
            f"{rows[2]['worst_leakage_frame']}. Для privacy-фильтра это требует отдельной проверки "
            "на целевых пользовательских видео перед включением по умолчанию.",
            "",
        ]
    )
    (root / "REPORT.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
