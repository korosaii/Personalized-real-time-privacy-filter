from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-weights", type=Path, required=True)
    parser.add_argument("--phase1-checkpoint", type=Path)
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs" / "yolo11n-face-pose.yaml")
    parser.add_argument("--pose-data", type=Path, default=ROOT / "data" / "processed" / "widerface_pose.yaml")
    parser.add_argument(
        "--detection-data",
        type=Path,
        default=ROOT / "data" / "processed" / "widerface_detection_official_val.yaml",
    )
    parser.add_argument(
        "--official-pose-data",
        type=Path,
        default=ROOT / "data" / "processed" / "widerface_pose_official_val.yaml",
    )
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--name")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--phase1-epochs", type=int, default=12)
    parser.add_argument("--phase2-epochs", type=int, default=15)
    parser.add_argument("--phase1-lr", type=float, default=1e-3)
    parser.add_argument("--phase2-lr", type=float, default=1e-4)
    parser.add_argument("--phase1-degrees", type=float, default=90.0)
    parser.add_argument("--phase2-degrees", type=float, default=75.0)
    parser.add_argument("--pose-gain", type=float, default=12.0)
    parser.add_argument("--kobj-gain", type=float, default=1.0)
    parser.add_argument("--box-gain", type=float, default=7.5)
    parser.add_argument("--cls-gain", type=float, default=0.5)
    parser.add_argument("--dfl-gain", type=float, default=1.5)
    parser.add_argument("--unfreeze-from-layer", type=int, default=7)
    parser.add_argument("--max-box-map50-drop", type=float, default=0.03)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("MPS is not available in this Python/PyTorch environment")
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"{label} not found: {resolved}")
    return resolved


def serializable_metrics(metrics) -> dict[str, float]:
    payload: dict[str, float] = {}
    for key, value in metrics.results_dict.items():
        try:
            payload[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return payload


def metric_value(metrics: dict[str, float], name: str) -> float:
    if name not in metrics:
        available = ", ".join(sorted(metrics))
        raise SystemExit(f"Metric {name} was not produced. Available metrics: {available}")
    return metrics[name]


def csv_loss_summary(path: Path) -> dict:
    if not path.is_file():
        return {"file": str(path), "rows": 0}
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keys = (
        "train/box_loss",
        "train/pose_loss",
        "train/kobj_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "val/box_loss",
        "val/pose_loss",
        "val/kobj_loss",
        "val/cls_loss",
        "val/dfl_loss",
    )
    summary: dict[str, object] = {"file": str(path), "rows": len(rows)}
    if not rows:
        return summary
    first: dict[str, float] = {}
    last: dict[str, float] = {}
    for key in keys:
        try:
            first[key] = float(rows[0][key])
            last[key] = float(rows[-1][key])
        except (KeyError, TypeError, ValueError):
            continue
    summary["first_epoch"] = first
    summary["last_epoch"] = last
    detection_keys = ("val/box_loss", "val/cls_loss", "val/dfl_loss")
    if all(key in first and key in last for key in detection_keys):
        start = sum(first[key] for key in detection_keys)
        end = sum(last[key] for key in detection_keys)
        summary["validation_detection_loss_sum_first"] = start
        summary["validation_detection_loss_sum_last"] = end
        summary["validation_detection_loss_ratio"] = end / start if start else None
    return summary


def train_arguments(
    args: argparse.Namespace,
    device: str,
    project: Path,
    degrees: float,
) -> dict:
    return {
        "data": str(args.pose_data),
        "device": device,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(project),
        "exist_ok": False,
        "optimizer": "AdamW",
        "cos_lr": True,
        "warmup_epochs": 1.0,
        "weight_decay": 5e-4,
        "amp": False if device == "mps" else True,
        "cache": False,
        "seed": args.seed,
        "deterministic": True,
        "fraction": args.fraction,
        "box": args.box_gain,
        "cls": args.cls_gain,
        "dfl": args.dfl_gain,
        "pose": args.pose_gain,
        "kobj": args.kobj_gain,
        "degrees": degrees,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5,
        "mosaic": 0.5,
        "mixup": 0.0,
        "plots": True,
        "verbose": True,
    }


def main() -> None:
    args = parse_args()
    args.source_weights = require_file(args.source_weights, "Source detector weights")
    if args.phase1_checkpoint is not None:
        args.phase1_checkpoint = require_file(
            args.phase1_checkpoint,
            "Phase 1 checkpoint",
        )
    args.model_config = require_file(args.model_config, "YOLO11 Pose config")
    args.pose_data = require_file(args.pose_data, "Pose dataset YAML")
    args.detection_data = require_file(args.detection_data, "Detection validation YAML")
    args.official_pose_data = require_file(args.official_pose_data, "Pose-compatible official validation YAML")
    if args.phase2_epochs < 1:
        raise SystemExit("Phase 2 must contain at least one epoch")
    if args.phase1_checkpoint is None and args.phase1_epochs < 1:
        raise SystemExit("Phase 1 must contain at least one epoch")
    if not 0.0 < args.fraction <= 1.0:
        raise SystemExit("--fraction must be in the (0, 1] range")
    if not 0.0 <= args.phase1_degrees <= 180.0:
        raise SystemExit("--phase1-degrees must be between 0 and 180")
    if not 0.0 <= args.phase2_degrees <= 180.0:
        raise SystemExit("--phase2-degrees must be between 0 and 180")
    if not 0 <= args.unfreeze_from_layer <= 22:
        raise SystemExit("--unfreeze-from-layer must be between 0 and 22")
    device = resolve_device(args.device)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = args.name or f"yolo11n-face-landmarks-{timestamp}"
    project = args.runs.expanduser().resolve() / run_name
    if project.exists():
        raise SystemExit(f"Run directory already exists: {project}")
    project.mkdir(parents=True)
    print(f"Device: {device}")
    print(f"Run: {project}")
    print("Evaluating the original detector on official WIDER FACE validation...")
    baseline_model = YOLO(str(args.source_weights))
    source_is_pose = baseline_model.task == "pose"
    baseline_data = args.official_pose_data if source_is_pose else args.detection_data
    baseline_result = baseline_model.val(
        data=str(baseline_data),
        device=device,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=str(project),
        name="baseline_detection",
        exist_ok=False,
        plots=True,
    )
    baseline_metrics = serializable_metrics(baseline_result)
    if args.phase1_checkpoint is not None:
        phase1_best = args.phase1_checkpoint
        phase1_last = args.phase1_checkpoint
        print(f"Phase 1: reusing checkpoint {phase1_last}")
    else:
        phase1_common = train_arguments(
            args,
            device,
            project,
            args.phase1_degrees,
        )
        print(
            "Phase 1: training only the landmark branch with random roll in "
            f"[-{args.phase1_degrees:.0f}, +{args.phase1_degrees:.0f}] degrees..."
        )
        if source_is_pose:
            phase1_model = YOLO(str(args.source_weights))
        else:
            phase1_model = YOLO(str(args.model_config))
            phase1_model.load(str(args.source_weights))
        phase1_freeze = [*range(23), "23.cv2", "23.cv3"]
        phase1_model.train(
            **phase1_common,
            name="phase1_landmarks",
            epochs=args.phase1_epochs,
            patience=args.phase1_epochs,
            lr0=args.phase1_lr,
            lrf=0.1,
            close_mosaic=min(3, args.phase1_epochs),
            freeze=phase1_freeze,
        )
        phase1_best = Path(phase1_model.trainer.best).resolve()
        phase1_last = Path(phase1_model.trainer.last).resolve()
        print(f"Phase 1 best upright-validation weights: {phase1_best}")
        print(f"Phase 1 selected large-roll weights: {phase1_last}")
    phase2_common = train_arguments(
        args,
        device,
        project,
        args.phase2_degrees,
    )
    print(
        "Phase 2: fine-tuning late backbone, neck, detection head, and landmark "
        f"head with roll in [-{args.phase2_degrees:.0f}, "
        f"+{args.phase2_degrees:.0f}] degrees..."
    )
    phase2_model = YOLO(str(phase1_last))
    phase2_model.train(
        **phase2_common,
        name="phase2_joint",
        epochs=args.phase2_epochs,
        patience=args.phase2_epochs,
        lr0=args.phase2_lr,
        lrf=0.1,
        close_mosaic=min(3, args.phase2_epochs),
        freeze=list(range(args.unfreeze_from_layer)),
    )
    phase2_best = Path(phase2_model.trainer.best).resolve()
    phase2_last = Path(phase2_model.trainer.last).resolve()
    phase2_selected = phase2_last
    print(f"Phase 2 best upright-validation weights: {phase2_best}")
    print(f"Phase 2 selected large-roll weights: {phase2_selected}")
    print("Evaluating bbox and landmarks on the held-out landmark images...")
    final_model = YOLO(str(phase2_selected))
    final_result = final_model.val(
        data=str(args.pose_data),
        device=device,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=str(project),
        name="final_pose_validation",
        exist_ok=False,
        plots=True,
    )
    final_metrics = serializable_metrics(final_result)
    print("Evaluating final bbox quality on official WIDER FACE validation...")
    official_result = final_model.val(
        data=str(args.official_pose_data),
        device=device,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=str(project),
        name="final_official_detection",
        exist_ok=False,
        plots=True,
    )
    official_metrics = serializable_metrics(official_result)
    baseline_map50 = metric_value(baseline_metrics, "metrics/mAP50(B)")
    final_map50 = metric_value(official_metrics, "metrics/mAP50(B)")
    map50_drop = baseline_map50 - final_map50
    loss_summary = csv_loss_summary(project / "phase2_joint" / "results.csv")
    loss_ratio = loss_summary.get("validation_detection_loss_ratio")
    loss_warning = isinstance(loss_ratio, float) and loss_ratio > 1.15
    quality_passed = map50_drop <= args.max_box_map50_drop
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "source_weights": str(args.source_weights),
        "source_task": baseline_model.task,
        "pose_model_config": str(args.model_config),
        "phase1_best": str(phase1_best),
        "phase1_last": str(phase1_last),
        "phase1_reused": args.phase1_checkpoint is not None,
        "phase2_best": str(phase2_best),
        "phase2_last": str(phase2_last),
        "phase2_selected": str(phase2_selected),
        "settings": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "phase1_epochs": args.phase1_epochs,
            "phase2_epochs": args.phase2_epochs,
            "phase1_lr": args.phase1_lr,
            "phase2_lr": args.phase2_lr,
            "phase1_degrees": args.phase1_degrees,
            "phase2_degrees": args.phase2_degrees,
            "pose_gain": args.pose_gain,
            "kobj_gain": args.kobj_gain,
            "box_gain": args.box_gain,
            "cls_gain": args.cls_gain,
            "dfl_gain": args.dfl_gain,
            "unfreeze_from_layer": args.unfreeze_from_layer,
            "max_box_map50_drop": args.max_box_map50_drop,
            "fraction": args.fraction,
            "seed": args.seed,
        },
        "baseline_detection_metrics": baseline_metrics,
        "landmark_holdout_metrics": final_metrics,
        "final_official_detection_metrics": official_metrics,
        "box_map50_drop": map50_drop,
        "box_quality_gate_passed": quality_passed,
        "phase2_losses": loss_summary,
        "detection_loss_warning": loss_warning,
    }
    summary_path = project / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary: {summary_path}")
    if loss_warning:
        print("Warning: validation box+class+DFL loss rose by more than 15% during phase 2", file=sys.stderr)
    if not quality_passed:
        print(
            f"Detection quality gate failed: box mAP50 drop {map50_drop:.4f} exceeds {args.max_box_map50_drop:.4f}",
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
