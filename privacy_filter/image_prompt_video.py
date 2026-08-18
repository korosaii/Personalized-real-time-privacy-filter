from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from .camera import Camera, VideoFile
from .redaction import blur_entire_frame, blur_mask, pixelate_mask, redact_entire_frame


WINDOW_TITLE = "YOLOE image-prompt privacy filter (Q/Esc to quit)"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = PROJECT_ROOT / "models"
DEFAULT_YOLOE_PATH = MODELS_ROOT / "yoloe" / "yoloe-26n-seg.pt"
DEFAULT_EDGETAM_REPO = "yonigozlan/EdgeTAM-hf"
DEFAULT_EDGETAM_PATH = MODELS_ROOT / "edgetam" / "EdgeTAM-hf"


@dataclass(frozen=True)
class RuntimeSelection:
    torch_device: str
    yolo_device: str | int
    precision: str


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    confidence: float
    reference_index: int
    mask: np.ndarray | None = None


@dataclass
class IouTrack:
    track_id: int
    detection: Detection
    missed_frames: int = 0
    last_iou: float | None = None


@dataclass(frozen=True)
class ObjectAnnotation:
    box: tuple[int, int, int, int]
    track_id: int
    reference_index: int
    detector_confidence: float
    tracker_score: float | None


@dataclass(frozen=True)
class FrameResult:
    mask: np.ndarray
    yolo_ms: float
    edgetam_ms: float
    detections: int
    keyframe: bool
    used_fallback: bool
    objects: tuple[ObjectAnnotation, ...]
    rejected_masks: int = 0
    tracker_kind: str = "edgetam"


@dataclass(frozen=True)
class ReferenceSegmentation:
    mask: np.ndarray
    predicted_iou: float
    stability_score: float
    area_ratio: float


@dataclass(frozen=True)
class ReferencePromptVariant:
    image: np.ndarray
    box: tuple[float, float, float, float]
    class_index: int
    path: Path
    kind: str


def resolve_yolo_model_source(value: str | Path) -> str:
    """Resolve relative YOLO weights under the project's models directory."""
    text = str(value)
    if "://" in text:
        return text
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (
            MODELS_ROOT / "yoloe" / candidate.name
            if candidate.parent == Path(".")
            else PROJECT_ROOT / candidate
        )
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return str(candidate.resolve())


def materialize_edgetam_source(model_id_or_path: str) -> str:
    """Download a Hub EdgeTAM snapshot into models/edgetam or use a local model."""
    candidate = Path(model_id_or_path).expanduser()
    if candidate.is_dir():
        return str(candidate.resolve())

    normalized = model_id_or_path.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) != 2 or Path(model_id_or_path).is_absolute():
        raise FileNotFoundError(f"Local EdgeTAM model directory not found: {candidate}")

    repo_id = "/".join(parts)
    local_dir = (
        DEFAULT_EDGETAM_PATH
        if repo_id == DEFAULT_EDGETAM_REPO
        else MODELS_ROOT / "edgetam" / parts[-1]
    )
    required_files = (
        local_dir / "config.json",
        local_dir / "preprocessor_config.json",
        local_dir / "model.safetensors",
    )
    if all(path.is_file() for path in required_files):
        return str(local_dir.resolve())

    try:
        from huggingface_hub import snapshot_download
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "huggingface_hub is required to download EdgeTAM into models/edgetam"
        ) from error

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading EdgeTAM {repo_id} to {local_dir.resolve()}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=("*.json", "*.safetensors"),
    )
    return str(local_dir.resolve())


def resolve_reference_groups(values: Iterable[str | Path]) -> list[tuple[Path, ...]]:
    """Treat every directory as one object class containing multiple views."""
    groups: list[tuple[Path, ...]] = []
    seen: set[Path] = set()
    for value in values:
        candidate = Path(value).expanduser().resolve()
        if candidate.is_dir():
            candidates = [
                path.resolve()
                for path in sorted(candidate.iterdir())
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ]
        elif candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            candidates = [candidate]
        elif not candidate.exists():
            raise FileNotFoundError(f"Reference image not found: {candidate}")
        else:
            raise ValueError(f"Unsupported reference image: {candidate}")
        group = tuple(path for path in candidates if path not in seen)
        if group:
            groups.append(group)
            seen.update(group)
    if not groups:
        raise ValueError("No supported reference images were found")
    return groups


def resolve_reference_paths(values: Iterable[str | Path]) -> list[Path]:
    """Compatibility helper returning the flattened reference group list."""
    return [path for group in resolve_reference_groups(values) for path in group]


def _build_reference_variants(
    reference_groups: Iterable[tuple[Path, ...]],
    reference_masks: dict[Path, np.ndarray] | None,
    *,
    crop_padding_ratio: float = 0.05,
) -> list[ReferencePromptVariant]:
    variants: list[ReferencePromptVariant] = []
    for class_index, group in enumerate(reference_groups):
        for path in group:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise ValueError(f"Could not decode reference image: {path}")
            mask = reference_masks.get(path) if reference_masks is not None else None
            if mask is None:
                variants.append(
                    ReferencePromptVariant(
                        image=image,
                        box=(0.0, 0.0, float(image.shape[1]), float(image.shape[0])),
                        class_index=class_index,
                        path=path,
                        kind="natural",
                    )
                )
                continue

            mask = np.asarray(mask, dtype=bool)
            if mask.shape != image.shape[:2]:
                raise ValueError(
                    f"Reference mask shape {mask.shape} does not match "
                    f"{path.name} shape {image.shape[:2]}"
                )
            if not np.any(mask):
                raise ValueError(f"Reference mask is empty: {path}")
            x, y, width, height = cv2.boundingRect(mask.astype(np.uint8))
            pad_x = max(2, round(width * crop_padding_ratio))
            pad_y = max(2, round(height * crop_padding_ratio))
            crop_x1 = max(0, x - pad_x)
            crop_y1 = max(0, y - pad_y)
            crop_x2 = min(image.shape[1], x + width + pad_x)
            crop_y2 = min(image.shape[0], y + height + pad_y)
            natural = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            mask_crop = mask[crop_y1:crop_y2, crop_x1:crop_x2]
            masked = np.full_like(natural, 114)
            masked[mask_crop] = natural[mask_crop]
            prompt_box = (
                float(x - crop_x1),
                float(y - crop_y1),
                float(x + width - crop_x1),
                float(y + height - crop_y1),
            )
            for kind, variant_image in (("natural", natural), ("masked", masked)):
                variants.append(
                    ReferencePromptVariant(
                        image=variant_image,
                        box=prompt_box,
                        class_index=class_index,
                        path=path,
                        kind=kind,
                    )
                )
    return variants


def build_reference_gallery(
    reference_groups: Iterable[tuple[Path, ...]],
    gallery_size: int,
    reference_masks: dict[Path, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place natural/masked views into a gallery with shared per-object IDs."""
    if gallery_size < 128:
        raise ValueError("reference gallery size must be at least 128 pixels")
    groups = list(reference_groups)
    variants = _build_reference_variants(groups, reference_masks)
    if not variants:
        raise ValueError("At least one reference image is required")

    columns = math.ceil(math.sqrt(len(variants)))
    rows = math.ceil(len(variants) / columns)
    cell_width = gallery_size // columns
    cell_height = gallery_size // rows
    padding = max(2, min(cell_width, cell_height) // 32)
    if cell_width <= padding * 2 or cell_height <= padding * 2:
        raise ValueError("Too many references for the selected gallery size")

    gallery = np.full((gallery_size, gallery_size, 3), 114, dtype=np.uint8)
    boxes: list[list[float]] = []
    classes: list[int] = []
    for index, variant in enumerate(variants):
        image = variant.image
        row, column = divmod(index, columns)
        available_width = cell_width - padding * 2
        available_height = cell_height - padding * 2
        scale = min(
            available_width / image.shape[1],
            available_height / image.shape[0],
        )
        resized_width = max(1, round(image.shape[1] * scale))
        resized_height = max(1, round(image.shape[0] * scale))
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        x1 = column * cell_width + (cell_width - resized_width) // 2
        y1 = row * cell_height + (cell_height - resized_height) // 2
        x2, y2 = x1 + resized_width, y1 + resized_height
        gallery[y1:y2, x1:x2] = resized
        box_x1, box_y1, box_x2, box_y2 = variant.box
        boxes.append(
            [
                float(x1 + box_x1 * scale),
                float(y1 + box_y1 * scale),
                float(x1 + box_x2 * scale),
                float(y1 + box_y2 * scale),
            ]
        )
        classes.append(variant.class_index)
    return (
        gallery,
        np.asarray(boxes, dtype=np.float32),
        np.asarray(classes, dtype=np.int32),
    )


def _mask_border_contact(mask: np.ndarray) -> float:
    border_pixels = np.concatenate((mask[0], mask[-1], mask[:, 0], mask[:, -1]))
    return float(np.count_nonzero(border_pixels)) / max(1, border_pixels.size)


def _highest_confidence_non_overlapping(
    boxes: np.ndarray,
    confidences: np.ndarray,
    *,
    overlap_threshold: float,
    maximum: int,
) -> np.ndarray:
    """Suppress nested/cross-class duplicates using overlap over the smaller box."""
    kept: list[int] = []
    for index in np.argsort(-confidences):
        x1, y1, x2, y2 = (float(value) for value in boxes[index])
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        duplicate = False
        for kept_index in kept:
            kx1, ky1, kx2, ky2 = (float(value) for value in boxes[kept_index])
            kept_area = max(0.0, kx2 - kx1) * max(0.0, ky2 - ky1)
            intersection = max(0.0, min(x2, kx2) - max(x1, kx1)) * max(
                0.0,
                min(y2, ky2) - max(y1, ky1),
            )
            smaller_area = min(area, kept_area)
            overlap = intersection / smaller_area if smaller_area > 0.0 else 0.0
            if overlap >= overlap_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(int(index))
            if len(kept) >= maximum:
                break
    return np.asarray(kept, dtype=np.int64)


def _select_reference_mask(
    annotations: Iterable[dict[str, Any]],
    image_shape: tuple[int, int],
    *,
    minimum_area_ratio: float,
    maximum_area_ratio: float,
    mask_decoder: Callable[[Any], np.ndarray] | None = None,
) -> ReferenceSegmentation | None:
    """Pick the stable, central foreground mask from SAM's automatic proposals."""
    height, width = image_shape
    image_area = float(height * width)
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    normalizer = max(1.0, math.hypot(width / 2.0, height / 2.0))
    best: tuple[float, ReferenceSegmentation] | None = None

    for annotation in annotations:
        raw_mask = annotation.get("segmentation")
        if isinstance(raw_mask, dict):
            if mask_decoder is None:
                continue
            raw_mask = mask_decoder(raw_mask)
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.shape != (height, width) or not np.any(mask):
            continue
        area_ratio = float(np.count_nonzero(mask)) / image_area
        if not minimum_area_ratio <= area_ratio <= maximum_area_ratio:
            continue
        x, y, box_width, box_height = cv2.boundingRect(mask.astype(np.uint8))
        box_center_x = x + box_width / 2.0
        box_center_y = y + box_height / 2.0
        center_distance = math.hypot(
            box_center_x - center_x,
            box_center_y - center_y,
        )
        center_proximity = max(0.0, 1.0 - center_distance / normalizer)
        contains_center = float(mask[round(center_y), round(center_x)])
        predicted_iou = float(annotation.get("predicted_iou", 0.0))
        stability = float(annotation.get("stability_score", 0.0))
        border_penalty = _mask_border_contact(mask)
        # Centrality identifies the intended object without a manual click. Quality
        # terms choose a complete SAM mask; border contact suppresses walls/background.
        score = (
            2.0 * contains_center
            + 0.80 * center_proximity
            + 0.60 * predicted_iou
            + 0.45 * stability
            + 0.45 * math.sqrt(area_ratio)
            - 0.35 * border_penalty
        )
        selected = ReferenceSegmentation(
            mask=mask,
            predicted_iou=predicted_iou,
            stability_score=stability,
            area_ratio=area_ratio,
        )
        if best is None or score > best[0]:
            best = (score, selected)
    return best[1] if best is not None else None


def extract_reference_masks_with_sam(
    reference_paths: Iterable[Path],
    *,
    model_id: str,
    runtime: RuntimeSelection,
    torch_module: Any,
    points_per_side: int,
    minimum_area_ratio: float,
    maximum_area_ratio: float,
) -> dict[Path, np.ndarray]:
    """Run a small SAM 2 model once, then release it before realtime inference."""
    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2_hf
        from sam2.utils.amg import rle_to_mask
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "Reference extraction requires the official facebookresearch/sam2 "
            "package. Reinstall requirements.txt or disable it with "
            "--no-image-reference-sam."
        ) from error

    print(f"Loading one-shot reference SAM 2: {model_id}")
    sam_model = build_sam2_hf(model_id, device=runtime.torch_device)
    points_per_batch = min(points_per_side * points_per_side, 32)
    if runtime.torch_device == "cpu":
        points_per_batch = min(points_per_batch, 8)
    generator = SAM2AutomaticMaskGenerator(
        sam_model,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=0.70,
        stability_score_thresh=0.85,
        crop_n_layers=0,
        min_mask_region_area=0,
        output_mode="uncompressed_rle",
    )
    autocast = (
        torch_module.autocast(
            device_type="cuda",
            dtype=_torch_dtype(torch_module, runtime.precision),
        )
        if runtime.torch_device == "cuda" and runtime.precision != "fp32"
        else nullcontext()
    )
    masks: dict[Path, np.ndarray] = {}
    try:
        with torch_module.inference_mode(), autocast:
            for path in reference_paths:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None or image.size == 0:
                    raise ValueError(f"Could not decode reference image: {path}")
                annotations = generator.generate(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                selected = _select_reference_mask(
                    annotations,
                    image.shape[:2],
                    minimum_area_ratio=minimum_area_ratio,
                    maximum_area_ratio=maximum_area_ratio,
                    mask_decoder=rle_to_mask,
                )
                if selected is None:
                    print(
                        f"Reference SAM warning: no usable foreground for {path.name}; "
                        "using the original image"
                    )
                    continue
                masks[path] = selected.mask
                print(
                    f"Reference SAM: {path.name} foreground={selected.area_ratio:.1%}, "
                    f"IoU={selected.predicted_iou:.3f}, "
                    f"stability={selected.stability_score:.3f}"
                )
    finally:
        del generator, sam_model
        if runtime.torch_device == "cuda":
            torch_module.cuda.empty_cache()
    return masks


def _fixed_prompt_onnx_path(
    model_source: str,
    gallery: np.ndarray,
    prompt_boxes: np.ndarray,
    prompt_classes: np.ndarray,
    *,
    yolo_imgsz: int,
    yolo_reference_imgsz: int,
) -> Path:
    """Return a content-addressed cache path for a fixed-prompt FP32 export."""
    digest = hashlib.sha256()
    digest.update(b"yoloe-fixed-visual-prompts-natural-masked-v1")
    digest.update(str(yolo_imgsz).encode())
    digest.update(str(yolo_reference_imgsz).encode())
    digest.update(gallery.tobytes())
    digest.update(prompt_boxes.tobytes())
    digest.update(prompt_classes.tobytes())
    model_path = Path(model_source)
    if model_path.is_file():
        with model_path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    else:
        digest.update(model_source.encode())
    cache_directory = MODELS_ROOT / "yoloe" / "onnx"
    cache_directory.mkdir(parents=True, exist_ok=True)
    model_stem = model_path.stem or "yoloe"
    return cache_directory / f"{model_stem}-{digest.hexdigest()[:16]}-fp32.onnx"


def select_runtime(torch_module: Any, device: str, precision: str) -> RuntimeSelection:
    requested_device = device.lower()
    if requested_device == "auto":
        if torch_module.cuda.is_available():
            requested_device = "cuda"
        elif (
            hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        ):
            requested_device = "mps"
        else:
            requested_device = "cpu"
    elif requested_device == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    elif requested_device == "mps" and not (
        hasattr(torch_module.backends, "mps")
        and torch_module.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested, but it is not available")

    requested_precision = precision.lower()
    if requested_precision == "auto":
        if requested_device == "cuda":
            requested_precision = (
                "bf16" if torch_module.cuda.is_bf16_supported() else "fp16"
            )
        elif requested_device == "mps":
            requested_precision = "fp16"
        else:
            requested_precision = "fp32"
    if requested_device == "cpu" and requested_precision == "fp16":
        raise ValueError("fp16 is not supported for this EdgeTAM CPU pipeline; use fp32")
    if requested_precision == "bf16" and requested_device == "mps":
        raise ValueError("bf16 is not supported by this MPS pipeline; use fp16")
    if (
        requested_precision == "bf16"
        and requested_device == "cuda"
        and not torch_module.cuda.is_bf16_supported()
    ):
        raise ValueError("bf16 was requested, but this CUDA device does not support it")

    yolo_device: str | int = 0 if requested_device == "cuda" else requested_device
    return RuntimeSelection(requested_device, yolo_device, requested_precision)


def _torch_dtype(torch_module: Any, precision: str) -> Any:
    return {
        "fp32": torch_module.float32,
        "fp16": torch_module.float16,
        "bf16": torch_module.bfloat16,
    }[precision]


def configure_edgetam_resolution(config: Any, image_size: int) -> Any:
    """Keep all EdgeTAM spatial config fields consistent for custom inference size."""
    if image_size < 256 or image_size > 1024 or image_size % 64 != 0:
        raise ValueError(
            "EdgeTAM input size must be from 256 to 1024 and divisible by 64"
        )
    config.image_size = image_size
    config.prompt_encoder_config.image_size = image_size
    config.vision_config.backbone_feature_sizes = [
        [image_size // 4, image_size // 4],
        [image_size // 8, image_size // 8],
        [image_size // 16, image_size // 16],
    ]
    config.memory_attention_rope_feat_sizes = [
        image_size // 16,
        image_size // 16,
    ]
    return config


def _validate_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0 or not np.any(mask):
        return mask
    kernel_size = pixels * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def draw_diagnostics(
    frame: np.ndarray,
    result: FrameResult | None,
    *,
    fps: float,
    processing_ms: float,
    runtime: RuntimeSelection,
    error: bool = False,
) -> np.ndarray:
    """Draw rolling performance and per-object detector/tracker values."""
    output = frame.copy()
    if result is None:
        status = "ERROR: fail-closed"
        yolo_text = "-"
        edge_text = "-"
        object_count = 0
    else:
        if result.rejected_masks:
            status = f"rejected-large-mask:{result.rejected_masks}"
        else:
            status = "fallback" if result.used_fallback else (
                "keyframe" if result.keyframe else "tracking"
            )
        yolo_text = f"{result.yolo_ms:.1f}" if result.keyframe else "skip"
        edge_text = f"{result.edgetam_ms:.1f}" if result.edgetam_ms else "-"
        object_count = len(result.objects)
    if error:
        status = "ERROR: fail-closed"
    tracker_metric = (
        f"EdgeTAM {edge_text} ms"
        if result is None or result.tracker_kind == "edgetam"
        else "IoU matching"
    )
    header = (
        f"FPS {fps:.1f} | frame {processing_ms:.1f} ms | "
        f"YOLOE {yolo_text} ms | {tracker_metric} | "
        f"objects {object_count} | {runtime.torch_device}/{runtime.precision} | {status}"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        header,
        font,
        font_scale,
        thickness,
    )
    cv2.rectangle(
        output,
        (5, 5),
        (min(output.shape[1] - 1, text_width + 17), text_height + baseline + 15),
        (20, 20, 20),
        thickness=-1,
    )
    cv2.putText(
        output,
        header,
        (11, text_height + 10),
        font,
        font_scale,
        (80, 230, 255) if error else (80, 255, 120),
        thickness,
        cv2.LINE_AA,
    )

    if result is None:
        return output
    for annotation in result.objects:
        x1, y1, x2, y2 = annotation.box
        color = (0, 200, 255) if result.used_fallback else (50, 230, 80)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        reference = (
            str(annotation.reference_index + 1)
            if annotation.reference_index >= 0
            else "?"
        )
        edge_score = (
            f"{annotation.tracker_score:.2f}"
            if annotation.tracker_score is not None
            else "n/a"
        )
        score_name = "edge" if result.tracker_kind == "edgetam" else "iou"
        label = (
            f"id={annotation.track_id} ref={reference} "
            f"yolo={annotation.detector_confidence:.2f} {score_name}={edge_score}"
        )
        if result.used_fallback:
            label += " fallback"
        (label_width, label_height), label_baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness,
        )
        label_y = max(label_height + label_baseline + 4, y1)
        cv2.rectangle(
            output,
            (x1, label_y - label_height - label_baseline - 4),
            (min(output.shape[1] - 1, x1 + label_width + 6), label_y + 2),
            (20, 20, 20),
            thickness=-1,
        )
        cv2.putText(
            output,
            label,
            (x1 + 3, label_y - label_baseline),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return output


class YoloEEdgeTamPipeline:
    def __init__(
        self,
        *,
        reference_groups: list[tuple[Path, ...]],
        yolo_model_id: str,
        yolo_onnx: bool,
        edgetam_model_id: str,
        device: str,
        precision: str,
        yolo_imgsz: int,
        yolo_reference_imgsz: int,
        edgetam_imgsz: int,
        reference_size: int,
        reference_sam: bool,
        reference_sam_model_id: str,
        reference_sam_points: int,
        reference_sam_min_area_ratio: float,
        reference_sam_max_area_ratio: float,
        yolo_confidence: float,
        yolo_iou: float,
        edgetam_score_threshold: float,
        mask_threshold: float,
        min_mask_area: int,
        max_mask_area_ratio: float,
        max_objects: int,
        redetect_interval: int,
        mask_dilation: int,
        fallback_frames: int,
        tracker_mode: str,
        iou_threshold: float,
        iou_max_missed: int,
    ) -> None:
        try:
            import torch
            from ultralytics import YOLO, YOLOE
            from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "YOLOE dependencies are missing. Install the project with "
                "the image-prompt extra: pip install -e '.[image-prompt]'"
            ) from error

        self.torch = torch
        self.runtime = select_runtime(torch, device, precision)
        self.tracker_mode = (
            "edgetam"
            if tracker_mode == "auto" and self.runtime.torch_device in {"cuda", "mps"}
            else "iou" if tracker_mode == "auto" else tracker_mode
        )
        if self.tracker_mode not in {"edgetam", "iou"}:
            raise ValueError("image tracker must be auto, edgetam, or iou")
        self.dtype = _torch_dtype(torch, self.runtime.precision)
        self.yolo_imgsz = yolo_imgsz
        self.yolo_reference_imgsz = yolo_reference_imgsz
        self.edgetam_imgsz = edgetam_imgsz
        self.yolo_confidence = yolo_confidence
        self.yolo_iou = yolo_iou
        self.edgetam_score_threshold = edgetam_score_threshold
        self.mask_threshold = mask_threshold
        self.min_mask_area = min_mask_area
        self.max_mask_area_ratio = max_mask_area_ratio
        self.max_objects = max_objects
        self.redetect_interval = redetect_interval
        self.mask_dilation = mask_dilation
        self.fallback_frames = fallback_frames
        self.iou_threshold = iou_threshold
        self.iou_max_missed = iou_max_missed
        self.frame_index = 0
        self.frames_since_detection = redetect_interval
        self.session: Any | None = None
        self.last_mask: np.ndarray | None = None
        self.last_objects: tuple[ObjectAnnotation, ...] = ()
        self.object_metadata: dict[int, Detection] = {}
        self.iou_tracks: dict[int, IouTrack] = {}
        self.next_iou_track_id = 1
        self.fallback_age = 0
        reference_paths = [path for group in reference_groups for path in group]

        reference_masks = (
            extract_reference_masks_with_sam(
                reference_paths,
                model_id=reference_sam_model_id,
                runtime=self.runtime,
                torch_module=torch,
                points_per_side=reference_sam_points,
                minimum_area_ratio=reference_sam_min_area_ratio,
                maximum_area_ratio=reference_sam_max_area_ratio,
            )
            if reference_sam
            else None
        )
        gallery, prompt_boxes, prompt_classes = build_reference_gallery(
            reference_groups,
            reference_size,
            reference_masks,
        )
        self.reference_sam_model = reference_sam_model_id if reference_sam else None
        self.segmented_references = len(reference_masks or ())
        visual_prompts = {
            "bboxes": prompt_boxes,
            "cls": prompt_classes,
        }

        print(
            "Image-prompt runtime: "
            f"device={self.runtime.torch_device}, precision={self.runtime.precision}, "
            f"tracker={self.tracker_mode}, groups={len(reference_groups)}, "
            f"references={len(reference_paths)}, prompts={len(prompt_boxes)}, "
            f"SAM-foreground={self.segmented_references}/{len(reference_paths)}"
        )
        self.yolo_model_source = resolve_yolo_model_source(yolo_model_id)
        self.yolo_runtime_source = self.yolo_model_source
        self.yolo_backend = "pytorch"
        self.edgetam_model_source: str | None = None
        print(f"YOLOE weights: {self.yolo_model_source}")
        print(
            "Model input sizes: "
            f"YOLOE frames={self.yolo_imgsz}x{self.yolo_imgsz}, "
            f"YOLOE references={self.yolo_reference_imgsz}x{self.yolo_reference_imgsz}, "
            + (
                f"EdgeTAM={self.edgetam_imgsz}x{self.edgetam_imgsz}"
                if self.tracker_mode == "edgetam"
                else "EdgeTAM=disabled"
            )
        )
        def initialize_visual_prompts(model: Any) -> None:
            model.predict(
                gallery,
                refer_image=gallery,
                visual_prompts=visual_prompts,
                predictor=YOLOEVPSegPredictor,
                imgsz=self.yolo_reference_imgsz,
                # This result is discarded; use a permissive value so prompt
                # extraction cannot be suppressed by the runtime threshold.
                conf=0.001,
                iou=self.yolo_iou,
                device=self.runtime.yolo_device,
                verbose=False,
            )

        if yolo_onnx:
            source_path = Path(self.yolo_model_source)
            if not source_path.is_file():
                raise ValueError("FP32 ONNX export requires a local YOLOE .pt file")
            onnx_path = _fixed_prompt_onnx_path(
                self.yolo_model_source,
                gallery,
                prompt_boxes,
                prompt_classes,
                yolo_imgsz=self.yolo_imgsz,
                yolo_reference_imgsz=self.yolo_reference_imgsz,
            )
            if not onnx_path.is_file():
                print(f"Exporting fixed-prompt YOLOE FP32 ONNX: {onnx_path}")
                with TemporaryDirectory(
                    prefix="yoloe-onnx-",
                    dir=onnx_path.parent,
                ) as temporary_directory_name:
                    temporary_directory = Path(temporary_directory_name).resolve()
                    staged_checkpoint = temporary_directory / source_path.name
                    shutil.copy2(source_path, staged_checkpoint)
                    export_model = YOLOE(str(staged_checkpoint))
                    initialize_visual_prompts(export_model)
                    exported = Path(
                        export_model.export(
                            format="onnx",
                            imgsz=self.yolo_imgsz,
                            batch=1,
                            dynamic=False,
                            half=False,
                            simplify=False,
                            device=self.runtime.yolo_device,
                        )
                    ).resolve()
                    if not exported.is_file() or not exported.is_relative_to(
                        temporary_directory
                    ):
                        raise RuntimeError(
                            f"Unexpected YOLOE ONNX export path: {exported}"
                        )
                    shutil.move(str(exported), str(onnx_path))
                    del export_model
                    if self.runtime.torch_device == "cuda":
                        torch.cuda.empty_cache()
            else:
                print(f"Using cached fixed-prompt YOLOE FP32 ONNX: {onnx_path}")
            self.yolo = YOLO(str(onnx_path), task="segment")
            self.yolo_runtime_source = str(onnx_path)
            self.yolo_backend = "onnx-fp32"
        else:
            self.yolo = YOLOE(self.yolo_model_source)
            initialize_visual_prompts(self.yolo)

        self.processor: Any | None = None
        self.edgetam: Any | None = None
        if self.tracker_mode == "edgetam":
            try:
                from transformers import (
                    EdgeTamVideoConfig,
                    EdgeTamVideoModel,
                    Sam2VideoProcessor,
                )
            except (ImportError, ModuleNotFoundError) as error:
                raise RuntimeError(
                    "EdgeTAM mode requires transformers>=4.57 and its dependencies"
                ) from error
            self.edgetam_model_source = materialize_edgetam_source(edgetam_model_id)
            print(f"EdgeTAM weights: {self.edgetam_model_source}")
            edgetam_config = EdgeTamVideoConfig.from_pretrained(
                self.edgetam_model_source
            )
            configure_edgetam_resolution(edgetam_config, self.edgetam_imgsz)
            self.processor = Sam2VideoProcessor.from_pretrained(
                self.edgetam_model_source,
                size={"height": self.edgetam_imgsz, "width": self.edgetam_imgsz},
                mask_size={
                    "height": self.edgetam_imgsz // 4,
                    "width": self.edgetam_imgsz // 4,
                },
            )
            self.edgetam = EdgeTamVideoModel.from_pretrained(
                self.edgetam_model_source,
                config=edgetam_config,
                torch_dtype=self.dtype,
            ).to(self.runtime.torch_device)
            self.edgetam.eval()

    def _new_session(self) -> Any:
        return self.processor.init_video_session(
            inference_device=self.runtime.torch_device,
            inference_state_device=self.runtime.torch_device,
            processing_device=self.runtime.torch_device,
            video_storage_device="cpu",
            max_vision_features_cache_size=1,
            dtype=self.dtype,
        )

    def _detect_objects(self, frame: np.ndarray) -> list[Detection]:
        results = self.yolo.predict(
            frame,
            imgsz=self.yolo_imgsz,
            conf=self.yolo_confidence,
            iou=self.yolo_iou,
            agnostic_nms=True,
            device=self.runtime.yolo_device,
            max_det=self.max_objects,
            retina_masks=self.tracker_mode == "iou",
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []
        boxes = results[0].boxes.xyxy.detach().float().cpu().numpy()
        confidences = results[0].boxes.conf.detach().float().cpu().numpy()
        classes = results[0].boxes.cls.detach().int().cpu().numpy()
        mask_values: np.ndarray | None = None
        if self.tracker_mode == "iou" and results[0].masks is not None:
            mask_values = results[0].masks.data.detach().float().cpu().numpy()
        order = _highest_confidence_non_overlapping(
            boxes,
            confidences,
            overlap_threshold=self.yolo_iou,
            maximum=self.max_objects,
        )
        detections: list[Detection] = []
        for index in order:
            x1, y1, x2, y2 = (float(value) for value in boxes[index])
            mask: np.ndarray | None = None
            if mask_values is not None and index < len(mask_values):
                raw_mask = mask_values[index]
                if raw_mask.shape != frame.shape[:2]:
                    raw_mask = cv2.resize(
                        raw_mask,
                        (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                mask = raw_mask > 0.5
            detections.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    confidence=float(confidences[index]),
                    reference_index=int(classes[index]),
                    mask=mask,
                )
            )
        return detections

    def invalidate_tracking(self) -> None:
        """Discard possibly inconsistent video memory after an inference failure."""
        self.session = None
        self.last_mask = None
        self.last_objects = ()
        self.object_metadata = {}
        self.iou_tracks = {}
        self.fallback_age = 0
        self.frames_since_detection = self.redetect_interval

    def _mask_from_output(
        self,
        output: Any,
        original_sizes: Any,
    ) -> tuple[np.ndarray, tuple[ObjectAnnotation, ...], int]:
        masks = self.processor.post_process_masks(
            [output.pred_masks],
            original_sizes=original_sizes,
            binarize=False,
        )[0]
        while masks.ndim > 3 and masks.shape[1] == 1:
            masks = masks[:, 0]

        scores = getattr(output, "object_score_logits", None)
        score_values: np.ndarray | None = None
        if scores is not None:
            score_values = scores.detach().float().sigmoid().cpu().reshape(-1).numpy()
        mask_values = masks.detach().float().cpu().numpy()
        if mask_values.ndim == 2:
            mask_values = mask_values[None, ...]
        if not len(mask_values):
            height, width = (int(value) for value in original_sizes[0])
            return np.zeros((height, width), dtype=bool), (), 0

        object_ids = getattr(output, "object_ids", None)
        if object_ids is None:
            object_ids = getattr(self.session, "obj_ids", range(1, len(mask_values) + 1))
        if hasattr(object_ids, "detach"):
            object_ids = object_ids.detach().cpu().reshape(-1).tolist()
        elif isinstance(object_ids, (int, np.integer)):
            object_ids = [int(object_ids)]
        else:
            object_ids = list(object_ids)

        instance_masks = mask_values > self.mask_threshold
        valid_masks: list[np.ndarray] = []
        annotations: list[ObjectAnnotation] = []
        rejected_masks = 0
        for index, mask in enumerate(instance_masks):
            tracker_score = (
                float(score_values[index])
                if score_values is not None and index < len(score_values)
                else None
            )
            if tracker_score is not None and tracker_score < self.edgetam_score_threshold:
                continue
            if int(np.count_nonzero(mask)) < self.min_mask_area:
                continue
            if float(np.count_nonzero(mask)) / float(mask.size) > self.max_mask_area_ratio:
                rejected_masks += 1
                continue
            track_id = int(object_ids[index]) if index < len(object_ids) else index + 1
            metadata = self.object_metadata.get(
                track_id,
                Detection((0.0, 0.0, 0.0, 0.0), 0.0, -1),
            )
            x, y, width, height = cv2.boundingRect(mask.astype(np.uint8))
            valid_masks.append(mask)
            annotations.append(
                ObjectAnnotation(
                    box=(x, y, x + width, y + height),
                    track_id=track_id,
                    reference_index=metadata.reference_index,
                    detector_confidence=metadata.confidence,
                    tracker_score=tracker_score,
                )
            )
        if not valid_masks:
            return np.zeros(instance_masks.shape[-2:], dtype=bool), (), rejected_masks
        return np.logical_or.reduce(valid_masks), tuple(annotations), rejected_masks

    @staticmethod
    def _box_iou(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(
            0.0, second[3] - second[1]
        )
        union = first_area + second_area - intersection
        return intersection / union if union > 0.0 else 0.0

    def _process_iou(self, frame: np.ndarray) -> FrameResult:
        started = perf_counter()
        detections = self._detect_objects(frame)
        yolo_ms = (perf_counter() - started) * 1000.0

        valid_detections: list[Detection] = []
        rejected_masks = 0
        for detection in detections:
            if detection.mask is None:
                continue
            mask_area = int(np.count_nonzero(detection.mask))
            if mask_area < self.min_mask_area:
                continue
            if mask_area / float(detection.mask.size) > self.max_mask_area_ratio:
                rejected_masks += 1
                continue
            valid_detections.append(detection)

        candidates: list[tuple[float, int, int]] = []
        track_ids = list(self.iou_tracks)
        for track_id in track_ids:
            track = self.iou_tracks[track_id]
            for detection_index, detection in enumerate(valid_detections):
                iou = self._box_iou(track.detection.box, detection.box)
                if iou >= self.iou_threshold:
                    candidates.append((iou, track_id, detection_index))
        candidates.sort(reverse=True)

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for iou, track_id, detection_index in candidates:
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            track = self.iou_tracks[track_id]
            track.detection = valid_detections[detection_index]
            track.missed_frames = 0
            track.last_iou = iou
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)

        for track_id in track_ids:
            if track_id in matched_tracks:
                continue
            track = self.iou_tracks[track_id]
            track.missed_frames += 1
            track.last_iou = None
            if track.missed_frames > self.iou_max_missed:
                del self.iou_tracks[track_id]

        for detection_index, detection in enumerate(valid_detections):
            if detection_index in matched_detections:
                continue
            track_id = self.next_iou_track_id
            self.next_iou_track_id += 1
            self.iou_tracks[track_id] = IouTrack(track_id, detection)

        masks: list[np.ndarray] = []
        annotations: list[ObjectAnnotation] = []
        used_fallback = False
        for track in self.iou_tracks.values():
            mask = track.detection.mask
            if mask is None:
                continue
            masks.append(mask)
            used_fallback = used_fallback or track.missed_frames > 0
            x, y, width, height = cv2.boundingRect(mask.astype(np.uint8))
            annotations.append(
                ObjectAnnotation(
                    box=(x, y, x + width, y + height),
                    track_id=track.track_id,
                    reference_index=track.detection.reference_index,
                    detector_confidence=track.detection.confidence,
                    tracker_score=track.last_iou,
                )
            )

        union_mask = (
            np.logical_or.reduce(masks)
            if masks
            else np.zeros(frame.shape[:2], dtype=bool)
        )
        union_mask = _dilate_mask(union_mask, self.mask_dilation)
        self.frame_index += 1
        return FrameResult(
            mask=union_mask,
            yolo_ms=yolo_ms,
            edgetam_ms=0.0,
            detections=len(valid_detections),
            keyframe=True,
            used_fallback=used_fallback,
            objects=tuple(annotations),
            rejected_masks=rejected_masks,
            tracker_kind="iou",
        )

    def process(self, frame: np.ndarray) -> FrameResult:
        if self.tracker_mode == "iou":
            return self._process_iou(frame)
        return self._process_edgetam(frame)

    def _process_edgetam(self, frame: np.ndarray) -> FrameResult:
        keyframe = (
            self.session is None
            or self.frames_since_detection >= self.redetect_interval
        )
        yolo_ms = 0.0
        detections = 0
        reseeded = False
        if keyframe:
            started = perf_counter()
            detected_objects = self._detect_objects(frame)
            yolo_ms = (perf_counter() - started) * 1000.0
            detections = len(detected_objects)
            self.frames_since_detection = 0
            if detections:
                self.session = self._new_session()
                self.object_metadata = {
                    track_id: detection
                    for track_id, detection in enumerate(detected_objects, start=1)
                }
                reseeded = True

        if self.session is None:
            empty = np.zeros(frame.shape[:2], dtype=bool)
            self.last_mask = None
            self.last_objects = ()
            self.fallback_age = 0
            self.frame_index += 1
            return FrameResult(
                empty,
                yolo_ms,
                0.0,
                detections,
                keyframe,
                False,
                (),
            )

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        started = perf_counter()
        inputs = self.processor(
            images=rgb_frame,
            device=self.runtime.torch_device,
            return_tensors="pt",
        ).to(self.runtime.torch_device)
        if reseeded:
            self.processor.add_inputs_to_inference_session(
                inference_session=self.session,
                frame_idx=0,
                obj_ids=list(range(1, detections + 1)),
                input_boxes=[[list(detection.box) for detection in detected_objects]],
                original_size=tuple(int(value) for value in inputs.original_sizes[0]),
            )
        with self.torch.inference_mode():
            output = self.edgetam(
                inference_session=self.session,
                frame=inputs.pixel_values[0].to(
                    device=self.runtime.torch_device,
                    dtype=self.dtype,
                ),
            )
        mask, objects, rejected_masks = self._mask_from_output(
            output,
            inputs.original_sizes,
        )
        edgetam_ms = (perf_counter() - started) * 1000.0
        used_fallback = False
        if np.any(mask):
            mask = _dilate_mask(mask, self.mask_dilation)
            self.last_mask = mask
            self.last_objects = objects
            self.fallback_age = 0
        elif self.last_mask is not None and self.fallback_age < self.fallback_frames:
            mask = self.last_mask.copy()
            objects = self.last_objects
            self.fallback_age += 1
            used_fallback = True
        else:
            self.last_mask = None
            self.last_objects = ()
            self.object_metadata = {}
            self.fallback_age = 0
            self.session = None

        self.frames_since_detection += 1
        self.frame_index += 1
        return FrameResult(
            mask,
            yolo_ms,
            edgetam_ms,
            detections,
            keyframe,
            used_fallback,
            objects,
            rejected_masks,
        )


def _create_writer(
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> tuple[cv2.VideoWriter, Path, Path]:
    final_path = output_path.expanduser().resolve()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if not final_path.suffix:
        final_path = final_path.with_suffix(".mp4")
    temporary_path = final_path.with_name(f"{final_path.stem}.part{final_path.suffix}")
    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not create output video: {final_path}")
    return writer, temporary_path, final_path


def process_image_prompt_stream(
    *,
    reference_images: Iterable[str | Path],
    video_path: Path | None,
    output_path: Path | None,
    output_size: tuple[int, int] | None,
    camera_index: int,
    camera_width: int,
    camera_height: int,
    camera_fps: float,
    mirror: bool,
    preview: bool,
    max_frames: int,
    benchmark_path: Path | None,
    yolo_model_id: str,
    yolo_onnx: bool,
    edgetam_model_id: str,
    device: str,
    precision: str,
    yolo_imgsz: int,
    yolo_reference_imgsz: int,
    edgetam_imgsz: int,
    reference_size: int,
    reference_sam: bool,
    reference_sam_model_id: str,
    reference_sam_points: int,
    reference_sam_min_area_ratio: float,
    reference_sam_max_area_ratio: float,
    yolo_confidence: float,
    yolo_iou: float,
    edgetam_score_threshold: float,
    mask_threshold: float,
    min_mask_area: int,
    max_mask_area_ratio: float,
    max_objects: int,
    redetect_interval: int,
    mask_dilation: int,
    fallback_frames: int,
    pixel_block_size: int,
    blur_kernel_size: int,
    redaction_mode: str,
    fail_closed: bool,
    diagnostic_overlay: bool,
    tracker_mode: str,
    iou_threshold: float,
    iou_max_missed: int,
) -> dict[str, Any]:
    _validate_unit_interval("YOLO confidence", yolo_confidence)
    _validate_unit_interval("YOLO IoU", yolo_iou)
    _validate_unit_interval("EdgeTAM score threshold", edgetam_score_threshold)
    _validate_unit_interval("IoU association threshold", iou_threshold)
    if not 0.0 < max_mask_area_ratio <= 1.0:
        raise ValueError("maximum mask area ratio must be greater than 0 and at most 1")
    if yolo_imgsz <= 0 or yolo_reference_imgsz <= 0 or reference_size <= 0:
        raise ValueError("YOLO frame, YOLO reference, and gallery sizes must be positive")
    if yolo_imgsz % 32 != 0 or yolo_reference_imgsz % 32 != 0:
        raise ValueError("YOLO input sizes must be divisible by 32")
    if reference_sam and not reference_sam_model_id.strip():
        raise ValueError("Reference SAM model ID cannot be empty")
    if not 2 <= reference_sam_points <= 64:
        raise ValueError("Reference SAM points per side must be from 2 to 64")
    if not 0.0 < reference_sam_min_area_ratio < reference_sam_max_area_ratio <= 1.0:
        raise ValueError(
            "Reference SAM area ratios must satisfy 0 < minimum < maximum <= 1"
        )
    if edgetam_imgsz < 256 or edgetam_imgsz > 1024 or edgetam_imgsz % 64 != 0:
        raise ValueError(
            "EdgeTAM input size must be from 256 to 1024 and divisible by 64"
        )
    if (
        min_mask_area < 0
        or mask_dilation < 0
        or fallback_frames < 0
        or iou_max_missed < 0
    ):
        raise ValueError(
            "Mask area, dilation, and missed/fallback frame counts cannot be negative"
        )
    if (
        max_objects <= 0
        or redetect_interval <= 0
        or pixel_block_size <= 0
        or blur_kernel_size <= 1
    ):
        raise ValueError(
            "Object count, redetection interval, pixel block size, and blur kernel "
            "must be positive"
        )

    reference_groups = resolve_reference_groups(reference_images)
    paths = [path for group in reference_groups for path in group]
    pipeline = YoloEEdgeTamPipeline(
        reference_groups=reference_groups,
        yolo_model_id=yolo_model_id,
        yolo_onnx=yolo_onnx,
        edgetam_model_id=edgetam_model_id,
        device=device,
        precision=precision,
        yolo_imgsz=yolo_imgsz,
        yolo_reference_imgsz=yolo_reference_imgsz,
        edgetam_imgsz=edgetam_imgsz,
        reference_size=reference_size,
        reference_sam=reference_sam,
        reference_sam_model_id=reference_sam_model_id,
        reference_sam_points=reference_sam_points,
        reference_sam_min_area_ratio=reference_sam_min_area_ratio,
        reference_sam_max_area_ratio=reference_sam_max_area_ratio,
        yolo_confidence=yolo_confidence,
        yolo_iou=yolo_iou,
        edgetam_score_threshold=edgetam_score_threshold,
        mask_threshold=mask_threshold,
        min_mask_area=min_mask_area,
        max_mask_area_ratio=max_mask_area_ratio,
        max_objects=max_objects,
        redetect_interval=redetect_interval,
        mask_dilation=mask_dilation,
        fallback_frames=fallback_frames,
        tracker_mode=tracker_mode,
        iou_threshold=iou_threshold,
        iou_max_missed=iou_max_missed,
    )

    source: Camera | VideoFile
    source = (
        VideoFile(video_path)
        if video_path is not None
        else Camera(camera_index, camera_width, camera_height, camera_fps)
    )
    writer: cv2.VideoWriter | None = None
    temporary_path: Path | None = None
    final_path: Path | None = None
    frames = 0
    keyframes = 0
    failures = 0
    fallback_count = 0
    rejected_mask_count = 0
    yolo_times: list[float] = []
    edgetam_times: list[float] = []
    total_times: list[float] = []
    recent_frame_times: deque[float] = deque(maxlen=30)
    started_at = perf_counter()
    completed = False
    try:
        if output_path is not None:
            frame_size = output_size or (source.info.width, source.info.height)
            output_fps = source.info.fps if source.info.fps > 0 else camera_fps
            writer, temporary_path, final_path = _create_writer(
                output_path,
                output_fps,
                frame_size,
            )
        while max_frames == 0 or frames < max_frames:
            frame = source.read()
            if frame is None:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            frame_started = perf_counter()
            result: FrameResult | None = None
            frame_error = False
            try:
                result = pipeline.process(frame)
                output = (
                    blur_mask(frame, result.mask, blur_kernel_size)
                    if redaction_mode == "blur"
                    else pixelate_mask(frame, result.mask, pixel_block_size)
                )
                keyframes += int(result.keyframe)
                fallback_count += int(result.used_fallback)
                rejected_mask_count += result.rejected_masks
                if result.keyframe:
                    yolo_times.append(result.yolo_ms)
                if result.edgetam_ms:
                    edgetam_times.append(result.edgetam_ms)
            except Exception as error:
                failures += 1
                frame_error = True
                print(f"Frame {frames}: image-prompt inference failed: {error}")
                if not fail_closed:
                    raise
                pipeline.invalidate_tracking()
                output = (
                    blur_entire_frame(frame, blur_kernel_size)
                    if redaction_mode == "blur"
                    else redact_entire_frame(frame)
                )
            processing_ms = (perf_counter() - frame_started) * 1000.0
            total_times.append(processing_ms)
            recent_frame_times.append(processing_ms)
            rolling_fps = 1000.0 / max(0.001, float(np.mean(recent_frame_times)))
            if diagnostic_overlay:
                output = draw_diagnostics(
                    output,
                    result,
                    fps=rolling_fps,
                    processing_ms=processing_ms,
                    runtime=pipeline.runtime,
                    error=frame_error,
                )
            if writer is not None:
                encoded = (
                    cv2.resize(output, output_size, interpolation=cv2.INTER_LINEAR)
                    if output_size is not None
                    else output
                )
                writer.write(encoded)
            frames += 1
            if preview:
                cv2.imshow(WINDOW_TITLE, output)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
        completed = True
    finally:
        source.close()
        if writer is not None:
            writer.release()
        if preview:
            cv2.destroyWindow(WINDOW_TITLE)
        if completed and temporary_path is not None and final_path is not None:
            temporary_path.replace(final_path)

    elapsed = perf_counter() - started_at
    summary: dict[str, Any] = {
        "mode": f"image-prompt-yoloe-{pipeline.tracker_mode}",
        "source": str(video_path.resolve()) if video_path is not None else f"camera:{camera_index}",
        "output": str(final_path) if final_path is not None else None,
        "reference_images": [str(path) for path in paths],
        "reference_groups": [
            [str(path) for path in group] for group in reference_groups
        ],
        "runtime": {
            "device": pipeline.runtime.torch_device,
            "precision": pipeline.runtime.precision,
            "tracker": pipeline.tracker_mode,
            "yolo_model": pipeline.yolo_model_source,
            "yolo_runtime_model": pipeline.yolo_runtime_source,
            "yolo_backend": pipeline.yolo_backend,
            "edgetam_model": pipeline.edgetam_model_source,
            "reference_sam_model": pipeline.reference_sam_model,
            "segmented_references": pipeline.segmented_references,
            "input_sizes": {
                "yoloe_frames": yolo_imgsz,
                "yoloe_references": yolo_reference_imgsz,
                "reference_gallery": reference_size,
                "edgetam": edgetam_imgsz if pipeline.tracker_mode == "edgetam" else None,
            },
        },
        "thresholds": {
            "yolo_confidence": yolo_confidence,
            "yolo_iou": yolo_iou,
            "edgetam_score": edgetam_score_threshold,
            "mask_logit": mask_threshold,
            "minimum_mask_area": min_mask_area,
            "maximum_mask_area_ratio": max_mask_area_ratio,
            "iou_association": iou_threshold,
            "iou_max_missed": iou_max_missed,
        },
        "frames": frames,
        "keyframes": keyframes,
        "fallback_frames": fallback_count,
        "rejected_large_masks": rejected_mask_count,
        "failures": failures,
        "elapsed_seconds": elapsed,
        "effective_fps": frames / elapsed if elapsed > 0 else 0.0,
        "latency_ms": {
            "yolo_keyframes": _distribution(yolo_times),
            "edgetam": _distribution(edgetam_times),
            "total": _distribution(total_times),
        },
    }
    if benchmark_path is not None:
        resolved_benchmark = benchmark_path.expanduser().resolve()
        resolved_benchmark.parent.mkdir(parents=True, exist_ok=True)
        resolved_benchmark.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
