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
from .virtual_camera import VirtualCameraSink, virtual_camera_fps


WINDOW_TITLE = "YOLOE image-prompt privacy filter (Q/Esc to quit)"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = PROJECT_ROOT / "models"
DEFAULT_YOLOE_PATH = (
    MODELS_ROOT / "yoloe" / "yoloe-26n-seg_int8_openvino_model"
)


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
    detections: int
    used_fallback: bool
    objects: tuple[ObjectAnnotation, ...]
    rejected_masks: int = 0


@dataclass(frozen=True)
class ReferenceSegmentation:
    mask: np.ndarray
    predicted_iou: float
    stability_score: float
    area_ratio: float


@dataclass(frozen=True)
class ReferencePrototype:
    image: np.ndarray
    mask: np.ndarray
    object_index: int
    path: Path
    uses_sam_mask: bool


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


def build_reference_prototypes(
    reference_groups: Iterable[tuple[Path, ...]],
    reference_masks: dict[Path, np.ndarray] | None,
    *,
    maximum_size: int,
    crop_padding_ratio: float = 0.05,
) -> list[ReferencePrototype]:
    """Create one independently encoded, exact-mask prototype per reference."""
    if maximum_size <= 0:
        raise ValueError("reference maximum size must be positive")
    prototypes: list[ReferencePrototype] = []
    for object_index, group in enumerate(reference_groups):
        for path in group:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise ValueError(f"Could not decode reference image: {path}")
            mask = reference_masks.get(path) if reference_masks is not None else None
            if mask is None:
                mask = np.ones(image.shape[:2], dtype=bool)
                uses_sam_mask = False
            else:
                mask = np.asarray(mask, dtype=bool)
                if mask.shape != image.shape[:2]:
                    raise ValueError(
                        f"Reference mask shape {mask.shape} does not match "
                        f"{path.name} shape {image.shape[:2]}"
                    )
                if not np.any(mask):
                    raise ValueError(f"Reference mask is empty: {path}")
                uses_sam_mask = True

                x, y, width, height = cv2.boundingRect(mask.astype(np.uint8))
                pad_x = max(2, round(width * crop_padding_ratio))
                pad_y = max(2, round(height * crop_padding_ratio))
                crop_x1 = max(0, x - pad_x)
                crop_y1 = max(0, y - pad_y)
                crop_x2 = min(image.shape[1], x + width + pad_x)
                crop_y2 = min(image.shape[0], y + height + pad_y)
                image = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                mask = mask[crop_y1:crop_y2, crop_x1:crop_x2].copy()

            height, width = image.shape[:2]
            scale = min(1.0, maximum_size / max(height, width))
            if scale < 1.0:
                resized_size = (
                    max(1, round(width * scale)),
                    max(1, round(height * scale)),
                )
                image = cv2.resize(image, resized_size, interpolation=cv2.INTER_AREA)
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    resized_size,
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            prototypes.append(
                ReferencePrototype(
                    image=image,
                    mask=mask,
                    object_index=object_index,
                    path=path,
                    uses_sam_mask=uses_sam_mask,
                )
            )
    if not prototypes:
        raise ValueError("At least one reference image is required")
    return prototypes


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
    mask_output_directory: Path | None = None,
) -> dict[Path, np.ndarray]:
    """Run a small SAM 2 model once, then release it before realtime inference."""
    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2_hf
        from sam2.utils.amg import rle_to_mask
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "Reference extraction requires the official facebookresearch/sam2 "
            "package. Reinstall requirements.txt."
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
    resolved_mask_directory = (
        mask_output_directory.expanduser().resolve()
        if mask_output_directory is not None
        else None
    )
    if resolved_mask_directory is not None:
        resolved_mask_directory.mkdir(parents=True, exist_ok=True)
        print(f"Reference SAM masks: {resolved_mask_directory}")
    try:
        with torch_module.inference_mode(), autocast:
            for reference_index, path in enumerate(reference_paths):
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
                saved_path: Path | None = None
                if resolved_mask_directory is not None:
                    saved_path = (
                        resolved_mask_directory
                        / f"{reference_index:03d}_{path.stem}.png"
                    )
                    save_mask_image(selected.mask, saved_path)
                print(
                    f"Reference SAM: {path.name} foreground={selected.area_ratio:.1%}, "
                    f"IoU={selected.predicted_iou:.3f}, "
                    f"stability={selected.stability_score:.3f}"
                    + (f", mask={saved_path}" if saved_path is not None else "")
                )
    finally:
        del generator, sam_model
        if runtime.torch_device == "cuda":
            torch_module.cuda.empty_cache()
    return masks


def _fixed_prompt_onnx_path(
    model_source: str,
    prototypes: Iterable[ReferencePrototype],
    *,
    yolo_imgsz: int,
    yolo_reference_imgsz: int,
) -> Path:
    """Return a content-addressed cache path for a fixed-prompt FP32 export."""
    digest = hashlib.sha256()
    digest.update(
        _reference_prompt_fingerprint(
            prototypes,
            yolo_imgsz=yolo_imgsz,
            yolo_reference_imgsz=yolo_reference_imgsz,
        ).encode()
    )
    model_path = Path(model_source)
    if model_path.is_file():
        digest.update(_sha256_file(model_path).encode())
    else:
        digest.update(model_source.encode())
    cache_directory = MODELS_ROOT / "yoloe" / "onnx"
    cache_directory.mkdir(parents=True, exist_ok=True)
    model_stem = model_path.stem or "yoloe"
    return cache_directory / f"{model_stem}-{digest.hexdigest()[:16]}-fp32.onnx"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_prompt_fingerprint(
    prototypes: Iterable[ReferencePrototype],
    *,
    yolo_imgsz: int,
    yolo_reference_imgsz: int,
) -> str:
    """Hash the exact visual-prompt inputs and their class grouping."""
    digest = hashlib.sha256()
    digest.update(b"yoloe-fixed-visual-prompts-exact-mask-per-reference-v3")
    digest.update(str(yolo_imgsz).encode())
    digest.update(str(yolo_reference_imgsz).encode())
    for prototype in prototypes:
        digest.update(np.asarray(prototype.image.shape, dtype=np.int32).tobytes())
        digest.update(prototype.image.tobytes())
        digest.update(np.asarray(prototype.mask.shape, dtype=np.int32).tobytes())
        digest.update(np.packbits(prototype.mask).tobytes())
        digest.update(str(prototype.object_index).encode())
    return digest.hexdigest()


def _openvino_reference_fingerprint(model_directory: Path) -> str | None:
    metadata_path = model_directory / "metadata.yaml"
    if not metadata_path.is_file():
        return None
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == "visual_prompt_sha256":
            candidate = value.strip().strip("'\"").lower()
            if len(candidate) == 64 and all(
                character in "0123456789abcdef" for character in candidate
            ):
                return candidate
            return None
    return None


def _append_openvino_reference_fingerprint(
    model_directory: Path,
    fingerprint: str,
) -> None:
    metadata_path = model_directory / "metadata.yaml"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"OpenVINO metadata not found: {metadata_path}")
    if _openvino_reference_fingerprint(model_directory) is not None:
        raise RuntimeError(f"OpenVINO metadata already has a fingerprint: {metadata_path}")
    with metadata_path.open("a", encoding="utf-8") as stream:
        stream.write(f"visual_prompt_sha256: {fingerprint}\n")


def encode_reference_prototypes(
    model: Any,
    prototypes: Iterable[ReferencePrototype],
    *,
    predictor_type: type,
    device: str | int,
    imgsz: int,
    torch_module: Any,
) -> tuple[int, ...]:
    """Extract one exact-mask VPE per photo and install them as pseudo-classes."""
    prototype_list = list(prototypes)
    if not prototype_list:
        raise ValueError("At least one reference prototype is required")

    class ExactMaskPredictor(predictor_type):
        """Rasterize masks with the same letterbox geometry as the reference."""

        def _process_single_image(
            self,
            dst_shape: tuple[int, int],
            src_shape: tuple[int, int],
            category: np.ndarray,
            bboxes: np.ndarray | None = None,
            masks: np.ndarray | None = None,
        ) -> Any:
            if masks is None:
                return super()._process_single_image(
                    dst_shape,
                    src_shape,
                    category,
                    bboxes,
                    masks,
                )
            mask_values = np.asarray(masks, dtype=np.uint8)
            if mask_values.ndim == 2:
                mask_values = mask_values[None, ...]
            if mask_values.ndim != 3 or mask_values.shape[1:] != src_shape:
                raise ValueError(
                    f"Expected reference masks shaped (N, {src_shape[0]}, {src_shape[1]}), "
                    f"got {mask_values.shape}"
                )
            categories = np.asarray(category, dtype=np.int32).reshape(-1)
            if len(categories) != len(mask_values):
                raise ValueError("Reference masks and classes must have equal length")

            dst_height, dst_width = dst_shape
            src_height, src_width = src_shape
            gain = min(dst_height / src_height, dst_width / src_width)
            resized_width = max(1, round(src_width * gain))
            resized_height = max(1, round(src_height * gain))
            left = round((dst_width - resized_width) / 2 - 0.1)
            top = round((dst_height - resized_height) / 2 - 0.1)
            prompt_height = max(1, int(dst_height / 8))
            prompt_width = max(1, int(dst_width / 8))
            unique_categories, inverse = np.unique(categories, return_inverse=True)
            visuals = np.zeros(
                (len(unique_categories), prompt_height, prompt_width),
                dtype=np.float32,
            )
            for class_index, mask in zip(inverse, mask_values):
                resized = cv2.resize(
                    mask,
                    (resized_width, resized_height),
                    interpolation=cv2.INTER_NEAREST,
                )
                letterboxed = np.zeros((dst_height, dst_width), dtype=np.uint8)
                letterboxed[
                    top : top + resized_height,
                    left : left + resized_width,
                ] = resized
                prompt_mask = cv2.resize(
                    letterboxed,
                    (prompt_width, prompt_height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                if not np.any(prompt_mask):
                    raise ValueError(
                        "Reference foreground disappeared at the YOLOE prompt "
                        f"resolution {prompt_width}x{prompt_height}"
                    )
                visuals[class_index] = np.logical_or(
                    visuals[class_index],
                    prompt_mask,
                )
            return torch_module.from_numpy(visuals)

    predictor = ExactMaskPredictor(
        overrides={
            "task": model.task,
            "mode": "predict",
            "save": False,
            "verbose": False,
            "batch": 1,
            "device": device,
            "imgsz": imgsz,
        }
    )
    predictor.setup_model(model=model.model, verbose=False)

    embeddings: list[Any] = []
    prototype_to_object: list[int] = []
    for prototype_index, prototype in enumerate(prototype_list):
        predictor.set_prompts(
            {
                "masks": np.asarray([prototype.mask], dtype=np.uint8),
                "cls": np.asarray([0], dtype=np.int32),
            }
        )
        with torch_module.inference_mode():
            embedding = predictor.get_vpe(prototype.image)
        if embedding.ndim != 3 or embedding.shape[0] != 1 or embedding.shape[1] != 1:
            raise RuntimeError(
                "Expected one visual embedding per reference image, got "
                f"{tuple(embedding.shape)} for {prototype.path.name}"
            )
        embeddings.append(embedding)
        prototype_to_object.append(prototype.object_index)
        foreground_ratio = float(np.count_nonzero(prototype.mask)) / prototype.mask.size
        print(
            f"YOLOE reference prototype {prototype_index}: {prototype.path.name}, "
            f"object={prototype.object_index}, exact-mask={prototype.uses_sam_mask}, "
            f"foreground={foreground_ratio:.1%}"
        )

    combined = torch_module.cat(embeddings, dim=1)
    model.set_classes(
        [f"prototype{index}" for index in range(len(prototype_list))],
        combined,
    )
    return tuple(prototype_to_object)


def _export_fixed_prompt_openvino_int8(
    *,
    yoloe_type: type,
    initialize_visual_prompts: Callable[[Any], None],
    source_model: Path,
    calibration_data: Path,
    cache_root: Path,
    reference_fingerprint: str,
    imgsz: int,
    device: str | int,
    torch_module: Any,
) -> Path:
    if not source_model.is_file():
        raise FileNotFoundError(
            "Automatic INT8 export requires the original YOLOE checkpoint: "
            f"{source_model}"
        )
    if not calibration_data.is_file():
        raise FileNotFoundError(
            "Automatic INT8 export requires a calibration dataset YAML: "
            f"{calibration_data}"
        )
    artifact_digest = hashlib.sha256()
    artifact_digest.update(reference_fingerprint.encode())
    artifact_digest.update(_sha256_file(source_model).encode())
    artifact_digest.update(_sha256_file(calibration_data).encode())
    artifact_key = artifact_digest.hexdigest()
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / (
        f"{source_model.stem}-{artifact_key[:20]}_int8_openvino_model"
    )
    if target.is_dir():
        if _openvino_reference_fingerprint(target) == reference_fingerprint:
            print(f"Using cached fixed-prompt YOLOE INT8: {target}")
            return target
        raise RuntimeError(
            "The content-addressed INT8 cache entry is invalid; remove it and retry: "
            f"{target}"
        )

    print(
        "Reference fingerprint does not match the configured OpenVINO model; "
        f"exporting a new INT8 cache entry: {reference_fingerprint}"
    )
    with TemporaryDirectory(
        prefix=f".{artifact_key[:12]}-",
        dir=cache_root,
    ) as temporary_directory_name:
        temporary_directory = Path(temporary_directory_name).resolve()
        staged_checkpoint = temporary_directory / source_model.name
        shutil.copy2(source_model, staged_checkpoint)
        export_model = yoloe_type(str(staged_checkpoint))
        initialize_visual_prompts(export_model)
        exported = Path(
            export_model.export(
                format="openvino",
                imgsz=imgsz,
                batch=1,
                dynamic=False,
                quantize=8,
                data=str(calibration_data),
                fraction=1.0,
                device=device,
            )
        ).resolve()
        if not exported.is_dir() or not exported.is_relative_to(temporary_directory):
            raise RuntimeError(f"Unexpected YOLOE OpenVINO export path: {exported}")
        _append_openvino_reference_fingerprint(exported, reference_fingerprint)
        shutil.move(str(exported), str(target))
        del export_model
        if str(device).startswith("cuda") or device == 0:
            torch_module.cuda.empty_cache()
    print(f"Saved fixed-prompt YOLOE INT8 cache: {target}")
    return target


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
        raise ValueError("fp16 is not supported for this CPU pipeline; use fp32")
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


def save_mask_image(mask: np.ndarray, output_path: Path) -> None:
    """Save a full-frame boolean segmentation mask as a viewable PNG image."""
    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {values.shape}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = values.astype(bool).astype(np.uint8) * 255
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write segmentation mask: {output_path}")


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
    show_bboxes: bool = True,
    show_statistics: bool = True,
    mirror_statistics: bool = False,
) -> np.ndarray:
    """Draw independently configurable boxes and diagnostic text."""
    output = frame.copy()
    if result is not None and show_bboxes:
        for annotation in result.objects:
            color = (0, 200, 255) if result.used_fallback else (50, 230, 80)
            x1, y1, x2, y2 = annotation.box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
    if not show_statistics:
        return output

    canvas = cv2.flip(output, 1) if mirror_statistics else output
    if result is None:
        status = "ERROR: fail-closed"
        yolo_text = "-"
        object_count = 0
    else:
        if result.rejected_masks:
            status = f"rejected-large-mask:{result.rejected_masks}"
        else:
            status = "fallback" if result.used_fallback else "detected"
        yolo_text = f"{result.yolo_ms:.1f}"
        object_count = len(result.objects)
    if error:
        status = "ERROR: fail-closed"
    header = (
        f"FPS {fps:.1f} | frame {processing_ms:.1f} ms | "
        f"YOLOE {yolo_text} ms | IoU matching | "
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
        canvas,
        (5, 5),
        (min(canvas.shape[1] - 1, text_width + 17), text_height + baseline + 15),
        (20, 20, 20),
        thickness=-1,
    )
    cv2.putText(
        canvas,
        header,
        (11, text_height + 10),
        font,
        font_scale,
        (80, 230, 255) if error else (80, 255, 120),
        thickness,
        cv2.LINE_AA,
    )

    if result is None:
        return cv2.flip(canvas, 1) if mirror_statistics else canvas
    for annotation in result.objects:
        original_x1, y1, original_x2, _ = annotation.box
        if mirror_statistics:
            x1 = canvas.shape[1] - original_x2
        else:
            x1 = original_x1
        color = (0, 200, 255) if result.used_fallback else (50, 230, 80)
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
        label = (
            f"id={annotation.track_id} ref={reference} "
            f"yolo={annotation.detector_confidence:.2f} iou={edge_score}"
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
            canvas,
            (x1, label_y - label_height - label_baseline - 4),
            (min(canvas.shape[1] - 1, x1 + label_width + 6), label_y + 2),
            (20, 20, 20),
            thickness=-1,
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 3, label_y - label_baseline),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return cv2.flip(canvas, 1) if mirror_statistics else canvas


class YoloESamPipeline:
    def __init__(
        self,
        *,
        reference_groups: list[tuple[Path, ...]],
        yolo_model_id: str,
        yolo_onnx: bool,
        device: str,
        precision: str,
        yolo_imgsz: int,
        yolo_reference_imgsz: int,
        reference_size: int,
        reference_sam_model_id: str,
        reference_sam_points: int,
        reference_sam_min_area_ratio: float,
        reference_sam_max_area_ratio: float,
        reference_sam_mask_output_directory: Path | None,
        yolo_confidence: float,
        yolo_iou: float,
        min_mask_area: int,
        max_mask_area_ratio: float,
        max_objects: int,
        mask_dilation: int,
        iou_threshold: float,
        iou_max_missed: int,
        yolo_auto_quantize: bool = False,
        yolo_source_model_id: str = "models/yoloe/yoloe-26n-seg.pt",
        yolo_int8_calibration_data: Path | None = None,
        yolo_int8_cache_directory: Path | None = None,
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
        self.dtype = _torch_dtype(torch, self.runtime.precision)
        self.yolo_imgsz = yolo_imgsz
        self.yolo_reference_imgsz = yolo_reference_imgsz
        self.yolo_confidence = yolo_confidence
        self.yolo_iou = yolo_iou
        self.min_mask_area = min_mask_area
        self.max_mask_area_ratio = max_mask_area_ratio
        self.max_objects = max_objects
        self.mask_dilation = mask_dilation
        self.iou_threshold = iou_threshold
        self.iou_max_missed = iou_max_missed
        self.frame_index = 0
        self.iou_tracks: dict[int, IouTrack] = {}
        self.next_iou_track_id = 1
        reference_paths = [path for group in reference_groups for path in group]

        reference_masks = extract_reference_masks_with_sam(
            reference_paths,
            model_id=reference_sam_model_id,
            runtime=self.runtime,
            torch_module=torch,
            points_per_side=reference_sam_points,
            minimum_area_ratio=reference_sam_min_area_ratio,
            maximum_area_ratio=reference_sam_max_area_ratio,
            mask_output_directory=reference_sam_mask_output_directory,
        )
        prototypes = build_reference_prototypes(
            reference_groups,
            reference_masks,
            maximum_size=reference_size,
        )
        self.prototype_to_reference = tuple(
            prototype.object_index for prototype in prototypes
        )
        self.reference_prompt_sha256 = _reference_prompt_fingerprint(
            prototypes,
            yolo_imgsz=yolo_imgsz,
            yolo_reference_imgsz=yolo_reference_imgsz,
        )
        self.reference_prototypes = len(prototypes)
        self.reference_sam_model = reference_sam_model_id
        self.segmented_references = len(reference_masks or ())
        self.reference_sam_mask_output_directory = (
            reference_sam_mask_output_directory.expanduser().resolve()
            if reference_sam_mask_output_directory is not None
            else None
        )
        self.saved_reference_sam_masks = (
            len(reference_masks)
            if self.reference_sam_mask_output_directory is not None
            else 0
        )

        print(
            "Image-prompt runtime: "
            f"device={self.runtime.torch_device}, precision={self.runtime.precision}, "
            f"tracker=iou, groups={len(reference_groups)}, "
            f"references={len(reference_paths)}, prototypes={len(prototypes)}, "
            f"SAM-foreground={self.segmented_references}/{len(reference_paths)}"
        )
        self.yolo_model_source = resolve_yolo_model_source(yolo_model_id)
        self.yolo_runtime_source = self.yolo_model_source
        self.yolo_backend = "pytorch"
        print(f"YOLOE weights: {self.yolo_model_source}")
        print(
            "Model input sizes: "
            f"YOLOE frames={self.yolo_imgsz}x{self.yolo_imgsz}, "
            f"YOLOE references={self.yolo_reference_imgsz}x{self.yolo_reference_imgsz}"
        )
        def initialize_visual_prompts(model: Any) -> None:
            mapping = encode_reference_prototypes(
                model,
                prototypes,
                predictor_type=YOLOEVPSegPredictor,
                device=self.runtime.yolo_device,
                imgsz=self.yolo_reference_imgsz,
                torch_module=torch,
            )
            if mapping != self.prototype_to_reference:
                raise RuntimeError("Reference prototype mapping changed during encoding")

        source_path = Path(self.yolo_model_source)
        openvino_ir_files = (
            list(source_path.glob("*.xml")) if source_path.is_dir() else []
        )
        if source_path.is_dir() and len(openvino_ir_files) == 1:
            if yolo_onnx:
                raise ValueError(
                    "A fixed-prompt OpenVINO model directory cannot be combined "
                    "with FP32 ONNX export"
                )
            embedded_fingerprint = _openvino_reference_fingerprint(source_path)
            if embedded_fingerprint != self.reference_prompt_sha256:
                if not yolo_auto_quantize:
                    raise ValueError(
                        "The reference fingerprint does not match the fixed-prompt "
                        f"OpenVINO model (expected {self.reference_prompt_sha256}, "
                        f"found {embedded_fingerprint or 'none'}). Enable automatic "
                        "INT8 quantization or select a matching model."
                    )
                source_model = Path(
                    resolve_yolo_model_source(yolo_source_model_id)
                )
                if yolo_int8_calibration_data is None:
                    raise ValueError(
                        "Automatic INT8 quantization requires calibration data"
                    )
                calibration_data = yolo_int8_calibration_data.expanduser()
                if not calibration_data.is_absolute():
                    calibration_data = PROJECT_ROOT / calibration_data
                cache_root = (
                    yolo_int8_cache_directory.expanduser()
                    if yolo_int8_cache_directory is not None
                    else PROJECT_ROOT / ".cache" / "yoloe" / "int8"
                )
                if not cache_root.is_absolute():
                    cache_root = PROJECT_ROOT / cache_root
                source_path = _export_fixed_prompt_openvino_int8(
                    yoloe_type=YOLOE,
                    initialize_visual_prompts=initialize_visual_prompts,
                    source_model=source_model.resolve(),
                    calibration_data=calibration_data.resolve(),
                    cache_root=cache_root.resolve(),
                    reference_fingerprint=self.reference_prompt_sha256,
                    imgsz=self.yolo_imgsz,
                    device=self.runtime.yolo_device,
                    torch_module=torch,
                )
            self.yolo = YOLO(str(source_path), task="segment")
            self.yolo_runtime_source = str(source_path)
            self.yolo_backend = "openvino-fixed-prompt"
            print(
                "Using fingerprint-matched fixed-prompt OpenVINO YOLOE: "
                f"{self.reference_prompt_sha256}"
            )
        elif yolo_onnx:
            if not source_path.is_file():
                raise ValueError("FP32 ONNX export requires a local YOLOE .pt file")
            onnx_path = _fixed_prompt_onnx_path(
                self.yolo_model_source,
                prototypes,
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

    def _detect_objects(self, frame: np.ndarray) -> list[Detection]:
        results = self.yolo.predict(
            frame,
            imgsz=self.yolo_imgsz,
            conf=self.yolo_confidence,
            iou=self.yolo_iou,
            agnostic_nms=True,
            device=self.runtime.yolo_device,
            max_det=self.max_objects,
            retina_masks=True,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []
        boxes = results[0].boxes.xyxy.detach().float().cpu().numpy()
        confidences = results[0].boxes.conf.detach().float().cpu().numpy()
        classes = results[0].boxes.cls.detach().int().cpu().numpy()
        mask_values: np.ndarray | None = None
        if results[0].masks is not None:
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
            prototype_index = int(classes[index])
            if not 0 <= prototype_index < len(self.prototype_to_reference):
                continue
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
                    reference_index=self.prototype_to_reference[prototype_index],
                    mask=mask,
                )
            )
        return detections

    def invalidate_tracking(self) -> None:
        """Discard IoU state after an inference failure."""
        self.iou_tracks = {}

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
            detections=len(valid_detections),
            used_fallback=used_fallback,
            objects=tuple(annotations),
            rejected_masks=rejected_masks,
        )

    def process(self, frame: np.ndarray) -> FrameResult:
        return self._process_iou(frame)


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
    virtual_camera: bool,
    max_frames: int,
    benchmark_path: Path | None,
    reference_sam_mask_output_directory: Path | None,
    yolo_model_id: str,
    yolo_onnx: bool,
    yolo_auto_quantize: bool,
    yolo_source_model_id: str,
    yolo_int8_calibration_data: Path | None,
    yolo_int8_cache_directory: Path | None,
    device: str,
    precision: str,
    yolo_imgsz: int,
    yolo_reference_imgsz: int,
    reference_size: int,
    reference_sam_model_id: str,
    reference_sam_points: int,
    reference_sam_min_area_ratio: float,
    reference_sam_max_area_ratio: float,
    yolo_confidence: float,
    yolo_iou: float,
    min_mask_area: int,
    max_mask_area_ratio: float,
    max_objects: int,
    mask_dilation: int,
    pixel_block_size: int,
    blur_kernel_size: int,
    redaction_mode: str,
    fail_closed: bool,
    show_bboxes: bool,
    show_statistics: bool,
    iou_threshold: float,
    iou_max_missed: int,
) -> dict[str, Any]:
    _validate_unit_interval("YOLO confidence", yolo_confidence)
    _validate_unit_interval("YOLO IoU", yolo_iou)
    _validate_unit_interval("IoU association threshold", iou_threshold)
    if not 0.0 < max_mask_area_ratio <= 1.0:
        raise ValueError("maximum mask area ratio must be greater than 0 and at most 1")
    if yolo_imgsz <= 0 or yolo_reference_imgsz <= 0 or reference_size <= 0:
        raise ValueError(
            "YOLO frame, YOLO reference, and reference maximum sizes must be positive"
        )
    if yolo_imgsz % 32 != 0 or yolo_reference_imgsz % 32 != 0:
        raise ValueError("YOLO input sizes must be divisible by 32")
    if not reference_sam_model_id.strip():
        raise ValueError("Reference SAM model ID cannot be empty")
    if not 2 <= reference_sam_points <= 64:
        raise ValueError("Reference SAM points per side must be from 2 to 64")
    if not 0.0 < reference_sam_min_area_ratio < reference_sam_max_area_ratio <= 1.0:
        raise ValueError(
            "Reference SAM area ratios must satisfy 0 < minimum < maximum <= 1"
        )
    if (
        min_mask_area < 0
        or mask_dilation < 0
        or iou_max_missed < 0
    ):
        raise ValueError(
            "Mask area, dilation, and missed/fallback frame counts cannot be negative"
        )
    if (
        max_objects <= 0
        or pixel_block_size <= 0
        or blur_kernel_size <= 1
    ):
        raise ValueError(
            "Object count, redetection interval, pixel block size, and blur kernel "
            "must be positive"
        )

    reference_groups = resolve_reference_groups(reference_images)
    paths = [path for group in reference_groups for path in group]
    pipeline = YoloESamPipeline(
        reference_groups=reference_groups,
        yolo_model_id=yolo_model_id,
        yolo_onnx=yolo_onnx,
        yolo_auto_quantize=yolo_auto_quantize,
        yolo_source_model_id=yolo_source_model_id,
        yolo_int8_calibration_data=yolo_int8_calibration_data,
        yolo_int8_cache_directory=yolo_int8_cache_directory,
        device=device,
        precision=precision,
        yolo_imgsz=yolo_imgsz,
        yolo_reference_imgsz=yolo_reference_imgsz,
        reference_size=reference_size,
        reference_sam_model_id=reference_sam_model_id,
        reference_sam_points=reference_sam_points,
        reference_sam_min_area_ratio=reference_sam_min_area_ratio,
        reference_sam_max_area_ratio=reference_sam_max_area_ratio,
        reference_sam_mask_output_directory=reference_sam_mask_output_directory,
        yolo_confidence=yolo_confidence,
        yolo_iou=yolo_iou,
        min_mask_area=min_mask_area,
        max_mask_area_ratio=max_mask_area_ratio,
        max_objects=max_objects,
        mask_dilation=mask_dilation,
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
    virtual_sink: VirtualCameraSink | None = None
    frames = 0
    failures = 0
    fallback_count = 0
    rejected_mask_count = 0
    yolo_times: list[float] = []
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
        if virtual_camera:
            sink_fps = virtual_camera_fps(source.info.fps, camera_fps)
            virtual_sink = VirtualCameraSink(
                source.info.width,
                source.info.height,
                sink_fps,
            )
            print(
                f"Virtual camera: {virtual_sink.device} "
                f"({source.info.width}x{source.info.height} at {sink_fps:.1f} FPS)"
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
                fallback_count += int(result.used_fallback)
                rejected_mask_count += result.rejected_masks
                yolo_times.append(result.yolo_ms)
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
            if show_bboxes or show_statistics:
                output = draw_diagnostics(
                    output,
                    result,
                    fps=rolling_fps,
                    processing_ms=processing_ms,
                    runtime=pipeline.runtime,
                    error=frame_error,
                    show_bboxes=show_bboxes,
                    show_statistics=show_statistics,
                    mirror_statistics=not mirror,
                )
            if virtual_sink is not None:
                virtual_sink.submit(output)
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
        if virtual_sink is not None:
            virtual_sink.close()
        if writer is not None:
            writer.release()
        if preview:
            cv2.destroyWindow(WINDOW_TITLE)
        if completed and temporary_path is not None and final_path is not None:
            temporary_path.replace(final_path)

    elapsed = perf_counter() - started_at
    summary: dict[str, Any] = {
        "mode": "image-prompt-sam-yoloe-iou",
        "source": str(video_path.resolve()) if video_path is not None else f"camera:{camera_index}",
        "output": str(final_path) if final_path is not None else None,
        "reference_sam_mask_output_directory": (
            str(pipeline.reference_sam_mask_output_directory)
            if pipeline.reference_sam_mask_output_directory is not None
            else None
        ),
        "saved_reference_sam_masks": pipeline.saved_reference_sam_masks,
        "virtual_camera": virtual_camera,
        "virtual_camera_device": (
            virtual_sink.device if virtual_sink is not None else None
        ),
        "reference_images": [str(path) for path in paths],
        "reference_groups": [
            [str(path) for path in group] for group in reference_groups
        ],
        "runtime": {
            "device": pipeline.runtime.torch_device,
            "precision": pipeline.runtime.precision,
            "tracker": "iou",
            "yolo_model": pipeline.yolo_model_source,
            "yolo_runtime_model": pipeline.yolo_runtime_source,
            "yolo_backend": pipeline.yolo_backend,
            "reference_sam_model": pipeline.reference_sam_model,
            "segmented_references": pipeline.segmented_references,
            "reference_prototypes": pipeline.reference_prototypes,
            "reference_prompt_sha256": pipeline.reference_prompt_sha256,
            "reference_prompt_policy": "exact_mask_per_photo_max_over_prototypes",
            "input_sizes": {
                "yoloe_frames": yolo_imgsz,
                "yoloe_references": yolo_reference_imgsz,
                "reference_maximum_side": reference_size,
            },
        },
        "thresholds": {
            "yolo_confidence": yolo_confidence,
            "yolo_iou": yolo_iou,
            "minimum_mask_area": min_mask_area,
            "maximum_mask_area_ratio": max_mask_area_ratio,
            "iou_association": iou_threshold,
            "iou_max_missed": iou_max_missed,
        },
        "frames": frames,
        "fallback_frames": fallback_count,
        "rejected_large_masks": rejected_mask_count,
        "failures": failures,
        "elapsed_seconds": elapsed,
        "effective_fps": frames / elapsed if elapsed > 0 else 0.0,
        "latency_ms": {
            "yolo": _distribution(yolo_times),
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
