from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DAVIS_ROOT = PROJECT_ROOT / "data" / "DAVIS" / "DAVIS"


def read_gt(path: Path, object_id: int) -> np.ndarray:
    values = np.asarray(Image.open(path))
    if values.ndim == 3:
        values = values[..., 0]
    return values == object_id


def boundary(mask: np.ndarray) -> np.ndarray:
    values = mask.astype(np.uint8)
    eroded = cv2.erode(
        values,
        np.ones((3, 3), np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (values - eroded) > 0


def boundary_f(prediction: np.ndarray, truth: np.ndarray) -> float:
    predicted_boundary = boundary(prediction)
    truth_boundary = boundary(truth)
    radius = max(1, int(np.ceil(0.008 * np.hypot(*truth.shape))))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    dilated_truth = cv2.dilate(truth_boundary.astype(np.uint8), kernel) > 0
    dilated_prediction = cv2.dilate(
        predicted_boundary.astype(np.uint8), kernel
    ) > 0
    predicted_count = int(np.count_nonzero(predicted_boundary))
    truth_count = int(np.count_nonzero(truth_boundary))
    precision = (
        np.count_nonzero(predicted_boundary & dilated_truth) / predicted_count
        if predicted_count
        else float(truth_count == 0)
    )
    recall = (
        np.count_nonzero(truth_boundary & dilated_prediction) / truth_count
        if truth_count
        else float(predicted_count == 0)
    )
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def frame_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    intersection = int(np.count_nonzero(prediction & truth))
    predicted = int(np.count_nonzero(prediction))
    target = int(np.count_nonzero(truth))
    union = predicted + target - intersection
    background = truth.size - target
    jaccard = intersection / union if union else 1.0
    contour = boundary_f(prediction, truth)
    return {
        "iou": jaccard,
        "boundary_f": contour,
        "j_and_f": (jaccard + contour) / 2.0,
        "dice": 2.0 * intersection / (predicted + target)
        if predicted + target
        else 1.0,
        "privacy_recall": intersection / target if target else 1.0,
        "leakage": (target - intersection) / target if target else 0.0,
        "over_redaction": (predicted - intersection) / background
        if background
        else 0.0,
    }


def union_ultralytics_masks(result: Any, shape: tuple[int, int]) -> np.ndarray:
    if not result or result[0].masks is None:
        return np.zeros(shape, dtype=bool)
    masks = result[0].masks.data.detach().float().cpu().numpy()
    union = np.any(masks > 0.5, axis=0)
    if union.shape != shape:
        union = cv2.resize(
            union.astype(np.uint8),
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return union


def build_yoloe_visual(args: argparse.Namespace) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    if args.onnx and args.onnx_provider == "dml":
        import ultralytics.nn.backends.onnx as ultralytics_onnx

        # onnxruntime-directml provides the same Python module but has a different
        # distribution name, so Ultralytics' package check would otherwise replace it.
        ultralytics_onnx.check_requirements = lambda *_args, **_kwargs: None

    from privacy_filter.image_prompt_video import YoloESamPipeline

    pipeline = YoloESamPipeline(
        reference_groups=[(args.reference.resolve(),)],
        yolo_model_id=str(args.weights),
        yolo_onnx=args.onnx,
        device="cpu",
        precision="fp32",
        yolo_imgsz=args.imgsz,
        yolo_reference_imgsz=args.imgsz,
        reference_size=1280,
        reference_sam_model_id="facebook/sam2.1-hiera-tiny",
        reference_sam_points=8,
        reference_sam_min_area_ratio=0.01,
        reference_sam_max_area_ratio=0.98,
        reference_sam_mask_output_directory=args.output / "reference_masks",
        yolo_confidence=args.confidence,
        yolo_iou=0.50,
        min_mask_area=64,
        max_mask_area_ratio=0.98,
        max_objects=20,
        mask_dilation=5,
        iou_threshold=0.30,
        iou_max_missed=0,
        yolo_auto_quantize=args.auto_quantize,
        yolo_source_model_id=str(args.source_model),
        yolo_int8_calibration_data=args.calibration_data,
        yolo_int8_cache_directory=args.int8_cache_dir,
    )
    provider = "pytorch"
    if args.onnx:
        provider = "CPUExecutionProvider"
    if args.onnx and args.onnx_provider == "dml":
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if pipeline.yolo.predictor is None:
            pipeline.process(cv2.imread(str(args.reference), cv2.IMREAD_COLOR))
        backend = pipeline.yolo.predictor.model
        backend.session = ort.InferenceSession(
            str(pipeline.yolo_runtime_source),
            sess_options=options,
            providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        )
        backend.output_names = [output.name for output in backend.session.get_outputs()]
        provider = backend.session.get_providers()[0]
    return (
        lambda frame: pipeline.process(frame).mask,
        {
            "backend": pipeline.yolo_backend,
            "execution_provider": provider,
            "runtime_source": pipeline.yolo_runtime_source,
            "parameters": sum(p.numel() for p in pipeline.yolo.model.parameters())
            if pipeline.yolo_backend == "pytorch"
            else None,
        },
    )


def build_yolo_static_seg(
    args: argparse.Namespace,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """Run an exported fixed-prompt YOLOE model with production postprocessing."""
    from ultralytics import YOLO

    from privacy_filter.image_prompt_video import (
        _dilate_mask,
        _highest_confidence_non_overlapping,
    )

    model = YOLO(str(args.weights), task="segment")

    def predict(frame: np.ndarray) -> np.ndarray:
        results = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.confidence,
            iou=0.50,
            agnostic_nms=True,
            device="cpu",
            max_det=20,
            retina_masks=True,
            verbose=False,
        )
        if not results or results[0].boxes is None or results[0].masks is None:
            return np.zeros(frame.shape[:2], dtype=bool)
        boxes = results[0].boxes.xyxy.detach().float().cpu().numpy()
        confidences = results[0].boxes.conf.detach().float().cpu().numpy()
        masks = results[0].masks.data.detach().float().cpu().numpy()
        order = _highest_confidence_non_overlapping(
            boxes,
            confidences,
            overlap_threshold=0.50,
            maximum=20,
        )
        accepted: list[np.ndarray] = []
        for index in order:
            if index >= len(masks):
                continue
            mask = masks[index]
            if mask.shape != frame.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            binary = mask > 0.5
            area = int(np.count_nonzero(binary))
            if area < 64 or area / float(binary.size) > 0.98:
                continue
            accepted.append(binary)
        union = (
            np.logical_or.reduce(accepted)
            if accepted
            else np.zeros(frame.shape[:2], dtype=bool)
        )
        return _dilate_mask(union, 5)

    return predict, {
        "backend": "openvino-fixed-visual-prompt",
        "runtime_source": str(args.weights.resolve()),
        "prompt_is_static": True,
    }


def build_yoloe_text(args: argparse.Namespace) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    from ultralytics import YOLOE

    model = YOLOE(str(args.weights))
    model.set_classes([args.prompt])

    def predict(frame: np.ndarray) -> np.ndarray:
        result = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.confidence,
            iou=0.50,
            agnostic_nms=True,
            device="cpu",
            max_det=20,
            retina_masks=True,
            verbose=False,
        )
        return union_ultralytics_masks(result, frame.shape[:2])

    return predict, {
        "backend": "pytorch",
        "parameters": sum(p.numel() for p in model.model.parameters()),
    }


def build_yolo_seg(args: argparse.Namespace) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    from ultralytics import YOLO

    model = YOLO(str(args.weights))

    def predict(frame: np.ndarray) -> np.ndarray:
        result = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.confidence,
            iou=0.50,
            classes=[args.class_id],
            device="cpu",
            retina_masks=True,
            verbose=False,
        )
        return union_ultralytics_masks(result, frame.shape[:2])

    return predict, {
        "backend": "pytorch",
        "parameters": sum(p.numel() for p in model.model.parameters()),
        "class_id": args.class_id,
    }


def build_fastsam_clip(args: argparse.Namespace) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    import clip
    import torch
    from ultralytics import FastSAM

    segmenter = FastSAM(str(args.weights))
    clip_model, preprocess = clip.load(
        "ViT-B/32", device="cpu", download_root=str(PROJECT_ROOT / ".cache" / "clip")
    )
    clip_model.eval()
    reference = Image.open(args.reference).convert("RGB")
    with torch.inference_mode():
        reference_embedding = clip_model.encode_image(preprocess(reference).unsqueeze(0))
        reference_embedding /= reference_embedding.norm(dim=-1, keepdim=True)

    def predict(frame: np.ndarray) -> np.ndarray:
        results = segmenter.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.confidence,
            iou=0.90,
            device="cpu",
            retina_masks=True,
            verbose=False,
        )
        if not results or results[0].masks is None:
            return np.zeros(frame.shape[:2], dtype=bool)
        masks = results[0].masks.data.detach().float().cpu().numpy() > 0.5
        if masks.shape[1:] != frame.shape[:2]:
            masks = np.stack(
                [
                    cv2.resize(
                        mask.astype(np.uint8),
                        (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    for mask in masks
                ]
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        candidates = []
        valid_masks = []
        for mask in masks:
            x, y, width, height = cv2.boundingRect(mask.astype(np.uint8))
            if width < 4 or height < 4:
                continue
            crop = rgb[y : y + height, x : x + width].copy()
            crop_mask = mask[y : y + height, x : x + width]
            crop[~crop_mask] = 127
            candidates.append(preprocess(Image.fromarray(crop)))
            valid_masks.append(mask)
        if not candidates:
            return np.zeros(frame.shape[:2], dtype=bool)
        with torch.inference_mode():
            embeddings = clip_model.encode_image(torch.stack(candidates))
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            scores = (embeddings @ reference_embedding.T).squeeze(1)
        return valid_masks[int(torch.argmax(scores).item())]

    return predict, {
        "backend": "fastsam+pytorch-clip",
        "segmenter_parameters": sum(p.numel() for p in segmenter.model.parameters()),
        "clip": "ViT-B/32",
    }


def build_yolo_edgetam(args: argparse.Namespace) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    import torch
    from transformers import EdgeTamModel, Sam2Processor
    from ultralytics import YOLO

    detector = YOLO(str(args.weights))
    edgetam_path = PROJECT_ROOT / "models" / "edgetam" / "EdgeTAM-hf"
    processor = Sam2Processor.from_pretrained(edgetam_path, local_files_only=True)
    refiner = EdgeTamModel.from_pretrained(edgetam_path, local_files_only=True).eval()

    def predict(frame: np.ndarray) -> np.ndarray:
        results = detector.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.confidence,
            iou=0.50,
            classes=[args.class_id],
            device="cpu",
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return np.zeros(frame.shape[:2], dtype=bool)
        boxes = results[0].boxes.xyxy.detach().float().cpu().tolist()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=image, input_boxes=[boxes], return_tensors="pt")
        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key not in {"original_sizes", "reshaped_input_sizes"}
        }
        with torch.inference_mode():
            outputs = refiner(**model_inputs, multimask_output=False)
        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]
        return np.any(masks.numpy() > 0, axis=tuple(range(masks.ndim - 2)))

    return predict, {
        "backend": "yolo+pytorch-edgetam",
        "detector_parameters": sum(p.numel() for p in detector.model.parameters()),
        "edgetam_parameters": sum(p.numel() for p in refiner.parameters()),
    }


def build_yoloworld_edgetam(
    args: argparse.Namespace,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    import torch
    from transformers import EdgeTamModel, Sam2Processor
    from ultralytics import YOLOWorld

    detector = YOLOWorld(str(args.weights))
    detector.set_classes([args.prompt])
    edgetam_path = PROJECT_ROOT / "models" / "edgetam" / "EdgeTAM-hf"
    processor = Sam2Processor.from_pretrained(edgetam_path, local_files_only=True)
    refiner = EdgeTamModel.from_pretrained(edgetam_path, local_files_only=True).eval()

    def predict(frame: np.ndarray) -> np.ndarray:
        results = detector.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.confidence,
            iou=0.50,
            device="cpu",
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return np.zeros(frame.shape[:2], dtype=bool)
        boxes = results[0].boxes.xyxy.detach().float().cpu().tolist()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=image, input_boxes=[boxes], return_tensors="pt")
        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key not in {"original_sizes", "reshaped_input_sizes"}
        }
        with torch.inference_mode():
            outputs = refiner(**model_inputs, multimask_output=False)
        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]
        return np.any(masks.numpy() > 0, axis=tuple(range(masks.ndim - 2)))

    return predict, {
        "backend": "yolo-world+pytorch-edgetam",
        "detector_parameters": sum(p.numel() for p in detector.model.parameters()),
        "edgetam_parameters": sum(p.numel() for p in refiner.parameters()),
    }


def build_groundingdino_edgetam(
    args: argparse.Namespace,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    import torch
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        EdgeTamModel,
        Sam2Processor,
    )

    detector_id = str(args.weights)
    detector_processor = AutoProcessor.from_pretrained(detector_id)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id).eval()
    edgetam_path = PROJECT_ROOT / "models" / "edgetam" / "EdgeTAM-hf"
    mask_processor = Sam2Processor.from_pretrained(edgetam_path, local_files_only=True)
    refiner = EdgeTamModel.from_pretrained(edgetam_path, local_files_only=True).eval()
    text = args.prompt.rstrip(". ") + "."

    def predict(frame: np.ndarray) -> np.ndarray:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detector_inputs = detector_processor(
            images=image, text=text, return_tensors="pt"
        )
        with torch.inference_mode():
            detector_outputs = detector(**detector_inputs)
        detections = detector_processor.post_process_grounded_object_detection(
            detector_outputs,
            detector_inputs.input_ids,
            threshold=args.confidence,
            text_threshold=args.confidence,
            target_sizes=[image.size[::-1]],
        )[0]
        if len(detections["boxes"]) == 0:
            return np.zeros(frame.shape[:2], dtype=bool)
        boxes = detections["boxes"].detach().float().cpu().tolist()
        inputs = mask_processor(images=image, input_boxes=[boxes], return_tensors="pt")
        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key not in {"original_sizes", "reshaped_input_sizes"}
        }
        with torch.inference_mode():
            outputs = refiner(**model_inputs, multimask_output=False)
        masks = mask_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]
        return np.any(masks.numpy() > 0, axis=tuple(range(masks.ndim - 2)))

    return predict, {
        "backend": "grounding-dino+pytorch-edgetam",
        "detector_parameters": sum(p.numel() for p in detector.parameters()),
        "edgetam_parameters": sum(p.numel() for p in refiner.parameters()),
    }


def build_persam_mobile(args: argparse.Namespace) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    import torch
    from torch.nn import functional as torch_functional

    persam_root = PROJECT_ROOT / ".cache" / "external" / "Personalize-SAM"
    sys.path.insert(0, str(persam_root))
    from per_segment_anything import SamPredictor, sam_model_registry

    checkpoint = PROJECT_ROOT / ".cache" / "models" / "mobile_sam.pt"
    sam = sam_model_registry["vit_t"](checkpoint=str(checkpoint)).to(device="cpu").eval()
    predictor = SamPredictor(sam)
    reference_image = cv2.cvtColor(
        cv2.imread(str(args.reference), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
    )
    reference_mask_values = cv2.imread(
        str(args.reference_mask), cv2.IMREAD_GRAYSCALE
    )
    reference_mask = np.repeat(
        (reference_mask_values > 0)[..., None].astype(np.uint8) * 255, 3, axis=2
    )
    transformed_mask = predictor.set_image(reference_image, reference_mask)
    reference_features = predictor.features.squeeze().permute(1, 2, 0)
    transformed_mask = torch_functional.interpolate(
        transformed_mask, size=reference_features.shape[:2], mode="bilinear"
    ).squeeze()[0]
    target_embedding = reference_features[transformed_mask > 0].mean(0).unsqueeze(0)
    target_features = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
    target_embedding = target_embedding.unsqueeze(0)

    def select_points(similarity: Any) -> tuple[np.ndarray, np.ndarray]:
        width, height = similarity.shape
        positive_index = similarity.flatten().topk(1)[1]
        negative_index = similarity.flatten().topk(1, largest=False)[1]
        indices = torch.cat((positive_index, negative_index))
        x_values = indices // height
        y_values = indices - x_values * height
        points = torch.stack((y_values, x_values), dim=1).cpu().numpy()
        return points, np.asarray([1, 0])

    def predict(frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)
        test_features = predictor.features.squeeze()
        channels, height, width = test_features.shape
        test_features = test_features / test_features.norm(dim=0, keepdim=True)
        similarity = target_features @ test_features.reshape(channels, height * width)
        similarity = similarity.reshape(1, 1, height, width)
        similarity = torch_functional.interpolate(
            similarity, scale_factor=4, mode="bilinear"
        )
        similarity = predictor.model.postprocess_masks(
            similarity,
            input_size=predictor.input_size,
            original_size=predictor.original_size,
        ).squeeze()
        points, labels = select_points(similarity)
        normalized_similarity = (similarity - similarity.mean()) / torch.std(similarity)
        attention_similarity = torch_functional.interpolate(
            normalized_similarity[None, None], size=(64, 64), mode="bilinear"
        ).sigmoid_()[None].flatten(3)
        masks, _scores, logits, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False,
            attn_sim=attention_similarity,
            target_embedding=target_embedding,
        )
        masks, scores, logits, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            mask_input=logits[0:1],
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        y_values, x_values = np.nonzero(masks[best])
        if not len(x_values):
            return masks[best]
        box = np.asarray(
            [x_values.min(), y_values.min(), x_values.max(), y_values.max()]
        )
        masks, scores, _logits, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=box[None],
            mask_input=logits[best : best + 1],
            multimask_output=True,
        )
        return masks[int(np.argmax(scores))]

    return predict, {
        "backend": "official-persam+mobile-sam",
        "sam_parameters": sum(parameter.numel() for parameter in sam.parameters()),
        "checkpoint": str(checkpoint),
    }


def percentile(values: list[float], level: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), level))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind",
        choices=(
            "yoloe-visual",
            "yolo-static-seg",
            "yoloe-text",
            "yolo-seg",
            "fastsam-clip",
            "yolo-edgetam",
            "yoloworld-edgetam",
            "groundingdino-edgetam",
            "persam-mobile",
        ),
        required=True,
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-mask", type=Path)
    parser.add_argument("--prompt", default="swan")
    parser.add_argument("--class-id", type=int, default=14)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--onnx", action="store_true")
    parser.add_argument("--auto-quantize", action="store_true")
    parser.add_argument(
        "--source-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "yoloe" / "yoloe-26n-seg.pt",
    )
    parser.add_argument("--calibration-data", type=Path)
    parser.add_argument(
        "--int8-cache-dir",
        type=Path,
        default=PROJECT_ROOT / ".cache" / "yoloe" / "int8",
    )
    parser.add_argument(
        "--onnx-provider", choices=("cpu", "dml"), default="cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    mask_output = args.output / "masks"
    mask_output.mkdir(exist_ok=True)
    frame_dir = DEFAULT_DAVIS_ROOT / "JPEGImages" / "480p" / "blackswan"
    truth_dir = DEFAULT_DAVIS_ROOT / "Annotations" / "480p" / "blackswan"
    frame_paths = sorted(frame_dir.glob("*.jpg"))[: args.frames]
    truth_paths = sorted(truth_dir.glob("*.png"))[: args.frames]

    setup_started = perf_counter()
    builders = {
        "yoloe-visual": build_yoloe_visual,
        "yolo-static-seg": build_yolo_static_seg,
        "yoloe-text": build_yoloe_text,
        "yolo-seg": build_yolo_seg,
        "fastsam-clip": build_fastsam_clip,
        "yolo-edgetam": build_yolo_edgetam,
        "yoloworld-edgetam": build_yoloworld_edgetam,
        "groundingdino-edgetam": build_groundingdino_edgetam,
        "persam-mobile": build_persam_mobile,
    }
    predict, model_metadata = builders[args.kind](args)
    setup_ms = (perf_counter() - setup_started) * 1000.0

    warmup_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    warmup_started = perf_counter()
    predict(warmup_frame)
    warmup_ms = (perf_counter() - warmup_started) * 1000.0

    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for index, (frame_path, truth_path) in enumerate(zip(frame_paths, truth_paths)):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        truth = read_gt(truth_path, args.object_id)
        started = perf_counter()
        prediction = np.asarray(predict(frame), dtype=bool)
        latency_ms = (perf_counter() - started) * 1000.0
        if prediction.shape != truth.shape:
            prediction = cv2.resize(
                prediction.astype(np.uint8),
                (truth.shape[1], truth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        cv2.imwrite(
            str(mask_output / f"{index:09d}.png"), prediction.astype(np.uint8) * 255
        )
        metrics = frame_metrics(prediction, truth)
        rows.append({"frame": index, "latency_ms": latency_ms, **metrics})
        latencies.append(latency_ms)

    with (args.output / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metric_names = (
        "iou",
        "boundary_f",
        "j_and_f",
        "dice",
        "privacy_recall",
        "leakage",
        "over_redaction",
    )
    summary = {
        "name": args.name,
        "kind": args.kind,
        "weights": str(args.weights.resolve()),
        "reference": str(args.reference.resolve()),
        "prompt": args.prompt,
        "frames": len(rows),
        "imgsz": args.imgsz,
        "confidence": args.confidence,
        "device": "cpu",
        "python": platform.python_version(),
        "setup_ms": setup_ms,
        "warmup_ms": warmup_ms,
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "min": min(latencies),
            "max": max(latencies),
        },
        "processing_fps": 1000.0 / float(np.mean(latencies)),
        "metrics": {
            key: float(np.mean([float(row[key]) for row in rows]))
            for key in metric_names
        },
        "worst_frame": {
            "iou": min(rows, key=lambda row: row["iou"]),
            "leakage": max(rows, key=lambda row: row["leakage"]),
        },
        "model": model_metadata,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
