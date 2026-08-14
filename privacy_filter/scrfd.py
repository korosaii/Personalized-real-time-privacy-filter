from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import onnx
import onnxruntime as ort


@dataclass(frozen=True)
class DetectionResult:
    detections: np.ndarray
    keypoints: np.ndarray | None
    latency_ms: float


def distance_to_bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            points[:, 0] - distance[:, 0],
            points[:, 1] - distance[:, 1],
            points[:, 0] + distance[:, 2],
            points[:, 1] + distance[:, 3],
        ),
        axis=-1,
    )


def distance_to_keypoints(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    decoded = np.empty_like(distance, dtype=np.float32)
    decoded[:, 0::2] = points[:, 0:1] + distance[:, 0::2]
    decoded[:, 1::2] = points[:, 1:2] + distance[:, 1::2]
    return decoded.reshape((-1, 5, 2))


def non_maximum_suppression(detections: np.ndarray, threshold: float) -> list[int]:
    if detections.size == 0:
        return []

    x1, y1, x2, y2, scores = detections.T
    areas = np.maximum(0.0, x2 - x1 + 1.0) * np.maximum(0.0, y2 - y1 + 1.0)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break

        others = order[1:]
        overlap_width = np.maximum(
            0.0,
            np.minimum(x2[index], x2[others]) - np.maximum(x1[index], x1[others]) + 1.0,
        )
        overlap_height = np.maximum(
            0.0,
            np.minimum(y2[index], y2[others]) - np.maximum(y1[index], y1[others]) + 1.0,
        )
        intersection = overlap_width * overlap_height
        union = areas[index] + areas[others] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = others[iou <= threshold]

    return keep


class SCRFDDetector:
    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int] = (640, 640),
        threshold: float = 0.4,
        nms_threshold: float = 0.4,
        provider: str = "auto",
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"SCRFD ONNX model not found: {self.model_path}")
        if self.model_path.suffix.lower() != ".onnx":
            raise ValueError(
                f"SCRFD inference requires an .onnx file, got {self.model_path.name}. "
                "The downloaded .pth file is a training checkpoint."
            )
        if input_size[0] <= 0 or input_size[1] <= 0:
            raise ValueError("input_size must contain positive width and height")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")

        self.input_size = input_size
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        self._anchor_cache: dict[tuple[int, int, int, int], np.ndarray] = {}
        self.provider_warning: str | None = None
        self.static_model_input_size = self._static_model_input_size()
        if self.static_model_input_size is not None and self.static_model_input_size != self.input_size:
            raise ValueError(
                f"Static model input is {self.static_model_input_size}, but requested input_size is "
                f"{self.input_size}. Use matching --model and --det-size values."
            )
        self.has_static_input = self.static_model_input_size is not None
        self.session = self._create_session(provider)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        output_count = len(self.output_names)
        if output_count in (6, 9):
            self.strides = (8, 16, 32)
            self.num_anchors = 2
        elif output_count in (10, 15):
            self.strides = (8, 16, 32, 64, 128)
            self.num_anchors = 1
        else:
            raise ValueError(
                f"Unsupported SCRFD graph with {output_count} outputs; expected 6, 9, 10, or 15"
            )
        self.feature_map_count = len(self.strides)
        self.uses_keypoints = output_count in (9, 15)

    @property
    def providers(self) -> list[str]:
        return self.session.get_providers()

    def _static_model_input_size(self) -> tuple[int, int] | None:
        model = onnx.load_model(self.model_path, load_external_data=False)
        dimensions = model.graph.input[0].type.tensor_type.shape.dim
        if len(dimensions) != 4:
            return None
        static_shape = [dimension.dim_value if dimension.HasField("dim_value") else 0 for dimension in dimensions]
        if static_shape[0:2] != [1, 3] or static_shape[2] <= 0 or static_shape[3] <= 0:
            return None
        return static_shape[3], static_shape[2]

    def _create_session(self, provider: str) -> ort.InferenceSession:
        provider = provider.lower()
        available = ort.get_available_providers()
        if provider not in {"auto", "cpu", "coreml"}:
            raise ValueError("provider must be one of: auto, cpu, coreml")

        cpu = ["CPUExecutionProvider"]
        coreml_cache = Path.cwd() / ".cache" / "coreml"
        coreml_cache.mkdir(parents=True, exist_ok=True)
        coreml: list[Any] = [
            (
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "RequireStaticInputShapes": "1" if self.has_static_input else "0",
                    "ModelCacheDirectory": str(coreml_cache),
                },
            ),
            "CPUExecutionProvider",
        ]
        if provider == "coreml" and "CoreMLExecutionProvider" not in available:
            raise RuntimeError(
                "This ONNX Runtime build does not include CoreMLExecutionProvider. "
                f"Available providers: {available}"
            )

        requested = coreml if provider == "coreml" else cpu
        if provider == "auto" and "CoreMLExecutionProvider" in available:
            requested = coreml

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            return ort.InferenceSession(
                str(self.model_path),
                sess_options=options,
                providers=requested,
            )
        except Exception as error:
            if provider == "auto" and requested is coreml:
                self.provider_warning = f"CoreML initialization failed; using CPU: {error}"
                return ort.InferenceSession(
                    str(self.model_path),
                    sess_options=options,
                    providers=cpu,
                )
            raise

    @staticmethod
    def _flatten_prediction(output: np.ndarray, values_per_anchor: int) -> np.ndarray:
        array = np.asarray(output)
        if array.ndim >= 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 1:
            if array.size % values_per_anchor:
                raise ValueError(f"Cannot reshape output {array.shape}")
            return array.reshape((-1, values_per_anchor))
        if array.ndim == 2:
            if array.shape[-1] == values_per_anchor:
                return array.reshape((-1, values_per_anchor))
            if array.shape[0] == values_per_anchor:
                return array.T.reshape((-1, values_per_anchor))
            if array.size % values_per_anchor == 0:
                return array.reshape((-1, values_per_anchor))
        if array.ndim == 3:
            if array.shape[0] % values_per_anchor == 0:
                return array.transpose((1, 2, 0)).reshape((-1, values_per_anchor))

            if array.shape[-1] % values_per_anchor == 0:
                return array.reshape((-1, values_per_anchor))
        raise ValueError(
            f"Unsupported output shape {array.shape} for {values_per_anchor} values per anchor"
        )

    def _anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride, self.num_anchors)
        cached = self._anchor_cache.get(key)
        if cached is not None:
            return cached

        centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        centers = (centers * stride).reshape((-1, 2))
        if self.num_anchors > 1:
            centers = np.repeat(centers, self.num_anchors, axis=0)
        self._anchor_cache[key] = centers
        return centers

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        input_width, input_height = self.input_size
        frame_height, frame_width = frame.shape[:2]
        scale = min(input_width / frame_width, input_height / frame_height)
        resized_width = max(1, int(round(frame_width * scale)))
        resized_height = max(1, int(round(frame_height * scale)))
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        detector_image = np.zeros((input_height, input_width, 3), dtype=np.uint8)
        detector_image[:resized_height, :resized_width] = resized
        blob = cv2.dnn.blobFromImage(
            detector_image,
            scalefactor=1.0 / 128.0,
            size=self.input_size,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        return blob, scale

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image with shape HxWx3")

        started = perf_counter()
        blob, scale = self._preprocess(frame)
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        scores_out: list[np.ndarray] = []
        boxes_out: list[np.ndarray] = []
        keypoints_out: list[np.ndarray] = []
        input_width, input_height = self.input_size

        for index, stride in enumerate(self.strides):
            scores = self._flatten_prediction(outputs[index], 1).reshape(-1)
            boxes = self._flatten_prediction(
                outputs[index + self.feature_map_count], 4
            ) * stride
            keypoint_offsets = None
            if self.uses_keypoints:
                keypoint_offsets = self._flatten_prediction(
                    outputs[index + 2 * self.feature_map_count], 10
                ) * stride

            feature_height = input_height // stride
            feature_width = input_width // stride
            centers = self._anchor_centers(feature_height, feature_width, stride)
            expected = centers.shape[0]
            if scores.shape[0] != expected or boxes.shape[0] != expected:
                raise ValueError(
                    f"Unexpected stride-{stride} outputs: scores={scores.shape}, "
                    f"boxes={boxes.shape}, anchors={centers.shape}"
                )

            positive = np.flatnonzero(scores >= self.threshold)
            if positive.size == 0:
                continue
            scores_out.append(scores[positive, None])
            boxes_out.append(distance_to_bbox(centers, boxes)[positive])
            if keypoint_offsets is not None:
                keypoints_out.append(
                    distance_to_keypoints(centers, keypoint_offsets)[positive]
                )

        if not scores_out:
            latency_ms = (perf_counter() - started) * 1000.0
            empty_detections = np.empty((0, 5), dtype=np.float32)
            empty_keypoints = (
                np.empty((0, 5, 2), dtype=np.float32) if self.uses_keypoints else None
            )
            return DetectionResult(empty_detections, empty_keypoints, latency_ms)

        scores = np.vstack(scores_out)
        boxes = np.vstack(boxes_out) / scale
        detections = np.hstack((boxes, scores)).astype(np.float32, copy=False)
        order = detections[:, 4].argsort()[::-1]
        detections = detections[order]
        keypoints = None
        if self.uses_keypoints:
            keypoints = np.vstack(keypoints_out)[order] / scale

        keep = non_maximum_suppression(detections, self.nms_threshold)
        detections = detections[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        frame_height, frame_width = frame.shape[:2]
        detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, frame_width - 1)
        detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, frame_height - 1)
        latency_ms = (perf_counter() - started) * 1000.0
        return DetectionResult(detections, keypoints, latency_ms)
