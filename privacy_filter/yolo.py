from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import onnx

from .ort_session import create_inference_session


@dataclass(frozen=True)
class DetectionResult:
    detections: np.ndarray
    latency_ms: float


def non_maximum_suppression(detections: np.ndarray, threshold: float) -> list[int]:
    if detections.size == 0:
        return []
    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 2]
    y2 = detections[:, 3]
    scores = detections[:, 4]
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


class YOLOFaceDetector:
    def __init__(
        self,
        model_path: str | Path,
        threshold: float = 0.25,
        nms_threshold: float = 0.45,
        provider: str = "auto",
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"YOLO face ONNX model not found: {self.model_path}")
        if self.model_path.suffix.lower() != ".onnx":
            raise ValueError("YOLO face inference requires an .onnx model")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if not 0.0 < nms_threshold < 1.0:
            raise ValueError("nms_threshold must be between 0 and 1")

        model = onnx.load_model(self.model_path, load_external_data=False)
        if len(model.graph.input) != 1 or len(model.graph.output) != 1:
            raise ValueError("Expected a YOLO detect graph with one input and one output")
        dimensions = model.graph.input[0].type.tensor_type.shape.dim
        shape = [value.dim_value if value.HasField("dim_value") else 0 for value in dimensions]
        if len(shape) != 4 or shape[0:2] != [1, 3] or min(shape[2:]) <= 0:
            raise ValueError(f"Expected a static YOLO input [1,3,H,W], got {shape}")

        self.input_size = (shape[3], shape[2])
        output_dimensions = model.graph.output[0].type.tensor_type.shape.dim
        output_shape = [
            value.dim_value if value.HasField("dim_value") else 0
            for value in output_dimensions
        ]
        self.has_landmarks = 20 in output_shape
        self.detection_width = 20 if self.has_landmarks else 5
        self.threshold = float(threshold)
        self.nms_threshold = float(nms_threshold)
        self._provider_policy = provider.lower()
        created = create_inference_session(
            self.model_path,
            provider,
            True,
            "yolo_detector",
        )
        self.session = created.session
        self.provider_warning = created.warning
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @property
    def providers(self) -> list[str]:
        return self.session.get_providers()

    def _run(self, blob: np.ndarray) -> np.ndarray:
        try:
            return self.session.run([self.output_name], {self.input_name: blob})[0]
        except Exception as error:
            providers = self.session.get_providers()
            if self._provider_policy != "auto" or providers[0] == "CPUExecutionProvider":
                raise
            failed_provider = providers[0]
            created = create_inference_session(
                self.model_path,
                "cpu",
                True,
                "yolo_detector",
            )
            self.session = created.session
            self.provider_warning = f"{failed_provider} inference failed; using CPU: {error}"
            return self.session.run([self.output_name], {self.input_name: blob})[0]

    def _preprocess(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, float, float, float]:
        input_width, input_height = self.input_size
        frame_height, frame_width = frame.shape[:2]
        scale = min(input_width / frame_width, input_height / frame_height)
        resized_width = max(1, int(round(frame_width * scale)))
        resized_height = max(1, int(round(frame_height * scale)))
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_x = (input_width - resized_width) / 2.0
        pad_y = (input_height - resized_height) / 2.0
        left = int(round(pad_x - 0.1))
        right = input_width - resized_width - left
        top = int(round(pad_y - 0.1))
        bottom = input_height - resized_height - top
        detector_image = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        blob = cv2.dnn.blobFromImage(
            detector_image,
            scalefactor=1.0 / 255.0,
            size=self.input_size,
            swapRB=True,
        )
        return blob, scale, float(left), float(top)

    @staticmethod
    def _prediction_rows(output: np.ndarray) -> np.ndarray:
        rows = np.asarray(output, dtype=np.float32)
        if rows.ndim == 3 and rows.shape[0] == 1:
            rows = rows[0]
        if rows.ndim != 2:
            raise ValueError(f"Unsupported YOLO output shape: {rows.shape}")
        if rows.shape[0] <= 128 and rows.shape[1] > rows.shape[0]:
            rows = rows.T
        if rows.shape[1] < 5:
            raise ValueError(f"Expected YOLO predictions with at least 5 values, got {rows.shape}")
        return rows

    def _decode(
        self,
        output: np.ndarray,
        scale: float,
        pad_x: float,
        pad_y: float,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        rows = self._prediction_rows(output)
        landmarks: np.ndarray | None = None
        if rows.shape[1] == 6 and np.all(rows[:, 2] >= rows[:, 0]) and np.all(rows[:, 3] >= rows[:, 1]):
            detections = rows[:, :5].copy()
            detections = detections[detections[:, 4] >= self.threshold]
        else:
            if self.has_landmarks:
                if rows.shape[1] != 20:
                    raise ValueError(
                        f"Expected YOLO Pose predictions with 20 values, got {rows.shape}"
                    )
                scores = rows[:, 4]
            else:
                scores = rows[:, 4:].max(axis=1)
            positive = scores >= self.threshold
            boxes = rows[positive, :4]
            scores = scores[positive]
            if self.has_landmarks:
                landmarks = rows[positive, 5:20].reshape(-1, 5, 3).copy()
            if not len(boxes):
                return np.empty((0, self.detection_width), dtype=np.float32)
            centers_x, centers_y, widths, heights = boxes.T
            detections = np.column_stack(
                (
                    centers_x - widths / 2.0,
                    centers_y - heights / 2.0,
                    centers_x + widths / 2.0,
                    centers_y + heights / 2.0,
                    scores,
                )
            ).astype(np.float32, copy=False)

        if not len(detections):
            return np.empty((0, self.detection_width), dtype=np.float32)
        detections[:, [0, 2]] = (detections[:, [0, 2]] - pad_x) / scale
        detections[:, [1, 3]] = (detections[:, [1, 3]] - pad_y) / scale
        detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, frame_width - 1)
        detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, frame_height - 1)
        valid = (detections[:, 2] > detections[:, 0]) & (detections[:, 3] > detections[:, 1])
        detections = detections[valid]
        if landmarks is not None:
            landmarks = landmarks[valid]
            landmarks[:, :, 0] = (landmarks[:, :, 0] - pad_x) / scale
            landmarks[:, :, 1] = (landmarks[:, :, 1] - pad_y) / scale
            landmarks[:, :, 0] = np.clip(landmarks[:, :, 0], 0, frame_width - 1)
            landmarks[:, :, 1] = np.clip(landmarks[:, :, 1], 0, frame_height - 1)
            landmarks[:, :, 2] = np.clip(landmarks[:, :, 2], 0.0, 1.0)
        order = detections[:, 4].argsort()[::-1]
        detections = detections[order]
        if landmarks is not None:
            landmarks = landmarks[order]
        keep = non_maximum_suppression(detections, self.nms_threshold)
        detections = detections[keep]
        if landmarks is not None:
            detections = np.column_stack((detections, landmarks[keep].reshape(-1, 15)))
        return detections.astype(np.float32, copy=False)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image with shape HxWx3")
        started = perf_counter()
        blob, scale, pad_x, pad_y = self._preprocess(frame)
        output = self._run(blob)
        frame_height, frame_width = frame.shape[:2]
        detections = self._decode(
            output,
            scale,
            pad_x,
            pad_y,
            frame_width,
            frame_height,
        )
        latency_ms = (perf_counter() - started) * 1000.0
        return DetectionResult(detections, latency_ms)
