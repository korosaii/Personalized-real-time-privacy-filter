from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import onnx
import onnxruntime as ort

from .face_align import align_face


@dataclass(frozen=True)
class EmbeddingResult:
    embedding: np.ndarray
    aligned_face: np.ndarray
    latency_ms: float


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Embedding has an invalid L2 norm")
    return value / norm


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.dot(l2_normalize(first), l2_normalize(second)))


class FaceEmbedder:
    def __init__(self, model_path: str | Path, provider: str = "auto") -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Recognition ONNX model not found: {self.model_path}")
        if self.model_path.suffix.lower() != ".onnx":
            raise ValueError("Recognition inference requires an .onnx model")

        model = onnx.load_model(self.model_path, load_external_data=False)
        self._validate_graph(model)
        self.input_mean, self.input_std = self._preprocessing_from_graph(model)
        self.has_static_batch = self._has_static_batch(model)
        self.provider_warning: str | None = None
        self.session = self._create_session(provider)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @staticmethod
    def _validate_graph(model: onnx.ModelProto) -> None:
        if len(model.graph.input) != 1 or len(model.graph.output) != 1:
            raise ValueError("Expected a recognition graph with one input and one output")
        shape = model.graph.input[0].type.tensor_type.shape.dim
        spatial = [dimension.dim_value if dimension.HasField("dim_value") else 0 for dimension in shape]
        if len(spatial) != 4 or spatial[1:] != [3, 112, 112]:
            raise ValueError(f"Expected recognition input [batch,3,112,112], got {spatial}")

    @staticmethod
    def _preprocessing_from_graph(model: onnx.ModelProto) -> tuple[float, float]:
        names = [node.name for node in model.graph.node[:8]]
        graph_normalizes = any(name.startswith(("Sub", "_minus")) for name in names) and any(
            name.startswith(("Mul", "_mul")) for name in names
        )
        return (0.0, 1.0) if graph_normalizes else (127.5, 127.5)

    @staticmethod
    def _has_static_batch(model: onnx.ModelProto) -> bool:
        batch = model.graph.input[0].type.tensor_type.shape.dim[0]
        return batch.HasField("dim_value") and batch.dim_value == 1

    @property
    def providers(self) -> list[str]:
        return self.session.get_providers()

    def _create_session(self, provider: str) -> ort.InferenceSession:
        provider = provider.lower()
        if provider not in {"auto", "cpu", "coreml"}:
            raise ValueError("provider must be one of: auto, cpu, coreml")
        available = ort.get_available_providers()
        cpu = ["CPUExecutionProvider"]
        cache = Path.cwd() / ".cache" / "coreml"
        cache.mkdir(parents=True, exist_ok=True)
        coreml: list[Any] = [
            (
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "RequireStaticInputShapes": "1" if self.has_static_batch else "0",
                    "ModelCacheDirectory": str(cache),
                },
            ),
            "CPUExecutionProvider",
        ]
        if provider == "coreml" and "CoreMLExecutionProvider" not in available:
            raise RuntimeError(f"CoreML is unavailable; providers: {available}")
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

    def embed_aligned(self, aligned_face: np.ndarray) -> tuple[np.ndarray, float]:
        if aligned_face.shape != (112, 112, 3):
            raise ValueError(f"Expected an aligned 112x112 BGR face, got {aligned_face.shape}")
        started = perf_counter()
        blob = cv2.dnn.blobFromImage(
            aligned_face,
            scalefactor=1.0 / self.input_std,
            size=(112, 112),
            mean=(self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        raw = self.session.run([self.output_name], {self.input_name: blob})[0]
        embedding = l2_normalize(raw)
        return embedding, (perf_counter() - started) * 1000.0

    def embed(self, frame: np.ndarray, keypoints: np.ndarray) -> EmbeddingResult:
        aligned = align_face(frame, keypoints)
        embedding, latency_ms = self.embed_aligned(aligned)
        return EmbeddingResult(embedding, aligned, latency_ms)

    def warmup(self, runs: int = 2) -> list[float]:
        sample = np.zeros((112, 112, 3), dtype=np.uint8)
        return [self.embed_aligned(sample)[1] for _ in range(runs)]
