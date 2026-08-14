from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path

import onnx
import onnxruntime as ort


DETECTOR_CANDIDATES = (
    Path("models/detector/det_10g.onnx"),
    Path("models/detector/det_10g_512.onnx"),
)
RECOGNITION_CANDIDATES = (
    Path("models/recognition/webface_r50.onnx"),
    Path("models/recognition/webface_r50_112.onnx"),
)


@dataclass(frozen=True)
class RuntimeModels:
    detector_source: Path
    recognition_source: Path
    detector_runtime: Path
    recognition_runtime: Path
    recognition_source_sha256: str
    coreml_enabled: bool
    generated: tuple[Path, ...]


def _resolve_model(
    requested: str | Path | None,
    candidates: tuple[Path, ...],
    label: str,
) -> Path:
    if requested is not None:
        path = Path(requested).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{label} ONNX model not found: {path}")
        return path
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if path.is_file():
            return path
    expected = " or ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"{label} ONNX model not found. Expected {expected}")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(value_info: onnx.ValueInfoProto) -> list[int]:
    dimensions = value_info.type.tensor_type.shape.dim
    return [
        dimension.dim_value if dimension.HasField("dim_value") else 0
        for dimension in dimensions
    ]


def _set_dimension(dimension, value: int) -> None:
    dimension.ClearField("dim_param")
    dimension.dim_value = value


def _cache_path(
    source: Path,
    suffix: str,
    cache_dir: Path,
    fingerprint: str | None = None,
) -> Path:
    model_fingerprint = fingerprint or _file_sha256(source)
    return cache_dir / f"{source.stem}-{model_fingerprint[:12]}-{suffix}.onnx"


def _save_model(model: onnx.ModelProto, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        onnx.checker.check_model(model)
        onnx.save_model(model, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_detector(source: Path, cache_dir: Path, size: int) -> tuple[Path, bool]:
    model = onnx.load_model(source)
    if _shape(model.graph.input[0]) == [1, 3, size, size]:
        return source, False
    if len(model.graph.input) != 1 or len(model.graph.output) != 9:
        raise ValueError("Expected an SCRFD_KPS ONNX graph with one input and nine outputs")
    input_shape = model.graph.input[0].type.tensor_type.shape.dim
    if len(input_shape) != 4:
        raise ValueError("SCRFD input must be four-dimensional")
    target = _cache_path(source, f"{size}x{size}", cache_dir)
    if target.is_file():
        return target, False
    for dimension, value in zip(input_shape, (1, 3, size, size), strict=True):
        _set_dimension(dimension, value)
    counts = [2 * (size // stride) ** 2 for stride in (8, 16, 32)]
    for output_index, value_info in enumerate(model.graph.output):
        output_shape = value_info.type.tensor_type.shape.dim
        if not output_shape:
            raise ValueError("SCRFD output shape is missing")
        _set_dimension(output_shape[0], counts[output_index % 3])
    _save_model(model, target)
    return target, True


def _prepare_recognition(
    source: Path,
    cache_dir: Path,
    fingerprint: str,
) -> tuple[Path, bool]:
    model = onnx.load_model(source)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("Expected a recognition ONNX graph with one input and one output")
    if _shape(model.graph.input[0]) == [1, 3, 112, 112] and _shape(
        model.graph.output[0]
    ) == [1, 512]:
        return source, False
    input_shape = model.graph.input[0].type.tensor_type.shape.dim
    if len(input_shape) != 4:
        raise ValueError("Recognition input must be four-dimensional")
    spatial = _shape(model.graph.input[0])
    if spatial[1:] != [3, 112, 112]:
        raise ValueError(f"Expected recognition input [batch,3,112,112], got {spatial}")
    target = _cache_path(source, "1x3x112x112", cache_dir, fingerprint)
    if target.is_file():
        return target, False
    for dimension, value in zip(input_shape, (1, 3, 112, 112), strict=True):
        _set_dimension(dimension, value)
    output_shape = model.graph.output[0].type.tensor_type.shape.dim
    if len(output_shape) != 2:
        raise ValueError("Recognition output must be two-dimensional")
    _set_dimension(output_shape[0], 1)
    _set_dimension(output_shape[1], 512)
    _save_model(model, target)
    return target, True


def coreml_enabled(provider: str) -> bool:
    normalized = provider.lower()
    if normalized not in {"auto", "cpu", "coreml"}:
        raise ValueError("provider must be one of: auto, cpu, coreml")
    available = ort.get_available_providers()
    if normalized == "coreml" and "CoreMLExecutionProvider" not in available:
        raise RuntimeError(
            "CoreML is unavailable on this system. Use --provider cpu or --provider auto."
        )
    return normalized == "coreml" or (
        normalized == "auto" and "CoreMLExecutionProvider" in available
    )


def prepare_runtime_models(
    detector_model: str | Path | None,
    recognition_model: str | Path | None,
    provider: str,
    cache_dir: str | Path = ".cache/models",
) -> RuntimeModels:
    detector_source = _resolve_model(detector_model, DETECTOR_CANDIDATES, "Detector")
    recognition_source = _resolve_model(
        recognition_model,
        RECOGNITION_CANDIDATES,
        "Recognition",
    )
    if detector_source.suffix.lower() != ".onnx":
        raise ValueError("Detector inference requires an ONNX model")
    if recognition_source.suffix.lower() != ".onnx":
        raise ValueError("Recognition inference requires an ONNX model")
    use_coreml = coreml_enabled(provider)
    resolved_cache = Path(cache_dir).expanduser().resolve()
    recognition_source_sha256 = _file_sha256(recognition_source)
    detector_runtime, detector_generated = _prepare_detector(
        detector_source,
        resolved_cache,
        512,
    )
    recognition_runtime, recognition_generated = _prepare_recognition(
        recognition_source,
        resolved_cache,
        recognition_source_sha256,
    )
    generated = tuple(
        path
        for path, was_generated in (
            (detector_runtime, detector_generated),
            (recognition_runtime, recognition_generated),
        )
        if was_generated
    )
    return RuntimeModels(
        detector_source,
        recognition_source,
        detector_runtime,
        recognition_runtime,
        recognition_source_sha256,
        use_coreml,
        generated,
    )
