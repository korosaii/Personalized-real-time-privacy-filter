from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path

import onnx

from .ort_session import preferred_execution_provider


DETECTOR_MODEL_ALIASES = {
    "yolo11": Path("models/detector/yolov11n-face.onnx"),
    "yolo11-pose": Path("models/detector/yolov11n-face-pose.onnx"),
    "yolo11-pose-roll90": Path(
        "models/detector/yolov11n-face-pose-roll90.onnx"
    ),
}
DEFAULT_DETECTOR_MODEL = "yolo11"
RECOGNITION_MODEL_ALIASES = {
    "r34-glint360k": Path("models/recognition/iresnet_r34_glint360k.onnx"),
    "r100-glint360k": Path("models/recognition/iresnet_r100_glint360k.onnx"),
    "r50-webface600k": Path("models/recognition/webface_r50.onnx"),
}
DEFAULT_RECOGNITION_MODEL = "r34-glint360k"


@dataclass(frozen=True)
class RuntimeModels:
    detector_source: Path
    detector_name: str
    recognition_source: Path
    recognition_name: str
    detector_runtime: Path
    recognition_runtime: Path
    recognition_source_sha256: str
    preferred_execution_provider: str
    generated: tuple[Path, ...]


def recognition_model_help() -> str:
    aliases = ", ".join(RECOGNITION_MODEL_ALIASES)
    return f"Recognition model alias ({aliases}) or path to an ONNX file"


def detector_model_help() -> str:
    aliases = ", ".join(DETECTOR_MODEL_ALIASES)
    return f"Detector model alias ({aliases}) or path to an ONNX file"


def _resolve_detector_model(
    requested: str | Path | None,
) -> tuple[Path, str]:
    value = DEFAULT_DETECTOR_MODEL if requested is None else str(requested)
    if value in DETECTOR_MODEL_ALIASES:
        path = DETECTOR_MODEL_ALIASES[value].expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Detector model '{value}' not found: {path}")
        return path, value
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        aliases = ", ".join(DETECTOR_MODEL_ALIASES)
        raise FileNotFoundError(
            f"Detector ONNX model not found: {path}. Available aliases: {aliases}"
        )
    return path, "custom"


def _resolve_recognition_model(
    requested: str | Path | None,
) -> tuple[Path, str]:
    value = DEFAULT_RECOGNITION_MODEL if requested is None else str(requested)
    if value in RECOGNITION_MODEL_ALIASES:
        path = RECOGNITION_MODEL_ALIASES[value].expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Recognition model '{value}' not found: {path}"
            )
        return path, value
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        aliases = ", ".join(RECOGNITION_MODEL_ALIASES)
        raise FileNotFoundError(
            f"Recognition ONNX model not found: {path}. Available aliases: {aliases}"
        )
    return path, "custom"


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


def _prepare_detector(source: Path) -> tuple[Path, bool]:
    model = onnx.load_model(source)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("Expected a YOLO detect graph with one input and one output")
    shape = _shape(model.graph.input[0])
    if len(shape) != 4 or shape[0:2] != [1, 3] or min(shape[2:]) <= 0:
        raise ValueError(f"Expected a static YOLO input [1,3,H,W], got {shape}")
    return source, False


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


def prepare_runtime_models(
    detector_model: str | Path | None,
    recognition_model: str | Path | None,
    provider: str,
    cache_dir: str | Path = ".cache/models",
) -> RuntimeModels:
    detector_source, detector_name = _resolve_detector_model(detector_model)
    recognition_source, recognition_name = _resolve_recognition_model(
        recognition_model
    )
    if detector_source.suffix.lower() != ".onnx":
        raise ValueError("Detector inference requires an ONNX model")
    if recognition_source.suffix.lower() != ".onnx":
        raise ValueError("Recognition inference requires an ONNX model")
    preferred_provider = preferred_execution_provider(provider)
    resolved_cache = Path(cache_dir).expanduser().resolve()
    recognition_source_sha256 = _file_sha256(recognition_source)
    detector_runtime, detector_generated = _prepare_detector(detector_source)
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
        detector_source=detector_source,
        detector_name=detector_name,
        recognition_source=recognition_source,
        recognition_name=recognition_name,
        detector_runtime=detector_runtime,
        recognition_runtime=recognition_runtime,
        recognition_source_sha256=recognition_source_sha256,
        preferred_execution_provider=preferred_provider,
        generated=generated,
    )
