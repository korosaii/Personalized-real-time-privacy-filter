from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnxruntime as ort


PROVIDER_CHOICES = ("auto", "cpu", "coreml", "directml", "cuda")
PROVIDER_NAMES = {
    "cpu": "CPUExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "directml": "DmlExecutionProvider",
    "cuda": "CUDAExecutionProvider",
}
AUTO_PROVIDER_ORDER = (
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
)


@dataclass(frozen=True)
class SessionCreation:
    session: ort.InferenceSession
    preferred_provider: str
    warning: str | None


def preferred_execution_provider(
    provider: str,
    available: list[str] | None = None,
) -> str:
    normalized = provider.lower()
    if normalized not in PROVIDER_CHOICES:
        raise ValueError(f"provider must be one of: {', '.join(PROVIDER_CHOICES)}")
    providers = available if available is not None else ort.get_available_providers()
    if normalized == "auto":
        return next(
            (candidate for candidate in AUTO_PROVIDER_ORDER if candidate in providers),
            "CPUExecutionProvider",
        )
    requested = PROVIDER_NAMES[normalized]
    if requested not in providers:
        raise RuntimeError(
            f"{requested} is unavailable. Available providers: {providers}. "
            "Use --provider auto to select the best available accelerator."
        )
    return requested


def _session_options(provider: str) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if provider == "DmlExecutionProvider":
        options.enable_mem_pattern = False
    return options


def _provider_chain(
    provider: str,
    has_static_input: bool,
    cache_name: str,
) -> list[Any]:
    if provider == "CoreMLExecutionProvider":
        cache = Path.cwd() / ".cache" / "coreml" / cache_name
        cache.mkdir(parents=True, exist_ok=True)
        return [
            (
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "RequireStaticInputShapes": "1" if has_static_input else "0",
                    "ModelCacheDirectory": str(cache),
                },
            ),
            "CPUExecutionProvider",
        ]
    if provider == "CUDAExecutionProvider":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if provider == "DmlExecutionProvider":
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def create_inference_session(
    model_path: str | Path,
    provider: str,
    has_static_input: bool,
    cache_name: str,
) -> SessionCreation:
    normalized = provider.lower()
    preferred = preferred_execution_provider(normalized)
    requested = _provider_chain(preferred, has_static_input, cache_name)
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=_session_options(preferred),
            providers=requested,
        )
        if preferred in session.get_providers():
            return SessionCreation(session, preferred, None)
        warning = (
            f"{preferred} was not activated; using "
            f"{session.get_providers()[0]} instead."
        )
        if normalized != "auto":
            raise RuntimeError(warning)
        return SessionCreation(session, preferred, warning)
    except Exception as error:
        if normalized != "auto" or preferred == "CPUExecutionProvider":
            raise
        session = ort.InferenceSession(
            str(model_path),
            sess_options=_session_options("CPUExecutionProvider"),
            providers=["CPUExecutionProvider"],
        )
        warning = f"{preferred} initialization failed; using CPU: {error}"
        return SessionCreation(session, preferred, warning)
