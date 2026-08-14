from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

import cv2
import numpy as np

from .recognition import l2_normalize


@dataclass(frozen=True)
class FaceQuality:
    accepted: bool
    reason: str
    face_size: float
    brightness: float
    sharpness: float


def assess_face_quality(
    aligned_face: np.ndarray,
    detection: np.ndarray,
    min_face_size: float = 96.0,
    min_sharpness: float = 25.0,
    min_brightness: float = 35.0,
    max_brightness: float = 220.0,
) -> FaceQuality:
    width = max(0.0, float(detection[2] - detection[0]))
    height = max(0.0, float(detection[3] - detection[1]))
    face_size = min(width, height)
    gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    reason = "accepted"
    if face_size < min_face_size:
        reason = f"move closer ({face_size:.0f}px < {min_face_size:.0f}px)"
    elif brightness < min_brightness:
        reason = f"too dark ({brightness:.0f})"
    elif brightness > max_brightness:
        reason = f"too bright ({brightness:.0f})"
    elif sharpness < min_sharpness:
        reason = f"hold still / improve focus ({sharpness:.1f})"

    return FaceQuality(
        accepted=reason == "accepted",
        reason=reason,
        face_size=face_size,
        brightness=brightness,
        sharpness=sharpness,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_identity_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")
    if not value:
        raise ValueError("Identity name must contain at least one letter or number")
    return value[:64]


@dataclass(frozen=True)
class EnrollmentTemplate:
    name: str
    embeddings: np.ndarray
    centroid: np.ndarray
    model_sha256: str
    threshold: float
    metadata: dict[str, object]

    def score(self, embedding: np.ndarray) -> float:
        return float(np.dot(self.centroid, l2_normalize(embedding)))

    @property
    def genuine_scores(self) -> np.ndarray:
        return self.embeddings @ self.centroid


def build_template(
    name: str,
    embeddings: list[np.ndarray] | np.ndarray,
    model_sha256: str,
    threshold: float = 0.50,
    minimum_samples: int = 1,
    source: str = "unknown",
) -> EnrollmentTemplate:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 512:
        raise ValueError(f"Expected Nx512 embeddings, got {matrix.shape}")
    if matrix.shape[0] < minimum_samples:
        raise ValueError(f"Need at least {minimum_samples} accepted samples, got {matrix.shape[0]}")
    matrix = np.vstack([l2_normalize(row) for row in matrix]).astype(np.float32)
    centroid = l2_normalize(matrix.mean(axis=0)).astype(np.float32)
    scores = matrix @ centroid
    metadata: dict[str, object] = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "samples": int(matrix.shape[0]),
        "genuine_score_min": round(float(scores.min()), 6),
        "genuine_score_mean": round(float(scores.mean()), 6),
        "genuine_score_max": round(float(scores.max()), 6),
    }
    return EnrollmentTemplate(
        name=safe_identity_name(name),
        embeddings=matrix,
        centroid=centroid,
        model_sha256=model_sha256,
        threshold=float(threshold),
        metadata=metadata,
    )


def save_template(template: EnrollmentTemplate, path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("wb") as destination:
        np.savez_compressed(
            destination,
            name=np.asarray(template.name),
            embeddings=template.embeddings.astype(np.float32),
            centroid=template.centroid.astype(np.float32),
            model_sha256=np.asarray(template.model_sha256),
            threshold=np.asarray(template.threshold, dtype=np.float32),
            metadata_json=np.asarray(json.dumps(template.metadata, ensure_ascii=False)),
        )
    os.chmod(temporary, 0o600)
    os.replace(temporary, output)
    return output


def load_template(path: str | Path) -> EnrollmentTemplate:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Enrollment template not found: {source}")
    with np.load(source, allow_pickle=False) as archive:
        required = {"name", "embeddings", "centroid", "model_sha256", "threshold", "metadata_json"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Enrollment template is missing fields: {sorted(missing)}")
        name = str(archive["name"].item())
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        centroid = l2_normalize(archive["centroid"]).astype(np.float32)
        model_hash = str(archive["model_sha256"].item())
        threshold = float(archive["threshold"].item())
        metadata = json.loads(str(archive["metadata_json"].item()))
    if embeddings.ndim != 2 or embeddings.shape[1] != 512:
        raise ValueError(f"Invalid enrollment embedding shape: {embeddings.shape}")
    embeddings = np.vstack([l2_normalize(row) for row in embeddings]).astype(np.float32)
    return EnrollmentTemplate(name, embeddings, centroid, model_hash, threshold, metadata)
