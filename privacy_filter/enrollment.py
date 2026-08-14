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

from .recognition import FACE_ROTATION_ANGLES, l2_normalize


@dataclass(frozen=True)
class FaceQuality:
    accepted: bool
    reason: str
    face_size: float
    brightness: float
    sharpness: float


def assess_face_quality(
    face_image: np.ndarray,
    detection: np.ndarray,
    min_face_size: float = 96.0,
    min_sharpness: float = 25.0,
    min_brightness: float = 35.0,
    max_brightness: float = 220.0,
) -> FaceQuality:
    width = max(0.0, float(detection[2] - detection[0]))
    height = max(0.0, float(detection[3] - detection[1]))
    face_size = min(width, height)
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
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
    centroids: np.ndarray
    rotation_angles: tuple[int, ...]
    model_sha256: str
    threshold: float
    metadata: dict[str, object]

    @property
    def centroid(self) -> np.ndarray:
        """The upright centroid retained for callers using the legacy API."""
        return self.centroids[0]

    def rotation_scores(self, embedding: np.ndarray) -> np.ndarray:
        return self.centroids @ l2_normalize(embedding)

    def score(self, embedding: np.ndarray) -> float:
        return float(self.rotation_scores(embedding).max())

    @property
    def genuine_scores(self) -> np.ndarray:
        return np.einsum("nad,ad->na", self.embeddings, self.centroids)


def build_template(
    name: str,
    embeddings: list[np.ndarray] | np.ndarray,
    model_sha256: str,
    threshold: float = 0.35,
    minimum_samples: int = 1,
    source: str = "unknown",
    face_preprocessing: str = "unknown",
) -> EnrollmentTemplate:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim == 2 and matrix.shape[1] == 512:
        matrix = matrix[:, None, :]
        rotation_angles = (0,)
    elif matrix.ndim == 3 and matrix.shape[2] == 512:
        if matrix.shape[1] != len(FACE_ROTATION_ANGLES):
            raise ValueError(
                f"Expected Nx{len(FACE_ROTATION_ANGLES)}x512 embeddings, got {matrix.shape}"
            )
        rotation_angles = FACE_ROTATION_ANGLES
    else:
        raise ValueError(f"Expected Nx512 or Nx4x512 embeddings, got {matrix.shape}")
    if matrix.shape[0] < minimum_samples:
        raise ValueError(f"Need at least {minimum_samples} accepted samples, got {matrix.shape[0]}")
    matrix = np.asarray(
        [
            [l2_normalize(embedding) for embedding in sample]
            for sample in matrix
        ],
        dtype=np.float32,
    )
    centroids = np.asarray(
        [l2_normalize(matrix[:, index, :].mean(axis=0)) for index in range(matrix.shape[1])],
        dtype=np.float32,
    )
    scores = np.einsum("nad,ad->na", matrix, centroids)
    metadata: dict[str, object] = {
        "version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "face_preprocessing": face_preprocessing,
        "samples": int(matrix.shape[0]),
        "rotation_angles": list(rotation_angles),
        "genuine_score_min": round(float(scores.min()), 6),
        "genuine_score_mean": round(float(scores.mean()), 6),
        "genuine_score_max": round(float(scores.max()), 6),
    }
    return EnrollmentTemplate(
        name=safe_identity_name(name),
        embeddings=matrix,
        centroids=centroids,
        rotation_angles=rotation_angles,
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
            centroids=template.centroids.astype(np.float32),
            rotation_angles=np.asarray(template.rotation_angles, dtype=np.int16),
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
        if "centroids" in archive.files:
            centroids = np.asarray(archive["centroids"], dtype=np.float32)
            rotation_angles = tuple(int(value) for value in archive["rotation_angles"])
        else:
            centroids = np.asarray(archive["centroid"], dtype=np.float32).reshape(1, -1)
            rotation_angles = (0,)
        model_hash = str(archive["model_sha256"].item())
        threshold = float(archive["threshold"].item())
        metadata = json.loads(str(archive["metadata_json"].item()))
    if embeddings.ndim == 2 and embeddings.shape[1] == 512:
        embeddings = embeddings[:, None, :]
    if embeddings.ndim != 3 or embeddings.shape[2] != 512:
        raise ValueError(f"Invalid enrollment embedding shape: {embeddings.shape}")
    if centroids.ndim != 2 or centroids.shape[1] != 512:
        raise ValueError(f"Invalid enrollment centroid shape: {centroids.shape}")
    if embeddings.shape[1] != len(centroids) or len(rotation_angles) != len(centroids):
        raise ValueError("Enrollment rotations, embeddings, and centroids do not match")
    embeddings = np.asarray(
        [
            [l2_normalize(embedding) for embedding in sample]
            for sample in embeddings
        ],
        dtype=np.float32,
    )
    centroids = np.vstack([l2_normalize(row) for row in centroids]).astype(np.float32)
    return EnrollmentTemplate(
        name,
        embeddings,
        centroids,
        rotation_angles,
        model_hash,
        threshold,
        metadata,
    )
