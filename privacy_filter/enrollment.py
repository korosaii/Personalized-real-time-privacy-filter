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
    rotation_centroids: np.ndarray
    model_sha256: str
    threshold: float
    metadata: dict[str, object]

    @property
    def centroid(self) -> np.ndarray:
        """The upright centroid retained for callers using the legacy API."""
        return self.rotation_centroids[0]

    @property
    def rotation_angles(self) -> tuple[int, ...]:
        stored = self.metadata.get("rotation_angles", [0])
        return tuple(int(value) for value in stored)

    def rotation_scores(self, embedding: np.ndarray) -> np.ndarray:
        return self.rotation_centroids @ l2_normalize(embedding)

    def best_rotation_match(self, embedding: np.ndarray) -> tuple[float, int, int]:
        scores = self.rotation_scores(embedding)
        centroid_index = int(np.argmax(scores))
        return (
            float(scores[centroid_index]),
            centroid_index,
            self.rotation_angles[centroid_index],
        )

    def score(self, embedding: np.ndarray) -> float:
        return self.best_rotation_match(embedding)[0]


def build_template(
    name: str,
    embeddings: list[np.ndarray] | np.ndarray,
    model_sha256: str,
    threshold: float = 0.35,
    minimum_samples: int = 1,
    source: str = "unknown",
    face_preprocessing: str = "unknown",
    rotation_angles: tuple[int, ...] | None = None,
) -> EnrollmentTemplate:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim == 3 and matrix.shape[2] == 512:
        source_photos = int(matrix.shape[0])
        rotations_per_photo = int(matrix.shape[1])
        cube = np.asarray(
            [[l2_normalize(embedding) for embedding in sample] for sample in matrix],
            dtype=np.float32,
        )
    elif matrix.ndim == 2 and matrix.shape[1] == 512:
        source_photos = int(matrix.shape[0])
        rotations_per_photo = 1
        cube = np.vstack([l2_normalize(row) for row in matrix]).astype(np.float32)
        cube = cube[:, None, :]
    else:
        raise ValueError(f"Expected Nx512 or NxRx512 embeddings, got {matrix.shape}")
    if source_photos < minimum_samples:
        raise ValueError(
            f"Need at least {minimum_samples} accepted samples, got {source_photos}"
        )
    matrix = cube.reshape(-1, 512)
    rotation_centroids = np.vstack(
        [
            l2_normalize(cube[:, rotation_index, :].mean(axis=0))
            for rotation_index in range(rotations_per_photo)
        ]
    ).astype(np.float32)
    resolved_rotation_angles = (
        tuple(rotation_angles)
        if rotation_angles is not None
        else ((0,) if rotations_per_photo == 1 else tuple(range(rotations_per_photo)))
    )
    if len(resolved_rotation_angles) != rotations_per_photo:
        raise ValueError(
            "The number of rotation angles must match embeddings per photo"
        )
    metadata: dict[str, object] = {
        "version": 9,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "face_preprocessing": face_preprocessing,
        "source_photos": source_photos,
        "embeddings": int(matrix.shape[0]),
        "rotations_per_photo": rotations_per_photo,
        "rotation_angles": list(resolved_rotation_angles),
        "matching_policy": "per_rotation_centroid_max",
    }
    return EnrollmentTemplate(
        name=safe_identity_name(name),
        embeddings=matrix,
        rotation_centroids=rotation_centroids,
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
            rotation_centroids=template.rotation_centroids.astype(np.float32),
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
        required = {"name", "embeddings", "model_sha256", "threshold", "metadata_json"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Enrollment template is missing fields: {sorted(missing)}")
        name = str(archive["name"].item())
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        stored_centroids = (
            np.asarray(archive["rotation_centroids"], dtype=np.float32)
            if "rotation_centroids" in archive.files
            else None
        )
        model_hash = str(archive["model_sha256"].item())
        threshold = float(archive["threshold"].item())
        metadata = json.loads(str(archive["metadata_json"].item()))
    if embeddings.ndim == 3 and embeddings.shape[2] == 512:
        cube = embeddings
        embeddings = cube.reshape(-1, 512)
    else:
        rotations_per_photo = int(metadata.get("rotations_per_photo", 1))
        source_photos = int(metadata.get("source_photos", 0))
        if source_photos * rotations_per_photo == len(embeddings):
            cube = embeddings.reshape(source_photos, rotations_per_photo, 512)
        else:
            cube = embeddings.reshape(len(embeddings), 1, 512)
    if embeddings.ndim != 2 or embeddings.shape[1] != 512:
        raise ValueError(f"Invalid enrollment embedding shape: {embeddings.shape}")
    embeddings = np.vstack([l2_normalize(row) for row in embeddings]).astype(np.float32)
    if stored_centroids is None:
        rotation_centroids = np.vstack(
            [
                l2_normalize(cube[:, rotation_index, :].mean(axis=0))
                for rotation_index in range(cube.shape[1])
            ]
        ).astype(np.float32)
    else:
        if stored_centroids.ndim != 2 or stored_centroids.shape[1] != 512:
            raise ValueError(
                f"Invalid rotation centroid shape: {stored_centroids.shape}"
            )
        rotation_centroids = np.vstack(
            [l2_normalize(row) for row in stored_centroids]
        ).astype(np.float32)
    return EnrollmentTemplate(
        name,
        embeddings,
        rotation_centroids,
        model_hash,
        threshold,
        metadata,
    )
