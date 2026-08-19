from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

from .enrollment import (
    assess_face_quality,
    build_template,
    load_template,
    safe_identity_name,
    save_template,
)
from .model_setup import (
    RuntimeModels,
    detector_model_help,
    prepare_runtime_models,
    recognition_model_help,
)
from .ort_session import PROVIDER_CHOICES
from .recognition import (
    FACE_PREPROCESSING,
    LANDMARK_FACE_PREPROCESSING,
    FaceEmbedder,
)
from .yolo import YOLOFaceDetector


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a local biometric template from owner photographs."
    )
    parser.add_argument("name", help="Local identity name, for example owner")
    parser.add_argument(
        "photos",
        nargs="+",
        type=Path,
        help="Photographs or folders containing photographs",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--detector-model",
        "--detector",
        default="yolo11face",
        help=detector_model_help(),
    )
    parser.add_argument(
        "--recognition-model",
        "--model",
        default="iresnet50",
        help=recognition_model_help(),
    )
    parser.add_argument("--provider", choices=PROVIDER_CHOICES, default="auto")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--min-face-size", type=float, default=80.0)
    parser.add_argument("--min-sharpness", type=float, default=25.0)
    return parser


def expand_photos(paths: list[Path]) -> list[Path]:
    discovered: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved.is_dir():
            discovered.extend(
                item.resolve()
                for item in sorted(resolved.rglob("*"))
                if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
            )
        elif resolved.is_file() and resolved.suffix.lower() in IMAGE_EXTENSIONS:
            discovered.append(resolved)
        elif resolved.is_file():
            print(f"Skipped unsupported file type: {resolved.name}", file=sys.stderr)
        else:
            print(f"Skipped missing path: {resolved}", file=sys.stderr)

    return list(dict.fromkeys(discovered))


def load_models(
    args: argparse.Namespace,
) -> tuple[YOLOFaceDetector, FaceEmbedder, RuntimeModels]:
    models = prepare_runtime_models(
        args.detector_model,
        args.recognition_model,
        args.provider,
    )
    if models.generated:
        print("Preparing optimized runtime model cache:")
        for path in models.generated:
            print(f"  {path}")
    detector = YOLOFaceDetector(
        models.detector_runtime,
        threshold=0.25,
        provider=args.provider,
    )
    embedder = FaceEmbedder(models.recognition_runtime, provider=args.provider)
    blank = np.zeros((detector.input_size[1], detector.input_size[0], 3), dtype=np.uint8)
    detector.detect(blank)
    detector.detect(blank)
    embedder.warmup(2)
    print(f"Recognition model: {models.recognition_name}")
    print(f"Detector model: {models.detector_name}")
    print(
        "Face preprocessing: "
        f"{'5-point alignment' if detector.has_landmarks else 'bbox crop'}"
    )
    print(f"Detector providers: {detector.providers}")
    print(f"Recognition providers: {embedder.providers}")
    for warning in (detector.provider_warning, embedder.provider_warning):
        if warning:
            print(f"Provider warning: {warning}", file=sys.stderr)
    return detector, embedder, models


def embedding_from_photo(
    path: Path,
    detector: YOLOFaceDetector,
    embedder: FaceEmbedder,
    min_face_size: float,
    min_sharpness: float,
) -> tuple[np.ndarray | None, str]:
    photo = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if photo is None:
        return None, "could not decode the image"

    detected = detector.detect(photo)
    face_count = len(detected.detections)
    if face_count != 1:
        return None, f"expected exactly one face, found {face_count}"
    try:
        face_image = embedder.extract_face(photo, detected.detections[0])
    except ValueError as error:
        return None, f"face alignment failed: {error}"
    quality = assess_face_quality(
        face_image,
        detected.detections[0],
        min_face_size=min_face_size,
        min_sharpness=min_sharpness,
    )
    if not quality.accepted:
        return None, quality.reason
    embedding = embedder.embed_face(face_image)[0]
    return embedding[None, :].astype(np.float32), "accepted"


def is_new_sample(candidate: np.ndarray, accepted: list[np.ndarray]) -> bool:
    if not accepted:
        return True
    accepted_upright = np.asarray(accepted, dtype=np.float32)[:, 0, :]
    similarities = accepted_upright @ candidate[0]
    return float(similarities.max()) < 0.9999


def enroll_photos(
    name: str,
    photos: list[Path],
    detector: YOLOFaceDetector,
    embedder: FaceEmbedder,
    models: RuntimeModels,
    output: Path,
    *,
    threshold: float = 0.35,
    min_face_size: float = 80.0,
    min_sharpness: float = 25.0,
) -> tuple[Path, int, int]:
    """Build and save one owner template with already initialized models."""
    accepted: list[np.ndarray] = []
    rejected = 0
    for path in photos:
        embedding, reason = embedding_from_photo(
            path,
            detector,
            embedder,
            min_face_size,
            min_sharpness,
        )
        if embedding is None:
            rejected += 1
            print(f"REJECTED  {path.name}: {reason}")
            continue
        if not is_new_sample(embedding, accepted):
            rejected += 1
            print(f"REJECTED  {path.name}: duplicate or nearly identical photo")
            continue
        accepted.append(embedding)
        print(f"ACCEPTED  {path.name}: {len(accepted)}")

    if not accepted:
        raise ValueError(
            f"No suitable face was found for owner '{name}'. Add at least one "
            "clear photograph containing exactly one visible face."
        )

    safe_name = safe_identity_name(name)
    template = build_template(
        safe_name,
        accepted,
        model_sha256=models.recognition_source_sha256,
        threshold=threshold,
        source="photos",
        face_preprocessing=(
            LANDMARK_FACE_PREPROCESSING
            if detector.has_landmarks
            else FACE_PREPROCESSING
        ),
        rotation_angles=(0,),
    )
    saved = save_template(template, output)
    return saved, len(accepted), rejected


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise SystemExit("--threshold must be between 0 and 1")
    photos = expand_photos(args.photos)
    if not photos:
        raise SystemExit(
            "No supported photographs found. Use JPG, JPEG, PNG, WEBP, BMP, TIFF, "
            "or a folder containing those files."
        )
    print(f"Found {len(photos)} photograph(s). Raw photos will not be copied.")

    detector, embedder, models = load_models(args)
    safe_name = safe_identity_name(args.name)
    output = args.output or Path("data/enrollments") / f"{safe_name}.npz"
    saved, accepted_count, rejected = enroll_photos(
        safe_name,
        photos,
        detector,
        embedder,
        models,
        output,
        threshold=args.threshold,
        min_face_size=args.min_face_size,
        min_sharpness=args.min_sharpness,
    )
    template = load_template(saved)
    print()
    print(f"Template saved: {saved}")
    print(f"Accepted photos: {accepted_count}; rejected: {rejected}")
    print(
        f"Template embeddings: {len(template.embeddings)} "
        f"({accepted_count} photo(s))"
    )
    print(f"Initial authorization threshold: {template.threshold:.3f}")
    print("No source photograph was copied into the template.")
    if accepted_count == 1:
        print(
            "Warning: the template contains one photo. It will work, but additional "
            "poses and lighting conditions usually improve recognition stability."
        )


if __name__ == "__main__":
    main()
