from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

from .enrollment import (
    assess_face_quality,
    build_template,
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
    FACE_ROTATION_ANGLES,
    LANDMARK_FACE_PREPROCESSING,
    FaceEmbedder,
    rotate_face,
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
        default="yolo11",
        help=detector_model_help(),
    )
    parser.add_argument(
        "--recognition-model",
        "--model",
        default="r34-glint360k",
        help=recognition_model_help(),
    )
    parser.add_argument("--provider", choices=PROVIDER_CHOICES, default="auto")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument(
        "--rotations",
        action="store_true",
        help="Add 30/90/180/270/330-degree enrollment rotations",
    )
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
    use_rotations: bool,
) -> tuple[np.ndarray | None, str]:
    photo = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if photo is None:
        return None, "could not decode the image"

    detected = detector.detect(photo)
    face_count = len(detected.detections)
    if face_count != 1:
        return None, f"expected exactly one face, found {face_count}"
    face_image = embedder.extract_face(photo, detected.detections[0])
    quality = assess_face_quality(
        face_image,
        detected.detections[0],
        min_face_size=min_face_size,
        min_sharpness=min_sharpness,
    )
    if not quality.accepted:
        return None, quality.reason
    rotation_angles = FACE_ROTATION_ANGLES if use_rotations else (0,)
    embeddings = [
        embedder.embed_face(rotate_face(face_image, angle))[0]
        for angle in rotation_angles
    ]
    return np.stack(embeddings).astype(np.float32), "accepted"


def is_new_sample(candidate: np.ndarray, accepted: list[np.ndarray]) -> bool:
    if not accepted:
        return True
    accepted_upright = np.asarray(accepted, dtype=np.float32)[:, 0, :]
    similarities = accepted_upright @ candidate[0]
    return float(similarities.max()) < 0.9999


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
    accepted: list[np.ndarray] = []
    rejected = 0
    for path in photos:
        embedding, reason = embedding_from_photo(
            path,
            detector,
            embedder,
            args.min_face_size,
            args.min_sharpness,
            args.rotations,
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
        raise SystemExit(
            "No suitable owner face was found. Add at least one clear photograph "
            "containing exactly one visible face."
        )

    safe_name = safe_identity_name(args.name)
    output = args.output or Path("data/enrollments") / f"{safe_name}.npz"
    template = build_template(
        safe_name,
        accepted,
        model_sha256=models.recognition_source_sha256,
        threshold=args.threshold,
        source="photos",
        face_preprocessing=(
            LANDMARK_FACE_PREPROCESSING
            if detector.has_landmarks
            else FACE_PREPROCESSING
        ),
        rotation_angles=FACE_ROTATION_ANGLES if args.rotations else (0,),
    )
    saved = save_template(template, output)
    print()
    print(f"Template saved: {saved}")
    print(f"Accepted photos: {len(accepted)}; rejected: {rejected}")
    rotation_angles = FACE_ROTATION_ANGLES if args.rotations else (0,)
    print(
        f"Template embeddings: {len(template.embeddings)} "
        f"({len(accepted)} photos x {len(rotation_angles)} orientation(s))"
    )
    print(
        "Rotation centroid indices: "
        + ", ".join(
            f"{index}={angle}deg"
            for index, angle in enumerate(template.rotation_angles)
        )
    )
    print(f"Rotation centroids: {len(template.rotation_centroids)}")
    print(f"Rotation angles: {list(rotation_angles)} degrees")
    print(f"Initial authorization threshold: {template.threshold:.3f}")
    print("No source photograph was copied into the template.")
    if len(accepted) == 1:
        print(
            "Warning: the template contains one photo. It will work, but additional "
            "poses and lighting conditions usually improve recognition stability."
        )


if __name__ == "__main__":
    main()
