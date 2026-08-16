from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Sequence

import cv2
import numpy as np



def _clean_prompts(prompt: str | Sequence[str]) -> tuple[str, ...]:
    candidates = (prompt,) if isinstance(prompt, str) else tuple(prompt)
    prompts: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        clean = candidate.strip()
        if not clean:
            raise ValueError("Grounded SAM 2 video prompts cannot be empty")
        key = clean.casefold()
        if key not in seen:
            seen.add(key)
            prompts.append(clean)
    if not prompts:
        raise ValueError("At least one Grounded SAM 2 video prompt is required")
    return tuple(prompts)


def _mask_path(mask_directory: Path, frame_index: int) -> Path:
    return mask_directory / f"{frame_index:09d}.png"


def _merge_mask(mask_path: Path, mask: np.ndarray) -> None:
    merged = np.asarray(mask, dtype=bool)
    if mask_path.exists():
        existing = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if existing is None:
            raise RuntimeError(f"Could not read temporary mask: {mask_path}")
        if existing.shape != merged.shape:
            raise RuntimeError(
                f"Temporary mask shape {existing.shape} does not match {merged.shape}"
            )
        merged = np.logical_or(merged, existing > 0)
    if not cv2.imwrite(str(mask_path), merged.astype(np.uint8) * 255):
        raise RuntimeError(f"Could not write temporary mask: {mask_path}")


def _pixelate_with_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    block_size: int,
) -> np.ndarray:
    height, width = frame.shape[:2]
    if mask.shape != (height, width):
        raise ValueError(
            f"Pixelation mask shape {mask.shape} does not match frame {(height, width)}"
        )
    if not bool(mask.any()):
        return frame.copy()
    reduced = cv2.resize(
        frame,
        (max(1, width // block_size), max(1, height // block_size)),
        interpolation=cv2.INTER_AREA,
    )
    pixelated = cv2.resize(
        reduced,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    output = frame.copy()
    output[mask] = pixelated[mask]
    return output


def _to_device(batch: Any, device: str) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch


def _detect_boxes(
    image_path: Path,
    prompts: tuple[str, ...],
    processor: Any,
    model: Any,
    torch: Any,
    device: str,
    box_threshold: float,
    text_threshold: float,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    from PIL import Image

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        inputs = _to_device(
            processor(
                images=image,
                text=[list(prompts)],
                return_tensors="pt",
            ),
            device,
        )
        with torch.inference_mode():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]

    boxes = result["boxes"].detach().cpu().numpy().astype(np.float32)
    scores = result["scores"].detach().cpu().numpy().astype(np.float32)
    raw_labels = result.get("text_labels", result.get("labels", []))
    labels = [str(label) for label in raw_labels]
    return boxes.reshape(-1, 4), labels, scores.reshape(-1)


def _union_sam2_logits(
    logits: Any,
    frame_height: int,
    frame_width: int,
) -> np.ndarray:
    if hasattr(logits, "detach"):
        logits = logits.detach()
    if hasattr(logits, "cpu"):
        logits = logits.cpu()
    values = np.asarray(logits)
    if values.size == 0:
        return np.zeros((frame_height, frame_width), dtype=bool)
    if values.ndim == 4 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim == 2:
        values = values[None]
    if values.ndim != 3:
        raise RuntimeError(f"Unexpected SAM 2 mask shape: {values.shape}")
    mask = np.any(values > 0.0, axis=0).astype(np.uint8)
    if mask.shape != (frame_height, frame_width):
        mask = cv2.resize(
            mask,
            (frame_width, frame_height),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask.astype(bool)


def _open_video(
    source: Path,
) -> tuple[Any, float, int, int, int]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open input video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Input video has an invalid frame size")
    return capture, fps, width, height, reported_frames


def _extract_chunk(
    capture: Any,
    frame_directory: Path,
    source_width: int,
    source_height: int,
    inference_max_side: int,
    chunk_limit: int,
) -> int:
    chunk_frames = 0
    while chunk_frames < chunk_limit:
        ok, frame = capture.read()
        if not ok:
            break
        if (
            inference_max_side > 0
            and max(source_width, source_height) > inference_max_side
        ):
            scale = inference_max_side / max(source_width, source_height)
            frame = cv2.resize(
                frame,
                (
                    max(1, round(source_width * scale)),
                    max(1, round(source_height * scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )
        frame_path = frame_directory / f"{chunk_frames:09d}.jpg"
        if not cv2.imwrite(
            str(frame_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        ):
            raise RuntimeError(f"Could not extract video frame: {frame_path}")
        chunk_frames += 1
    return chunk_frames


def _resolve_device(torch: Any, requested: str) -> str:
    normalized = requested.strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError("Offline device must be one of: auto, cpu, cuda")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--offline-device cuda requested, but CUDA is unavailable")
        return "cuda"
    if normalized == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def process_video_with_grounded_sam2(
    video_path: str | Path,
    prompt: str | list[str] | tuple[str, ...],
    output_path: str | Path | None = None,
    grounding_model_id: str = "IDEA-Research/grounding-dino-tiny",
    sam2_model_id: str = "facebook/sam2.1-hiera-small",
    sam2_checkpoint_path: str | Path | None = None,
    sam2_model_config: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
    box_threshold: float = 0.20,
    text_threshold: float = 0.20,
    redetect_interval: int = 25,
    inference_max_side: int = 1280,
    max_frames: int = 0,
    device: str = "auto",
    pixel_block_size: int = 16,
    output_size: tuple[int, int] | None = None,
) -> dict[str, object]:
    source = Path(video_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video not found: {source}")
    prompts = _clean_prompts(prompt)
    if not grounding_model_id.strip():
        raise ValueError("Grounding DINO model id cannot be empty")
    if not sam2_model_id.strip():
        raise ValueError("SAM 2 model id cannot be empty")
    if not 0.0 < box_threshold < 1.0:
        raise ValueError("Grounding DINO box threshold must be between 0 and 1")
    if not 0.0 < text_threshold < 1.0:
        raise ValueError("Grounding DINO text threshold must be between 0 and 1")
    if redetect_interval < 1:
        raise ValueError("Grounding DINO redetect interval must be positive")
    if inference_max_side != 0 and inference_max_side < 320:
        raise ValueError("Video inference max side must be 0 or at least 320 pixels")
    if max_frames < 0:
        raise ValueError("Video max frames cannot be negative")
    if pixel_block_size < 2:
        raise ValueError("Video pixel block size must be at least 2")
    if output_size is not None and (output_size[0] <= 0 or output_size[1] <= 0):
        raise ValueError("Video output width and height must be positive")

    output = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else source.with_name(f"{source.stem}.redacted.mp4")
    )
    if output.suffix.lower() != ".mp4":
        raise ValueError("Offline video output must have the .mp4 extension")
    if output == source:
        raise ValueError("Input and output video paths must be different")
    if output.exists():
        raise FileExistsError(f"Output video already exists: {output}")

    checkpoint: Path | None = None
    if sam2_checkpoint_path is not None:
        checkpoint = Path(sam2_checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM 2.1 checkpoint not found: {checkpoint}")

    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as error:
        raise RuntimeError(
            "Grounded SAM 2 mode requires PyTorch, transformers and the official "
            "facebookresearch/sam2 package"
        ) from error

    resolved_device = _resolve_device(torch, device)
    if resolved_device == "cpu":
        print("Grounded SAM 2 warning: using CPU; processing will be slow")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.partial.mp4")
    started = perf_counter()
    processed_frames = 0
    frames_with_masks = 0
    failed_frame_indexes: set[int] = set()
    detected_objects = 0
    detections_by_label: dict[str, int] = {}
    detection_keyframes = 0
    frame_count = 0
    reported_frames = 0
    completed = False

    try:
        with TemporaryDirectory(
            prefix=f".{output.stem}.grounded-sam2-",
            dir=output.parent,
        ) as work_directory_name:
            work_directory = Path(work_directory_name)
            mask_directory = work_directory / "masks"
            mask_directory.mkdir()
            source_capture, fps, width, height, reported_frames = _open_video(source)

            print(f"Loading Grounding DINO: {grounding_model_id}")
            processor = AutoProcessor.from_pretrained(grounding_model_id)
            grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                grounding_model_id
            ).to(resolved_device)
            grounding_model.eval()

            print(
                "Loading SAM 2.1: "
                + (str(checkpoint) if checkpoint is not None else sam2_model_id)
            )
            if checkpoint is None:
                video_predictor = SAM2VideoPredictor.from_pretrained(
                    sam2_model_id,
                    device=resolved_device,
                )
            else:
                video_predictor = build_sam2_video_predictor(
                    sam2_model_config,
                    str(checkpoint),
                    device=resolved_device,
                )

            processed_mask_frames: set[int] = set()
            inference_context = torch.inference_mode()
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if resolved_device == "cuda"
                else torch.autocast(device_type="cpu", enabled=False)
            )
            try:
                with inference_context, autocast_context:
                    start_frame = 0
                    while max_frames == 0 or start_frame < max_frames:
                        remaining = (
                            redetect_interval
                            if max_frames == 0
                            else min(redetect_interval, max_frames - start_frame)
                        )
                        with TemporaryDirectory(
                            prefix=f"frames-{start_frame:09d}-",
                            dir=work_directory,
                        ) as frame_directory_name:
                            frame_directory = Path(frame_directory_name)
                            chunk_count = _extract_chunk(
                                source_capture,
                                frame_directory,
                                width,
                                height,
                                inference_max_side,
                                remaining,
                            )
                            if chunk_count == 0:
                                break
                            frame_count += chunk_count
                            detection_keyframes += 1
                            frame_path = frame_directory / "000000000.jpg"
                            boxes, labels, scores = _detect_boxes(
                                frame_path,
                                prompts,
                                processor,
                                grounding_model,
                                torch,
                                resolved_device,
                                box_threshold,
                                text_threshold,
                            )
                            detected_objects += len(boxes)
                            for label in labels:
                                detections_by_label[label] = (
                                    detections_by_label.get(label, 0) + 1
                                )
                            chunk_end = start_frame + chunk_count
                            total_label = (
                                max_frames
                                if max_frames > 0
                                else reported_frames or "?"
                            )
                            print(
                                f"Grounding DINO frame {start_frame}/{total_label}: "
                                f"{len(boxes)} objects "
                                f"({', '.join(f'{score:.3f}' for score in scores) or 'none'})"
                            )
                            if len(boxes) == 0:
                                processed_mask_frames.update(
                                    range(start_frame, chunk_end)
                                )
                            else:
                                inference_state = video_predictor.init_state(
                                    video_path=str(frame_directory),
                                    offload_video_to_cpu=True,
                                    offload_state_to_cpu=resolved_device == "cpu",
                                )
                                try:
                                    for object_id, box in enumerate(boxes, start=1):
                                        video_predictor.add_new_points_or_box(
                                            inference_state=inference_state,
                                            frame_idx=0,
                                            obj_id=object_id,
                                            box=box,
                                        )
                                    expected_local_frames = set(range(chunk_count))
                                    returned_local_frames: set[int] = set()
                                    for local_frame, _object_ids, mask_logits in (
                                        video_predictor.propagate_in_video(
                                            inference_state,
                                            start_frame_idx=0,
                                            max_frame_num_to_track=chunk_count - 1,
                                        )
                                    ):
                                        local_frame = int(local_frame)
                                        if local_frame not in expected_local_frames:
                                            continue
                                        global_frame = start_frame + local_frame
                                        mask = _union_sam2_logits(
                                            mask_logits,
                                            height,
                                            width,
                                        )
                                        _merge_mask(
                                            _mask_path(mask_directory, global_frame),
                                            mask,
                                        )
                                        returned_local_frames.add(local_frame)
                                        processed_mask_frames.add(global_frame)
                                    missing = (
                                        expected_local_frames - returned_local_frames
                                    )
                                    failed_frame_indexes.update(
                                        start_frame + index for index in missing
                                    )
                                finally:
                                    video_predictor.reset_state(inference_state)
                                    del inference_state
                            start_frame = chunk_end
                            if chunk_count < remaining:
                                break
                            if resolved_device == "cuda":
                                torch.cuda.empty_cache()
            finally:
                source_capture.release()
            if frame_count == 0:
                raise RuntimeError("Input video contains no readable frames")

            capture = cv2.VideoCapture(str(source))
            if not capture.isOpened():
                raise RuntimeError(f"OpenCV could not reopen input video: {source}")
            output_width, output_height = output_size or (width, height)
            writer = cv2.VideoWriter(
                str(temporary),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (output_width, output_height),
            )
            if not writer.isOpened():
                capture.release()
                raise RuntimeError(f"OpenCV could not create output video: {temporary}")
            full_mask = np.ones((height, width), dtype=bool)
            empty_mask = np.zeros((height, width), dtype=bool)
            try:
                while True:
                    if processed_frames >= frame_count:
                        break
                    ok, frame = capture.read()
                    if not ok:
                        break
                    raw_mask = cv2.imread(
                        str(_mask_path(mask_directory, processed_frames)),
                        cv2.IMREAD_GRAYSCALE,
                    )
                    if raw_mask is not None:
                        mask = raw_mask > 0
                    elif processed_frames in processed_mask_frames:
                        mask = empty_mask
                    else:
                        mask = full_mask
                        failed_frame_indexes.add(processed_frames)
                    if (output_width, output_height) != (width, height):
                        frame = cv2.resize(
                            frame,
                            (output_width, output_height),
                            interpolation=cv2.INTER_AREA,
                        )
                        mask = cv2.resize(
                            mask.astype(np.uint8),
                            (output_width, output_height),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    if bool(mask.any()):
                        frames_with_masks += 1
                    writer.write(_pixelate_with_mask(frame, mask, pixel_block_size))
                    processed_frames += 1
            finally:
                capture.release()
                writer.release()

            del video_predictor, grounding_model
            if resolved_device == "cuda":
                torch.cuda.empty_cache()
        completed = True
    finally:
        if not completed:
            temporary.unlink(missing_ok=True)

    temporary.replace(output)
    elapsed = perf_counter() - started
    summary: dict[str, object] = {
        "mode": "offline_grounded_sam2_video",
        "input": str(source),
        "output": str(output),
        "prompts": list(prompts),
        "grounding_model": grounding_model_id,
        "sam2_model": str(checkpoint) if checkpoint is not None else sam2_model_id,
        "device": resolved_device,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "redetect_interval": redetect_interval,
        "chunk_frames": redetect_interval,
        "inference_max_side": inference_max_side,
        "output_size": [output_width, output_height],
        "max_frames": max_frames,
        "source_reported_frames": reported_frames,
        "detection_keyframes": detection_keyframes,
        "detected_objects": detected_objects,
        "detections_by_label": detections_by_label,
        "frames": processed_frames,
        "frames_with_masks": frames_with_masks,
        "fail_closed_frames": len(failed_frame_indexes),
        "fps": round(fps, 3),
        "elapsed_seconds": round(elapsed, 3),
        "processing_fps": round(processed_frames / elapsed, 3) if elapsed else 0.0,
        "audio_preserved": False,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
