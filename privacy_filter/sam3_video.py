from __future__ import annotations

from collections.abc import Iterator, Sequence
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import cv2
import numpy as np


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _union_output_masks(
    outputs: dict[str, Any],
    frame_height: int,
    frame_width: int,
) -> np.ndarray:
    raw_masks = outputs.get("out_binary_masks")
    if raw_masks is None:
        raw_masks = outputs.get("masks")
    if raw_masks is None:
        raise RuntimeError(
            "SAM 3 output has no 'out_binary_masks' or 'masks' field"
        )

    masks = _as_numpy(raw_masks)
    if masks.size == 0:
        return np.zeros((frame_height, frame_width), dtype=bool)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3:
        raise RuntimeError(f"Unexpected SAM 3 mask shape: {masks.shape}")

    union = np.any(masks > 0, axis=0).astype(np.uint8)
    if union.shape != (frame_height, frame_width):
        union = cv2.resize(
            union,
            (frame_width, frame_height),
            interpolation=cv2.INTER_NEAREST,
        )
    return union.astype(bool)


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


def _sam3_responses(
    predictor: Any,
    session_id: str,
    initial_response: dict[str, Any],
    score_threshold: float,
) -> Iterator[dict[str, Any]]:
    yield initial_response
    yield from predictor.handle_stream_request(
        request={
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "forward",
            "start_frame_index": 0,
            "output_prob_thresh": score_threshold,
        }
    )


def _clean_prompts(prompt: str | Sequence[str]) -> tuple[str, ...]:
    candidates = (prompt,) if isinstance(prompt, str) else tuple(prompt)
    prompts: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        clean = candidate.strip()
        if not clean:
            raise ValueError("SAM 3 video prompts cannot be empty")
        key = clean.casefold()
        if key not in seen:
            seen.add(key)
            prompts.append(clean)
    if not prompts:
        raise ValueError("At least one SAM 3 video prompt is required")
    return tuple(prompts)


def _mask_path(mask_directory: Path, frame_index: int) -> Path:
    return mask_directory / f"{frame_index:09d}.png"


def _merge_mask(mask_path: Path, mask: np.ndarray) -> None:
    merged = np.asarray(mask, dtype=bool)
    if mask_path.exists():
        existing = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if existing is None:
            raise RuntimeError(f"Could not read temporary SAM 3 mask: {mask_path}")
        if existing.shape != merged.shape:
            raise RuntimeError(
                f"Temporary mask shape {existing.shape} does not match {merged.shape}"
            )
        merged = np.logical_or(merged, existing > 0)
    if not cv2.imwrite(str(mask_path), merged.astype(np.uint8) * 255):
        raise RuntimeError(f"Could not write temporary SAM 3 mask: {mask_path}")


def process_video_with_sam3(
    video_path: str | Path,
    prompt: str | Sequence[str],
    output_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    score_threshold: float = 0.50,
    pixel_block_size: int = 16,
    max_frames: int = 0,
    output_size: tuple[int, int] | None = None,
) -> dict[str, object]:
    source = Path(video_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video not found: {source}")
    prompts = _clean_prompts(prompt)
    if not 0.0 < score_threshold < 1.0:
        raise ValueError("SAM 3 score threshold must be between 0 and 1")
    if pixel_block_size < 2:
        raise ValueError("Video pixel block size must be at least 2")
    if max_frames < 0:
        raise ValueError("Video max frames cannot be negative")
    if output_size is not None and (output_size[0] <= 0 or output_size[1] <= 0):
        raise ValueError("Video output width and height must be positive")

    output = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else source.with_name(f"{source.stem}.redacted.mp4")
    )
    if output.suffix.lower() != ".mp4":
        raise ValueError("SAM 3 video output must have the .mp4 extension")
    if output == source:
        raise ValueError("Input and output video paths must be different")
    if output.exists():
        raise FileExistsError(f"Output video already exists: {output}")

    checkpoint: Path | None = None
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM 3 checkpoint not found: {checkpoint}")

    try:
        import torch
        from sam3.model_builder import build_sam3_video_predictor
    except ImportError as error:
        raise RuntimeError(
            "Offline video mode requires PyTorch and the official facebookresearch/sam3 package"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("SAM 3 video mode requires a CUDA GPU")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open input video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    target_frames = (
        min(expected_frames, max_frames)
        if max_frames > 0 and expected_frames > 0
        else max_frames or expected_frames
    )
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Input video has an invalid frame size")

    capture.release()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.partial.mp4")
    predictor: Any | None = None
    processed_frames = 0
    masked_frames = 0
    failed_frame_indexes: set[int] = set()
    prompt_frame_counts: dict[str, int] = {}
    completed = False
    started = perf_counter()
    try:
        build_kwargs: dict[str, Any] = {
            "gpus_to_use": list(range(torch.cuda.device_count())),
        }
        if checkpoint is not None:
            build_kwargs.update(
                checkpoint_path=str(checkpoint),
                load_from_HF=False,
            )
        predictor = build_sam3_video_predictor(**build_kwargs)
        full_mask = np.ones((height, width), dtype=bool)
        with TemporaryDirectory(
            prefix=f".{output.stem}.sam3-masks-",
            dir=output.parent,
        ) as mask_directory_name:
            mask_directory = Path(mask_directory_name)
            for prompt_index, clean_prompt in enumerate(prompts, start=1):
                print(
                    f"SAM 3 prompt {prompt_index}/{len(prompts)}: {clean_prompt!r}"
                )
                session_id: str | None = None
                next_frame_index = 0
                try:
                    session = predictor.handle_request(
                        request={
                            "type": "start_session",
                            "resource_path": str(source),
                            "offload_video_to_cpu": True,
                        }
                    )
                    session_id = str(session["session_id"])
                    initial_response = predictor.handle_request(
                        request={
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": 0,
                            "text": clean_prompt,
                            "output_prob_thresh": score_threshold,
                        }
                    )
                    for response in _sam3_responses(
                        predictor,
                        session_id,
                        initial_response,
                        score_threshold,
                    ):
                        frame_index = int(response["frame_index"])
                        if max_frames > 0 and frame_index >= max_frames:
                            break
                        if frame_index < next_frame_index:
                            continue
                        while next_frame_index < frame_index:
                            _merge_mask(
                                _mask_path(mask_directory, next_frame_index),
                                full_mask,
                            )
                            failed_frame_indexes.add(next_frame_index)
                            next_frame_index += 1
                        mask = _union_output_masks(
                            response["outputs"],
                            height,
                            width,
                        )
                        _merge_mask(
                            _mask_path(mask_directory, frame_index),
                            mask,
                        )
                        next_frame_index += 1
                        if next_frame_index % 25 == 0:
                            print(
                                "SAM 3 masks: "
                                f"{next_frame_index}/{expected_frames or '?'} frames "
                                f"for {clean_prompt!r}"
                            )
                    while next_frame_index < target_frames:
                        _merge_mask(
                            _mask_path(mask_directory, next_frame_index),
                            full_mask,
                        )
                        failed_frame_indexes.add(next_frame_index)
                        next_frame_index += 1
                    prompt_frame_counts[clean_prompt] = next_frame_index
                finally:
                    if session_id is not None:
                        try:
                            predictor.handle_request(
                                request={
                                    "type": "close_session",
                                    "session_id": session_id,
                                }
                            )
                        except Exception as error:
                            print(f"SAM 3 session close warning: {error}")

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
            try:
                while True:
                    if max_frames > 0 and processed_frames >= max_frames:
                        break
                    ok, frame = capture.read()
                    if not ok:
                        break
                    mask_file = _mask_path(mask_directory, processed_frames)
                    raw_mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    if raw_mask is None:
                        mask = full_mask
                        failed_frame_indexes.add(processed_frames)
                    else:
                        if raw_mask.shape != (height, width):
                            raise RuntimeError(
                                f"Temporary SAM 3 mask has invalid shape: {raw_mask.shape}"
                            )
                        mask = raw_mask > 0
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
                        masked_frames += 1
                    writer.write(_pixelate_with_mask(frame, mask, pixel_block_size))
                    processed_frames += 1
            finally:
                capture.release()
                writer.release()
        completed = True
    finally:
        if predictor is not None and hasattr(predictor, "shutdown"):
            try:
                predictor.shutdown()
            except Exception as error:
                print(f"SAM 3 shutdown warning: {error}")
        if not completed:
            temporary.unlink(missing_ok=True)

    temporary.replace(output)
    elapsed = perf_counter() - started
    summary: dict[str, object] = {
        "mode": "offline_sam3_video",
        "input": str(source),
        "output": str(output),
        "prompts": list(prompts),
        "prompt_frame_counts": prompt_frame_counts,
        "score_threshold": score_threshold,
        "pixel_block_size": pixel_block_size,
        "max_frames": max_frames,
        "output_size": [output_width, output_height],
        "frames": processed_frames,
        "frames_with_masks": masked_frames,
        "fail_closed_frames": len(failed_frame_indexes),
        "fps": round(fps, 3),
        "elapsed_seconds": round(elapsed, 3),
        "processing_fps": round(processed_frames / elapsed, 3) if elapsed else 0.0,
        "audio_preserved": False,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
