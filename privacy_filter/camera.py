from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraInfo:
    width: int
    height: int
    fps: float
    backend: str
    frames: int | None = None


def camera_backends(system: str | None = None) -> tuple[int, ...]:
    operating_system = system or platform.system()
    if operating_system == "Darwin":
        return cv2.CAP_AVFOUNDATION, cv2.CAP_ANY
    if operating_system == "Windows":
        return cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY
    if operating_system == "Linux":
        return cv2.CAP_V4L2, cv2.CAP_ANY
    return (cv2.CAP_ANY,)


class Camera:
    def __init__(
        self,
        index: int,
        width: int,
        height: int,
        requested_fps: float,
    ) -> None:
        self._capture = None
        for backend in camera_backends():
            capture = cv2.VideoCapture(index, backend)
            if capture.isOpened():
                self._capture = capture
                break
            capture.release()
        if self._capture is None:
            raise RuntimeError(
                f"Could not open webcam {index}. Allow camera access in the operating "
                "system privacy settings, close other apps using the camera, and retry."
            )
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if requested_fps > 0:
            self._capture.set(cv2.CAP_PROP_FPS, requested_fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            backend_name = self._capture.getBackendName()
        except cv2.error:
            backend_name = "unknown"
        self.info = CameraInfo(
            width=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(self._capture.get(cv2.CAP_PROP_FPS)),
            backend=backend_name,
        )

    def read(self) -> np.ndarray:
        ok, frame = self._capture.read()
        if not ok:
            raise RuntimeError(
                "The webcam stopped returning frames. Close other apps using the camera "
                "and retry."
            )
        return frame

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class VideoFile:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Input video not found: {self.path}")
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open input video: {self.path}")
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        if width <= 0 or height <= 0:
            self._capture.release()
            raise RuntimeError(f"Input video has an invalid frame size: {self.path}")
        if not np.isfinite(fps) or fps <= 0.0:
            fps = 25.0
        frame_count = max(0, int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        try:
            backend_name = self._capture.getBackendName()
        except cv2.error:
            backend_name = "unknown"
        self.info = CameraInfo(
            width=width,
            height=height,
            fps=fps,
            backend=f"file/{backend_name}",
            frames=frame_count or None,
        )

    def read(self) -> np.ndarray | None:
        ok, frame = self._capture.read()
        return frame if ok else None

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> "VideoFile":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
