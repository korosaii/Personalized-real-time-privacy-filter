from __future__ import annotations

from dataclasses import dataclass
import platform

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraInfo:
    width: int
    height: int
    fps: float


class Camera:
    def __init__(
        self,
        index: int,
        width: int,
        height: int,
        requested_fps: float,
    ) -> None:
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
        self._capture = cv2.VideoCapture(index, backend)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if requested_fps > 0:
            self._capture.set(cv2.CAP_PROP_FPS, requested_fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(
                "Could not open the webcam. On macOS, allow Camera access for VS Code "
                "(System Settings → Privacy & Security → Camera), then retry."
            )

        self.info = CameraInfo(
            width=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(self._capture.get(cv2.CAP_PROP_FPS)),
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
