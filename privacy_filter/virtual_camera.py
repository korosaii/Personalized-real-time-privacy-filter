from __future__ import annotations

from threading import Event, Lock, Thread
from types import TracebackType
from typing import Any

import cv2
import numpy as np


class VirtualCameraSink:
    """Publish the latest processed BGR frame at a stable virtual-camera FPS."""

    def __init__(self, width: int, height: int, fps: float) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Virtual-camera width and height must be positive")
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("Virtual-camera FPS must be positive")

        try:
            import pyvirtualcam
        except ImportError as error:
            raise RuntimeError(
                "Virtual-camera support is not installed. Run "
                "'python -m pip install -e \".[virtual-camera]\"' and install "
                "a supported virtual-camera device such as OBS Virtual Camera."
            ) from error

        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        try:
            self._camera: Any = pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                fmt=pyvirtualcam.PixelFormat.BGR,
                print_fps=False,
            )
        except Exception as error:
            raise RuntimeError(
                "Could not open a virtual camera. Install/enable OBS Virtual "
                "Camera (or another pyvirtualcam-compatible device) and retry. "
                f"Backend error: {error}"
            ) from error

        # Start fail-closed: until the first processed frame arrives, consumers
        # receive a black frame rather than any unredacted camera content.
        self._latest = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._lock = Lock()
        self._stop = Event()
        self._failure: BaseException | None = None
        self._thread = Thread(
            target=self._publish_loop,
            name="privacy-virtual-camera",
            daemon=True,
        )
        self._thread.start()

    @property
    def device(self) -> str:
        return str(self._camera.device)

    @property
    def frames_sent(self) -> int:
        return int(self._camera.frames_sent)

    def submit(self, frame: np.ndarray) -> None:
        self.raise_if_failed()
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Virtual-camera frames must have shape HxWx3")
        if frame.dtype != np.uint8:
            raise ValueError("Virtual-camera frames must use uint8 pixels")

        prepared = frame
        if (frame.shape[1], frame.shape[0]) != (self.width, self.height):
            prepared = cv2.resize(
                frame,
                (self.width, self.height),
                interpolation=cv2.INTER_AREA,
            )
        prepared = np.ascontiguousarray(prepared)
        with self._lock:
            self._latest = prepared.copy()

    def raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError(
                f"Virtual camera stopped publishing: {self._failure}"
            ) from self._failure

    def _publish_loop(self) -> None:
        try:
            while not self._stop.is_set():
                with self._lock:
                    frame = self._latest
                self._camera.send(frame)
                self._camera.sleep_until_next_frame()
        except BaseException as error:
            self._failure = error
            self._stop.set()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, 3.0 / self.fps))
        self._camera.close()

    def __enter__(self) -> "VirtualCameraSink":
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


def virtual_camera_fps(reported_fps: float, requested_fps: float) -> float:
    """Choose a usable output FPS when a webcam backend reports zero/NaN."""
    if np.isfinite(reported_fps) and reported_fps > 0.0:
        return float(reported_fps)
    if np.isfinite(requested_fps) and requested_fps > 0.0:
        return float(requested_fps)
    return 30.0
