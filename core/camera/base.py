"""Camera interface — every concrete driver returns a CameraFrame.

Programming to this interface keeps YOLO, grasping and calibration independent
of the specific sensor (Orbbec today, something else tomorrow).
"""

from __future__ import annotations

import abc

from core.types import CameraFrame, CameraIntrinsics


class Camera(abc.ABC):
    def __init__(self, camera_id: str):
        self.camera_id = camera_id

    @abc.abstractmethod
    def start(self) -> None:
        """Open the device and begin streaming."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop streaming and release the device."""

    @abc.abstractmethod
    def capture(self) -> CameraFrame:
        """Grab one synchronized RGB + aligned-depth frame."""

    @abc.abstractmethod
    def intrinsics(self) -> CameraIntrinsics:
        """Return the (color) intrinsics used for deprojection."""

    def __enter__(self) -> "Camera":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
