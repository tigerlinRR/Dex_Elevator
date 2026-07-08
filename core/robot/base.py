"""Robot arm interface.

The pipeline depends only on this abstraction; the concrete RealMan SDK is
wired in :mod:`core.robot.realman`. This keeps motion control swappable and
makes the press logic testable without hardware (see :mod:`core.robot.sim`).
"""

from __future__ import annotations

import abc

import numpy as np


class RobotArm(abc.ABC):
    @abc.abstractmethod
    def connect(self) -> None:
        ...

    @abc.abstractmethod
    def disconnect(self) -> None:
        ...

    @abc.abstractmethod
    def get_tcp_pose(self) -> np.ndarray:
        """Current TCP pose as a 4x4 transform in the robot base frame."""

    @abc.abstractmethod
    def move_to_pose(self, pose: np.ndarray, speed: float = 0.1) -> bool:
        """Move the TCP to a 4x4 pose (base frame). Returns success."""

    @abc.abstractmethod
    def open_gripper(self) -> None:
        ...

    @abc.abstractmethod
    def close_gripper(self, width: float | None = None) -> None:
        """Close gripper, optionally to a target opening (meters)."""

    def set_manual_mode(self, enable: bool) -> None:
        """Enter/leave backdrivable teach mode for hand-guiding (optional).

        Concrete arms that support freedrive (e.g. the xArm) override this; it is
        used by eye-to-hand calibration to set poses by hand. Default: unsupported.
        """
        raise NotImplementedError("This arm does not support manual/teach mode.")

    def __enter__(self) -> "RobotArm":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()
