"""Simulated robot arm — records motions, never moves hardware.

Lets the elevator pipeline and tests run end-to-end with no arm SDK: every
move/gripper call succeeds and is logged to ``history`` for assertions.
"""

from __future__ import annotations

import numpy as np

from core.robot.base import RobotArm
from core.transforms import identity


class SimRobot(RobotArm):
    def __init__(self, fail_moves: bool = False):
        self._tcp = identity()
        self._connected = False
        self.fail_moves = fail_moves
        self.history: list[tuple[str, object]] = []

    def connect(self) -> None:
        self._connected = True
        self.history.append(("connect", None))

    def disconnect(self) -> None:
        self._connected = False
        self.history.append(("disconnect", None))

    def get_tcp_pose(self) -> np.ndarray:
        return self._tcp.copy()

    def move_to_pose(self, pose: np.ndarray, speed: float = 0.1) -> bool:
        if self.fail_moves:
            self.history.append(("move_failed", np.asarray(pose, dtype=np.float64).copy()))
            return False
        self._tcp = np.asarray(pose, dtype=np.float64).copy()
        self.history.append(("move", self._tcp.copy()))
        return True

    def open_gripper(self) -> None:
        self.history.append(("open", None))

    def close_gripper(self, width: float | None = None) -> None:
        self.history.append(("close", width))
