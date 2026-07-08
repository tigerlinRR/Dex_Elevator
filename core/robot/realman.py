"""RealMan RM-series arm adapter — the only code that talks to the real arm.

The DEX robot's arms are **RealMan RM** units (NOT the xArm the original scaffold
assumed), driven by the ``Robotic_Arm`` Python SDK (``rm_robot_interface``). This
module is the single place that imports the SDK; it implements the
:class:`RobotArm` interface so the calibration capture, the press pipeline and the
tests can all treat it interchangeably with :class:`core.robot.sim.SimRobot`.

Cell facts (see the project notes / `ssh dex4`):
- Dual arm: LEFT ``192.168.11.42``, RIGHT ``192.168.11.43``, port ``8080``.
  Button pressing uses the **right** arm.
- A vertical **lift column** (torso) is reached through the LEFT arm's IP; the
  chest camera and the arm bases ride the lift together, so ``base_T_camera`` is a
  constant (eye-to-hand calibrated once).
- The end-effector is a **LinkerHand** dexterous hand, controlled by its own
  bridge (``linker_hand_bridge.py`` on the robot) — NOT the RealMan gripper. So
  ``open_gripper``/``close_gripper`` here are intentionally no-ops; pressing uses a
  fixed "pointing" hand pose set out-of-band.

Conventions (matching :mod:`core.transforms`):
- TCP pose is a 4x4 ``base_T_gripper`` in **meters**.
- The RealMan SDK reports/consumes a pose as ``[x, y, z, rx, ry, rz]`` with
  **position in meters and orientation in radians** (confirmed from the SDK
  ``rm_movej_p``/``rm_movel`` docstrings). We convert orientation via the SDK's own
  euler<->quaternion helpers so we never hard-code the Euler axis order.

Install (on the robot): the SDK is packaged as ``Robotic_Arm`` and lives in the
``richtech-v3`` conda env. It is an optional runtime dep here — the module imports
and degrades cleanly when the SDK is absent (e.g. on the dev laptop).
"""

from __future__ import annotations

import numpy as np

from core.robot.base import RobotArm
from core.transforms import make_transform, matrix_to_quat, quat_to_matrix

try:
    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e  # type: ignore

    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - SDK is an optional runtime dep
    _SDK_AVAILABLE = False


# Known arm endpoints on the DEX robot.
ARM_IPS = {"left": "192.168.11.42", "right": "192.168.11.43"}


class RealmanArm(RobotArm):
    """One RealMan RM arm reached over TCP/IP via the ``Robotic_Arm`` SDK.

    ``side`` picks the default IP ("right" is the button-pressing arm); pass an
    explicit ``ip`` to override. ``max_speed_pct`` caps every move for safety
    during bring-up — raise it once motion is validated.
    """

    def __init__(
        self,
        side: str = "right",
        ip: str | None = None,
        port: int = 8080,
        max_speed_pct: int = 20,
        clear_errors_on_connect: bool = True,
    ):
        self.side = side
        self.ip = ip or ARM_IPS.get(side)
        if self.ip is None:
            raise ValueError(f"unknown arm side {side!r}; pass an explicit ip=")
        self.port = port
        self.max_speed_pct = int(np.clip(max_speed_pct, 1, 100))
        self.clear_errors_on_connect = clear_errors_on_connect
        self._arm: "RoboticArm | None" = None

    # -- lifecycle ----------------------------------------------------------
    def connect(self) -> None:
        if not _SDK_AVAILABLE:
            raise ImportError(
                "RealMan SDK not importable. It ships as `Robotic_Arm` and lives in "
                "the robot's `richtech-v3` conda env; run calibration/motion on the "
                "robot (ssh dex4), not the laptop."
            )
        # RM_TRIPLE_MODE_E matches the working init scripts on the robot. The
        # RealMan controller allows multiple client connections, but drag-teach and
        # motion still contend with whatever else drives the arm — make sure the
        # richtech stack isn't commanding this arm during calibration.
        arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = arm.rm_create_robot_arm(self.ip, self.port)
        if handle is None or getattr(handle, "id", 0) <= 0:
            raise RuntimeError(
                f"Could not connect to the RealMan arm at {self.ip}:{self.port} "
                f"(handle={handle}). Is the controller powered and reachable?"
            )
        self._arm = arm
        if self.clear_errors_on_connect:
            arm.rm_clear_system_err()

    def disconnect(self) -> None:
        if self._arm is not None:
            self._arm.rm_delete_robot_arm()
            self._arm = None

    def _require(self) -> "RoboticArm":
        if self._arm is None:
            raise RuntimeError("Arm not connected; call connect() first.")
        return self._arm

    # -- pose <-> matrix (convention-proof via the SDK's own helpers) -------
    def _matrix_from_pose6(self, pose6) -> np.ndarray:
        """``[x,y,z,rx,ry,rz]`` (m, rad) -> 4x4 base_T_gripper (m)."""
        x, y, z, rx, ry, rz = [float(v) for v in pose6[:6]]
        # SDK euler(rad) -> quaternion [w,x,y,z]; then our unambiguous quat->R.
        quat = self._require().rm_algo_euler2quaternion([rx, ry, rz])
        R = quat_to_matrix(*[float(q) for q in quat])
        return make_transform(R, np.array([x, y, z], dtype=np.float64))

    def _pose6_from_matrix(self, T: np.ndarray) -> list:
        """4x4 base_T_gripper (m) -> ``[x,y,z,rx,ry,rz]`` (m, rad) for the SDK."""
        T = np.asarray(T, dtype=np.float64)
        w, x, y, z = matrix_to_quat(T[:3, :3])
        rx, ry, rz = self._require().rm_algo_quaternion2euler([float(w), float(x), float(y), float(z)])
        t = T[:3, 3]
        return [float(t[0]), float(t[1]), float(t[2]), float(rx), float(ry), float(rz)]

    # -- state --------------------------------------------------------------
    def get_tcp_pose(self) -> np.ndarray:
        """Current TCP pose as a 4x4 ``base_T_gripper`` (meters).

        Reads ``rm_get_current_arm_state()['pose']`` = ``[x,y,z,rx,ry,rz]``. The
        pose is reported in the currently-active tool frame; for eye-to-hand it
        only needs to be *consistent* across samples (the board is rigid to it), so
        don't change the tool frame mid-capture.
        """
        arm = self._require()
        code, state = arm.rm_get_current_arm_state()
        if code != 0 or not state or "pose" not in state:
            raise RuntimeError(
                f"rm_get_current_arm_state failed (code={code}, state={state!r})."
            )
        pose = state["pose"]
        if len(pose) < 6:
            raise RuntimeError(f"unexpected pose length {len(pose)}: {pose!r}")
        return self._matrix_from_pose6(pose)

    def get_joint_angles(self) -> np.ndarray:
        """Current joint angles (degrees)."""
        code, joints = self._require().rm_get_joint_degree()
        if code != 0 or joints is None:
            raise RuntimeError(f"rm_get_joint_degree failed (code={code}).")
        return np.asarray(joints, dtype=np.float64)

    # -- motion -------------------------------------------------------------
    def move_to_pose(self, pose: np.ndarray, speed: float = 0.1) -> bool:
        """Move to a 4x4 base-frame pose via joint-space planning (``rm_movej_p``).

        ``speed`` is a fraction 0..1 mapped to the SDK's 1..100% and capped at
        ``max_speed_pct`` for safety. Blocks until the move completes.
        """
        arm = self._require()
        v = int(np.clip(round(speed * 100), 1, self.max_speed_pct))
        pose6 = self._pose6_from_matrix(pose)
        # connect=0 (plan+execute now), r=0 (no blend), block=1 (wait).
        code = arm.rm_movej_p(pose6, v, 0, 0, 1)
        return code == 0

    def move_line_to_pose(self, pose: np.ndarray, speed: float = 0.05) -> bool:
        """Straight-line Cartesian move (``rm_movel``) — used for press approach/retract."""
        arm = self._require()
        v = int(np.clip(round(speed * 100), 1, self.max_speed_pct))
        pose6 = self._pose6_from_matrix(pose)
        code = arm.rm_movel(pose6, v, 0, 0, 1)
        return code == 0

    # -- lift (torso column; reached through the LEFT arm's controller) -----
    def get_lift_height(self) -> float:
        """Current lift height in **mm**."""
        code, state = self._require().rm_get_lift_state()
        if code != 0 or not state:
            raise RuntimeError(f"rm_get_lift_state failed (code={code}).")
        return float(state.get("pos", 0))

    def set_lift_height(self, height_mm: float, speed: int = 50) -> bool:
        """Move the lift to ``height_mm``. Blocks. NOTE: the lift lives on the LEFT
        controller — construct a ``RealmanArm(side='left')`` for lift control."""
        code = self._require().rm_set_lift_height(int(np.clip(speed, 1, 100)), int(height_mm), 0)
        return code == 0

    # -- teach / drag (hand-guiding for eye-to-hand capture) ----------------
    def set_manual_mode(self, enable: bool) -> None:
        """Enter/leave drag-teach (zero-gravity) mode so poses can be set by hand.

        Mirrors the xArm teach mode used by the calibration capture. ``get_tcp_pose``
        keeps reading the live pose while dragging.
        """
        arm = self._require()
        code = arm.rm_start_drag_teach(0) if enable else arm.rm_stop_drag_teach()
        if code != 0:
            raise RuntimeError(
                f"rm_{'start' if enable else 'stop'}_drag_teach failed (code={code})."
            )

    # -- gripper (LinkerHand is controlled separately; these are no-ops) ----
    def open_gripper(self) -> None:
        # The end-effector is a LinkerHand dexterous hand driven by its own bridge,
        # not the RealMan gripper. Pressing uses a fixed "pointing" hand pose set
        # out-of-band, so there is nothing to do here.
        pass

    def close_gripper(self, width: float | None = None) -> None:
        pass
