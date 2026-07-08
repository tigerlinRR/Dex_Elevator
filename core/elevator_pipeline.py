"""Elevator button-press pipeline orchestrator (SKELETON).

The button-press orchestrator (interface-first, like the rest of ``core``). Wires
the stages for pressing one floor button with the right arm:

    capture (chest Orbbec 335)
        -> YOLO button detect  (yolo.button_detector.ButtonDetector)
        -> read floor labels   (read_floor_label — STUB)
        -> select the button whose label == requested floor
        -> ray ∩ panel-plane -> press target (base frame)   (core.press)
        -> RealMan right arm: standoff -> press -> retract

Assumptions baked into the current bring-up plan:
  * the base has already docked to a repeatable pose so the panel is inside the
    camera FOV and the right arm's reach (the base does NOT move during a press);
  * the panel plane ``(point, outward_normal)`` in the base frame is known from
    config (measured once for the docked pose) — a live plane fit is a follow-up;
  * the LinkerHand is preset to a rigid "pointing" pose out-of-band.

Plain Python so it runs headless and can be wrapped node-by-node in ROS 2 later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.camera.manager import CameraHandle
from core.press import PressTarget, press_target_from_pixel, press_waypoints
from core.robot.base import RobotArm
from core.types import Detection
from yolo.button_detector import ButtonDetector, centroid_pixel, read_floor_label


@dataclass
class PressResult:
    target: Optional[PressTarget]
    num_buttons: int
    matched_floor: Optional[str] = None


class ElevatorPipeline:
    def __init__(
        self,
        camera: CameraHandle,
        detector: ButtonDetector,
        robot: RobotArm | None,
        plane_point,                          # (3,) point on the panel plane (base frame)
        plane_normal,                         # (3,) OUTWARD panel normal toward the robot (base)
        standoff: float = 0.06,
        push_depth: float = 0.010,
        up_hint=(0.0, 0.0, 1.0),
    ):
        self.camera = camera
        self.detector = detector
        self.robot = robot
        self.plane_point = np.asarray(plane_point, dtype=np.float64)
        self.plane_normal = np.asarray(plane_normal, dtype=np.float64)
        self.standoff = standoff
        self.push_depth = push_depth
        self.up_hint = up_hint

    # -- perception ---------------------------------------------------------
    def perceive(self, target_floor: str) -> PressResult:
        """Find the requested floor's button and compute its press target (base frame)."""
        handle = self.camera
        if handle.extrinsic is None:
            raise RuntimeError(
                f"Camera {handle.camera.camera_id} has no eye-to-hand extrinsic. "
                "Run initialization/run_calibration.py first."
            )
        frame = handle.camera.capture()
        detections = self.detector.detect(frame.rgb)
        if not detections:
            return PressResult(None, 0)

        target_det = self._match_floor(frame.rgb, detections, target_floor)
        if target_det is None:
            return PressResult(None, len(detections))

        pt = press_target_from_pixel(
            centroid_pixel(target_det), frame.intrinsics, handle.extrinsic,
            self.plane_point, self.plane_normal, up_hint=self.up_hint,
            detection=target_det, floor=target_floor, confidence=target_det.confidence,
        )
        return PressResult(pt, len(detections), matched_floor=target_floor)

    def _match_floor(self, rgb, detections: list[Detection], target_floor: str) -> Optional[Detection]:
        """Pick the detection whose read label matches ``target_floor``.

        Uses :func:`read_floor_label` (currently a STUB → always ""), so until the
        label reader is trained this returns ``None`` (no match). Swap in the
        trained reader and this selection works unchanged.
        """
        want = str(target_floor).strip().upper()
        for det in detections:
            if read_floor_label(rgb, det).strip().upper() == want:
                return det
        return None

    # -- execution ----------------------------------------------------------
    def press(self, target: PressTarget) -> bool:
        """Standoff -> press -> retract, in straight Cartesian moves. Needs a robot."""
        if self.robot is None:
            raise RuntimeError("No robot configured; cannot press.")
        standoff, press_pose, retract = press_waypoints(
            target, standoff=self.standoff, push_depth=self.push_depth,
        )
        mover = getattr(self.robot, "move_line_to_pose", self.robot.move_to_pose)
        if not self.robot.move_to_pose(standoff):     # get to standoff via joint plan
            return False
        if not mover(press_pose):                     # straight-line push into the button
            return False
        # TODO (hardware): replace the blind push with a force-limited press
        # (RealMan rm_force_position_move_pose) so contact force is bounded.
        return mover(retract)

    def run_once(self, target_floor: str) -> PressResult:
        result = self.perceive(target_floor)
        if result.target is not None and self.robot is not None:
            self.press(result.target)
        return result
