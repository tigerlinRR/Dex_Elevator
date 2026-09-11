"""Multi-camera manager — we run two Orbbec 335L units.

Builds cameras from ``configs/cameras.yaml`` and loads each camera's calibration
**once**, sharing it with the whole runtime:
  * Part 1 — intrinsics  (``<cam>_intrinsics.npz``): K + distortion, handed to the
    camera driver so it does not re-read the SDK factory values.
  * Part 2 — extrinsics  (``<cam>.npy``): 4x4 extrinsic, exposed on the handle.
Then exposes capture-by-role / -by-id helpers.

**Two camera mounts.** ``cameras.yaml`` ``mount:`` says what the extrinsic file
holds, because the two are the same shape and cannot be told apart by content:

* ``fixed`` (default, the chest 335) — ``base_T_camera``, a constant. The camera
  and the arm base ride the torso lift together, so it never changes.
* ``arm`` — ``gripper_T_camera``; the live ``base_T_camera`` is only defined
  together with an arm pose. Ask :meth:`CameraHandle.base_T_camera` for it.

``handle.extrinsic`` keeps its original meaning for ``fixed`` cameras, so every
existing caller is unchanged. Arm-mounted cameras must go through
``base_T_camera(arm_pose)``, which refuses to guess when the pose is missing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibration.frames import check_frame
from calibration.io import load_camera_intrinsics, load_extrinsic
from core.camera.base import Camera
from core.camera.orbbec import OrbbecCamera
from core.config import load_cameras
from core.types import CameraFrame, CameraIntrinsics


@dataclass
class CameraHandle:
    camera: Camera
    role: str
    extrinsic: np.ndarray | None              # 4x4 (Part 2), None until calibrated.
                                              # mount "fixed" -> base_T_camera (unchanged meaning)
                                              # mount "arm"   -> gripper_T_camera
    intrinsics: CameraIntrinsics | None = None  # K + distortion (Part 1), None -> SDK fallback
    mount: str = "fixed"                      # "fixed" (torso/world) | "arm" (rides the arm)

    def base_T_camera(self, arm_pose: np.ndarray | None = None) -> np.ndarray | None:
        """The camera's pose in the arm base frame — a CONSTANT only for a fixed mount.

        Args:
            arm_pose: 4x4 ``base_T_gripper`` read at the instant the frame was
                captured, with the arm stationary. Required for ``mount: arm``,
                ignored for ``mount: fixed``.

        Returns None if the camera is not calibrated yet. Raises for an
        arm-mounted camera with no pose rather than returning ``gripper_T_camera``
        — that would be a wrong answer of exactly the right shape, and everything
        downstream (ray-plane intersection, plane fit, obstacle check) would
        happily consume it.
        """
        if self.extrinsic is None:
            return None
        if self.mount == "fixed":
            return self.extrinsic
        if self.mount == "arm":
            if arm_pose is None:
                raise ValueError(
                    f"camera mount is 'arm': base_T_camera needs the arm pose at capture time "
                    f"(pass arm.get_tcp_pose()); there is no constant extrinsic for this mount."
                )
            return np.asarray(arm_pose, dtype=np.float64) @ self.extrinsic
        raise ValueError(f"unknown mount {self.mount!r} (expected 'fixed' or 'arm')")


class CameraManager:
    """Owns the lifecycle of all configured cameras."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_cameras()
        self.handles: dict[str, CameraHandle] = {}

    def build(self) -> "CameraManager":
        for cam_cfg in self.config.get("cameras", []):
            cam_id = cam_cfg["id"]
            width = cam_cfg.get("width", 1280)
            height = cam_cfg.get("height", 720)

            # Part 1 — shared intrinsics (K + distortion), loaded once at the
            # resolution they were calibrated at. If that differs from the stream
            # resolution, warn loudly and rescale K (rather than silently use a
            # wrong K) — the correct practice is to calibrate at the stream res.
            intrinsics = None
            intr_file = cam_cfg.get("intrinsics_file")
            if intr_file:
                try:
                    intrinsics = load_camera_intrinsics(intr_file)
                    if (intrinsics.width, intrinsics.height) != (width, height):
                        print(
                            f"⚠ {cam_id}: intrinsics calibrated at "
                            f"{intrinsics.width}x{intrinsics.height} but stream is "
                            f"{width}x{height}; rescaling K (assumes pure resize). "
                            f"Recalibrate at the stream resolution to be safe."
                        )
                        intrinsics = intrinsics.scaled_to(width, height)
                except FileNotFoundError:
                    pass  # not calibrated yet — run calibrate_intrinsics.py (falls back to SDK)

            camera = OrbbecCamera(
                camera_id=cam_id,
                serial=cam_cfg.get("serial"),
                match_name=cam_cfg.get("match_name"),
                width=width,
                height=height,
                fps=cam_cfg.get("fps", 30),
                intrinsics=intrinsics,
                exposure=cam_cfg.get("exposure"),
                gain=cam_cfg.get("gain"),
            )

            # Part 2 — shared extrinsic, loaded once. What it MEANS depends on the
            # mount, and the file cannot say (both are a bare 4x4), so the sidecar
            # frame tag is checked against `mount:` and a mismatch is fatal —
            # silently mixing them up yields plausible coordinates that are wrong
            # by the length of the arm.
            mount = cam_cfg.get("mount", "fixed")
            extrinsic = None
            extr_file = cam_cfg.get("extrinsic_file")
            if extr_file:
                try:
                    extrinsic = load_extrinsic(extr_file)
                    check_frame(extr_file, mount)
                except FileNotFoundError:
                    pass  # not calibrated yet — run the matching run_calibration script

            self.handles[cam_id] = CameraHandle(
                camera=camera, role=cam_cfg.get("role", cam_id),
                extrinsic=extrinsic, intrinsics=intrinsics, mount=mount,
            )
        return self

    def start(self) -> None:
        for h in self.handles.values():
            h.camera.start()

    def stop(self) -> None:
        for h in self.handles.values():
            h.camera.stop()

    def get(self, camera_id: str) -> CameraHandle:
        return self.handles[camera_id]

    def by_role(self, role: str) -> CameraHandle:
        for h in self.handles.values():
            if h.role == role:
                return h
        raise KeyError(f"No camera with role {role!r}")

    def capture_all(self) -> dict[str, CameraFrame]:
        return {cid: h.camera.capture() for cid, h in self.handles.items()}

    def __enter__(self) -> "CameraManager":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
