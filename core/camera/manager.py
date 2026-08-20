"""Multi-camera manager — we run two Orbbec 335L units.

Builds cameras from ``configs/cameras.yaml`` and loads each camera's calibration
**once**, sharing it with the whole runtime:
  * Part 1 — intrinsics  (``<cam>_intrinsics.npz``): K + distortion, handed to the
    camera driver so it does not re-read the SDK factory values.
  * Part 2 — extrinsics  (``<cam>.npy``): 4x4 base_T_camera, exposed on the handle.
Then exposes capture-by-role / -by-id helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibration.io import load_camera_intrinsics, load_extrinsic
from core.camera.base import Camera
from core.camera.orbbec import OrbbecCamera
from core.config import load_cameras
from core.types import CameraFrame, CameraIntrinsics


@dataclass
class CameraHandle:
    camera: Camera
    role: str
    extrinsic: np.ndarray | None              # 4x4 camera->base (Part 2), None until calibrated
    intrinsics: CameraIntrinsics | None = None  # K + distortion (Part 1), None -> SDK fallback


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

            # Part 2 — shared extrinsic (base_T_camera), loaded once.
            extrinsic = None
            extr_file = cam_cfg.get("extrinsic_file")
            if extr_file:
                try:
                    extrinsic = load_extrinsic(extr_file)
                except FileNotFoundError:
                    pass  # not calibrated yet — run run_calibration.py

            self.handles[cam_id] = CameraHandle(
                camera=camera, role=cam_cfg.get("role", cam_id),
                extrinsic=extrinsic, intrinsics=intrinsics,
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
