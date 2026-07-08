"""Calibration — PART 1: camera intrinsics (K + distortion).

Run **once per camera**; the result is saved to ``<cam>_intrinsics.npz`` and then
shared by everything (the runtime camera driver loads it, and Part 2 / extrinsics
consumes it). Two ways to produce it:

* :func:`intrinsics_from_camera` — read the factory intrinsics the Orbbec SDK
  reports (fast; good enough for the 335L in most cases).
* :func:`calibrate_intrinsics_from_images` — a proper ChArUco calibration via
  ``cv2.calibrateCamera`` over captured board images (use when you need the best
  distortion estimate or don't trust the factory values).

Self-contained: depends only on ``cv2``, ``numpy``, :mod:`calibration.board`,
:mod:`calibration.io` and :mod:`core`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from calibration.board import BoardDetector, CharucoBoard
from calibration.io import load_camera_intrinsics, save_intrinsics
from core.types import CameraIntrinsics

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass
class IntrinsicCalibration:
    """The intrinsics result for one camera."""

    K: np.ndarray              # 3x3 camera matrix
    dist: np.ndarray           # (5,) [k1, k2, p1, p2, k3]
    width: int
    height: int
    rms: float = float("nan")  # mean reprojection error (px), if image-calibrated
    num_images: int = 0

    def to_intrinsics(self) -> CameraIntrinsics:
        return CameraIntrinsics.from_matrix(self.K, self.width, self.height, dist=self.dist)

    def save(self, path: str | Path) -> Path:
        return save_intrinsics(path, self.K, self.dist, self.width, self.height, self.rms)

    @classmethod
    def load(cls, path: str | Path) -> "IntrinsicCalibration":
        intr = load_camera_intrinsics(path)
        return cls(K=intr.K, dist=intr.D, width=intr.width, height=intr.height)


def intrinsics_from_camera(camera) -> IntrinsicCalibration:
    """Read factory intrinsics from a *started* :class:`~core.camera.base.Camera`.

    The Orbbec driver pulls K + distortion from the SDK on ``start()``.
    """
    intr = camera.intrinsics()
    return IntrinsicCalibration(K=intr.K, dist=intr.D, width=intr.width, height=intr.height)


def calibrate_intrinsics_from_images(
    images: list[np.ndarray],
    board: Optional[CharucoBoard] = None,
    accurate: bool = True,
    image_is_rgb: bool = True,
) -> IntrinsicCalibration:
    """Calibrate intrinsics from ChArUco board images via ``cv2.calibrateCamera``.

    Args:
        images: list of full-resolution color images of the board at varied poses
                (aim for 15+, covering the frame corners and a range of tilts).
        board:  the ChArUco board (defaults to :class:`CharucoBoard`).
        accurate: subpixel corner refinement.
        image_is_rgb: images are RGB (this repo's CameraFrame convention) vs BGR.

    Returns an :class:`IntrinsicCalibration` (with the RMS reprojection error).
    """
    if cv2 is None:
        raise ImportError("opencv-contrib-python is required for calibration.")
    if not images:
        raise ValueError("No images provided.")

    board = board or CharucoBoard()
    detector = BoardDetector(board=board, accurate=accurate, image_is_rgb=image_is_rgb)

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    for img in images:
        found = detector.detect_corners(img)
        if found is None:
            continue
        corners, charuco_ids, opencv_board = found
        objp = opencv_board.getChessboardCorners()[charuco_ids.flatten()].astype(np.float32)
        obj_points.append(objp)
        img_points.append(np.asarray(corners, dtype=np.float32))

    if len(obj_points) < 5:
        raise RuntimeError(
            f"Only {len(obj_points)} usable board views (need >= 5). "
            "Capture more images with the board clearly visible."
        )

    h, w = images[0].shape[:2]
    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None
    )
    return IntrinsicCalibration(
        K=np.asarray(K, dtype=np.float64),
        dist=np.asarray(dist, dtype=np.float64).reshape(-1),
        width=w, height=h, rms=float(rms), num_images=len(obj_points),
    )
