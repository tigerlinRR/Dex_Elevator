"""Shared data types passed between pipeline stages.

Deliberately framework-agnostic (plain dataclasses + numpy) so the same objects
flow through camera -> YOLO -> press geometry -> robot, and can later be wrapped
in ROS 2 messages without changing the core logic.

Convention: ``CameraFrame.rgb`` is **RGB** (not BGR); depth is in **meters**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics for a single camera stream.

    ``dist`` holds the OpenCV radial/tangential distortion coefficients
    ``[k1, k2, p1, p2, k3]`` when known (e.g. reported by the Orbbec SDK). It is
    optional — leave it ``None`` for an already-rectified stream — but supplying
    it makes ChArUco ``solvePnP`` during calibration more accurate.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: Optional[np.ndarray] = None     # (5,) [k1, k2, p1, p2, k3] or None

    @property
    def K(self) -> np.ndarray:
        """3x3 camera matrix."""
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def D(self) -> np.ndarray:
        """Distortion coefficients as a (5,) float array (zeros if unknown)."""
        if self.dist is None:
            return np.zeros(5, dtype=np.float64)
        return np.asarray(self.dist, dtype=np.float64).reshape(-1)

    @classmethod
    def from_matrix(
        cls,
        K: np.ndarray,
        width: int,
        height: int,
        dist: Optional[np.ndarray] = None,
    ) -> "CameraIntrinsics":
        return cls(
            fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
            width=width, height=height,
            dist=None if dist is None else np.asarray(dist, dtype=np.float64).reshape(-1),
        )

    def scaled_to(self, width: int, height: int) -> "CameraIntrinsics":
        """Return intrinsics rescaled to a new resolution (assumes a pure resize).

        ``fx, cx`` scale with the width ratio and ``fy, cy`` with the height ratio;
        distortion coefficients are resolution-independent and pass through. Only
        valid when the new resolution is a uniform resize of the same field of
        view — NOT for a cropped/binned sensor profile.
        """
        sx = width / self.width
        sy = height / self.height
        return CameraIntrinsics(
            fx=self.fx * sx, fy=self.fy * sy,
            cx=self.cx * sx, cy=self.cy * sy,
            width=width, height=height,
            dist=None if self.dist is None else np.asarray(self.dist, dtype=np.float64).reshape(-1),
        )


@dataclass
class CameraFrame:
    """A single synchronized capture from one camera.

    Depth is in meters. ``point_cloud`` is organized HxWx3 (camera frame, meters)
    or None if not computed. ``camera_id`` ties the frame back to its extrinsics.
    """

    camera_id: str
    rgb: np.ndarray                       # (H, W, 3) uint8
    depth: np.ndarray                     # (H, W) float32, meters
    intrinsics: CameraIntrinsics
    point_cloud: Optional[np.ndarray] = None   # (H, W, 3) or (N, 3) float32, camera frame
    timestamp: float = 0.0


@dataclass
class Detection:
    """One object detected by YOLO — here, an elevator button.

    ``bbox_xyxy`` is in image pixels; ``mask`` (optional, from a -seg model) gives
    a tighter center; ``track_id`` is a persistent id across frames when tracking.
    """

    label: str
    confidence: float
    bbox_xyxy: np.ndarray                 # (4,) float [x1, y1, x2, y2] in pixels
    class_id: int = 0
    mask: Optional[np.ndarray] = None     # (H, W) bool, optional segmentation
    track_id: Optional[int] = None        # persistent id across frames (tracking)

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox_xyxy
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)
