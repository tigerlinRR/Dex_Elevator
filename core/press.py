"""Button-press geometry — pure functions, no hardware, no learned model.

Button-press geometry (the pressing analogue of a grasp). A button is a
**point target on a known vertical plane** (the panel). Given the button's pixel
and the eye-to-hand extrinsic we:

  1. back-project the pixel to a ray in the base frame, intersect it with the
     panel plane -> the 3D contact point (depth-independent ray-plane trick,
     robust to the panel's small/low-contrast buttons where stereo depth is
     unreliable);
  2. orient the tool so its approach axis (tool +Z) points **into** the panel
     along the inward normal;
  3. hand the pipeline three waypoints: STANDOFF (backed off the panel) ->
     PRESS (advanced past contact by ``push_depth``) -> RETRACT (back to standoff).

The panel plane is given in the base frame as ``(point_on_plane, outward_normal)``
— measured once for a docked pose, or fit from the panel-region depth later
(:func:`fit_panel_plane_from_depth`, TODO). Nothing here selects *which* button is
the target floor; that is the detector/OCR layer's job (see
:mod:`yolo.button_detector`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.transforms import deproject_pixel, make_transform, transform_point
from core.types import CameraIntrinsics, Detection


@dataclass
class PressTarget:
    """Where and how to press one button, in the robot base frame."""

    contact_pose: np.ndarray              # (4,4) tool pose at the button face; +Z = inward normal
    point_base: np.ndarray                # (3,) contact point on the panel, meters
    normal_base: np.ndarray               # (3,) INWARD unit normal (approach direction)
    pixel: np.ndarray                     # (2,) [u, v] button center pixel (provenance)
    floor: str = ""                       # which floor/label this button is (from detector/OCR)
    confidence: float = 0.0
    detection: Optional[Detection] = field(default=None, repr=False)


def _orthonormal_frame(approach: np.ndarray, up_hint=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Right-handed rotation whose +Z is ``approach`` (unit inward normal).

    ``up_hint`` fixes the roll about the approach axis; the finger orientation of
    the LinkerHand about +Z must still be confirmed on the real cell.
    """
    z = np.asarray(approach, dtype=np.float64)
    z = z / (np.linalg.norm(z) + 1e-12)
    up = np.asarray(up_hint, dtype=np.float64)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:            # approach ~parallel to up_hint
        x = np.cross(np.array([1.0, 0.0, 0.0]), z)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def ray_plane_intersection(
    pixel,
    intrinsics: CameraIntrinsics,
    extrinsic: np.ndarray,                # (4,4) base_T_camera
    plane_point,                          # (3,) any point on the panel plane (base)
    plane_normal,                         # (3,) panel normal (base); need not be unit
) -> Optional[np.ndarray]:
    """Intersect the pixel's back-projected ray with the panel plane. (3,) base or None."""
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    u, v = float(pixel[0]), float(pixel[1])
    d_cam = deproject_pixel(intrinsics, u, v, 1.0)      # ray direction in camera frame
    d_base = extrinsic[:3, :3] @ d_cam
    o_base = extrinsic[:3, 3]
    n = np.asarray(plane_normal, dtype=np.float64)
    p0 = np.asarray(plane_point, dtype=np.float64)
    denom = float(d_base @ n)
    if abs(denom) < 1e-9:                                # ray parallel to the panel
        return None
    t = float((p0 - o_base) @ n) / denom
    if t <= 0:                                           # panel behind the camera
        return None
    return o_base + t * d_base


def press_target_from_pixel(
    pixel,
    intrinsics: CameraIntrinsics,
    extrinsic: np.ndarray,                # (4,4) base_T_camera
    plane_point,                          # (3,) point on the panel plane (base)
    plane_normal,                         # (3,) OUTWARD panel normal, toward the robot (base)
    up_hint=(0.0, 0.0, 1.0),
    detection: Optional[Detection] = None,
    floor: str = "",
    confidence: float = 0.0,
) -> Optional[PressTarget]:
    """Full press target for one button pixel, or ``None`` if the ray misses the panel."""
    contact = ray_plane_intersection(pixel, intrinsics, extrinsic, plane_point, plane_normal)
    if contact is None:
        return None
    outward = np.asarray(plane_normal, dtype=np.float64)
    outward = outward / (np.linalg.norm(outward) + 1e-12)
    inward = -outward                                    # approach axis (into the panel)
    R = _orthonormal_frame(inward, up_hint=up_hint)
    pose = make_transform(R, contact)
    return PressTarget(
        contact_pose=pose, point_base=contact, normal_base=inward,
        pixel=np.asarray([float(pixel[0]), float(pixel[1])], dtype=np.float64),
        floor=floor, confidence=confidence, detection=detection,
    )


def press_waypoints(
    target: PressTarget,
    standoff: float = 0.06,               # meters backed off the panel before/after pressing
    push_depth: float = 0.010,            # meters advanced past the panel face to actuate the button
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STANDOFF -> PRESS -> RETRACT poses (4x4 each), all with the target's orientation.

    Approach axis is ``target.normal_base`` (inward). STANDOFF/RETRACT sit
    ``standoff`` m outside the panel; PRESS sits ``push_depth`` m inside the face
    so the fingertip actuates the button. ``push_depth`` must be small and,
    ideally, force-limited (RealMan ``rm_force_position_move_pose``) on hardware.
    """
    R = target.contact_pose[:3, :3]
    contact = target.point_base
    inward = target.normal_base
    standoff_pt = contact - inward * standoff
    press_pt = contact + inward * push_depth
    return (
        make_transform(R, standoff_pt),
        make_transform(R, press_pt),
        make_transform(R, standoff_pt),
    )


def fit_panel_plane_from_depth(*args, **kwargs):
    """TODO: estimate ``(plane_point, outward_normal)`` from the panel-region depth.

    For a docked pose the panel plane can be measured once and kept in config
    (simplest, most robust). A live plane fit (RANSAC over the panel ROI's depth
    points, mapped to the base frame) would let the base dock less precisely.
    Not needed for first bring-up — kept as a stub.
    """
    raise NotImplementedError("panel-plane fitting is a follow-up; measure the plane for now.")
