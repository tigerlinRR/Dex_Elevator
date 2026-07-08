"""Rigid-body (SE3) transform helpers and 2D->3D deprojection.

Kept dependency-light (numpy only) so it imports cleanly on the robot
controller without pulling in the full vision/ML stack.
"""

from __future__ import annotations

import numpy as np


def identity() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from rotation (3x3) and translation (3,)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert(T: np.ndarray) -> np.ndarray:
    """Inverse of a homogeneous transform (cheaper and more stable than np.linalg.inv)."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply T to an (N, 3) array of points, returning (N, 3)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return (pts @ T[:3, :3].T) + T[:3, 3]


def transform_point(T: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Apply T to a single (3,) point, returning (3,).

    Convenience wrapper over :func:`transform_points` for the common case of a
    single 3D position (e.g. a deprojected pixel: camera frame -> base frame).
    """
    p = np.asarray(point, dtype=np.float64).reshape(3)
    return T[:3, :3] @ p + T[:3, 3]


def deproject_pixel(intr, u: float, v: float, depth: float) -> np.ndarray:
    """Back-project a pixel (u, v) with metric depth to a 3D point in the camera frame.

    ``intr`` is a CameraIntrinsics. Returns (3,) in meters.
    """
    x = (u - intr.cx) * depth / intr.fx
    y = (v - intr.cy) * depth / intr.fy
    return np.array([x, y, depth], dtype=np.float64)


def assert_valid_transform(T: np.ndarray, atol: float = 1e-3) -> None:
    """Sanity check that T is a proper SE3 matrix (orthonormal rotation, det +1)."""
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=atol):
        raise ValueError("Rotation block is not orthonormal")
    if not np.isclose(np.linalg.det(R), 1.0, atol=atol):
        raise ValueError("Rotation determinant is not +1")


# ---------------------------------------------------------------------------
# Rotation / relative-motion helpers
#
# These are used by the eye-to-hand calibration (board pose diversity gating and
# residual diagnostics). They live here, in numpy-only core, so the calibration
# package depends solely on this repo's core — never on any external utility module.
# ---------------------------------------------------------------------------


def rot_angle_deg(R: np.ndarray) -> float:
    """Geodesic rotation angle (degrees) of a 3x3 rotation matrix.

    Accepts a 4x4 transform too (uses its rotation block).
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape == (4, 4):
        R = R[:3, :3]
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def rel_motion(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """Relative motion ``T_a -> T_b`` expressed in the common (base) frame.

    Returns ``T_b @ inv(T_a)`` as a 4x4 transform. Useful both for measuring how
    different two robot poses are (pose-diversity gating) and for the AX = XB
    consistency check.
    """
    return np.asarray(T_b, dtype=np.float64) @ invert(np.asarray(T_a, dtype=np.float64))


def rpy_to_matrix(roll: float, pitch: float, yaw: float, degrees: bool = True) -> np.ndarray:
    """Build a 3x3 rotation from roll/pitch/yaw using the ZYX convention.

    ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` — the convention most arm controllers
    (e.g. xArm / many industrial arms) report TCP orientation in. Mirrors
    :func:`matrix_to_rpy`.
    """
    if degrees:
        roll, pitch, yaw = np.radians([roll, pitch, yaw])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def matrix_to_rpy(R: np.ndarray, degrees: bool = True) -> np.ndarray:
    """Inverse of :func:`rpy_to_matrix` — extract (roll, pitch, yaw), ZYX order.

    Accepts a 3x3 rotation or a 4x4 transform. Returns a (3,) array.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape == (4, 4):
        R = R[:3, :3]
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy >= 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    rpy = np.array([roll, pitch, yaw], dtype=np.float64)
    return np.degrees(rpy) if degrees else rpy


def pose_from_rpy(x: float, y: float, z: float,
                  roll: float, pitch: float, yaw: float,
                  degrees: bool = True) -> np.ndarray:
    """Convenience: build a 4x4 pose from a translation + roll/pitch/yaw.

    Handy for turning an arm controller's ``[x, y, z, r, p, y]`` TCP readout into
    the ``base_T_gripper`` transform the calibration consumes.
    """
    return make_transform(rpy_to_matrix(roll, pitch, yaw, degrees=degrees), np.array([x, y, z]))


# ---------------------------------------------------------------------------
# Quaternion helpers
#
# Some arm SDKs (e.g. RealMan) report/consume orientation as a quaternion, or as
# Euler angles whose exact axis order is SDK-specific. Converting via a
# quaternion is convention-proof: build R from a unit quaternion (unambiguous)
# rather than guessing the Euler order. Order is [w, x, y, z] (scalar first),
# matching RealMan's ``rm_quat_t`` field layout.
# ---------------------------------------------------------------------------


def quat_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    """3x3 rotation from a quaternion ``[w, x, y, z]`` (scalar first).

    The quaternion is normalized first, so a slightly non-unit input (float
    round-trip through an SDK) still yields a proper rotation.
    """
    n = float(np.sqrt(w * w + x * x + y * y + z * z))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Quaternion ``[w, x, y, z]`` (scalar first) from a 3x3 rotation or 4x4 pose.

    Uses the branch-by-largest-diagonal method for numerical stability.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape == (4, 4):
        R = R[:3, :3]
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)
