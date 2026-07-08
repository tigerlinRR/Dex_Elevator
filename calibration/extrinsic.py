"""Calibration — PART 2: camera extrinsics (eye-to-hand ``base_T_camera``).

Run **once per camera**, *after* Part 1 (intrinsics). The camera is fixed in the
workspace and a ChArUco board is attached rigidly to the gripper; we jog the arm
to varied poses and at each one pair ``base_T_gripper`` (robot) with
``cam_T_target`` (board pose from Part 1's intrinsics). OpenCV's
``calibrateHandEye`` solves the eye-IN-hand problem, so for eye-to-hand we feed
the **inverted** gripper poses; the result is ``base_T_camera``. It is saved to
``<cam>.npy`` and shared by the runtime (``CameraManager`` loads it once).

Self-contained: depends only on ``cv2``, ``numpy``, :mod:`calibration.board`,
:mod:`calibration.io` and :mod:`core`.

Reference: ``cv2.calibrateHandEye``, AX = XB formulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from calibration.board import BoardDetector, CharucoBoard
from calibration.io import save_calibration, save_extrinsic
from core.transforms import invert, make_transform, rel_motion, rot_angle_deg
from core.types import CameraIntrinsics

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


# ==========================================================================
# Solver
# ==========================================================================
def _methods() -> dict[str, int]:
    return {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }


def _check(base_T_gripper: list[np.ndarray], cam_T_target: list[np.ndarray]) -> None:
    if cv2 is None:
        raise ImportError("opencv-python is required for calibration.")
    if len(base_T_gripper) != len(cam_T_target):
        raise ValueError("Pose lists must have equal length.")
    if len(base_T_gripper) < 3:
        raise ValueError("Need at least 3 (ideally 10+) varied poses.")


def _solve(base_T_gripper, cam_T_target, method: int) -> np.ndarray:
    """Run one OpenCV hand-eye method; return ``base_T_camera`` (4x4)."""
    # Eye-to-hand trick: invert the gripper poses so calibrateHandEye yields the
    # camera-to-base transform directly.
    gripper_T_base = [invert(np.asarray(T, dtype=np.float64)) for T in base_T_gripper]
    R_g2b = [T[:3, :3] for T in gripper_T_base]
    t_g2b = [T[:3, 3] for T in gripper_T_base]
    R_t2c = [np.asarray(T, dtype=np.float64)[:3, :3] for T in cam_T_target]
    t_t2c = [np.asarray(T, dtype=np.float64)[:3, 3] for T in cam_T_target]
    R_cam, t_cam = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)
    return make_transform(R_cam, t_cam.reshape(3))


def solve_eye_to_hand(
    base_T_gripper: list[np.ndarray],
    cam_T_target: list[np.ndarray],
    method: int | None = None,
) -> np.ndarray:
    """Return ``base_T_camera`` (4x4) from a single method (PARK by default).

    Args:
        base_T_gripper: list of 4x4 robot TCP poses (base frame), one per sample.
        cam_T_target:   list of 4x4 board poses in the camera frame, one per sample.
    """
    _check(base_T_gripper, cam_T_target)
    # PARK is the most numerically robust OpenCV hand-eye method across arbitrary
    # rotations (incl. down-looking cameras); TSAI/DANIILIDIS can drift badly.
    method = cv2.CALIB_HAND_EYE_PARK if method is None else method
    return _solve(base_T_gripper, cam_T_target, method)


def solve_eye_to_hand_all_methods(
    base_T_gripper: list[np.ndarray],
    cam_T_target: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Solve with every OpenCV method; return ``{name: base_T_camera}``.

    Methods that fail on a degenerate set are omitted. Cross-method agreement is a
    cheap sanity check on pose conditioning.
    """
    _check(base_T_gripper, cam_T_target)
    results: dict[str, np.ndarray] = {}
    for name, method in _methods().items():
        try:
            results[name] = _solve(base_T_gripper, cam_T_target, method)
        except cv2.error:  # pragma: no cover - depends on pose conditioning
            continue
    if not results:
        raise RuntimeError("All hand-eye methods failed; check the pose set.")
    return results


def select_best(
    candidates: dict[str, np.ndarray],
    base_T_gripper: list[np.ndarray],
    cam_T_target: list[np.ndarray],
    prefer: str = "PARK",
) -> tuple[str, np.ndarray]:
    """Pick the method with the lowest translation consistency residual.

    ``prefer`` only breaks near-ties (within 0.1 mm). Returns ``(name, X)``.
    """
    scored = {
        name: consistency_residual(X, base_T_gripper, cam_T_target)["trans_m"]["mean"]
        for name, X in candidates.items()
    }
    best = min(scored, key=scored.get)
    if prefer in scored and scored[prefer] - scored[best] < 1e-4:
        best = prefer
    return best, candidates[best]


def consistency_residual(
    base_T_camera: np.ndarray,
    base_T_gripper: list[np.ndarray],
    cam_T_target: list[np.ndarray],
) -> dict:
    """Rotation + translation consistency of the recovered extrinsic.

    The board is fixed to the (moving) gripper, so ``gripper_T_target`` is
    constant. For a correct ``X = base_T_camera``,
        gripper_T_target = inv(base_T_gripper) @ base_T_camera @ cam_T_target
    is identical at every pose. We report the spread about the mean (translation
    in meters, rotation in degrees). A good calibration is a few mm.
    """
    grip_T_target = [
        invert(np.asarray(bg, dtype=np.float64)) @ base_T_camera @ np.asarray(ct, dtype=np.float64)
        for bg, ct in zip(base_T_gripper, cam_T_target)
    ]
    positions = np.array([T[:3, 3] for T in grip_T_target])
    trans_err = np.linalg.norm(positions - positions.mean(axis=0), axis=1)
    R_ref = grip_T_target[0][:3, :3]
    rot_err = np.array([rot_angle_deg(R_ref.T @ T[:3, :3]) for T in grip_T_target])

    def _stats(a: np.ndarray) -> dict:
        return {"mean": float(np.mean(a)), "median": float(np.median(a)), "max": float(np.max(a))}

    return {"trans_m": _stats(trans_err), "rot_deg": _stats(rot_err)}


def reprojection_residual(base_T_camera, base_T_gripper, cam_T_target) -> float:
    """Mean translational consistency error (meters). See :func:`consistency_residual`."""
    return consistency_residual(base_T_camera, base_T_gripper, cam_T_target)["trans_m"]["mean"]


# ==========================================================================
# Capture session
# ==========================================================================
@dataclass
class ExtrinsicCalibrationSession:
    """Accumulates synchronized (robot pose, board pose) samples for one camera."""

    board: CharucoBoard = field(default_factory=CharucoBoard)
    accurate: bool = True
    # Distortion override; when None we use the camera's own intrinsics.D (Part 1).
    dist_coeffs: Optional[np.ndarray] = None
    # Pose-diversity gating — reject samples too close to the previous one so the
    # AX = XB problem stays well-conditioned (varied rotations + translations).
    min_angle_deg: float = 8.0
    min_trans_m: float = 0.03
    # When set, each accepted sample is written here at capture time so a session
    # can be re-solved offline (or audited) without re-jogging the arm.
    sample_dir: Optional[Path] = None

    base_T_gripper: list[np.ndarray] = field(default_factory=list)
    cam_T_target: list[np.ndarray] = field(default_factory=list)

    # Populated by solve() — exposed so validation can reuse them.
    last_candidates: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    last_method: Optional[str] = field(default=None, init=False)

    _detector: BoardDetector = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._detector = BoardDetector(board=self.board, accurate=self.accurate)
        if self.sample_dir is not None:
            self.sample_dir = Path(self.sample_dir)

    @classmethod
    def from_samples(cls, sample_dir: str | Path, **kwargs) -> "ExtrinsicCalibrationSession":
        """Rebuild a session from samples saved by :meth:`add_sample` (offline re-solve)."""
        sample_dir = Path(sample_dir)
        files = sorted(sample_dir.glob("sample*.npz"))
        if not files:
            raise FileNotFoundError(f"No sample*.npz under {sample_dir}")
        session = cls(sample_dir=sample_dir, **kwargs)
        for f in files:
            data = np.load(f)
            session.base_T_gripper.append(np.asarray(data["base_T_gripper"], dtype=np.float64))
            session.cam_T_target.append(np.asarray(data["cam_T_target"], dtype=np.float64))
        print(f"[extrinsics] loaded {session.num_samples} samples from {sample_dir}")
        return session

    # -- detection ----------------------------------------------------------
    def detect_board(self, rgb: np.ndarray, intr: CameraIntrinsics) -> Optional[np.ndarray]:
        """Return ``cam_T_target`` (4x4) or ``None``. Uses Part 1 intrinsics (K + dist)."""
        dist = intr.D if self.dist_coeffs is None else np.asarray(self.dist_coeffs, dtype=np.float64)
        return self._detector.detect(rgb, intr.K, dist)

    # -- sample bookkeeping -------------------------------------------------
    def is_diverse(self, base_T_gripper: np.ndarray) -> bool:
        """True if this robot pose differs enough from the last accepted one."""
        if not self.base_T_gripper:
            return True
        delta = rel_motion(self.base_T_gripper[-1], base_T_gripper)
        return rot_angle_deg(delta) >= self.min_angle_deg or \
            float(np.linalg.norm(delta[:3, 3])) >= self.min_trans_m

    def add_sample(
        self,
        base_T_gripper: np.ndarray,
        cam_T_target: np.ndarray,
        rgb: np.ndarray | None = None,
    ) -> None:
        bg = np.asarray(base_T_gripper, dtype=np.float64)
        ct = np.asarray(cam_T_target, dtype=np.float64)
        self.base_T_gripper.append(bg)
        self.cam_T_target.append(ct)
        if self.sample_dir is not None:
            self._persist_sample(self.num_samples, bg, ct, rgb)

    def _persist_sample(self, n: int, bg, ct, rgb) -> None:
        """Write one sample (transforms + optional RGB) for offline re-solve/audit."""
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        np.savez(self.sample_dir / f"sample{n:03d}.npz", base_T_gripper=bg, cam_T_target=ct)
        if rgb is not None and cv2 is not None:
            cv2.imwrite(str(self.sample_dir / f"sample{n:03d}.jpg"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def pop_sample(self) -> bool:
        """Remove the most recent sample (and its persisted files). For 'delete last'."""
        if not self.base_T_gripper:
            return False
        removed_idx = self.num_samples  # 1-based filename of the sample being removed
        self.base_T_gripper.pop()
        self.cam_T_target.pop()
        if self.sample_dir is not None:
            for ext in (".npz", ".jpg"):
                f = self.sample_dir / f"sample{removed_idx:03d}{ext}"
                try:
                    f.unlink()
                except OSError:
                    pass
        return True

    @property
    def num_samples(self) -> int:
        return len(self.base_T_gripper)

    # -- solve --------------------------------------------------------------
    def solve(self, save_as: str | None = None, prefer: str = "PARK") -> np.ndarray:
        """Solve with all methods, pick the most consistent, optionally save it."""
        candidates = solve_eye_to_hand_all_methods(self.base_T_gripper, self.cam_T_target)
        method, base_T_camera = select_best(
            candidates, self.base_T_gripper, self.cam_T_target, prefer=prefer
        )
        self.last_candidates = candidates
        self.last_method = method
        res = consistency_residual(base_T_camera, self.base_T_gripper, self.cam_T_target)
        print(
            f"[extrinsics] {self.num_samples} samples | method={method} | "
            f"consistency: trans mean={res['trans_m']['mean'] * 1000:.2f} mm "
            f"(max {res['trans_m']['max'] * 1000:.2f}), "
            f"rot mean={res['rot_deg']['mean']:.3f}° (max {res['rot_deg']['max']:.3f})"
        )
        for name, X in candidates.items():
            tm = consistency_residual(X, self.base_T_gripper, self.cam_T_target)["trans_m"]["mean"]
            print(f"    {name:<11} trans residual = {tm * 1000:.2f} mm")

        if save_as:
            path = save_extrinsic(save_as, base_T_camera)
            save_calibration(save_as, base_T_camera, method, candidates, res, self.num_samples)
            print(f"[extrinsics] saved base_T_camera -> {path}")
        return base_T_camera


# Backward-compatible alias.
CalibrationSession = ExtrinsicCalibrationSession
