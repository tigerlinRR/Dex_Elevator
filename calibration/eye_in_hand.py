"""Calibration — PART 2b: extrinsics for an ARM-MOUNTED camera (eye-IN-hand).

The counterpart to :mod:`calibration.extrinsic`, for the case where the camera
is bolted to the arm instead of the torso. Same two-part pipeline (Part 1
intrinsics first, unchanged), same ChArUco board, same sample bookkeeping — a
DIFFERENT geometry, so it is a separate module and the eye-to-hand path is left
exactly as it is.

What changes:

===================  ==================================  ==================================
                     eye-to-hand (chest, existing)       eye-IN-hand (arm, here)
===================  ==================================  ==================================
board                bolted to the FLANGE, moves         FIXED in the scene, never moves
camera               fixed on the torso                  rides the arm
solved quantity      ``base_T_camera``  (a constant)     ``gripper_T_camera`` (a constant)
runtime use          use it directly                     ``base_T_gripper(t) @ gripper_T_camera``
OpenCV call          gripper poses INVERTED first        gripper poses passed AS-IS
what must be rigid   ``gripper_T_board``                 ``base_T_board``
===================  ==================================  ==================================

``cv2.calibrateHandEye`` solves the eye-in-hand problem natively: fed
``base_T_gripper`` and ``cam_T_target`` it returns ``gripper_T_camera``. The
eye-to-hand module only inverts the gripper poses to bend that same routine to
the other geometry — so here the inversion simply goes away.

**Observability.** ``AX = XB`` determines ``X``'s translation only from the
ROTATION between poses: two poses related by a pure translation carry no
information about where the camera sits, and a pose set whose rotation axes are
all parallel leaves the translation along that axis unobservable. That degeneracy
is easy to walk into when hand-guiding an arm ("move it around a lot" is mostly
translation), and it does not announce itself — the solve returns a clean-looking
matrix. :func:`rotation_diversity` measures it so the capture UI can say "rotate
about a different axis" while the operator is still standing at the robot.

Self-contained: ``cv2``, ``numpy``, :mod:`calibration.board`,
:mod:`calibration.io`, :mod:`calibration.frames`, :mod:`calibration.validate`
and :mod:`core`.

Reference: ``cv2.calibrateHandEye``, AX = XB formulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from calibration.extrinsic import ExtrinsicCalibrationSession
from calibration.frames import FRAME_GRIPPER, resolve, save_frame_tag
from calibration.io import save_extrinsic
from calibration.validate import ExtrinsicThresholds, ValidationReport
from core.transforms import make_transform, rel_motion, rot_angle_deg

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
    """Run one OpenCV hand-eye method; return ``gripper_T_camera`` (4x4)."""
    # Eye-IN-hand is what calibrateHandEye solves directly: pass the gripper poses
    # as they come off the controller (NO inversion — that is the eye-to-hand trick).
    R_g2b = [np.asarray(T, dtype=np.float64)[:3, :3] for T in base_T_gripper]
    t_g2b = [np.asarray(T, dtype=np.float64)[:3, 3] for T in base_T_gripper]
    R_t2c = [np.asarray(T, dtype=np.float64)[:3, :3] for T in cam_T_target]
    t_t2c = [np.asarray(T, dtype=np.float64)[:3, 3] for T in cam_T_target]
    R_cam, t_cam = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)
    X = make_transform(R_cam, t_cam.reshape(3))
    # A degenerate pose set (all rotations about one axis, or near-zero rotation)
    # makes the OpenCV solve return NaN rather than raising. Measured on a
    # synthetic yaw-only set: PARK returns an all-NaN matrix. Left to propagate it
    # is WORSE than an exception, because every downstream threshold test is a
    # comparison and `nan > limit` is False — the residual checks then fall
    # through to their "pass" branch and the calibration reports clean. Turn it
    # into a failure here, where it is still attributable to the solve.
    if not np.isfinite(X).all():
        raise RuntimeError(
            "hand-eye solve returned a non-finite transform — the pose set is degenerate "
            "(check rotation_diversity: rotations about a single axis, or too little rotation)"
        )
    return X


def solve_eye_in_hand(
    base_T_gripper: list[np.ndarray],
    cam_T_target: list[np.ndarray],
    method: int | None = None,
) -> np.ndarray:
    """Return ``gripper_T_camera`` (4x4) from a single method (PARK by default).

    Args:
        base_T_gripper: list of 4x4 robot TCP poses (base frame), one per sample.
        cam_T_target:   list of 4x4 board poses in the camera frame, one per sample.
    """
    _check(base_T_gripper, cam_T_target)
    # PARK is the method that proved reliable on this cell's eye-to-hand solve
    # (consistency 0.67 mm); TSAI/DANIILIDIS drifted badly for angled cameras.
    method = cv2.CALIB_HAND_EYE_PARK if method is None else method
    return _solve(base_T_gripper, cam_T_target, method)


def solve_eye_in_hand_all_methods(
    base_T_gripper: list[np.ndarray],
    cam_T_target: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Solve with every OpenCV method; return ``{name: gripper_T_camera}``.

    Methods that fail on a degenerate set are omitted. Cross-method agreement is a
    cheap sanity check on pose conditioning.
    """
    _check(base_T_gripper, cam_T_target)
    results: dict[str, np.ndarray] = {}
    for name, method in _methods().items():
        try:
            results[name] = _solve(base_T_gripper, cam_T_target, method)
        except (cv2.error, RuntimeError):  # pragma: no cover - depends on pose conditioning
            continue
    if not results:
        raise RuntimeError("All hand-eye methods failed; check the pose set.")
    return results


def consistency_residual(
    gripper_T_camera: np.ndarray,
    base_T_gripper: list[np.ndarray],
    cam_T_target: list[np.ndarray],
) -> dict:
    """Rotation + translation consistency of the recovered extrinsic.

    The board is fixed in the WORLD here (the mirror image of eye-to-hand, where
    it is fixed to the gripper), so ``base_T_target`` is the constant. For a
    correct ``X = gripper_T_camera``,
        base_T_target = base_T_gripper @ gripper_T_camera @ cam_T_target
    is identical at every pose. We report the spread about the mean (translation
    in meters, rotation in degrees). A good calibration is a few mm.

    Note this residual is NOT directly comparable to the eye-to-hand one: it also
    absorbs the arm's forward-kinematics and joint repeatability error, which the
    fixed-camera solve never sees. Expect it to be the larger of the two.
    """
    base_T_target = [
        np.asarray(bg, dtype=np.float64) @ gripper_T_camera @ np.asarray(ct, dtype=np.float64)
        for bg, ct in zip(base_T_gripper, cam_T_target)
    ]
    positions = np.array([T[:3, 3] for T in base_T_target])
    trans_err = np.linalg.norm(positions - positions.mean(axis=0), axis=1)
    R_ref = base_T_target[0][:3, :3]
    rot_err = np.array([rot_angle_deg(R_ref.T @ T[:3, :3]) for T in base_T_target])

    def _stats(a: np.ndarray) -> dict:
        return {"mean": float(np.mean(a)), "median": float(np.median(a)), "max": float(np.max(a))}

    return {"trans_m": _stats(trans_err), "rot_deg": _stats(rot_err)}


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
    # Drop non-finite scores before ranking: min() would happily return a NaN
    # entry (every comparison against NaN is False), i.e. silently pick the one
    # candidate that is not a solution at all.
    scored = {k: v for k, v in scored.items() if np.isfinite(v)}
    if not scored:
        raise RuntimeError("no hand-eye candidate produced a finite residual; pose set is degenerate")
    best = min(scored, key=scored.get)
    if prefer in scored and scored[prefer] - scored[best] < 1e-4:
        best = prefer
    return best, candidates[best]


def base_T_camera(gripper_T_camera: np.ndarray, base_T_gripper: np.ndarray) -> np.ndarray:
    """Compose the live ``base_T_camera`` for an arm-mounted camera.

    The whole point of the mount change: this is a FUNCTION OF THE ARM POSE, not a
    constant. ``base_T_gripper`` must be the pose read at the instant the frame was
    captured, with the arm stationary — see the module docs of
    ``initialization/run_calibration_arm_cam.py``.
    """
    return np.asarray(base_T_gripper, dtype=np.float64) @ np.asarray(gripper_T_camera, dtype=np.float64)


# ==========================================================================
# Pose-set conditioning (the eye-in-hand failure mode)
# ==========================================================================
def rotation_diversity(base_T_gripper: list[np.ndarray], min_angle_deg: float = 5.0) -> dict:
    """How well-conditioned the pose set is for recovering X's TRANSLATION.

    ``AX = XB`` recovers the translation of ``X`` from the rotations between
    poses. Collect the rotation axis of every pairwise relative motion that turns
    by more than ``min_angle_deg``, and look at how those axes are spread:
    ``M = sum(a a^T)`` (an outer-product sum, so the arbitrary sign of each axis
    cannot matter). Its eigenvalues rank the axis directions.

    Returns ``axis_ratio`` = lambda2 / lambda1 — 0 means every rotation shares one
    axis (translation along it is unobservable), ~1 means the axes span the space.
    Also returns ``max_angle_deg``, the largest relative rotation in the set.

    The thresholds in :class:`EyeInHandThresholds` are engineering judgement, NOT
    measured on this robot — treat a warning as "go rotate about another axis",
    and trust the consistency + hold-out numbers as the actual verdict.
    """
    n = len(base_T_gripper)
    axes = []
    angles = []
    for i in range(n):
        for j in range(i + 1, n):
            delta = rel_motion(
                np.asarray(base_T_gripper[i], dtype=np.float64),
                np.asarray(base_T_gripper[j], dtype=np.float64),
            )
            ang = rot_angle_deg(delta)
            angles.append(ang)
            if ang < min_angle_deg:
                continue
            R = delta[:3, :3]
            # Rotation axis from the skew-symmetric part; robust enough away from
            # 180 deg, and near-180 pairs still give a usable axis direction.
            a = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
            norm = float(np.linalg.norm(a))
            if norm < 1e-9:                      # exactly 0 or 180 deg — skip
                continue
            axes.append(a / norm)
    if not axes:
        return {"axis_ratio": 0.0, "num_axes": 0,
                "max_angle_deg": float(max(angles)) if angles else 0.0}
    A = np.asarray(axes)
    eig = np.linalg.eigvalsh(A.T @ A)[::-1]      # descending
    ratio = float(eig[1] / eig[0]) if eig[0] > 1e-12 else 0.0
    return {"axis_ratio": ratio, "num_axes": len(axes), "max_angle_deg": float(max(angles))}


@dataclass
class EyeInHandThresholds(ExtrinsicThresholds):
    """Eye-to-hand thresholds plus the conditioning checks specific to this geometry.

    ``trans_resid_*`` are inherited (5 mm warn / 15 mm fail). They are the right
    ORDER of magnitude but the arm-mounted residual also carries the arm's own FK
    error, so re-tune them against the first real calibration rather than treating
    a warning as a defect.
    """
    axis_ratio_fail: float = 0.03    # all rotations about one axis — X's translation is unobservable
    axis_ratio_warn: float = 0.15
    min_rotation_deg: float = 30.0   # largest relative rotation in the set


# ==========================================================================
# Validation
# ==========================================================================
def validate_eye_in_hand(
    gripper_T_camera: np.ndarray,
    base_T_gripper: list[np.ndarray],
    cam_T_target: list[np.ndarray],
    all_methods: Optional[dict[str, np.ndarray]] = None,
    thresholds: Optional[EyeInHandThresholds] = None,
) -> ValidationReport:
    """Validate an eye-in-hand extrinsic. Mirrors ``validate_extrinsics`` for this geometry."""
    th = thresholds or EyeInHandThresholds()
    rep = ValidationReport(part="extrinsics (eye-in-hand)")
    n = len(base_T_gripper)
    if n < 3:
        rep.check("fail", f"only {n} samples (need >= 3)")
        return rep
    # Independent of the solve, because X can arrive from a saved file. Every
    # check below is a threshold comparison, and `nan > limit` is False, so a
    # non-finite X would sail through all of them and report PASS.
    if not np.isfinite(np.asarray(gripper_T_camera, dtype=np.float64)).all():
        rep.check("fail", "extrinsic contains NaN/inf — the solve did not converge "
                          "(degenerate pose set); every residual below is meaningless")
        return rep

    res = consistency_residual(gripper_T_camera, base_T_gripper, cam_T_target)
    if not np.isfinite([res["trans_m"]["mean"], res["rot_deg"]["mean"]]).all():
        rep.check("fail", "consistency residual is NaN/inf — a sample pose or board pose "
                          "is not a valid transform")
        return rep
    tmean, tmax = res["trans_m"]["mean"], res["trans_m"]["max"]
    rmean, rmax = res["rot_deg"]["mean"], res["rot_deg"]["max"]
    rep.metrics.update({
        "num_samples": n,
        "trans_resid_mean_mm": round(tmean * 1000, 3), "trans_resid_max_mm": round(tmax * 1000, 3),
        "rot_resid_mean_deg": round(rmean, 4), "rot_resid_max_deg": round(rmax, 4),
    })
    if tmean > th.trans_resid_fail_m:
        rep.check("fail", f"base_T_board consistency {tmean*1000:.2f} mm > {th.trans_resid_fail_m*1000:.0f} mm")
    elif tmean > th.trans_resid_warn_m:
        rep.check("warn", f"base_T_board consistency {tmean*1000:.2f} mm > {th.trans_resid_warn_m*1000:.0f} mm")
    else:
        rep.check("pass", f"base_T_board consistency {tmean*1000:.2f} mm (rot {rmean:.3f} deg)")
    if rmean > th.rot_resid_fail_deg:
        rep.check("fail", f"rotation consistency {rmean:.3f} deg > {th.rot_resid_fail_deg} deg")
    elif rmean > th.rot_resid_warn_deg:
        rep.check("warn", f"rotation consistency {rmean:.3f} deg > {th.rot_resid_warn_deg} deg")

    # Pose-set conditioning — the eye-in-hand-specific degeneracy (see rotation_diversity).
    div = rotation_diversity(base_T_gripper)
    rep.metrics.update({
        "rotation_axis_ratio": round(div["axis_ratio"], 4),
        "max_relative_rotation_deg": round(div["max_angle_deg"], 2),
    })
    if div["axis_ratio"] < th.axis_ratio_fail:
        rep.check("fail", f"rotation axes are near-collinear (ratio {div['axis_ratio']:.3f} < "
                          f"{th.axis_ratio_fail}) — X's translation along that axis is "
                          f"UNOBSERVABLE; recapture with rotations about a second axis")
    elif div["axis_ratio"] < th.axis_ratio_warn:
        rep.check("warn", f"rotation axes poorly spread (ratio {div['axis_ratio']:.3f} < "
                          f"{th.axis_ratio_warn}) — add poses rotated about a different axis")
    else:
        rep.check("pass", f"rotation axes spread (ratio {div['axis_ratio']:.3f})")
    if div["max_angle_deg"] < th.min_rotation_deg:
        rep.check("warn", f"largest relative rotation is only {div['max_angle_deg']:.1f} deg "
                          f"(< {th.min_rotation_deg}) — small rotations make X's translation "
                          f"ill-conditioned")

    # Leave-one-out hold-out: does the solve GENERALIZE to a pose it never saw?
    # Refit on the other n-1 samples, build the base_T_board reference from those,
    # then measure how far the HELD-OUT sample's prediction lands from it.
    if n >= 6:
        worst_holdout = 0.0
        for i in range(n):
            idx = [k for k in range(n) if k != i]
            bg = [base_T_gripper[k] for k in idx]
            ct = [cam_T_target[k] for k in idx]
            try:
                X = solve_eye_in_hand(bg, ct)
            except Exception:
                continue
            ref = np.mean([(bg[j] @ X @ ct[j])[:3, 3] for j in range(len(bg))], axis=0)
            held = base_T_gripper[i] @ X @ cam_T_target[i]
            worst_holdout = max(worst_holdout, float(np.linalg.norm(held[:3, 3] - ref)))
        rep.metrics["holdout_worst_mm"] = round(worst_holdout * 1000, 3)
        if worst_holdout > th.holdout_warn_m:
            rep.check("warn", f"leave-one-out hold-out worst {worst_holdout*1000:.2f} mm "
                              f"> {th.holdout_warn_m*1000:.0f} mm — solve may not generalize / too few poses")

    # Cross-method spread — only among CONVERGED methods (some diverge by hundreds
    # of mm and would trip this check every time).
    if all_methods and len(all_methods) > 1:
        method_res = {
            name: consistency_residual(X, base_T_gripper, cam_T_target)["trans_m"]["mean"]
            for name, X in all_methods.items()
        }
        best = min(method_res.values())
        cap = max(3.0 * best, 0.02)               # within 3x the best, or 20 mm absolute
        converged = {k: all_methods[k] for k, r in method_res.items() if r <= cap}
        rep.metrics["crossmethod_converged"] = sorted(converged)
        if len(converged) >= 2:
            ts = np.array([X[:3, 3] for X in converged.values()])
            spread = float(np.linalg.norm(ts - ts.mean(axis=0), axis=1).max())
            rep.metrics["crossmethod_spread_mm"] = round(spread * 1000, 3)
            if spread > th.crossmethod_warn_m:
                rep.check("warn", f"converged methods disagree by {spread*1000:.1f} mm "
                                  f"> {th.crossmethod_warn_m*1000:.0f} mm — poorly conditioned pose set")
            else:
                rep.check("pass", f"{len(converged)} methods agree within {spread*1000:.2f} mm")
        else:
            rep.metrics["crossmethod_spread_mm"] = 0.0
            rep.check("pass", f"only {sorted(converged)} converged; cross-method check skipped")
    return rep


# ==========================================================================
# Persistence
# ==========================================================================
def save_eye_in_hand(path: str | Path, gripper_T_camera: np.ndarray) -> Path:
    """Save ``gripper_T_camera`` to ``<cam>.npy`` **and tag its frame**.

    The tag is what stops the runtime from silently reading this as a fixed
    camera's ``base_T_camera`` (see :mod:`calibration.frames`).
    """
    out = save_extrinsic(path, gripper_T_camera)
    save_frame_tag(path, FRAME_GRIPPER)
    return out


def save_eye_in_hand_record(
    path: str | Path,
    gripper_T_camera: np.ndarray,
    method: str,
    all_methods: dict[str, np.ndarray],
    residual: dict,
    num_samples: int,
    diversity: Optional[dict] = None,
) -> Path:
    """Write the full audit ``.calib.npz`` next to the ``.npy`` (all methods + residuals)."""
    out = resolve(path).with_suffix(".calib.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gripper_T_camera": np.asarray(gripper_T_camera, dtype=np.float64),
        "frame": FRAME_GRIPPER,
        "method": method,
        "num_samples": int(num_samples),
        "trans_residual_m": float(residual["trans_m"]["mean"]),
        "rot_residual_deg": float(residual["rot_deg"]["mean"]),
    }
    if diversity is not None:
        payload["rotation_axis_ratio"] = float(diversity["axis_ratio"])
        payload["max_relative_rotation_deg"] = float(diversity["max_angle_deg"])
    for name, X in all_methods.items():
        payload[f"X_{name}"] = np.asarray(X, dtype=np.float64)
    np.savez(out, **payload)
    return out


# ==========================================================================
# Capture session
# ==========================================================================
class EyeInHandSession(ExtrinsicCalibrationSession):
    """Eye-in-hand capture session.

    Reuses the eye-to-hand session's capture machinery unchanged — board
    detection, pose-diversity gating, per-sample persistence, delete-last,
    ``from_samples`` offline re-solve — because none of that depends on which
    geometry is being solved. Only :meth:`solve` differs.

    Samples written by either session are interchangeable on disk (both store
    ``base_T_gripper`` + ``cam_T_target``), so a set captured with the board on
    the flange can never be re-solved as eye-in-hand by accident — the difference
    is in where the board physically was, which no file can record. Keep the two
    cameras' sample directories separate (``cam_arm_samples`` vs
    ``cam_chest_samples``), which the default naming already does.
    """

    def solve(self, save_as: str | None = None, prefer: str = "PARK") -> np.ndarray:
        """Solve with all methods, pick the most consistent, optionally save it."""
        candidates = solve_eye_in_hand_all_methods(self.base_T_gripper, self.cam_T_target)
        method, X = select_best(candidates, self.base_T_gripper, self.cam_T_target, prefer=prefer)
        self.last_candidates = candidates
        self.last_method = method
        res = consistency_residual(X, self.base_T_gripper, self.cam_T_target)
        div = rotation_diversity(self.base_T_gripper)
        print(
            f"[eye-in-hand] {self.num_samples} samples | method={method} | "
            f"consistency: trans mean={res['trans_m']['mean'] * 1000:.2f} mm "
            f"(max {res['trans_m']['max'] * 1000:.2f}), "
            f"rot mean={res['rot_deg']['mean']:.3f}° (max {res['rot_deg']['max']:.3f})"
        )
        print(f"[eye-in-hand] pose conditioning: rotation-axis ratio {div['axis_ratio']:.3f} "
              f"(0 = all one axis), largest relative rotation {div['max_angle_deg']:.1f}°")
        for name, cand in candidates.items():
            tm = consistency_residual(cand, self.base_T_gripper, self.cam_T_target)["trans_m"]["mean"]
            print(f"    {name:<11} trans residual = {tm * 1000:.2f} mm")
        t = X[:3, 3] * 1000.0
        print(f"[eye-in-hand] camera sits at [{t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}] mm from the "
              f"TCP frame — sanity-check that against a tape measure before trusting it.")

        if save_as:
            path = save_eye_in_hand(save_as, X)
            save_eye_in_hand_record(save_as, X, method, candidates, res, self.num_samples, div)
            print(f"[eye-in-hand] saved gripper_T_camera -> {path} (frame tag: {FRAME_GRIPPER})")
        return X
