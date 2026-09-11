#!/usr/bin/env python3
"""Self-test for the eye-in-hand solver — SYNTHETIC data, no camera and no arm.

Run this before touching hardware. It builds a world where the right answer is
known exactly, and checks that :mod:`calibration.eye_in_hand` recovers it:

1. **Exact recovery.** With noiseless poses the solver must return the true
   ``gripper_T_camera`` to numerical precision. Anything else is a wiring bug
   (the classic one being an inverted or transposed pose convention).
2. **Behaviour under realistic noise.** Board detection and the arm's reported
   pose both carry error; this reports what that does to the solve, so the
   numbers coming off the real robot have something to be compared against.
3. **The degenerate pose set is REFUSED.** Rotations about a single axis leave
   the camera's translation along that axis unobservable — the solver still
   returns a tidy matrix, so the validator has to be the thing that catches it.
4. **The two geometries are not interchangeable.** Solving eye-in-hand data with
   the eye-to-hand routine yields a metre-scale error, which is why the extrinsic
   files carry a frame tag (:mod:`calibration.frames`).

Needs cv2 (so: on the robot, or any machine with OpenCV). No hardware.

    python3 initialization/selftest_eye_in_hand.py
"""

from __future__ import annotations

import numpy as np

from calibration.extrinsic import consistency_residual as eth_residual
from calibration.extrinsic import solve_eye_to_hand
from calibration.eye_in_hand import (
    consistency_residual,
    rotation_diversity,
    solve_eye_in_hand,
    solve_eye_in_hand_all_methods,
    validate_eye_in_hand,
)
from core.transforms import invert, make_transform, rot_angle_deg, rpy_to_matrix

RNG = np.random.default_rng(7)

# Ground truth: a camera bracketed off the TCP, and a board on a wall in front.
X_TRUE = make_transform(rpy_to_matrix(8.0, -4.0, 90.0), np.array([0.045, -0.020, 0.085]))
BASE_T_BOARD = make_transform(rpy_to_matrix(1.5, 88.0, 0.5), np.array([0.70, -0.18, 0.90]))


def _pose_set(n: int, single_axis: bool = False) -> list[np.ndarray]:
    """Plausible hand-guided poses. ``single_axis`` builds the degenerate case."""
    poses = []
    for _ in range(n):
        if single_axis:
            rpy = (0.0, 0.0, float(RNG.uniform(-70, 70)))     # yaw only
        else:
            rpy = tuple(RNG.uniform(-45, 45, 3))
        t = np.array([0.35, 0.0, 0.55]) + RNG.uniform(-0.12, 0.12, 3)
        poses.append(make_transform(rpy_to_matrix(*rpy), t))
    return poses


def _observations(grips, pos_noise_m=0.0, rot_noise_deg=0.0):
    """cam_T_target for each pose, optionally perturbed like a real detection."""
    cams = []
    for g in grips:
        ct = invert(g @ X_TRUE) @ BASE_T_BOARD
        if pos_noise_m or rot_noise_deg:
            noise = make_transform(
                rpy_to_matrix(*RNG.normal(0, rot_noise_deg, 3)),
                RNG.normal(0, pos_noise_m, 3),
            )
            ct = ct @ noise
        cams.append(ct)
    return cams


def _err(X: np.ndarray) -> tuple[float, float]:
    """(translation mm, rotation deg) of a solved X against the truth."""
    d = invert(X_TRUE) @ X
    return float(np.linalg.norm(d[:3, 3])) * 1000.0, rot_angle_deg(d[:3, :3])


def main() -> None:
    ok = True

    # --- 1. exact recovery -------------------------------------------------
    grips = _pose_set(18)
    cams = _observations(grips)
    X = solve_eye_in_hand(grips, cams)
    t_mm, r_deg = _err(X)
    print(f"1. noiseless recovery : {t_mm:.6f} mm, {r_deg:.6f} deg")
    if t_mm > 0.01 or r_deg > 0.001:
        print("   ✗ FAIL — the solver does not recover the known answer")
        ok = False
    else:
        print("   ✓ recovers gripper_T_camera exactly")

    all_m = solve_eye_in_hand_all_methods(grips, cams)
    spread = max(_err(Xm)[0] for Xm in all_m.values())
    print(f"   all methods {sorted(all_m)} agree to {spread:.4f} mm")

    # --- 2. realistic noise ------------------------------------------------
    # 0.5 mm / 0.1 deg board pose, which is the order of a good ChArUco solve at
    # this distance; the arm pose itself is left exact here so the two error
    # sources stay separable.
    cams_n = _observations(grips, pos_noise_m=0.0005, rot_noise_deg=0.1)
    Xn = solve_eye_in_hand(grips, cams_n)
    t_mm, r_deg = _err(Xn)
    res = consistency_residual(Xn, grips, cams_n)
    print(f"2. with 0.5 mm / 0.1 deg detection noise: X off by {t_mm:.2f} mm, {r_deg:.3f} deg; "
          f"base_T_board consistency {res['trans_m']['mean']*1000:.2f} mm")
    print("   (the consistency residual is what the real run prints — note it does NOT equal "
          "the error in X)")

    rep = validate_eye_in_hand(Xn, grips, cams_n, all_methods=solve_eye_in_hand_all_methods(grips, cams_n))
    print("   " + rep.summary().replace("\n", "\n   "))

    # --- 3. degenerate pose set must be refused ----------------------------
    dgrips = _pose_set(14, single_axis=True)
    dcams = _observations(dgrips, pos_noise_m=0.0005, rot_noise_deg=0.1)
    Xd = solve_eye_in_hand(dgrips, dcams)
    t_mm, _ = _err(Xd)
    div = rotation_diversity(dgrips)
    drep = validate_eye_in_hand(Xd, dgrips, dcams)
    print(f"3. single-axis pose set: X off by {t_mm:.1f} mm, axis ratio {div['axis_ratio']:.4f}, "
          f"validator says {drep.status.upper()}")
    if drep.status != "fail":
        print("   ✗ FAIL — a degenerate pose set must be refused, not warned about")
        ok = False
    else:
        print("   ✓ refused (the solve looks tidy; only the conditioning check sees it)")

    # --- 4. the geometries are not interchangeable -------------------------
    X_wrong = solve_eye_to_hand(grips, cams)         # wrong routine, same data
    wrong_res = eth_residual(X_wrong, grips, cams)["trans_m"]["mean"] * 1000.0
    right_res = consistency_residual(X, grips, cams)["trans_m"]["mean"] * 1000.0
    print(f"4. eye-to-hand routine on eye-in-hand data: residual {wrong_res:.1f} mm "
          f"vs {right_res:.3f} mm for the correct one")
    if wrong_res < 100.0:
        print("   ⚠ the two are closer than expected on this pose set — the frame tag still "
              "matters, but this check proved less than it should")
    else:
        print("   ✓ mixing them up is loud in the residual — and the frame tag stops it earlier")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
